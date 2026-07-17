"""Reproducible full-tensor, factor, and response-operator training."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from .data import PiezoDataset, create_or_load_splits, graph_cache_key, load_gmtnet_records
from .jarvis_dfpt import JarvisDFPTCache
from .model import AtomCoordinateResponsePotential, PiezoJet, model_from_config
from .pretraining_protocol import validate_inductive_checkpoint
from .project_config import load_project_config
from .projectors import translation_projector
from .metrics import response_tensor_skill
from .tensor_ops import (
    piezo_scale,
    piezo_voigt_to_cartesian,
)
from .data import RESPONSE_NORM_BOUNDS
from .elastic_dielectric_ops import elastic_voigt_to_cartesian
from .operator_losses import (
    born_charge_probe_loss,
    born_oracle_piezo_loss,
    internal_strain_probe_loss,
    ionic_elastic_response_loss,
    low_mode_operator_action_losses,
    mixed_force_constant_probe_loss,
    phi_oracle_normal_equation_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_config(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _response_bins(target: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=-1)
    positive_bins = 1 + torch.bucketize(norm, torch.tensor(RESPONSE_NORM_BOUNDS[1:], dtype=norm.dtype, device=norm.device), right=False)
    return torch.where(norm == 0, torch.zeros_like(positive_bins), positive_bins)


def response_bin_weights(target: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency, mean-one weights for invariant response strata."""
    bins = _response_bins(target)
    counts = torch.bincount(bins, minlength=5).to(dtype=target.dtype)
    weights = target.shape[0] / (counts.clamp_min(1.0) * 5.0)
    return weights


def full_loss(prediction: torch.Tensor, target: torch.Tensor, scale: torch.Tensor, bin_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Balanced, O(3)-invariant robust loss for zero-heavy long-tail tensors.

    Pseudo-Huber residuals stop a few extreme labels from dominating the
    gradient.  A stabilized relative term preserves pressure to fit genuinely
    responsive tensors, while zero labels remain explicit negative examples.
    Every weight depends only on the target Frobenius norm and therefore does
    not privilege a Cartesian component or break rotational equivariance.
    """
    residual_norm = torch.linalg.vector_norm((prediction - target).reshape(target.shape[0], -1), dim=-1)
    target_norm = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=-1)
    normalized = residual_norm / (scale * (18.0 ** 0.5)).clamp_min(torch.finfo(target.dtype).eps)
    pseudo_huber = torch.sqrt(1.0 + normalized.square()) - 1.0
    relative = (residual_norm / (target_norm + 0.5)).square()
    per_sample = 0.5 * pseudo_huber + 0.5 * relative
    if bin_weights is None:
        return per_sample.mean()
    weights = bin_weights[_response_bins(target)]
    return (weights * per_sample).sum() / weights.sum().clamp_min(torch.finfo(target.dtype).eps)


def invariant_pseudo_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    scale_floor: float = 0.0,
    sample_ndim: int = 0,
) -> torch.Tensor:
    """Rotation-invariant robust loss from whole-tensor Frobenius norms.

    The residual is reduced only after forming its invariant Cartesian norm;
    unlike componentwise SmoothL1 this value cannot change under a common
    orthogonal change of coordinates. ``scale_floor`` has the same units as a
    single Cartesian component and prevents near-zero labels from dominating.
    """
    if prediction.shape != target.shape:
        raise ValueError("Invariant loss requires matching prediction/target shapes")
    if sample_ndim not in {0, 1}:
        raise ValueError("sample_ndim must be 0 (one whole tensor) or 1 (leading sample axis)")
    samples = prediction.shape[0] if sample_ndim == 1 else 1
    residual = (prediction - target).reshape(samples, -1)
    targets = target.reshape(samples, -1)
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    target_norm = torch.linalg.vector_norm(targets, dim=-1)
    component_floor = prediction.new_tensor(scale_floor) * max(targets.shape[-1], 1) ** 0.5
    scale = torch.maximum(target_norm, component_floor).clamp_min(
        torch.finfo(target.dtype).eps
    )
    normalized = residual_norm / scale
    return (torch.sqrt(1.0 + normalized.square()) - 1.0).mean()


def dielectric_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Robust auxiliary supervision for the relaxed dielectric response."""
    mask = mask.reshape(-1).to(dtype=torch.bool)
    if not mask.any():
        return prediction.sum() * 0.0
    return invariant_pseudo_huber(prediction[mask], target[mask], sample_ndim=1)


def elastic_auxiliary_loss(prediction_gpa: torch.Tensor, target_gpa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Availability-masked elastic loss in complete Cartesian tensor space."""
    mask = mask.reshape(-1).to(dtype=torch.bool)
    if not mask.any():
        return prediction_gpa.sum() * 0.0
    return invariant_pseudo_huber(
        elastic_voigt_to_cartesian(prediction_gpa[mask]),
        elastic_voigt_to_cartesian(target_gpa[mask]),
        sample_ndim=1,
    )


def born_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    batch_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale-robust supervision of same-source, atom-resolved Born tensors."""
    mask = mask.reshape(-1).to(dtype=torch.bool)
    if not mask.any():
        return prediction.sum() * 0.0
    if batch_index is None:
        return invariant_pseudo_huber(prediction[mask], target[mask], sample_ndim=1)
    # A capacity/training batch is a set of crystals, not a set of atoms.
    # Reducing all selected atoms together changes material weighting with
    # batch composition and makes batch gradients differ from the mean of the
    # corresponding single-crystal gradients.  Project each material to its
    # BEC acoustic sum rule and reduce its atom-resolved Frobenius errors first.
    graphs = int(batch_index.max()) + 1
    target = target - scatter(target, batch_index, dim=0, dim_size=graphs, reduce="mean")[batch_index]
    losses = []
    for graph_index in range(graphs):
        graph_mask = mask & (batch_index == graph_index)
        if graph_mask.any():
            losses.append(
                invariant_pseudo_huber(
                    prediction[graph_mask], target[graph_mask], sample_ndim=1
                )
            )
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def macroscopic_piezo_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Robust physical loss for a masked JARVIS macroscopic piezo branch."""
    mask = mask.reshape(-1).to(dtype=torch.bool)
    if not mask.any():
        return prediction.sum() * 0.0
    # Centrosymmetric and near-cancelling crystals can have printed values at
    # 1e-5 C/m^2.  Dividing by that numerical residue would dominate every
    # other physical target, so 0.05 C/m^2 is the robust resolution floor.
    return invariant_pseudo_huber(
        prediction[mask], target[mask], scale_floor=0.05, sample_ndim=1
    )


def ionic_piezo_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Supervise the independent ``Z*^T U_eta`` ionic response."""
    return macroscopic_piezo_loss(prediction, target, mask)


def displacement_macro_piezo_loss(
    displacement_response: torch.Tensor,
    true_born: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    batch,
    response: AtomCoordinateResponsePotential,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isolate U_eta supervision using the observed Born charges on 610 rows."""
    prediction = response.ionic_piezo_from_displacement_response(
        true_born, displacement_response, batch
    )
    return macroscopic_piezo_loss(prediction, target, mask), prediction


def response_active_internal_strain_loss(
    prediction: torch.Tensor,
    true_born: torch.Tensor,
    true_force_constants_flat: torch.Tensor,
    target_ionic_piezo: torch.Tensor,
    mask: torch.Tensor,
    node_ptr: torch.Tensor,
    cells: torch.Tensor,
    response,
    force_constant_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise only the observable part of Lambda with true Z* and Phi.

    Macroscopic labels cannot identify every component of a ``3N x 6``
    strain-force tensor.  This oracle-isolated loss therefore propagates the
    predicted Lambda through the *true* DFPT Born charges and force constants,
    and compares ``Z*^T D_delta(Phi) Lambda`` with the OUTCAR ionic tensor.
    The continuous signed regularized operator is used deliberately: no
    predicted stability threshold participates in this supervision path.
    """
    mask = mask.reshape(-1).to(dtype=torch.bool)
    force_mask = mask if force_constant_mask is None else force_constant_mask.reshape(-1).to(dtype=torch.bool)
    losses, force_offset = [], 0
    for graph_index in range(node_ptr.numel() - 1):
        atoms = int(node_ptr[graph_index + 1] - node_ptr[graph_index])
        block_values = 9 * atoms * atoms
        blocks = None
        if bool(force_mask[graph_index]):
            blocks = true_force_constants_flat[
                force_offset : force_offset + block_values
            ].reshape(atoms, atoms, 3, 3)
            force_offset += block_values
        if not bool(mask[graph_index]):
            continue
        if blocks is None:
            raise ValueError("Ionic response supervision requires a matching force-constant label")
        start, stop = int(node_ptr[graph_index]), int(node_ptr[graph_index + 1])
        coupling = response._coupling_voigt(prediction[start:stop]).reshape(3 * atoms, 6)
        charge = true_born[start:stop].reshape(3 * atoms, 3)
        volume = torch.linalg.det(cells[graph_index]).abs().clamp_min(
            torch.finfo(cells.dtype).eps
        )
        predicted = response.PIEZO_C_PER_M2 * (
            charge.transpose(0, 1) @ response.apply_optical_operator(
                blocks, coupling, solve_policy="regularized"
            )
        ) / volume
        losses.append(
            invariant_pseudo_huber(
                piezo_voigt_to_cartesian(predicted),
                target_ionic_piezo[graph_index],
                scale_floor=0.05,
            )
        )
    if not losses:
        return prediction.sum() * 0.0
    if force_offset != true_force_constants_flat.numel():
        raise ValueError("Ragged force-constant labels did not match response-active mask")
    return torch.stack(losses).mean()


def displacement_response_target_loss(
    prediction: torch.Tensor,
    true_force_constants_flat: torch.Tensor,
    true_internal_strain: torch.Tensor,
    strict_mask: torch.Tensor,
    force_constant_mask: torch.Tensor,
    node_ptr: torch.Tensor,
    response: AtomCoordinateResponsePotential,
) -> torch.Tensor:
    """Supervise independent ``U_eta`` with true regularized DFPT factors.

    This label construction is explicit about its semantics: on all strata it
    is the displacement minimizing the current Tikhonov force-residual
    objective. Exact stationary propagation remains a stable-only diagnostic.
    No inverse involving predicted factors participates in this loss.
    """
    strict_mask = strict_mask.reshape(-1).to(dtype=torch.bool)
    force_constant_mask = force_constant_mask.reshape(-1).to(dtype=torch.bool)
    losses: list[torch.Tensor] = []
    force_offset = 0
    for graph_index in range(node_ptr.numel() - 1):
        start, stop = int(node_ptr[graph_index]), int(node_ptr[graph_index + 1])
        atoms = stop - start
        values = 9 * atoms * atoms
        blocks = None
        if bool(force_constant_mask[graph_index]):
            blocks = true_force_constants_flat[
                force_offset : force_offset + values
            ].reshape(atoms, atoms, 3, 3)
            force_offset += values
        if not bool(strict_mask[graph_index]):
            continue
        if blocks is None:
            raise ValueError("A strict U_eta target requires true force constants")
        target_lambda = response._coupling_voigt(
            true_internal_strain[start:stop]
        ).reshape(3 * atoms, 6)
        target_u = response.apply_optical_operator(
            blocks, target_lambda, solve_policy="regularized"
        )
        predicted_u = response._coupling_voigt(
            prediction[start:stop]
        ).reshape(3 * atoms, 6)
        losses.append(
            invariant_pseudo_huber(predicted_u, target_u, scale_floor=1e-4)
        )
    if force_offset != true_force_constants_flat.numel():
        raise ValueError("Ragged force constants did not match U_eta target masks")
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def displacement_first_order_block_loss(
    displacement_response: torch.Tensor,
    auxiliary_response: torch.Tensor,
    force_constants_flat: torch.Tensor,
    internal_strain: torch.Tensor,
    graph_mask: torch.Tensor,
    force_constant_mask: torch.Tensor,
    node_ptr: torch.Tensor,
    response: AtomCoordinateResponsePotential,
) -> torch.Tensor:
    """First-order real block residual for the signed regularized response.

    Writing ``(Phi + i delta I)(U + i V) = Lambda`` gives
    ``Phi U - delta V = Lambda`` and ``Phi V + delta U = 0``.  Unlike the
    historical squared normal equation, this system does not square the
    spectral condition number or overweight hard modes by ``lambda^4``.
    """
    graph_mask = graph_mask.reshape(-1).to(dtype=torch.bool)
    force_constant_mask = force_constant_mask.reshape(-1).to(dtype=torch.bool)
    losses: list[torch.Tensor] = []
    force_offset = 0
    delta = response.optical_regularization
    for graph_index in range(node_ptr.numel() - 1):
        start, stop = int(node_ptr[graph_index]), int(node_ptr[graph_index + 1])
        atoms = stop - start
        values = 9 * atoms * atoms
        blocks = None
        if bool(force_constant_mask[graph_index]):
            blocks = force_constants_flat[
                force_offset : force_offset + values
            ].reshape(atoms, atoms, 3, 3)
            force_offset += values
        if not bool(graph_mask[graph_index]):
            continue
        if blocks is None:
            raise ValueError("First-order U/V consistency requires true Phi")
        matrix = response._matrix_from_blocks(blocks)
        u = response._coupling_voigt(
            displacement_response[start:stop]
        ).reshape(3 * atoms, 6)
        v = response._coupling_voigt(
            auxiliary_response[start:stop]
        ).reshape(3 * atoms, 6)
        coupling = response._coupling_voigt(
            internal_strain[start:stop]
        ).reshape(3 * atoms, 6)
        real_residual = matrix @ u - delta * v - coupling
        imaginary_residual = matrix @ v + delta * u
        residual = torch.cat((real_residual, imaginary_residual), dim=-1)
        target_scale = torch.linalg.vector_norm(coupling).clamp_min(1e-6)
        relative = torch.linalg.vector_norm(residual) / target_scale
        losses.append(torch.sqrt(1.0 + relative.square()) - 1.0)
    if force_offset != force_constants_flat.numel():
        raise ValueError("True force constants did not match first-order masks")
    return torch.stack(losses).mean() if losses else (
        displacement_response.sum() + auxiliary_response.sum()
    ) * 0.0


def displacement_consistency_weight_for_epoch(
    base_weight: float,
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    """Warm up and linearly ramp the first-order U/V consistency residual."""
    if base_weight < 0.0 or warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError("Consistency schedule parameters must be non-negative")
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return float(base_weight)
    fraction = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return float(base_weight) * fraction


def first_order_spectral_residual_diagnostics(
    displacement_response: torch.Tensor,
    auxiliary_response: torch.Tensor,
    predicted_force_constants_flat: torch.Tensor,
    predicted_internal_strain: torch.Tensor,
    true_force_constants_flat: torch.Tensor,
    strict_mask: torch.Tensor,
    force_constant_mask: torch.Tensor,
    node_ptr: torch.Tensor,
    response: AtomCoordinateResponsePotential,
) -> dict[str, float | int]:
    """Resolve the first-order U/V block residual in true optical modes.

    The diagnostic is read-only and does not alter the registered loss.  True
    ``|lambda|`` defines four fixed regions so a hard-mode-dominated residual
    cannot be hidden by the whole-tensor reduction.
    """
    strict_mask = strict_mask.reshape(-1).to(dtype=torch.bool)
    force_constant_mask = force_constant_mask.reshape(-1).to(dtype=torch.bool)
    delta = response.optical_regularization
    region_names = (
        "below_delta",
        "delta_to_3delta",
        "3delta_to_10delta",
        "above_10delta",
    )
    energy = {name: 0.0 for name in region_names}
    counts = {name: 0 for name in region_names}
    predicted_offset = 0
    true_offset = 0
    materials = 0
    for graph_index in range(node_ptr.numel() - 1):
        start, stop = int(node_ptr[graph_index]), int(node_ptr[graph_index + 1])
        atoms = stop - start
        values = 9 * atoms * atoms
        predicted_blocks = predicted_force_constants_flat[
            predicted_offset : predicted_offset + values
        ].reshape(atoms, atoms, 3, 3)
        predicted_offset += values
        true_blocks = None
        if bool(force_constant_mask[graph_index]):
            true_blocks = true_force_constants_flat[
                true_offset : true_offset + values
            ].reshape(atoms, atoms, 3, 3)
            true_offset += values
        if not bool(strict_mask[graph_index]):
            continue
        if true_blocks is None:
            raise ValueError("A strict spectral diagnostic requires true force constants")
        materials += 1
        predicted_matrix = response._matrix_from_blocks(predicted_blocks.detach()).to(
            torch.float64
        )
        true_matrix = response._matrix_from_blocks(true_blocks.detach()).to(torch.float64)
        u = response._coupling_voigt(
            displacement_response[start:stop].detach()
        ).reshape(3 * atoms, 6).to(torch.float64)
        v = response._coupling_voigt(
            auxiliary_response[start:stop].detach()
        ).reshape(3 * atoms, 6).to(torch.float64)
        coupling = response._coupling_voigt(
            predicted_internal_strain[start:stop].detach()
        ).reshape(3 * atoms, 6).to(torch.float64)
        residual_real = predicted_matrix @ u - delta * v - coupling
        residual_imaginary = predicted_matrix @ v + delta * u
        basis = response._optical_basis(atoms, true_matrix)
        if basis.shape[1] == 0:
            continue
        eigenvalues, reduced_vectors = torch.linalg.eigh(basis.T @ true_matrix @ basis)
        optical_vectors = basis @ reduced_vectors
        per_mode_energy = (
            (optical_vectors.T @ residual_real).square().sum(dim=1)
            + (optical_vectors.T @ residual_imaginary).square().sum(dim=1)
        )
        absolute = eigenvalues.abs()
        masks = {
            "below_delta": absolute < delta,
            "delta_to_3delta": (absolute >= delta) & (absolute < 3.0 * delta),
            "3delta_to_10delta": (absolute >= 3.0 * delta) & (absolute < 10.0 * delta),
            "above_10delta": absolute >= 10.0 * delta,
        }
        for name, mask in masks.items():
            counts[name] += int(mask.sum())
            energy[name] += float(per_mode_energy[mask].sum())
    if predicted_offset != predicted_force_constants_flat.numel():
        raise ValueError("Predicted force constants did not match graph boundaries")
    if true_offset != true_force_constants_flat.numel():
        raise ValueError("True force constants did not match graph masks")
    total_energy = sum(energy.values())
    return {
        "materials": materials,
        "total_optical_residual_squared": total_energy,
        **{f"{name}_mode_count": counts[name] for name in region_names},
        **{f"{name}_residual_squared": energy[name] for name in region_names},
        **{
            f"{name}_residual_fraction": energy[name] / max(total_energy, 1e-30)
            for name in region_names
        },
    }


def _parameter_gradient_norm(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    squared = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.square().sum()
    return squared.sqrt()


def _paired_parameter_gradient_metrics(
    first_loss: torch.Tensor,
    second_loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> dict[str, float]:
    """Measure two task gradients on one parameter group without mutation."""
    first_gradients = torch.autograd.grad(
        first_loss, parameters, retain_graph=True, allow_unused=True
    )
    second_gradients = torch.autograd.grad(
        second_loss, parameters, retain_graph=True, allow_unused=True
    )
    first_squared = first_loss.new_zeros(())
    second_squared = first_loss.new_zeros(())
    dot = first_loss.new_zeros(())
    for parameter, first, second in zip(parameters, first_gradients, second_gradients):
        first_value = torch.zeros_like(parameter) if first is None else first
        second_value = torch.zeros_like(parameter) if second is None else second
        first_squared = first_squared + first_value.square().sum()
        second_squared = second_squared + second_value.square().sum()
        dot = dot + (first_value * second_value).sum()
    first_norm = first_squared.sqrt()
    second_norm = second_squared.sqrt()
    epsilon = torch.finfo(first_norm.dtype).eps
    cosine = dot / (first_norm * second_norm).clamp_min(epsilon)
    return {
        "direct_u_gradient_norm": float(first_norm.detach()),
        "true_born_ionic_gradient_norm": float(second_norm.detach()),
        "gradient_cosine": float(cosine.detach()),
    }


def _unique_trainable_parameters(*modules: torch.nn.Module) -> tuple[torch.nn.Parameter, ...]:
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return tuple(parameters)


def _displacement_core_parameters(model: PiezoJet) -> tuple[torch.nn.Parameter, ...]:
    """Parameters trained by the teacher-U stage, in checkpoint order."""
    return _unique_trainable_parameters(
        model.displacement_encoder,
        model.displacement_local_polar,
        model.displacement_global_context,
        model.displacement_response_head,
    )


def _displacement_auxiliary_parameters(model: PiezoJet) -> tuple[torch.nn.Parameter, ...]:
    return _unique_trainable_parameters(model.displacement_auxiliary_head)


def _joint_optimizer(
    model: PiezoJet,
    cfg: dict,
    *,
    displacement_optimizer_state: dict | None,
) -> tuple[torch.optim.AdamW, bool]:
    """Build joint AdamW while optionally retaining teacher-U moments.

    Loading a whole teacher optimizer into a newly constructed all-model
    optimizer is invalid because the parameter groups differ.  We instead
    recreate its exact U-only group, restore that state, then append the new
    auxiliary-V and non-U groups.  This preserves moments without coupling the
    independent macro/factor towers to teacher-stage history.
    """
    core = _displacement_core_parameters(model)
    auxiliary = _displacement_auxiliary_parameters(model)
    displacement_ids = {id(parameter) for parameter in core + auxiliary}
    other = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in displacement_ids
    )
    learning_rate = float(cfg["learning_rate"])
    displacement_learning_rate = float(
        cfg.get("joint_displacement_learning_rate", learning_rate)
    )
    weight_decay = float(cfg["weight_decay"])
    preserve = bool(cfg.get("preserve_displacement_optimizer_state", False))
    restored = preserve and displacement_optimizer_state is not None
    if restored:
        optimizer = torch.optim.AdamW(
            core,
            lr=float(cfg.get("displacement_pretrain_learning_rate", displacement_learning_rate)),
            weight_decay=weight_decay,
        )
        optimizer.load_state_dict(displacement_optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = displacement_learning_rate
            group["weight_decay"] = weight_decay
        if auxiliary:
            optimizer.add_param_group(
                {"params": auxiliary, "lr": displacement_learning_rate, "weight_decay": weight_decay}
            )
        if other:
            optimizer.add_param_group(
                {"params": other, "lr": learning_rate, "weight_decay": weight_decay}
            )
        return optimizer, True
    groups: list[dict] = []
    if core or auxiliary:
        groups.append(
            {
                "params": core + auxiliary,
                "lr": displacement_learning_rate,
                "weight_decay": weight_decay,
            }
        )
    if other:
        groups.append({"params": other, "lr": learning_rate, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups), False


def _optimizer_for_resume(
    model: PiezoJet,
    cfg: dict,
    optimizer_state: dict,
) -> torch.optim.AdamW:
    """Recreate the exact saved parameter-group topology before loading AdamW."""
    core = _displacement_core_parameters(model)
    auxiliary = _displacement_auxiliary_parameters(model)
    displacement_ids = {id(parameter) for parameter in core + auxiliary}
    other = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in displacement_ids
    )
    group_count = len(optimizer_state.get("param_groups", []))
    if group_count == 1:
        groups = [tuple(parameter for parameter in model.parameters() if parameter.requires_grad)]
    elif group_count == 2:
        groups = [core + auxiliary, other]
    elif group_count == 3:
        groups = [core, auxiliary, other]
    else:
        raise ValueError(f"Unsupported saved optimizer group count: {group_count}")
    expected = [len(group["params"]) for group in optimizer_state["param_groups"]]
    observed = [len(group) for group in groups]
    if observed != expected:
        raise ValueError(
            "Saved optimizer parameter groups do not match the current model: "
            f"expected {expected}, observed {observed}"
        )
    optimizer = torch.optim.AdamW(
        [{"params": group} for group in groups],
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    optimizer.load_state_dict(optimizer_state)
    return optimizer


def _read_metric_rows(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed: dict[str, float | int] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                parsed[key] = int(float(value)) if key == "epoch" else float(value)
            rows.append(parsed)
    return rows


def force_constant_loss(
    prediction_flat: torch.Tensor,
    target_flat: torch.Tensor,
    node_ptr: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Supervise physical Hessians after cleaning numerical ASR violations."""
    mask = mask.reshape(-1).to(dtype=torch.bool)
    losses, prediction_offset, target_offset = [], 0, 0
    for graph_index in range(node_ptr.numel() - 1):
        atoms = int(node_ptr[graph_index + 1] - node_ptr[graph_index])
        values = 9 * atoms * atoms
        predicted = prediction_flat[prediction_offset : prediction_offset + values].reshape(atoms, atoms, 3, 3)
        prediction_offset += values
        if not bool(mask[graph_index]):
            continue
        target = target_flat[target_offset : target_offset + values].reshape(atoms, atoms, 3, 3)
        target_offset += values
        target_matrix = target.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
        target_matrix = 0.5 * (target_matrix + target_matrix.transpose(0, 1))
        projector, _ = translation_projector(atoms, target_matrix)
        target_matrix = projector @ target_matrix @ projector
        cleaned = target_matrix.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)
        losses.append(invariant_pseudo_huber(predicted, cleaned))
    if not losses:
        return prediction_flat.sum() * 0.0
    if target_offset != target_flat.numel():
        raise ValueError("Ragged force-constant labels did not match graph sizes")
    return torch.stack(losses).mean()


def soft_optical_eigenvalue_loss(
    prediction_flat: torch.Tensor,
    target_flat: torch.Tensor,
    node_ptr: torch.Tensor,
    mask: torch.Tensor,
    mode_count: int = 3,
) -> torch.Tensor:
    """Directly supervise the response-dominant lowest optical eigenvalues."""
    mask = mask.reshape(-1).to(dtype=torch.bool)
    losses, prediction_offset, target_offset = [], 0, 0
    for graph_index in range(node_ptr.numel() - 1):
        atoms = int(node_ptr[graph_index + 1] - node_ptr[graph_index])
        values = 9 * atoms * atoms
        predicted = prediction_flat[prediction_offset : prediction_offset + values]
        prediction_offset += values
        if not bool(mask[graph_index]):
            continue
        target = target_flat[target_offset : target_offset + values]
        target_offset += values
        predicted = predicted.reshape(atoms, atoms, 3, 3).permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
        target = target.reshape(atoms, atoms, 3, 3).permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
        predicted = 0.5 * (predicted + predicted.T)
        target = 0.5 * (target + target.T)
        basis = AtomCoordinateResponsePotential._optical_basis(atoms, target)
        if basis.shape[1] == 0:
            continue
        predicted_values = torch.linalg.eigvalsh(basis.T @ predicted @ basis)
        target_values = torch.linalg.eigvalsh(basis.T @ target @ basis)
        count = min(mode_count, predicted_values.numel(), target_values.numel())
        if count:
            scale = target_values.abs().mean().clamp_min(0.1)
            losses.append(
                torch.nn.functional.smooth_l1_loss(
                    predicted_values[:count] / scale,
                    target_values[:count] / scale,
                )
            )
    if not losses:
        return prediction_flat.sum() * 0.0
    if target_offset != target_flat.numel():
        raise ValueError("Ragged force-constant labels did not match graph sizes")
    return torch.stack(losses).mean()


def internal_strain_loss(
    prediction: torch.Tensor,
    target_flat: torch.Tensor,
    ions: torch.Tensor,
    directions: torch.Tensor,
    counts: torch.Tensor,
    node_ptr: torch.Tensor,
) -> torch.Tensor:
    """Supervise only VASP-printed internal-strain blocks.

    The public OUTCAR omits symmetry-equivalent perturbations.  Their values
    are intentionally not fabricated; observed 3x3 strain blocks are merely
    symmetrized to remove small finite-difference/print noise.
    """
    target_blocks = target_flat.reshape(-1, 3, 3)
    losses, offset = [], 0
    for graph_index, count_value in enumerate(counts.reshape(-1)):
        count = int(count_value)
        if count == 0:
            continue
        selected = prediction[
            node_ptr[graph_index] + ions[offset : offset + count],
            directions[offset : offset + count],
        ]
        target = target_blocks[offset : offset + count]
        target = 0.5 * (target + target.transpose(-1, -2))
        losses.append(invariant_pseudo_huber(selected, target, sample_ndim=1))
        offset += count
    if not losses:
        return prediction.sum() * 0.0
    if offset != target_blocks.shape[0]:
        raise ValueError("Ragged internal-strain labels did not match block counts")
    return torch.stack(losses).mean()


def full_internal_strain_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    graph_mask: torch.Tensor,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    """Supervise only strictly audited, symmetry-completed Lambda tensors."""
    graph_mask = graph_mask.reshape(-1).to(dtype=torch.bool)
    if not graph_mask.any():
        return prediction.sum() * 0.0
    losses = []
    for graph_index in torch.nonzero(graph_mask, as_tuple=False).reshape(-1):
        node_mask = batch_index == graph_index
        losses.append(invariant_pseudo_huber(prediction[node_mask], target[node_mask]))
    return torch.stack(losses).mean()


def _epoch(
    model,
    loader,
    optimizer,
    scale: torch.Tensor,
    bin_weights: torch.Tensor,
    device: torch.device,
    *,
    macro_weight: float = 1.0,
    macro_dielectric_weight: float = 0.0,
    macro_elastic_weight: float = 0.0,
    dielectric_weight: float = 0.0,
    ionic_dielectric_weight: float = 0.0,
    ionic_elastic_weight: float = 0.0,
    elastic_weight: float = 0.0,
    born_weight: float = 0.0,
    ionic_weight: float = 0.0,
    displacement_response_weight: float = 0.0,
    displacement_consistency_weight: float = 0.0,
    max_consistency_gradient_ratio: float = 1.0,
    electronic_weight: float = 0.0,
    branch_sum_weight: float = 0.0,
    force_weight: float = 0.0,
    internal_strain_weight: float = 0.0,
    internal_strain_full_weight: float = 0.0,
    soft_mode_weight: float = 0.0,
    low_mode_action_weight: float = 0.0,
    low_mode_leak_weight: float = 0.0,
    phi_probe_weight: float = 0.0,
    lambda_probe_weight: float = 0.0,
    born_probe_weight: float = 0.0,
    born_oracle_weight: float = 0.0,
    phi_oracle_normal_weight: float = 0.0,
    response_active_strain_weight: float = 0.0,
    max_train_updates: int | None = None,
    collect_conditioning_diagnostics: bool = False,
    collect_u_gradient_diagnostics: bool = False,
) -> tuple[float, float, dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    evaluate_all_components = not training
    compute_macro_response = evaluate_all_components or any(
        weight != 0.0
        for weight in (macro_weight, macro_dielectric_weight, macro_elastic_weight)
    )
    compute_factorized_response = evaluate_all_components or any(
        weight != 0.0
        for weight in (
            dielectric_weight,
            ionic_dielectric_weight,
            ionic_elastic_weight,
            elastic_weight,
        )
    )
    total, count, elapsed = 0.0, 0, 0.0
    response_predictions, response_targets = [], []
    component_totals = {
        "piezo_full": 0.0,
        "dielectric": 0.0,
        "macro_dielectric": 0.0,
        "macro_elastic": 0.0,
        "ionic_dielectric": 0.0,
        "ionic_elastic": 0.0,
        "elastic_auxiliary": 0.0,
        "born": 0.0,
        "force_constant": 0.0,
        "soft_optical": 0.0,
        "low_mode_action": 0.0,
        "low_mode_leak": 0.0,
        "phi_probe": 0.0,
        "lambda_probe": 0.0,
        "born_probe": 0.0,
        "born_oracle": 0.0,
        "phi_oracle_normal": 0.0,
        "internal_strain": 0.0,
        "internal_strain_full": 0.0,
        "response_active_strain": 0.0,
        "ionic_piezo": 0.0,
        "factorized_ionic_piezo": 0.0,
        "displacement_response": 0.0,
        "displacement_first_order_consistency": 0.0,
        "electronic_piezo": 0.0,
        "branch_sum": 0.0,
    }
    conditioning_totals = {
        "materials": 0,
        "consistency_u_head_gradient_norm_ratio_sum": 0.0,
        "weighted_consistency_u_head_gradient_norm_ratio_sum": 0.0,
        "total_optical_residual_squared": 0.0,
        "u_gradient_materials": 0,
        "direct_u_gradient_norm_sum": 0.0,
        "true_born_ionic_gradient_norm_sum": 0.0,
        "direct_u_true_born_ionic_gradient_cosine_sum": 0.0,
        "branch_sum_gradient_norm_sum": 0.0,
        "direct_u_branch_sum_gradient_cosine_sum": 0.0,
        "combined_branch_response_gradient_norm_sum": 0.0,
        "direct_u_combined_branch_response_gradient_cosine_sum": 0.0,
        **{
            f"{name}_mode_count": 0
            for name in (
                "below_delta",
                "delta_to_3delta",
                "3delta_to_10delta",
                "above_10delta",
            )
        },
        **{
            f"{name}_residual_squared": 0.0
            for name in (
                "below_delta",
                "delta_to_3delta",
                "3delta_to_10delta",
                "above_10delta",
            )
        },
    }
    updates = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            components = model.predict_components(
                batch,
                compute_macro_response=compute_macro_response,
                compute_factorized_response=compute_factorized_response,
            )
            prediction = components.tensor
            # A constant zero keeps inactive towers out of autograd entirely.
            # Using ``prediction.sum() * 0`` would create explicit zero
            # gradients, causing AdamW to apply weight decay to parameters
            # that this data stream is meant to leave untouched.
            zero = torch.zeros(
                (), dtype=components.physical_tensor.dtype,
                device=components.physical_tensor.device,
            )
            full = (
                full_loss(prediction, batch.y, scale, bin_weights)
                if compute_macro_response else zero
            )
            dielectric_component = (
                dielectric_loss(
                    components.dielectric, batch.y_dielectric,
                    batch.dielectric_mask,
                )
                if evaluate_all_components or dielectric_weight != 0.0 else zero
            )
            macro_dielectric_component = (
                dielectric_loss(
                    components.macro_dielectric, batch.y_dielectric,
                    batch.dielectric_mask,
                )
                if compute_macro_response else zero
            )
            macro_elastic_component = (
                elastic_auxiliary_loss(
                    components.macro_elastic, batch.y_elastic_gpa,
                    batch.elastic_mask,
                )
                if compute_macro_response else zero
            )
            ionic_dielectric_component = (
                dielectric_loss(
                    components.ionic_dielectric,
                    batch.y_dfpt_ionic_dielectric,
                    batch.dfpt_ionic_dielectric_mask,
                )
                if evaluate_all_components or ionic_dielectric_weight != 0.0
                else zero
            )
            ionic_elastic_component = (
                ionic_elastic_response_loss(
                    components.elastic_softening,
                    batch.dfpt_force_constants_flat,
                    batch.dfpt_internal_strain_full,
                    batch.internal_strain_full_mask,
                    batch.force_constant_mask,
                    batch.ptr,
                    batch.cell.reshape(-1, 3, 3),
                    model.response,
                )
                if ionic_elastic_weight != 0.0
                else zero
            )
            elastic_component = (
                elastic_auxiliary_loss(
                    components.elastic, batch.y_elastic_gpa,
                    batch.elastic_mask,
                )
                if evaluate_all_components or elastic_weight != 0.0 else zero
            )
            born_component = (
                born_loss(
                    components.born_charges, batch.y_born,
                    batch.born_mask, batch.batch,
                )
                if evaluate_all_components or born_weight != 0.0 else zero
            )
            need_ionic = (
                evaluate_all_components or ionic_weight != 0.0
                or branch_sum_weight != 0.0 or collect_u_gradient_diagnostics
            )
            if need_ionic:
                ionic_component, supervised_ionic_prediction = displacement_macro_piezo_loss(
                    components.displacement_response,
                    batch.y_born,
                    batch.y_ionic_piezo,
                    batch.ionic_piezo_mask,
                    batch,
                    model.response,
                )
            else:
                ionic_component = zero
                supervised_ionic_prediction = components.ionic_piezo * 0.0
            electronic_component = (
                macroscopic_piezo_loss(
                    components.electronic_piezo,
                    batch.y_electronic_piezo,
                    batch.dfpt_branch_mask,
                )
                if evaluate_all_components or electronic_weight != 0.0
                or branch_sum_weight != 0.0 or collect_u_gradient_diagnostics
                else zero
            )
            branch_sum_component = (
                macroscopic_piezo_loss(
                    components.electronic_piezo + supervised_ionic_prediction,
                    batch.y_dfpt_total_piezo,
                    batch.dfpt_branch_mask,
                )
                if evaluate_all_components or branch_sum_weight != 0.0
                or collect_u_gradient_diagnostics else zero
            )
            force_component = (
                force_constant_loss(
                    components.force_constants_flat,
                    batch.dfpt_force_constants_flat, batch.ptr,
                    batch.force_constant_mask,
                )
                if evaluate_all_components or force_weight != 0.0 else zero
            )
            soft_component = (
                soft_optical_eigenvalue_loss(
                    components.force_constants_flat,
                    batch.dfpt_force_constants_flat, batch.ptr,
                    batch.force_constant_mask,
                )
                if evaluate_all_components or soft_mode_weight != 0.0 else zero
            )
            low_action_component, low_leak_component = (
                low_mode_operator_action_losses(
                    components.force_constants_flat,
                    batch.dfpt_force_constants_flat,
                    batch.ptr,
                    batch.force_constant_mask,
                    mode_count=int(getattr(model, "operator_low_mode_count", 6)),
                )
                if low_mode_action_weight != 0.0 or low_mode_leak_weight != 0.0
                else (zero, zero)
            )
            phi_probe_component = (
                mixed_force_constant_probe_loss(
                    components.force_constants_flat,
                    batch.dfpt_force_constants_flat,
                    batch.dfpt_internal_strain_full,
                    batch.ptr,
                    batch.force_constant_mask,
                    batch.internal_strain_full_mask,
                    model.response,
                    material_ids=batch.material_id,
                )
                if phi_probe_weight != 0.0
                else zero
            )
            internal_component = (
                internal_strain_loss(
                    components.internal_strain, batch.dfpt_internal_strain_flat,
                    batch.dfpt_internal_strain_ions,
                    batch.dfpt_internal_strain_directions,
                    batch.dfpt_internal_strain_count, batch.ptr,
                )
                if evaluate_all_components or internal_strain_weight != 0.0
                else zero
            )
            full_internal_component = (
                full_internal_strain_loss(
                    components.internal_strain,
                    batch.dfpt_internal_strain_full,
                    batch.internal_strain_full_mask,
                    batch.batch,
                )
                if evaluate_all_components or internal_strain_full_weight != 0.0
                else zero
            )
            lambda_probe_component = (
                internal_strain_probe_loss(
                    components.internal_strain,
                    batch.dfpt_internal_strain_full,
                    batch.internal_strain_full_mask,
                    batch.batch,
                    material_ids=batch.material_id,
                )
                if lambda_probe_weight != 0.0
                else zero
            )
            born_probe_component = (
                born_charge_probe_loss(
                    components.born_charges,
                    batch.y_born,
                    batch.born_mask,
                    batch.batch,
                    material_ids=batch.material_id,
                )
                if born_probe_weight != 0.0
                else zero
            )
            oracle_mask = batch.internal_strain_full_mask.reshape(-1) & batch.ionic_piezo_mask.reshape(-1)
            born_oracle_component = (
                born_oracle_piezo_loss(
                    components.born_charges,
                    batch.dfpt_force_constants_flat,
                    batch.dfpt_internal_strain_full,
                    batch.y_ionic_piezo,
                    oracle_mask,
                    batch.force_constant_mask,
                    batch.ptr,
                    batch.cell.reshape(-1, 3, 3),
                    model.response,
                )
                if born_oracle_weight != 0.0
                else zero
            )
            phi_oracle_normal_component = (
                phi_oracle_normal_equation_loss(
                    components.force_constants_flat,
                    batch.dfpt_force_constants_flat,
                    batch.dfpt_internal_strain_full,
                    batch.internal_strain_full_mask,
                    batch.force_constant_mask,
                    batch.ptr,
                    model.response,
                )
                if phi_oracle_normal_weight != 0.0
                else zero
            )
            displacement_component = (
                displacement_response_target_loss(
                    components.displacement_response,
                    batch.dfpt_force_constants_flat,
                    batch.dfpt_internal_strain_full,
                    batch.internal_strain_full_mask,
                    batch.force_constant_mask,
                    batch.ptr,
                    model.response,
                )
                if evaluate_all_components
                or displacement_response_weight != 0.0
                or displacement_consistency_weight != 0.0
                else zero
            )
            if displacement_consistency_weight != 0.0:
                _, auxiliary_displacement = model.predict_displacement_block_response(
                    batch
                )
                displacement_consistency_component = displacement_first_order_block_loss(
                    components.displacement_response,
                    auxiliary_displacement,
                    batch.dfpt_force_constants_flat,
                    batch.dfpt_internal_strain_full,
                    batch.internal_strain_full_mask,
                    batch.force_constant_mask,
                    batch.ptr,
                    model.response,
                )
            else:
                auxiliary_displacement = components.displacement_response * 0.0
                displacement_consistency_component = zero
            active_strain_component = (
                response_active_internal_strain_loss(
                    components.internal_strain,
                    batch.y_born,
                    batch.dfpt_force_constants_flat,
                    batch.y_ionic_piezo,
                    batch.ionic_piezo_mask,
                    batch.ptr,
                    batch.cell.reshape(-1, 3, 3),
                    model.response,
                    batch.force_constant_mask,
                )
                if evaluate_all_components
                or response_active_strain_weight != 0.0 else zero
            )
            effective_consistency_weight = displacement_consistency_weight
            strict_materials = int(
                batch.internal_strain_full_mask.reshape(-1).sum()
            )
            direct_norm = components.displacement_response.new_zeros(())
            consistency_norm = components.displacement_response.new_zeros(())
            if (
                training
                and strict_materials
                and displacement_consistency_weight != 0.0
            ):
                head_parameters = tuple(model.displacement_response_head.parameters())
                direct_norm = _parameter_gradient_norm(
                    displacement_component, head_parameters
                )
                consistency_norm = _parameter_gradient_norm(
                    displacement_consistency_component, head_parameters
                )
                epsilon = torch.finfo(direct_norm.dtype).eps
                cap = (
                    max_consistency_gradient_ratio
                    * displacement_response_weight
                    * float(direct_norm.detach())
                    / max(float(consistency_norm.detach()), float(epsilon))
                )
                effective_consistency_weight = min(
                    displacement_consistency_weight, cap
                )
            gradient_materials = int(
                (
                    batch.internal_strain_full_mask.reshape(-1).to(dtype=torch.bool)
                    & batch.ionic_piezo_mask.reshape(-1).to(dtype=torch.bool)
                ).sum()
            )
            if training and collect_u_gradient_diagnostics and gradient_materials:
                gradient_metrics = _paired_parameter_gradient_metrics(
                    displacement_component,
                    ionic_component,
                    _displacement_core_parameters(model),
                )
                conditioning_totals["u_gradient_materials"] += gradient_materials
                conditioning_totals["direct_u_gradient_norm_sum"] += (
                    gradient_metrics["direct_u_gradient_norm"] * gradient_materials
                )
                conditioning_totals["true_born_ionic_gradient_norm_sum"] += (
                    gradient_metrics["true_born_ionic_gradient_norm"] * gradient_materials
                )
                conditioning_totals[
                    "direct_u_true_born_ionic_gradient_cosine_sum"
                ] += gradient_metrics["gradient_cosine"] * gradient_materials
                branch_sum_metrics = _paired_parameter_gradient_metrics(
                    displacement_component,
                    branch_sum_component,
                    _displacement_core_parameters(model),
                )
                conditioning_totals["branch_sum_gradient_norm_sum"] += (
                    branch_sum_metrics["true_born_ionic_gradient_norm"]
                    * gradient_materials
                )
                conditioning_totals[
                    "direct_u_branch_sum_gradient_cosine_sum"
                ] += branch_sum_metrics["gradient_cosine"] * gradient_materials
                combined_metrics = _paired_parameter_gradient_metrics(
                    displacement_component,
                    ionic_component + branch_sum_component,
                    _displacement_core_parameters(model),
                )
                conditioning_totals[
                    "combined_branch_response_gradient_norm_sum"
                ] += (
                    combined_metrics["true_born_ionic_gradient_norm"]
                    * gradient_materials
                )
                conditioning_totals[
                    "direct_u_combined_branch_response_gradient_cosine_sum"
                ] += combined_metrics["gradient_cosine"] * gradient_materials
            auxiliary = (
                dielectric_weight * dielectric_component
                + macro_dielectric_weight * macro_dielectric_component
                + macro_elastic_weight * macro_elastic_component
                + ionic_dielectric_weight * ionic_dielectric_component
                + ionic_elastic_weight * ionic_elastic_component
                + elastic_weight * elastic_component
                + born_weight * born_component
                + ionic_weight * ionic_component
                + displacement_response_weight * displacement_component
                + effective_consistency_weight * displacement_consistency_component
                + electronic_weight * electronic_component
                + branch_sum_weight * branch_sum_component
                + force_weight * force_component
                + soft_mode_weight * soft_component
                + low_mode_action_weight * low_action_component
                + low_mode_leak_weight * low_leak_component
                + phi_probe_weight * phi_probe_component
                + lambda_probe_weight * lambda_probe_component
                + born_probe_weight * born_probe_component
                + born_oracle_weight * born_oracle_component
                + phi_oracle_normal_weight * phi_oracle_normal_component
                + internal_strain_weight * internal_component
                + internal_strain_full_weight * full_internal_component
                + response_active_strain_weight * active_strain_component
            )
            loss = macro_weight * full + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite optimization loss encountered")
            if training:
                if collect_conditioning_diagnostics and strict_materials:
                    epsilon = torch.finfo(direct_norm.dtype).eps
                    if displacement_consistency_weight == 0.0:
                        head_parameters = tuple(model.displacement_response_head.parameters())
                        direct_norm = _parameter_gradient_norm(
                            displacement_component, head_parameters
                        )
                        consistency_norm = direct_norm * 0.0
                    raw_ratio = consistency_norm / (direct_norm + epsilon)
                    weighted_ratio = (
                        effective_consistency_weight * consistency_norm
                        / (displacement_response_weight * direct_norm + epsilon)
                    )
                    conditioning_totals["materials"] += strict_materials
                    conditioning_totals[
                        "consistency_u_head_gradient_norm_ratio_sum"
                    ] += float(raw_ratio.detach()) * strict_materials
                    conditioning_totals[
                        "weighted_consistency_u_head_gradient_norm_ratio_sum"
                    ] += float(weighted_ratio.detach()) * strict_materials
                    spectral = first_order_spectral_residual_diagnostics(
                        components.displacement_response,
                        auxiliary_displacement,
                        components.force_constants_flat,
                        components.internal_strain,
                        batch.dfpt_force_constants_flat,
                        batch.internal_strain_full_mask,
                        batch.force_constant_mask,
                        batch.ptr,
                        model.response,
                    )
                    conditioning_totals["total_optical_residual_squared"] += float(
                        spectral["total_optical_residual_squared"]
                    )
                    for key in tuple(conditioning_totals):
                        if key.endswith("_mode_count") or key.endswith(
                            "_residual_squared"
                        ):
                            if key != "total_optical_residual_squared":
                                conditioning_totals[key] += float(spectral[key])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                    raise FloatingPointError("Non-finite parameter gradient encountered")
                optimizer.step()
                updates += 1
        total += float(loss.detach()) * batch.num_graphs
        if compute_macro_response:
            response_predictions.append(prediction.detach().cpu())
            response_targets.append(batch.y.detach().cpu())
        detached_components = {
            "piezo_full": full,
            "dielectric": dielectric_component,
            "macro_dielectric": macro_dielectric_component,
            "macro_elastic": macro_elastic_component,
            "ionic_dielectric": ionic_dielectric_component,
            "ionic_elastic": ionic_elastic_component,
            "elastic_auxiliary": elastic_component,
            "born": born_component,
            "force_constant": force_component,
            "soft_optical": soft_component,
            "low_mode_action": low_action_component,
            "low_mode_leak": low_leak_component,
            "phi_probe": phi_probe_component,
            "lambda_probe": lambda_probe_component,
            "born_probe": born_probe_component,
            "born_oracle": born_oracle_component,
            "phi_oracle_normal": phi_oracle_normal_component,
            "internal_strain": internal_component,
            "internal_strain_full": full_internal_component,
            "response_active_strain": active_strain_component,
            "ionic_piezo": ionic_component,
            "factorized_ionic_piezo": (
                components.factorized_ionic_piezo.abs().mean()
                if compute_factorized_response else zero
            ),
            "displacement_response": displacement_component,
            "displacement_first_order_consistency": displacement_consistency_component,
            "electronic_piezo": electronic_component,
            "branch_sum": branch_sum_component,
        }
        for name, value in detached_components.items():
            component_totals[name] += float(value.detach()) * batch.num_graphs
        count += batch.num_graphs
        elapsed += time.perf_counter() - start
        if training and max_train_updates is not None and updates >= max_train_updates:
            break
    denominator = max(count, 1)
    component_summary = {
        name: value / denominator for name, value in component_totals.items()
    }
    component_summary["macro_response_evaluated"] = float(
        compute_macro_response
    )
    component_summary["factorized_response_evaluated"] = float(
        compute_factorized_response
    )
    conditioning_materials = int(conditioning_totals["materials"])
    if conditioning_materials:
        component_summary.update(
            conditioning_consistency_u_head_gradient_norm_ratio=(
                conditioning_totals["consistency_u_head_gradient_norm_ratio_sum"]
                / conditioning_materials
            ),
            conditioning_weighted_consistency_u_head_gradient_norm_ratio=(
                conditioning_totals["weighted_consistency_u_head_gradient_norm_ratio_sum"]
                / conditioning_materials
            ),
            conditioning_materials=float(conditioning_materials),
        )
        residual_total = float(conditioning_totals["total_optical_residual_squared"])
        for name in (
            "below_delta",
            "delta_to_3delta",
            "3delta_to_10delta",
            "above_10delta",
        ):
            component_summary[f"conditioning_{name}_mode_count"] = float(
                conditioning_totals[f"{name}_mode_count"]
            )
            component_summary[f"conditioning_{name}_residual_fraction"] = float(
                conditioning_totals[f"{name}_residual_squared"]
            ) / max(residual_total, 1e-30)
    gradient_materials = int(conditioning_totals["u_gradient_materials"])
    if gradient_materials:
        component_summary.update(
            u_gradient_materials=float(gradient_materials),
            direct_u_gradient_norm=(
                conditioning_totals["direct_u_gradient_norm_sum"] / gradient_materials
            ),
            true_born_ionic_gradient_norm=(
                conditioning_totals["true_born_ionic_gradient_norm_sum"]
                / gradient_materials
            ),
            direct_u_true_born_ionic_gradient_cosine=(
                conditioning_totals[
                    "direct_u_true_born_ionic_gradient_cosine_sum"
                ]
                / gradient_materials
            ),
            branch_sum_gradient_norm=(
                conditioning_totals["branch_sum_gradient_norm_sum"]
                / gradient_materials
            ),
            direct_u_branch_sum_gradient_cosine=(
                conditioning_totals["direct_u_branch_sum_gradient_cosine_sum"]
                / gradient_materials
            ),
            combined_branch_response_gradient_norm=(
                conditioning_totals["combined_branch_response_gradient_norm_sum"]
                / gradient_materials
            ),
            direct_u_combined_branch_response_gradient_cosine=(
                conditioning_totals[
                    "direct_u_combined_branch_response_gradient_cosine_sum"
                ]
                / gradient_materials
            ),
        )
    component_summary["tensor_response_skill_vs_zero"] = (
        float(
            response_tensor_skill(
                torch.cat(response_predictions), torch.cat(response_targets)
            )["tensor_response_skill_vs_zero"]
        )
        if response_predictions else float("nan")
    )
    return total / denominator, elapsed, component_summary, updates


def _macro_epoch(
    model: PiezoJet,
    loader,
    optimizer,
    scale: torch.Tensor,
    bin_weights: torch.Tensor,
    device: torch.device,
    dielectric_weight: float = 0.0,
    elastic_weight: float = 0.0,
) -> tuple[float, float, list[dict[str, float | int]], dict[str, float], int]:
    """Run the total-only tower without constructing any physical factors."""
    training = optimizer is not None
    model.train(training)
    total, count, elapsed, updates = 0.0, 0, 0.0, 0
    component_totals = {"piezo_full": 0.0, "macro_dielectric": 0.0, "macro_elastic": 0.0}
    predictions, targets = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            prediction, macro_dielectric, macro_elastic = model.predict_macro_responses(batch)
            piezo_component = full_loss(prediction, batch.y, scale, bin_weights)
            dielectric_component = dielectric_loss(
                macro_dielectric, batch.y_dielectric, batch.dielectric_mask
            )
            elastic_component = elastic_auxiliary_loss(
                macro_elastic, batch.y_elastic_gpa, batch.elastic_mask
            )
            loss = (
                piezo_component
                + dielectric_weight * dielectric_component
                + elastic_weight * elastic_component
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite macro-only loss encountered")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ):
                    raise FloatingPointError("Non-finite macro-only gradient encountered")
                optimizer.step()
                updates += 1
        elapsed += time.perf_counter() - start
        total += float(loss.detach()) * batch.num_graphs
        component_totals["piezo_full"] += float(piezo_component.detach()) * batch.num_graphs
        component_totals["macro_dielectric"] += float(dielectric_component.detach()) * batch.num_graphs
        component_totals["macro_elastic"] += float(elastic_component.detach()) * batch.num_graphs
        count += batch.num_graphs
        predictions.append(prediction.detach().cpu())
        targets.append(batch.y.detach().cpu())
    denominator = max(count, 1)
    return (
        total / denominator,
        elapsed,
        [],
        {
            **{name: value / denominator for name, value in component_totals.items()},
            "tensor_response_skill_vs_zero": float(
                response_tensor_skill(torch.cat(predictions), torch.cat(targets))[
                    "tensor_response_skill_vs_zero"
                ]
            ),
        },
        updates,
    )


def _factor_epoch(
    model: PiezoJet,
    loader,
    optimizer,
    device: torch.device,
    born_weight: float,
    force_weight: float,
    internal_strain_weight: float,
    internal_strain_full_weight: float,
    soft_mode_weight: float,
    low_mode_action_weight: float,
    low_mode_leak_weight: float,
    phi_probe_weight: float,
    lambda_probe_weight: float,
    born_probe_weight: float,
    born_oracle_weight: float,
    phi_oracle_normal_weight: float,
    response_active_strain_weight: float,
    max_train_updates: int | None = None,
) -> tuple[float, float, dict[str, float], int]:
    """Train direct DFPT factors without the ill-conditioned inverse response path."""
    training = optimizer is not None
    model.train(training)
    total, count, elapsed = 0.0, 0, 0.0
    component_totals = {
        "born": 0.0, "force_constant": 0.0, "soft_optical": 0.0,
        "low_mode_action": 0.0, "low_mode_leak": 0.0, "phi_probe": 0.0,
        "lambda_probe": 0.0, "born_probe": 0.0, "born_oracle": 0.0,
        "phi_oracle_normal": 0.0, "internal_strain": 0.0,
        "internal_strain_full": 0.0, "response_active_strain": 0.0,
    }
    updates = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            factors = model.predict_factors(batch)
            low_action, low_leak = (
                low_mode_operator_action_losses(
                    factors.force_constants_flat,
                    batch.dfpt_force_constants_flat,
                    batch.ptr,
                    batch.force_constant_mask,
                    mode_count=6,
                )
                if low_mode_action_weight != 0.0 or low_mode_leak_weight != 0.0
                else (factors.force_constants_flat.sum() * 0.0, factors.force_constants_flat.sum() * 0.0)
            )
            phi_probe = (
                mixed_force_constant_probe_loss(
                    factors.force_constants_flat,
                    batch.dfpt_force_constants_flat,
                    batch.dfpt_internal_strain_full,
                    batch.ptr,
                    batch.force_constant_mask,
                    batch.internal_strain_full_mask,
                    model.response,
                    material_ids=batch.material_id,
                )
                if phi_probe_weight != 0.0
                else factors.force_constants_flat.sum() * 0.0
            )
            lambda_probe = (
                internal_strain_probe_loss(
                    factors.internal_strain,
                    batch.dfpt_internal_strain_full,
                    batch.internal_strain_full_mask,
                    batch.batch,
                    material_ids=batch.material_id,
                )
                if lambda_probe_weight != 0.0
                else factors.internal_strain.sum() * 0.0
            )
            born_probe = (
                born_charge_probe_loss(
                    factors.born_charges,
                    batch.y_born,
                    batch.born_mask,
                    batch.batch,
                    material_ids=batch.material_id,
                )
                if born_probe_weight != 0.0
                else factors.born_charges.sum() * 0.0
            )
            oracle_mask = batch.internal_strain_full_mask.reshape(-1) & batch.ionic_piezo_mask.reshape(-1)
            born_oracle = (
                born_oracle_piezo_loss(
                    factors.born_charges,
                    batch.dfpt_force_constants_flat,
                    batch.dfpt_internal_strain_full,
                    batch.y_ionic_piezo,
                    oracle_mask,
                    batch.force_constant_mask,
                    batch.ptr,
                    batch.cell.reshape(-1, 3, 3),
                    model.response,
                )
                if born_oracle_weight != 0.0
                else factors.born_charges.sum() * 0.0
            )
            phi_oracle_normal = (
                phi_oracle_normal_equation_loss(
                    factors.force_constants_flat,
                    batch.dfpt_force_constants_flat,
                    batch.dfpt_internal_strain_full,
                    batch.internal_strain_full_mask,
                    batch.force_constant_mask,
                    batch.ptr,
                    model.response,
                )
                if phi_oracle_normal_weight != 0.0
                else factors.force_constants_flat.sum() * 0.0
            )
            components = {
                "born": born_loss(factors.born_charges, batch.y_born, batch.born_mask, batch.batch),
                "force_constant": force_constant_loss(
                    factors.force_constants_flat,
                    batch.dfpt_force_constants_flat,
                    batch.ptr,
                    batch.force_constant_mask,
                ),
                "soft_optical": soft_optical_eigenvalue_loss(
                    factors.force_constants_flat,
                    batch.dfpt_force_constants_flat,
                    batch.ptr,
                    batch.force_constant_mask,
                ),
                "low_mode_action": low_action,
                "low_mode_leak": low_leak,
                "phi_probe": phi_probe,
                "lambda_probe": lambda_probe,
                "born_probe": born_probe,
                "born_oracle": born_oracle,
                "phi_oracle_normal": phi_oracle_normal,
                "internal_strain": internal_strain_loss(
                    factors.internal_strain,
                    batch.dfpt_internal_strain_flat,
                    batch.dfpt_internal_strain_ions,
                    batch.dfpt_internal_strain_directions,
                    batch.dfpt_internal_strain_count,
                    batch.ptr,
                ),
                "internal_strain_full": full_internal_strain_loss(
                    factors.internal_strain,
                    batch.dfpt_internal_strain_full,
                    batch.internal_strain_full_mask,
                    batch.batch,
                ),
                "response_active_strain": response_active_internal_strain_loss(
                    factors.internal_strain,
                    batch.y_born,
                    batch.dfpt_force_constants_flat,
                    batch.y_ionic_piezo,
                    batch.ionic_piezo_mask,
                    batch.ptr,
                    batch.cell.reshape(-1, 3, 3),
                    model.response,
                    batch.force_constant_mask,
                ),
            }
            loss = (
                born_weight * components["born"]
                + force_weight * components["force_constant"]
                + soft_mode_weight * components["soft_optical"]
                + low_mode_action_weight * components["low_mode_action"]
                + low_mode_leak_weight * components["low_mode_leak"]
                + phi_probe_weight * components["phi_probe"]
                + lambda_probe_weight * components["lambda_probe"]
                + born_probe_weight * components["born_probe"]
                + born_oracle_weight * components["born_oracle"]
                + phi_oracle_normal_weight * components["phi_oracle_normal"]
                + internal_strain_weight * components["internal_strain"]
                + internal_strain_full_weight * components["internal_strain_full"]
                + response_active_strain_weight * components["response_active_strain"]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite direct-factor loss encountered")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ):
                    raise FloatingPointError("Non-finite direct-factor gradient encountered")
                optimizer.step()
                updates += 1
        total += float(loss.detach()) * batch.num_graphs
        for name, value in components.items():
            component_totals[name] += float(value.detach()) * batch.num_graphs
        count += batch.num_graphs
        elapsed += time.perf_counter() - start
        if training and max_train_updates is not None and updates >= max_train_updates:
            break
    denominator = max(count, 1)
    return total / denominator, elapsed, {
        name: value / denominator for name, value in component_totals.items()
    }, updates


def _displacement_epoch(
    model: PiezoJet,
    loader,
    optimizer,
    device: torch.device,
    *,
    target_weight: float,
    true_born_macro_weight: float,
    max_train_updates: int | None = None,
) -> tuple[float, float, dict[str, float], int]:
    """Teacher-force the independent U head before coupled response training.

    The target loss uses true ``Phi,Lambda`` only to construct the declared
    regularized coordinate target; the macro loss uses true BECs.  Thus neither
    objective can be satisfied by shrinking a predicted BEC, Hessian, Lambda,
    or U field together.  This is a curriculum stage, not a source of a new
    inference-time operator.
    """
    if target_weight < 0.0 or true_born_macro_weight < 0.0:
        raise ValueError("Teacher-forced displacement weights must be non-negative")
    training = optimizer is not None
    model.train(training)
    total, count, elapsed, updates = 0.0, 0, 0.0, 0
    component_totals = {"displacement_response": 0.0, "true_born_ionic": 0.0}
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            prediction = model.predict_displacement_response(batch)
            target_component = displacement_response_target_loss(
                prediction,
                batch.dfpt_force_constants_flat,
                batch.dfpt_internal_strain_full,
                batch.internal_strain_full_mask,
                batch.force_constant_mask,
                batch.ptr,
                model.response,
            )
            macro_component, _ = displacement_macro_piezo_loss(
                prediction,
                batch.y_born,
                batch.y_ionic_piezo,
                batch.ionic_piezo_mask,
                batch,
                model.response,
            )
            loss = target_weight * target_component + true_born_macro_weight * macro_component
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite teacher-forced displacement loss encountered")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ):
                    raise FloatingPointError("Non-finite teacher-forced displacement gradient encountered")
                optimizer.step()
                updates += 1
        total += float(loss.detach()) * batch.num_graphs
        component_totals["displacement_response"] += float(target_component.detach()) * batch.num_graphs
        component_totals["true_born_ionic"] += float(macro_component.detach()) * batch.num_graphs
        count += batch.num_graphs
        elapsed += time.perf_counter() - start
        if training and max_train_updates is not None and updates >= max_train_updates:
            break
    denominator = max(count, 1)
    return total / denominator, elapsed, {
        name: value / denominator for name, value in component_totals.items()
    }, updates


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _data_commit(data_root: str | Path) -> str:
    path = Path(data_root) / "SOURCE_COMMIT.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Reproducible training requires the exact GMTNet repository commit SHA; "
            "write it to SOURCE_COMMIT.txt before starting an experiment."
        )
    commit = path.read_text(encoding="utf-8").strip()
    if len(commit) != 40 or any(char not in "0123456789abcdefABCDEF" for char in commit):
        raise ValueError(f"{path} must contain one 40-character Git commit SHA")
    return commit


def restrict_splits_to_material_ids(
    splits: dict[str, list[str]], selected_ids: list[str], mode: str
) -> dict[str, list[str]]:
    """Restrict training to audited IDs without silently leaking validation.

    ``same`` preserves the historical plumbing-smoke behavior. ``global``
    intersects the selected population with the persisted formula-disjoint
    train/validation/test split and is required for accuracy-oriented runs.
    """
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Material-ID file must contain a non-empty list of unique IDs")
    if mode == "same":
        return {"train": selected_ids.copy(), "val": selected_ids.copy(), "test": []}
    if mode != "global":
        raise ValueError(f"Unknown material-ID split mode: {mode}")
    selected = set(selected_ids)
    restricted = {name: [jid for jid in splits[name] if jid in selected] for name in ("train", "val", "test")}
    restored = set(restricted["train"] + restricted["val"] + restricted["test"])
    if restored != selected:
        missing = sorted(selected - restored)
        raise ValueError(f"Selected IDs are missing from the persisted global split: {missing[:5]}")
    if not restricted["train"] or not restricted["val"]:
        raise ValueError("Global material-ID restriction requires non-empty train and validation subsets")
    if set(restricted["train"]) & set(restricted["val"]):
        raise RuntimeError("Persisted global split leaks material IDs between train and validation")
    return restricted


def load_explicit_splits(path: Path, known_ids: set[str]) -> dict[str, list[str]]:
    """Load a frozen, auditable train/validation/test assignment.

    This is distinct from ``--material-ids-file``: benchmark splits are
    already formula-disjoint and must not be intersected with the unrelated
    4,998-material global split.
    """
    parsed = json.loads(path.read_text(encoding="utf-8"))
    splits = parsed.get("splits", parsed) if isinstance(parsed, dict) else None
    if not isinstance(splits, dict):
        raise ValueError("Explicit split file must contain a splits object")
    restored: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        values = splits.get(name)
        if not isinstance(values, list):
            raise ValueError(f"Explicit split file is missing a {name} list")
        restored[name] = [str(value) for value in values]
    all_ids = restored["train"] + restored["val"] + restored["test"]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Explicit split file leaks material IDs across splits")
    unknown = sorted(set(all_ids) - known_ids)
    if unknown:
        raise ValueError(f"Explicit split file contains unknown IDs: {unknown[:5]}")
    if not restored["train"] or not restored["val"] or not restored["test"]:
        raise ValueError("Explicit split file requires non-empty train, val, and test splits")
    return restored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="Override config epochs for a bounded smoke run")
    parser.add_argument("--train-updates-per-epoch", type=int, help="Cap optimizer updates per joint epoch; enables exact fixed-update full-corpus protocols")
    parser.add_argument("--factor-train-updates-per-epoch", type=int, help="Cap optimizer updates per direct-factor epoch in a fixed-update protocol")
    parser.add_argument("--batch-size", type=int, help="Override config batch size")
    parser.add_argument("--learning-rate", type=float, help="Override config learning rate")
    parser.add_argument("--weight-decay", type=float, help="Override config weight decay")
    parser.add_argument("--early-stopping-patience", type=int, help="Override early stopping patience; set 0 to disable for a controlled diagnostic run")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader workers; set 0 for constrained Windows shared-memory diagnostics")
    parser.add_argument("--device", type=str, help="Override device (for example cpu for an isolated diagnostic)")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument(
        "--material-ids-file", type=Path,
        help="JSON list or newline-delimited material IDs for a bounded, auditable training/validation smoke run",
    )
    parser.add_argument(
        "--material-ids-split", choices=("same", "global"), default="same",
        help="Use the same IDs for a plumbing smoke or intersect them with the persisted formula-disjoint global split",
    )
    parser.add_argument(
        "--splits-file", type=Path,
        help="Frozen explicit train/val/test JSON; do not combine with --material-ids-file",
    )
    parser.add_argument("--resume", type=Path, help="Resume a saved checkpoint at its next epoch")
    parser.add_argument(
        "--factor-checkpoint",
        type=Path,
        help="Initialize a fresh joint stage from a selected factor checkpoint without restoring its optimizer/epoch",
    )
    parser.add_argument(
        "--displacement-checkpoint",
        type=Path,
        help=(
            "Initialize a fresh joint-stage adjudication from a selected "
            "teacher-forced displacement checkpoint, including its AdamW state"
        ),
    )
    parser.add_argument(
        "--pretrained-encoder",
        type=Path,
        help="Explicit inductive Cartesian structural-pretraining checkpoint; overrides config without altering the split.",
    )
    parser.add_argument("--seed", type=int, help="Override config seed for multi-seed experiments")
    parser.add_argument("--factor-pretrain-epochs", type=int, help="Direct Z*, Phi, Lambda curriculum epochs before joint response training")
    parser.add_argument("--factor-pretrain-learning-rate", type=float, help="Learning rate for direct-factor curriculum")
    parser.add_argument("--factor-pretrain-patience", type=int, help="Early-stopping patience for direct-factor validation loss")
    parser.add_argument(
        "--displacement-pretrain-epochs", type=int,
        help="Teacher-forced U_eta epochs before bilinear ionic and first-order-consistency joint training",
    )
    parser.add_argument(
        "--displacement-pretrain-learning-rate", type=float,
        help="Learning rate for the teacher-forced U_eta curriculum",
    )
    parser.add_argument(
        "--joint-displacement-learning-rate",
        type=float,
        help="Joint-stage learning rate for the isolated U/V tower",
    )
    parser.add_argument(
        "--preserve-displacement-optimizer-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain teacher-U AdamW moments when entering the joint stage",
    )
    parser.add_argument(
        "--collect-u-gradient-diagnostics",
        action="store_true",
        help="Record same-batch strict direct-U versus true-BEC ionic U-tower gradients",
    )
    parser.add_argument(
        "--displacement-consistency-warmup-epochs", type=int,
        help="Joint epochs with zero first-order U/V consistency weight",
    )
    parser.add_argument(
        "--displacement-consistency-ramp-epochs", type=int,
        help="Joint epochs for the linear first-order-consistency ramp",
    )
    parser.add_argument(
        "--allow-noninductive-overfit", action="store_true",
        help="Permit same-ID validation only for an explicitly labelled memorization diagnostic; never a benchmark run.",
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=("trs", "loss"),
        help="Preregister the joint-stage early-stopping and primary-checkpoint metric",
    )
    args = parser.parse_args()
    cfg = load_project_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.pretrained_encoder is not None:
        cfg["pretrained_encoder"] = str(args.pretrained_encoder)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        cfg["epochs"] = args.epochs
    if args.train_updates_per_epoch is not None:
        if args.train_updates_per_epoch < 1:
            raise ValueError("--train-updates-per-epoch must be positive")
        cfg["train_updates_per_epoch"] = args.train_updates_per_epoch
    if args.factor_train_updates_per_epoch is not None:
        if args.factor_train_updates_per_epoch < 1:
            raise ValueError("--factor-train-updates-per-epoch must be positive")
        cfg["factor_train_updates_per_epoch"] = args.factor_train_updates_per_epoch
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        cfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive")
        cfg["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        if args.weight_decay < 0:
            raise ValueError("--weight-decay must be non-negative")
        cfg["weight_decay"] = args.weight_decay
    if args.early_stopping_patience is not None:
        if args.early_stopping_patience < 0:
            raise ValueError("--early-stopping-patience must be non-negative")
        cfg["early_stopping_patience"] = args.early_stopping_patience
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        cfg["num_workers"] = args.num_workers
    if args.device is not None:
        cfg["device"] = args.device
    if args.factor_pretrain_epochs is not None:
        if args.factor_pretrain_epochs < 0:
            raise ValueError("--factor-pretrain-epochs must be non-negative")
        cfg["factor_pretrain_epochs"] = args.factor_pretrain_epochs
    if args.factor_pretrain_learning_rate is not None:
        if args.factor_pretrain_learning_rate <= 0:
            raise ValueError("--factor-pretrain-learning-rate must be positive")
        cfg["factor_pretrain_learning_rate"] = args.factor_pretrain_learning_rate
    if args.factor_pretrain_patience is not None:
        if args.factor_pretrain_patience < 0:
            raise ValueError("--factor-pretrain-patience must be non-negative")
        cfg["factor_pretrain_patience"] = args.factor_pretrain_patience
    if args.displacement_pretrain_epochs is not None:
        if args.displacement_pretrain_epochs < 0:
            raise ValueError("--displacement-pretrain-epochs must be non-negative")
        cfg["displacement_pretrain_epochs"] = args.displacement_pretrain_epochs
    if args.displacement_pretrain_learning_rate is not None:
        if args.displacement_pretrain_learning_rate <= 0:
            raise ValueError("--displacement-pretrain-learning-rate must be positive")
        cfg["displacement_pretrain_learning_rate"] = args.displacement_pretrain_learning_rate
    if args.joint_displacement_learning_rate is not None:
        if args.joint_displacement_learning_rate <= 0:
            raise ValueError("--joint-displacement-learning-rate must be positive")
        cfg["joint_displacement_learning_rate"] = args.joint_displacement_learning_rate
    if args.preserve_displacement_optimizer_state is not None:
        cfg["preserve_displacement_optimizer_state"] = (
            args.preserve_displacement_optimizer_state
        )
    if args.collect_u_gradient_diagnostics:
        cfg["collect_u_gradient_diagnostics"] = True
    if args.displacement_consistency_warmup_epochs is not None:
        if args.displacement_consistency_warmup_epochs < 0:
            raise ValueError("--displacement-consistency-warmup-epochs must be non-negative")
        cfg["displacement_consistency_warmup_epochs"] = args.displacement_consistency_warmup_epochs
    if args.displacement_consistency_ramp_epochs is not None:
        if args.displacement_consistency_ramp_epochs < 0:
            raise ValueError("--displacement-consistency-ramp-epochs must be non-negative")
        cfg["displacement_consistency_ramp_epochs"] = args.displacement_consistency_ramp_epochs
    if args.checkpoint_selection_metric is not None:
        cfg["checkpoint_selection_metric"] = args.checkpoint_selection_metric
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    seed_everything(int(cfg["seed"]))
    device = device_from_config(cfg["device"])
    runtime_device = str(device)
    if device.type == "cuda":
        runtime_device = f"{device} ({torch.cuda.get_device_name(device)})"
    print(f"runtime_device={runtime_device}")
    data_commit = _data_commit(cfg["data_root"])
    records = load_gmtnet_records(cfg["data_root"])
    splits = create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
    if args.material_ids_file is not None and args.splits_file is not None:
        raise ValueError("--material-ids-file and --splits-file are mutually exclusive")
    known_ids = {str(record["JARVIS_ID"]) for record in records}
    if args.splits_file is not None:
        if not args.splits_file.is_file():
            raise FileNotFoundError(f"Explicit split file does not exist: {args.splits_file}")
        splits = load_explicit_splits(args.splits_file, known_ids)
        cfg["splits_file"] = str(args.splits_file)
        cfg["restricted_split_counts"] = {name: len(values) for name, values in splits.items()}
    elif args.material_ids_file is not None:
        if not args.material_ids_file.is_file():
            raise FileNotFoundError(f"Material-ID file does not exist: {args.material_ids_file}")
        # Windows PowerShell 5 writes UTF-8 JSON with a BOM by default.
        # Accept it so a bounded diagnostic does not silently change its first
        # material ID into a different string.
        text = args.material_ids_file.read_text(encoding="utf-8-sig")
        try:
            parsed = json.loads(text)
            selected_ids = [str(value) for value in parsed]
        except json.JSONDecodeError:
            selected_ids = [line.strip() for line in text.splitlines() if line.strip()]
        unknown = sorted(set(selected_ids) - known_ids)
        if unknown:
            raise ValueError(f"Material-ID file contains unknown IDs: {unknown[:5]}")
        splits = restrict_splits_to_material_ids(splits, selected_ids, args.material_ids_split)
        cfg["material_ids_file"] = str(args.material_ids_file)
        cfg["material_ids_split"] = args.material_ids_split
        cfg["bounded_smoke_material_count"] = len(selected_ids)
        cfg["restricted_split_counts"] = {name: len(values) for name, values in splits.items()}
    cache_key = graph_cache_key(records, cfg["cutoff"], cfg["max_neighbors"])
    dfpt_dir = cfg.get("jarvis_dfpt_dir")
    strain_completion_dir = cfg.get("jarvis_strain_completion_dir")
    dataset_kwargs = {
        "processed_dir": cfg["processed_dir"],
        "cache_key": cache_key,
        "dfpt_dir": dfpt_dir,
        "strain_completion_dir": strain_completion_dir,
        "elastic_targets_path": cfg.get("elastic_targets_path"),
        "dfpt_total_consistency_absolute_tolerance": float(
            cfg.get("dfpt_total_consistency_absolute_tolerance_c_per_m2", 0.05)
        ),
        "dfpt_total_consistency_relative_tolerance": float(
            cfg.get("dfpt_total_consistency_relative_tolerance", 0.05)
        ),
    }
    train_set = PiezoDataset(records, splits["train"], cfg["cutoff"], cfg["max_neighbors"], **dataset_kwargs)
    val_set = PiezoDataset(records, splits["val"], cfg["cutoff"], cfg["max_neighbors"], **dataset_kwargs)
    dfpt_cache = JarvisDFPTCache(dfpt_dir) if dfpt_dir is not None else None
    completion_dir = Path(strain_completion_dir) if strain_completion_dir is not None else None
    branch_ids = [
        jid for jid in splits["train"]
        if dfpt_cache is not None and dfpt_cache.path(jid).is_file()
    ]
    strict_ids = [
        jid for jid in branch_ids
        if completion_dir is not None and (completion_dir / f"{jid}.pt").is_file()
    ]
    if bool(cfg.get("multistream_training", True)) and (not branch_ids or not strict_ids):
        raise ValueError(
            "Exposure-matched training requires non-empty macro, DFPT-branch, and strict streams"
        )
    branch_set = PiezoDataset(
        records, branch_ids, cfg["cutoff"], cfg["max_neighbors"], **dataset_kwargs
    ) if branch_ids else None
    strict_set = PiezoDataset(
        records, strict_ids, cfg["cutoff"], cfg["max_neighbors"], **dataset_kwargs
    ) if strict_ids else None
    loader_options = {"num_workers": cfg["num_workers"], "pin_memory": device.type == "cuda"}
    if cfg["num_workers"] > 0:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, **loader_options)
    branch_loader = (
        DataLoader(branch_set, batch_size=cfg["batch_size"], shuffle=True, **loader_options)
        if branch_set is not None else None
    )
    strict_loader = (
        DataLoader(strict_set, batch_size=cfg["batch_size"], shuffle=True, **loader_options)
        if strict_set is not None else None
    )
    first_target = torch.cat([train_set[index].y_voigt for index in range(len(train_set))])
    scale = piezo_scale(first_target).to(device)
    bin_weights = response_bin_weights(torch.stack([train_set[index].y.squeeze(0) for index in range(len(train_set))])).to(device)
    processed = Path(cfg["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "stats.json").write_text(json.dumps({"piezo_scale": float(scale), "source": "train split only"}, indent=2) + "\n", encoding="utf-8")
    model = model_from_config(cfg).to(device)
    pretraining_path = Path(cfg["pretrained_encoder"])
    if not pretraining_path.is_file():
        raise FileNotFoundError(
            f"PiezoJet fine-tuning requires the structural pretraining checkpoint {pretraining_path}. "
            "Run `python -m piezojet.pretrain --config config.yaml` or use `python scripts/run_pipeline.py --config config.yaml`."
        )
    pretrained = torch.load(pretraining_path, map_location=device, weights_only=False)
    if "encoder" not in pretrained:
        raise ValueError(f"Pretraining checkpoint {pretraining_path} has no encoder state")
    if pretrained.get("architecture") != "cartesian_local_environment_v1":
        raise ValueError(
            "The requested checkpoint predates PiezoJet's Cartesian local-environment encoder. "
            "Run structural pretraining with the current config before fine-tuning."
        )
    if args.allow_noninductive_overfit:
        if args.material_ids_file is None or args.material_ids_split != "same":
            raise ValueError(
                "--allow-noninductive-overfit requires --material-ids-file with --material-ids-split same"
            )
        pretraining_provenance = {
            "mode": "noninductive_memorization_diagnostic",
            "heldout_ids": [],
            "warning": "Validation IDs equal training IDs; this run cannot support a generalization claim.",
        }
    else:
        pretraining_provenance = validate_inductive_checkpoint(
            pretrained, splits["train"], splits["val"] + splits["test"]
        )
    model.encoder.load_state_dict(pretrained["encoder"], strict=True)
    model.macro_encoder.load_state_dict(pretrained["encoder"], strict=True)
    model.displacement_encoder.load_state_dict(pretrained["encoder"], strict=True)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    # The maintained CLI has one optimization objective. Sketch, mode-aware,
    # and architecture-switch diagnostics are intentionally not user options.
    cfg["loss"] = "full"
    cfg["pretrained_encoder"] = str(pretraining_path)
    cfg["pretraining_epoch"] = pretrained.get("epoch")
    cfg["pretraining_provenance"] = pretraining_provenance
    cfg["noninductive_overfit_diagnostic"] = bool(args.allow_noninductive_overfit)
    cfg["git_commit"] = _git_commit()
    cfg["data_commit"] = data_commit
    cfg["runtime_device"] = runtime_device
    cfg["exposure_protocol"] = {
        "unit": "complete_stream_passes",
        "macro_materials": len(train_set),
        "branch_materials": len(branch_set) if branch_set is not None else 0,
        "strict_materials": len(strict_set) if strict_set is not None else 0,
    }
    factor_rows: list[dict[str, float | int]] = (
        _read_metric_rows(output / "factor_metrics.csv") if args.resume is not None else []
    )
    factor_best = float("inf")
    factor_best_epoch = 0
    if factor_rows:
        factor_best_row = min(factor_rows, key=lambda row: float(row["val_loss"]))
        factor_best = float(factor_best_row["val_loss"])
        factor_best_epoch = int(factor_best_row["epoch"])
    factor_epochs = int(cfg.get("factor_pretrain_epochs", 0))
    factor_branch_weights = {
        "born_weight": float(cfg.get("factor_pretrain_born_weight", 1.0)),
        "force_weight": float(cfg.get("factor_pretrain_force_weight", 1.0)),
        "internal_strain_weight": float(cfg.get("factor_pretrain_internal_strain_weight", 5.0)),
        "internal_strain_full_weight": 0.0,
        "soft_mode_weight": float(cfg.get("factor_pretrain_soft_mode_weight", 1.0)),
        "low_mode_action_weight": float(cfg.get("factor_pretrain_low_mode_action_weight", 0.0)),
        "low_mode_leak_weight": float(cfg.get("factor_pretrain_low_mode_leak_weight", 0.0)),
        "phi_probe_weight": float(cfg.get("factor_pretrain_phi_probe_weight", 0.0)),
        "lambda_probe_weight": 0.0,
        "born_probe_weight": float(cfg.get("factor_pretrain_born_probe_weight", 0.0)),
        "born_oracle_weight": 0.0,
        "phi_oracle_normal_weight": 0.0,
        "response_active_strain_weight": float(
            cfg.get("factor_pretrain_response_active_strain_weight", 0.0)
        ),
    }
    factor_strict_weights = {
        "born_weight": 0.0,
        "force_weight": 0.0,
        "internal_strain_weight": 0.0,
        "internal_strain_full_weight": float(
            cfg.get("factor_pretrain_internal_strain_full_weight", 0.0)
        ),
        "soft_mode_weight": 0.0,
        "low_mode_action_weight": 0.0,
        "low_mode_leak_weight": 0.0,
        "phi_probe_weight": 0.0,
        "lambda_probe_weight": float(cfg.get("factor_pretrain_lambda_probe_weight", 0.0)),
        "born_probe_weight": 0.0,
        "born_oracle_weight": float(cfg.get("factor_pretrain_born_oracle_weight", 0.0)),
        "phi_oracle_normal_weight": float(cfg.get("factor_pretrain_phi_oracle_normal_weight", 0.0)),
        "response_active_strain_weight": 0.0,
    }
    factor_weights = {
        "branch_stream": factor_branch_weights,
        "strict_stream": factor_strict_weights,
    }
    if (
        factor_epochs > 0
        and args.resume is None
        and args.factor_checkpoint is None
        and args.displacement_checkpoint is None
    ):
        factor_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.get("factor_pretrain_learning_rate", cfg["learning_rate"])),
            weight_decay=float(cfg["weight_decay"]),
        )
        factor_wait, factor_patience = 0, int(cfg.get("factor_pretrain_patience", 0))
        for factor_epoch in range(1, factor_epochs + 1):
            assert branch_loader is not None and strict_loader is not None
            branch_factor, branch_factor_seconds, branch_factor_components, branch_factor_updates = _factor_epoch(
                model, branch_loader, factor_optimizer, device, **factor_branch_weights
            )
            strict_factor, strict_factor_seconds, strict_factor_components, strict_factor_updates = _factor_epoch(
                model, strict_loader, factor_optimizer, device, **factor_strict_weights
            )
            train_factor = branch_factor + strict_factor
            train_factor_seconds = branch_factor_seconds + strict_factor_seconds
            factor_train_updates = branch_factor_updates + strict_factor_updates
            train_factor_components = {
                **{f"branch_stream_{name}": value for name, value in branch_factor_components.items()},
                **{f"strict_stream_{name}": value for name, value in strict_factor_components.items()},
            }
            val_factor, val_factor_seconds, val_factor_components, _ = _factor_epoch(
                model, val_loader, None, device,
                **{
                    name: factor_branch_weights[name] + factor_strict_weights[name]
                    for name in factor_branch_weights
                },
            )
            factor_row = {
                "epoch": factor_epoch,
                "train_loss": train_factor,
                "val_loss": val_factor,
                "train_seconds": train_factor_seconds,
                "val_seconds": val_factor_seconds,
                "train_optimizer_updates": factor_train_updates,
                "branch_stream_optimizer_updates": branch_factor_updates,
                "strict_stream_optimizer_updates": strict_factor_updates,
                "branch_effective_passes": factor_epoch,
                "strict_effective_passes": factor_epoch,
                "branch_examples_seen": factor_epoch * len(branch_set),
                "strict_examples_seen": factor_epoch * len(strict_set),
                "branch_unique_samples_seen": len(branch_set),
                "strict_unique_samples_seen": len(strict_set),
            }
            factor_row.update({f"train_{name}_loss": value for name, value in train_factor_components.items()})
            factor_row.update({f"val_{name}_loss": value for name, value in val_factor_components.items()})
            factor_rows.append(factor_row)
            factor_checkpoint = {
                "model": model.state_dict(),
                "optimizer": factor_optimizer.state_dict(),
                "config": cfg,
                "piezo_scale": float(scale),
                "epoch": factor_epoch,
                "stage": "direct_factor_pretraining",
            }
            torch.save(factor_checkpoint, output / "factor_last.pt")
            if val_factor < factor_best:
                factor_best = val_factor
                factor_best_epoch = factor_epoch
                factor_wait = 0
                torch.save(factor_checkpoint, output / "factor_best.pt")
            else:
                factor_wait += 1
            print(f"factor_epoch={factor_epoch} train={train_factor:.6g} val={val_factor:.6g}")
            if factor_patience > 0 and factor_wait >= factor_patience:
                print(
                    f"factor_early_stop epoch={factor_epoch} best_val={factor_best:.6g} "
                    f"patience={factor_patience}"
                )
                break
        with (output / "factor_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=factor_rows[0].keys())
            writer.writeheader()
            writer.writerows(factor_rows)
        factor_saved = torch.load(output / "factor_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(factor_saved["model"])
        cfg["factor_pretraining_best_epoch"] = factor_best_epoch
        cfg["factor_pretraining_best_val_loss"] = factor_best
        cfg["factor_pretraining_epochs_completed"] = len(factor_rows)

    if args.factor_checkpoint is not None:
        if args.resume is not None or args.displacement_checkpoint is not None:
            raise ValueError(
                "--factor-checkpoint, --displacement-checkpoint, and --resume are mutually exclusive"
            )
        if not args.factor_checkpoint.is_file():
            raise FileNotFoundError(
                f"Factor checkpoint does not exist: {args.factor_checkpoint}"
            )
        factor_saved = torch.load(
            args.factor_checkpoint, map_location=device, weights_only=False
        )
        if factor_saved.get("stage") != "direct_factor_pretraining":
            raise ValueError("--factor-checkpoint must come from direct factor pretraining")
        model.load_state_dict(factor_saved["model"])
        torch.save(factor_saved, output / "factor_best.pt")
        cfg["initialized_from_factor_checkpoint"] = str(args.factor_checkpoint)
        cfg["initialized_from_factor_epoch"] = int(factor_saved["epoch"])

    displacement_rows: list[dict[str, float | int]] = (
        _read_metric_rows(output / "displacement_metrics.csv")
        if args.resume is not None
        else []
    )
    displacement_best = float("inf")
    displacement_best_epoch = 0
    if displacement_rows:
        displacement_best_row = min(
            displacement_rows, key=lambda row: float(row["val_loss"])
        )
        displacement_best = float(displacement_best_row["val_loss"])
        displacement_best_epoch = int(displacement_best_row["epoch"])
    displacement_optimizer_state: dict | None = None
    displacement_epochs = int(cfg.get("displacement_pretrain_epochs", 0))
    if (
        displacement_epochs > 0
        and args.resume is None
        and args.displacement_checkpoint is None
    ):
        assert branch_loader is not None and strict_loader is not None
        target_weight = float(cfg.get("displacement_pretrain_target_weight", 1.0))
        macro_weight = float(cfg.get("displacement_pretrain_true_born_macro_weight", 1.0))
        if target_weight <= 0.0 and macro_weight <= 0.0:
            raise ValueError("Teacher-forced displacement pretraining needs a positive target or macro weight")
        displacement_optimizer = torch.optim.AdamW(
            list(model.displacement_encoder.parameters())
            + list(model.displacement_local_polar.parameters())
            + list(model.displacement_global_context.parameters())
            + list(model.displacement_response_head.parameters()),
            lr=float(cfg.get("displacement_pretrain_learning_rate", cfg["learning_rate"])),
            weight_decay=float(cfg["weight_decay"]),
        )
        for displacement_epoch in range(1, displacement_epochs + 1):
            branch_value, branch_seconds, branch_components, branch_updates = _displacement_epoch(
                model,
                branch_loader,
                displacement_optimizer,
                device,
                target_weight=0.0,
                true_born_macro_weight=macro_weight,
            )
            strict_value, strict_seconds, strict_components, strict_updates = _displacement_epoch(
                model,
                strict_loader,
                displacement_optimizer,
                device,
                target_weight=target_weight,
                true_born_macro_weight=0.0,
            )
            val_value, val_seconds, val_components, _ = _displacement_epoch(
                model,
                val_loader,
                None,
                device,
                target_weight=target_weight,
                true_born_macro_weight=macro_weight,
            )
            row = {
                "epoch": displacement_epoch,
                "train_loss": branch_value + strict_value,
                "val_loss": val_value,
                "train_seconds": branch_seconds + strict_seconds,
                "val_seconds": val_seconds,
                "branch_stream_optimizer_updates": branch_updates,
                "strict_stream_optimizer_updates": strict_updates,
                "branch_effective_passes": displacement_epoch,
                "strict_effective_passes": displacement_epoch,
                "branch_examples_seen": displacement_epoch * len(branch_set),
                "strict_examples_seen": displacement_epoch * len(strict_set),
            }
            row.update({f"train_branch_{name}_loss": value for name, value in branch_components.items()})
            row.update({f"train_strict_{name}_loss": value for name, value in strict_components.items()})
            row.update({f"val_{name}_loss": value for name, value in val_components.items()})
            displacement_rows.append(row)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": displacement_optimizer.state_dict(),
                "config": cfg,
                "piezo_scale": float(scale),
                "epoch": displacement_epoch,
                "stage": "teacher_forced_displacement_pretraining",
            }
            torch.save(checkpoint, output / "displacement_last.pt")
            if val_value < displacement_best:
                displacement_best = val_value
                displacement_best_epoch = displacement_epoch
                torch.save(checkpoint, output / "displacement_best.pt")
            print(
                f"displacement_epoch={displacement_epoch} "
                f"train={branch_value + strict_value:.6g} val={val_value:.6g}"
            )
        with (output / "displacement_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=displacement_rows[0].keys())
            writer.writeheader()
            writer.writerows(displacement_rows)
        saved = torch.load(output / "displacement_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        displacement_optimizer_state = saved.get("optimizer")
        cfg["displacement_pretraining"] = {
            "epochs_completed": len(displacement_rows),
            "best_epoch": displacement_best_epoch,
            "best_val_loss": displacement_best,
            "target_weight": target_weight,
            "true_born_macro_weight": macro_weight,
            "objective": "true-Phi/Lambda U target plus true-BEC ionic macro supervision",
        }

    if args.displacement_checkpoint is not None:
        if args.resume is not None or args.factor_checkpoint is not None:
            raise ValueError(
                "--factor-checkpoint, --displacement-checkpoint, and --resume are mutually exclusive"
            )
        if not args.displacement_checkpoint.is_file():
            raise FileNotFoundError(
                f"Displacement checkpoint does not exist: {args.displacement_checkpoint}"
            )
        saved = torch.load(
            args.displacement_checkpoint, map_location=device, weights_only=False
        )
        if saved.get("stage") != "teacher_forced_displacement_pretraining":
            raise ValueError(
                "--displacement-checkpoint must come from teacher-forced displacement pretraining"
            )
        model.load_state_dict(saved["model"])
        displacement_optimizer_state = saved.get("optimizer")
        cfg["initialized_from_displacement_checkpoint"] = str(
            args.displacement_checkpoint
        )
        cfg["initialized_from_displacement_epoch"] = int(saved["epoch"])
        cfg["initialized_from_displacement_commit"] = saved.get("config", {}).get(
            "git_commit", "unknown"
        )

    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        if "optimizer" not in saved:
            raise ValueError("Resume checkpoint has no optimizer state")
        optimizer = _optimizer_for_resume(model, cfg, saved["optimizer"])
        displacement_optimizer_state_restored = bool(
            saved.get("config", {})
            .get("joint_optimizer", {})
            .get("teacher_state_restored", False)
        )
    else:
        optimizer, displacement_optimizer_state_restored = _joint_optimizer(
            model,
            cfg,
            displacement_optimizer_state=displacement_optimizer_state,
        )
    cfg["joint_optimizer"] = {
        "base_learning_rate": float(cfg["learning_rate"]),
        "displacement_learning_rate": float(
            cfg.get("joint_displacement_learning_rate", cfg["learning_rate"])
        ),
        "requested_teacher_state_preservation": bool(
            cfg.get("preserve_displacement_optimizer_state", False)
        ),
        "teacher_state_restored": displacement_optimizer_state_restored,
        "parameter_groups": "teacher-U core, auxiliary V, remaining model",
    }
    start_epoch = 1
    resumed_from = None
    if args.resume is not None:
        resumed_from = int(saved["epoch"])
        start_epoch = resumed_from + 1
        cfg["resumed_from_epoch"] = resumed_from
        cfg["resumed_from_commit"] = saved.get("config", {}).get("git_commit", "unknown")
    selection_metric = str(cfg.get("checkpoint_selection_metric", "trs"))
    if selection_metric not in {"trs", "loss"}:
        raise ValueError("checkpoint_selection_metric must be 'trs' or 'loss'")
    cfg["primary_checkpoint"] = "trs_best.pt" if selection_metric == "trs" else "loss_best.pt"
    cfg["displacement_consistency_schedule"] = {
        "base_weight": float(cfg.get("displacement_first_order_consistency_loss_weight", 0.0)),
        "warmup_epochs": int(cfg.get("displacement_consistency_warmup_epochs", 0)),
        "ramp_epochs": int(cfg.get("displacement_consistency_ramp_epochs", 0)),
        "max_u_head_gradient_ratio": float(cfg.get("max_displacement_consistency_gradient_ratio", 1.0)),
        "rule": "zero through warmup, then linear ramp with an actual U-head gradient-ratio cap",
    }
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8"
    )
    loss_best = float("inf")
    loss_best_epoch = 0
    trs_best = float("-inf")
    trs_best_epoch = 0
    epochs_without_improvement = 0
    patience = int(cfg.get("early_stopping_patience", 0))
    rows: list[dict[str, float | int]] = []
    existing_metrics = output / "metrics.csv"
    if args.resume is not None and existing_metrics.is_file():
        with existing_metrics.open(newline="", encoding="utf-8") as handle:
            rows = [{key: (int(value) if key == "epoch" else float(value)) for key, value in row.items()} for row in csv.DictReader(handle)]
        if rows:
            loss_row = min(rows, key=lambda row: float(row["val_loss"]))
            trs_row = max(
                rows,
                key=lambda row: float(row["val_tensor_response_skill_vs_zero_loss"]),
            )
            loss_best, loss_best_epoch = float(loss_row["val_loss"]), int(loss_row["epoch"])
            trs_best = float(trs_row["val_tensor_response_skill_vs_zero_loss"])
            trs_best_epoch = int(trs_row["epoch"])
            selected_epoch = trs_best_epoch if selection_metric == "trs" else loss_best_epoch
            epochs_without_improvement = int(rows[-1]["epoch"]) - selected_epoch
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        scheduled_consistency_weight = displacement_consistency_weight_for_epoch(
            float(cfg.get("displacement_first_order_consistency_loss_weight", 0.0)),
            epoch,
            int(cfg.get("displacement_consistency_warmup_epochs", 0)),
            int(cfg.get("displacement_consistency_ramp_epochs", 0)),
        )
        response_weights = dict(
            macro_dielectric_weight=float(cfg.get("macro_dielectric_loss_weight", 0.0)),
            macro_elastic_weight=float(cfg.get("macro_elastic_loss_weight", 0.0)),
            dielectric_weight=float(cfg.get("dielectric_loss_weight", 0.0)),
            ionic_dielectric_weight=float(cfg.get("ionic_dielectric_loss_weight", 0.0)),
            ionic_elastic_weight=float(cfg.get("ionic_elastic_loss_weight", 0.0)),
            elastic_weight=float(cfg.get("elastic_loss_weight", 0.0)),
            born_weight=float(cfg.get("born_loss_weight", 0.0)),
            ionic_weight=float(cfg.get("ionic_piezo_loss_weight", 0.0)),
            displacement_response_weight=float(
                cfg.get("displacement_response_loss_weight", 0.0)
            ),
            displacement_consistency_weight=scheduled_consistency_weight,
            max_consistency_gradient_ratio=float(
                cfg.get("max_displacement_consistency_gradient_ratio", 1.0)
            ),
            electronic_weight=float(cfg.get("electronic_piezo_loss_weight", 0.0)),
            branch_sum_weight=float(cfg.get("branch_sum_loss_weight", 0.0)),
            force_weight=float(cfg.get("force_constant_loss_weight", 0.0)),
            internal_strain_weight=float(cfg.get("internal_strain_loss_weight", 0.0)),
            internal_strain_full_weight=float(
                cfg.get("internal_strain_full_loss_weight", 0.0)
            ),
            soft_mode_weight=float(cfg.get("soft_mode_loss_weight", 0.0)),
            low_mode_action_weight=float(cfg.get("low_mode_action_loss_weight", 0.0)),
            low_mode_leak_weight=float(cfg.get("low_mode_leak_loss_weight", 0.0)),
            phi_probe_weight=float(cfg.get("phi_probe_loss_weight", 0.0)),
            lambda_probe_weight=float(cfg.get("lambda_probe_loss_weight", 0.0)),
            born_probe_weight=float(cfg.get("born_probe_loss_weight", 0.0)),
            born_oracle_weight=float(cfg.get("born_oracle_loss_weight", 0.0)),
            phi_oracle_normal_weight=float(cfg.get("phi_oracle_normal_loss_weight", 0.0)),
            response_active_strain_weight=float(
                cfg.get("response_active_strain_loss_weight", 0.0)
            ),
            collect_conditioning_diagnostics=bool(
                cfg.get("collect_conditioning_diagnostics", True)
            ),
        )
        branch_weights = dict(response_weights)
        branch_weights.update(
            macro_dielectric_weight=0.0,
            macro_elastic_weight=0.0,
            displacement_response_weight=0.0,
            displacement_consistency_weight=0.0,
            internal_strain_full_weight=0.0,
            lambda_probe_weight=0.0,
            born_oracle_weight=0.0,
            phi_oracle_normal_weight=0.0,
            ionic_elastic_weight=0.0,
            collect_conditioning_diagnostics=False,
            collect_u_gradient_diagnostics=False,
        )
        strict_weights = {name: 0.0 for name in response_weights}
        strict_weights.update(
            displacement_response_weight=response_weights["displacement_response_weight"],
            displacement_consistency_weight=response_weights["displacement_consistency_weight"],
            max_consistency_gradient_ratio=response_weights["max_consistency_gradient_ratio"],
            internal_strain_full_weight=response_weights["internal_strain_full_weight"],
            lambda_probe_weight=response_weights["lambda_probe_weight"],
            born_oracle_weight=response_weights["born_oracle_weight"],
            phi_oracle_normal_weight=response_weights["phi_oracle_normal_weight"],
            ionic_elastic_weight=response_weights["ionic_elastic_weight"],
            collect_conditioning_diagnostics=bool(
                cfg.get("collect_conditioning_diagnostics", True)
            ),
            collect_u_gradient_diagnostics=bool(
                cfg.get("collect_u_gradient_diagnostics", False)
            ),
        )
        if bool(cfg.get("multistream_training", True)):
            if cfg.get("train_updates_per_epoch") is not None:
                raise ValueError(
                    "Exposure-matched multistream training cannot be capped by optimizer updates"
                )
            macro_value, macro_seconds, _, macro_components, macro_updates = _macro_epoch(
                model, train_loader, optimizer, scale, bin_weights, device,
                dielectric_weight=response_weights["macro_dielectric_weight"],
                elastic_weight=response_weights["macro_elastic_weight"],
            )
            assert branch_loader is not None and strict_loader is not None
            branch_value, branch_seconds, branch_components, branch_updates = _epoch(
                model, branch_loader, optimizer, scale, bin_weights, device,
                macro_weight=0.0, **branch_weights,
            )
            strict_value, strict_seconds, strict_components, strict_updates = _epoch(
                model, strict_loader, optimizer, scale, bin_weights, device,
                macro_weight=0.0, **strict_weights,
            )
            train_value = macro_value + branch_value + strict_value
            train_seconds = macro_seconds + branch_seconds + strict_seconds
            train_updates = macro_updates + branch_updates + strict_updates
            train_components = {
                **{f"macro_stream_{key}": value for key, value in macro_components.items()},
                **{f"branch_stream_{key}": value for key, value in branch_components.items()},
                **{f"strict_stream_{key}": value for key, value in strict_components.items()},
            }
        else:
            train_value, train_seconds, train_components, train_updates = _epoch(
                model, train_loader, optimizer, scale, bin_weights, device,
                max_train_updates=cfg.get("train_updates_per_epoch"), **response_weights,
            )
        val_value, val_seconds, val_components, _ = _epoch(
            model, val_loader, None, scale, bin_weights, device, **response_weights,
        )
        row = {
            "epoch": epoch,
            "scheduled_displacement_consistency_weight": scheduled_consistency_weight,
            "train_loss": train_value,
            "val_loss": val_value,
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "train_optimizer_updates": train_updates,
            "macro_stream_optimizer_updates": macro_updates if bool(cfg.get("multistream_training", True)) else train_updates,
            "branch_stream_optimizer_updates": branch_updates if bool(cfg.get("multistream_training", True)) else 0,
            "strict_stream_optimizer_updates": strict_updates if bool(cfg.get("multistream_training", True)) else 0,
            "macro_effective_passes": epoch if bool(cfg.get("multistream_training", True)) else float("nan"),
            "branch_effective_passes": len(factor_rows) + len(displacement_rows) + epoch if bool(cfg.get("multistream_training", True)) else float("nan"),
            "strict_effective_passes": len(factor_rows) + len(displacement_rows) + epoch if bool(cfg.get("multistream_training", True)) else float("nan"),
            "factor_branch_effective_passes": len(factor_rows),
            "factor_strict_effective_passes": len(factor_rows),
            "joint_branch_effective_passes": epoch,
            "joint_strict_effective_passes": epoch,
            "macro_examples_seen": epoch * len(train_set),
            "branch_examples_seen": (len(factor_rows) + len(displacement_rows) + epoch) * (len(branch_set) if branch_set is not None else 0),
            "strict_examples_seen": (len(factor_rows) + len(displacement_rows) + epoch) * (len(strict_set) if strict_set is not None else 0),
            "macro_unique_samples_seen": len(train_set) if epoch > 0 else 0,
            "branch_unique_samples_seen": len(branch_set) if branch_set is not None and (factor_rows or epoch > 0) else 0,
            "strict_unique_samples_seen": len(strict_set) if strict_set is not None and (factor_rows or epoch > 0) else 0,
        }
        row.update({f"train_{name}_loss": value for name, value in train_components.items()})
        row.update({f"val_{name}_loss": value for name, value in val_components.items()})
        rows.append(row)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "piezo_scale": float(scale),
            "epoch": epoch,
            "validation": {
                "loss": float(val_value),
                "tensor_response_skill_vs_zero": float(
                    val_components["tensor_response_skill_vs_zero"]
                ),
            },
        }
        torch.save(checkpoint, output / "last.pt")
        loss_improved = val_value < loss_best
        if loss_improved:
            loss_best = val_value
            loss_best_epoch = epoch
            torch.save(checkpoint, output / "loss_best.pt")
        val_response_skill = float(val_components["tensor_response_skill_vs_zero"])
        trs_improved = val_response_skill > trs_best
        if trs_improved:
            trs_best = val_response_skill
            trs_best_epoch = epoch
            torch.save(checkpoint, output / "trs_best.pt")
        selected_improved = trs_improved if selection_metric == "trs" else loss_improved
        epochs_without_improvement = 0 if selected_improved else epochs_without_improvement + 1
        print(f"epoch={epoch} train={train_value:.6g} val={val_value:.6g}")
        if patience > 0 and epochs_without_improvement >= patience:
            selected_value = trs_best if selection_metric == "trs" else loss_best
            print(
                f"early_stop epoch={epoch} metric={selection_metric} "
                f"best={selected_value:.6g} patience={patience}"
            )
            break
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint_selection_metric": selection_metric,
        "primary_checkpoint": cfg["primary_checkpoint"],
        "best_val_loss": loss_best,
        "loss_best_epoch": loss_best_epoch,
        "best_val_response_skill": trs_best,
        "trs_best_epoch": trs_best_epoch,
        "loss": "full",
        "epochs": int(cfg["epochs"]),
        "epochs_completed": len(rows),
        "metrics_rows": len(rows),
        "optimization_loss": "full",
        "runtime_device": runtime_device,
        "all_finite": True,
    }
    if rows:
        factor_branch_updates = sum(
            int(row.get("branch_stream_optimizer_updates", 0)) for row in factor_rows
        )
        factor_strict_updates = sum(
            int(row.get("strict_stream_optimizer_updates", 0)) for row in factor_rows
        )
        displacement_branch_updates = sum(
            int(row.get("branch_stream_optimizer_updates", 0)) for row in displacement_rows
        )
        displacement_strict_updates = sum(
            int(row.get("strict_stream_optimizer_updates", 0)) for row in displacement_rows
        )
        joint_macro_updates = sum(
            int(row.get("macro_stream_optimizer_updates", 0)) for row in rows
        )
        joint_branch_updates = sum(
            int(row.get("branch_stream_optimizer_updates", 0)) for row in rows
        )
        joint_strict_updates = sum(
            int(row.get("strict_stream_optimizer_updates", 0)) for row in rows
        )
        consistency_updates = sum(
            int(row.get("strict_stream_optimizer_updates", 0))
            for row in rows
            if float(row.get("scheduled_displacement_consistency_weight", 0.0)) > 0.0
        )

        def active_updates(weight_name: str, updates: int) -> int:
            return updates if float(cfg.get(weight_name, 0.0)) > 0.0 else 0

        summary["exposure"] = {
            "contract": "complete independent stream passes; no update-capped multistream fallback",
            "stream_materials": {
                "macro": len(train_set),
                "branch": len(branch_set) if branch_set is not None else 0,
                "strict": len(strict_set) if strict_set is not None else 0,
            },
            "unique_samples_seen": {
                "macro": len(train_set),
                "branch": len(branch_set) if branch_set is not None else 0,
                "strict": len(strict_set) if strict_set is not None else 0,
            },
            "examples_seen": {
                "macro": len(rows) * len(train_set),
                "branch": (len(factor_rows) + len(displacement_rows) + len(rows))
                * (len(branch_set) if branch_set is not None else 0),
                "strict": (len(factor_rows) + len(displacement_rows) + len(rows))
                * (len(strict_set) if strict_set is not None else 0),
            },
            "optimizer_updates": {
                "factor_branch": factor_branch_updates,
                "factor_strict": factor_strict_updates,
                "teacher_forced_displacement_branch": displacement_branch_updates,
                "teacher_forced_displacement_strict": displacement_strict_updates,
                "joint_macro": joint_macro_updates,
                "joint_branch": joint_branch_updates,
                "joint_strict": joint_strict_updates,
            },
            "label_gradient_updates": {
                "macro_total": joint_macro_updates,
                "branch_born": active_updates("born_loss_weight", joint_branch_updates)
                + (factor_branch_updates if float(cfg.get("factor_pretrain_born_weight", 0.0)) > 0.0 else 0),
                "branch_force_constant": active_updates("force_constant_loss_weight", joint_branch_updates)
                + (factor_branch_updates if float(cfg.get("factor_pretrain_force_weight", 0.0)) > 0.0 else 0),
                "branch_printed_lambda": active_updates("internal_strain_loss_weight", joint_branch_updates)
                + (factor_branch_updates if float(cfg.get("factor_pretrain_internal_strain_weight", 0.0)) > 0.0 else 0),
                "branch_ionic": active_updates("ionic_piezo_loss_weight", joint_branch_updates),
                "branch_electronic": active_updates("electronic_piezo_loss_weight", joint_branch_updates),
                "branch_sum": active_updates("branch_sum_loss_weight", joint_branch_updates),
                "strict_full_lambda": active_updates("internal_strain_full_loss_weight", joint_strict_updates)
                + (factor_strict_updates if float(cfg.get("factor_pretrain_internal_strain_full_weight", 0.0)) > 0.0 else 0),
                "strict_displacement_response": active_updates("displacement_response_loss_weight", joint_strict_updates)
                + (displacement_strict_updates if float(cfg.get("displacement_pretrain_target_weight", 1.0)) > 0.0 else 0),
                "branch_teacher_forced_true_born_ionic": displacement_branch_updates
                if float(cfg.get("displacement_pretrain_true_born_macro_weight", 1.0)) > 0.0 else 0,
                "strict_first_order_consistency": consistency_updates,
            },
            "interpretation": (
                "label_gradient_updates counts optimizer steps containing that supervised objective; "
                "examples_seen counts material occurrences and unique coverage is exact because every "
                "registered stream pass exhausts its DataLoader"
            ),
        }
    if factor_rows:
        summary["factor_pretraining"] = {
            "epochs_completed": len(factor_rows),
            "best_epoch": factor_best_epoch,
            "best_val_loss": factor_best,
            "objective": "weighted direct-factor losses; see objective_weights",
            "objective_weights": factor_weights,
        }
    if displacement_rows:
        summary["displacement_pretraining"] = {
            "epochs_completed": len(displacement_rows),
            "best_epoch": displacement_best_epoch,
            "best_val_loss": displacement_best,
            "target_weight": float(
                cfg.get("displacement_pretrain_target_weight", 1.0)
            ),
            "true_born_macro_weight": float(
                cfg.get("displacement_pretrain_true_born_macro_weight", 1.0)
            ),
            "objective": (
                "true-Phi/Lambda U target plus true-BEC ionic macro supervision"
            ),
        }
    summary["joint_optimizer"] = cfg["joint_optimizer"]
    if resumed_from is not None:
        summary["resumed_from_epoch"] = resumed_from
        summary["resumed_from_commit"] = cfg.get("resumed_from_commit")
        summary["metrics_coverage"] = [int(rows[0]["epoch"]), int(rows[-1]["epoch"])] if rows else []
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
