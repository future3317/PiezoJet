"""Reproducible full-tensor and random mixed-Hessian-sketch training."""

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
from torch.func import jvp
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from .data import PiezoDataset, create_or_load_splits, graph_cache_key, load_gmtnet_records
from .model import PiezoJet, model_from_config
from .metrics import response_tensor_skill
from .tensor_ops import cartesian_to_piezo_voigt, piezo_scale, piezo_to_irreps
from .data import RESPONSE_NORM_BOUNDS


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


def dielectric_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Robust auxiliary supervision for the relaxed dielectric response."""
    mask = mask.reshape(-1).to(dtype=torch.bool)
    if not mask.any():
        return prediction.sum() * 0.0
    residual = prediction[mask] - target[mask]
    scale = target[mask].abs().mean().clamp_min(torch.finfo(target.dtype).eps)
    return torch.nn.functional.smooth_l1_loss(residual / scale, torch.zeros_like(residual))


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
    if batch_index is not None:
        graphs = int(batch_index.max()) + 1
        target = target - scatter(target, batch_index, dim=0, dim_size=graphs, reduce="mean")[batch_index]
    residual = prediction[mask] - target[mask]
    scale = target[mask].abs().mean().clamp_min(torch.finfo(target.dtype).eps)
    return torch.nn.functional.smooth_l1_loss(residual / scale, torch.zeros_like(residual))


def ionic_piezo_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Supervise the DFPT ionic response, i.e. the full Z* Phi^-1 Lambda product."""
    mask = mask.reshape(-1).to(dtype=torch.bool)
    if not mask.any():
        return prediction.sum() * 0.0
    residual = prediction[mask] - target[mask]
    # Centrosymmetric and near-cancelling crystals can have printed values at
    # 1e-5 C/m^2.  Dividing by that numerical residue would dominate every
    # other physical target, so 0.05 C/m^2 is the robust resolution floor.
    scale = target[mask].abs().mean().clamp_min(0.05)
    return torch.nn.functional.smooth_l1_loss(residual / scale, torch.zeros_like(residual))


def response_active_internal_strain_loss(
    prediction: torch.Tensor,
    true_born: torch.Tensor,
    true_force_constants_flat: torch.Tensor,
    target_ionic_piezo: torch.Tensor,
    mask: torch.Tensor,
    node_ptr: torch.Tensor,
    cells: torch.Tensor,
    response,
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
    losses, force_offset = [], 0
    for graph_index in range(node_ptr.numel() - 1):
        atoms = int(node_ptr[graph_index + 1] - node_ptr[graph_index])
        block_values = 9 * atoms * atoms
        if not bool(mask[graph_index]):
            continue
        blocks = true_force_constants_flat[
            force_offset : force_offset + block_values
        ].reshape(atoms, atoms, 3, 3)
        force_offset += block_values
        operator = response.optical_operator(blocks, solve_policy="regularized")
        start, stop = int(node_ptr[graph_index]), int(node_ptr[graph_index + 1])
        coupling = response._coupling_voigt(prediction[start:stop]).reshape(3 * atoms, 6)
        charge = true_born[start:stop].reshape(3 * atoms, 3)
        volume = torch.linalg.det(cells[graph_index]).abs().clamp_min(
            torch.finfo(cells.dtype).eps
        )
        predicted = response.PIEZO_C_PER_M2 * (
            charge.transpose(0, 1) @ operator @ coupling
        ) / volume
        target = cartesian_to_piezo_voigt(target_ionic_piezo[graph_index])
        scale = target.abs().mean().clamp_min(0.05)
        losses.append(
            torch.nn.functional.smooth_l1_loss(
                predicted / scale,
                target / scale,
            )
        )
    if not losses:
        return prediction.sum() * 0.0
    if force_offset != true_force_constants_flat.numel():
        raise ValueError("Ragged force-constant labels did not match response-active mask")
    return torch.stack(losses).mean()


def _translation_projector(atoms: int, reference: torch.Tensor) -> torch.Tensor:
    """Project Cartesian displacements onto the optical subspace."""
    translation = reference.new_zeros(3 * atoms, 3)
    for axis in range(3):
        translation[axis::3, axis] = atoms ** -0.5
    return torch.eye(3 * atoms, dtype=reference.dtype, device=reference.device) - translation @ translation.T


def _near_degenerate_mode_blocks(
    force_constants: torch.Tensor,
    tolerance: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return gauge-safe optical eigenvalue/vector blocks from true DFPT Phi.

    Exact degeneracies have an arbitrary orthogonal basis.  Nearby modes are
    grouped conservatively as well, so the later loss compares a projector
    onto their subspace rather than attempting a fragile eigenvector match.
    """
    if tolerance < 0:
        raise ValueError("Mode degeneracy tolerance must be non-negative")
    atoms = force_constants.shape[0]
    matrix = force_constants.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    projector = _translation_projector(atoms, matrix)
    matrix = projector @ matrix @ projector
    values, vectors = torch.linalg.eigh(matrix)
    if values.numel() <= 3:
        return []
    optical = torch.argsort(values.abs())[3:]
    optical = optical[torch.argsort(values[optical])]
    values, vectors = values[optical], vectors[:, optical]
    blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
    start = 0
    for index in range(1, values.numel() + 1):
        boundary = index == values.numel()
        if not boundary:
            left, right = values[index - 1], values[index]
            scale = torch.maximum(torch.ones((), dtype=values.dtype, device=values.device), torch.maximum(left.abs(), right.abs()))
            boundary = bool((right - left).abs() > tolerance * scale)
        if boundary:
            blocks.append((values[start:index], vectors[:, start:index]))
            start = index
    return blocks


def mode_aware_internal_strain_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    true_born: torch.Tensor,
    true_force_constants_flat: torch.Tensor,
    graph_mask: torch.Tensor,
    node_ptr: torch.Tensor,
    response: AtomCoordinateResponsePotential,
    degeneracy_tolerance: float = 1e-2,
    epsilon: float = 1e-6,
    direction_weight: float = 1.0,
    amplitude_weight: float = 1.0,
    sign_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Response-weighted, degeneracy-safe supervision on complete Lambda.

    For each true-DFPT optical subspace ``V_g`` this compares
    ``V_g.T Lambda`` rather than an arbitrary individual eigenvector.  The
    block is weighted by its mode-effective charge and the declared signed
    regularization.  Direction and amplitude are Frobenius comparisons in the
    block; the signed-hinge term acts on the gauge-invariant ionic block
    ``Z*.T V_g V_g.T Lambda``.  Thus no per-mode sign/eigenvector convention is
    imposed on exact or near-degenerate branches.
    """
    if epsilon <= 0:
        raise ValueError("Mode-aware epsilon must be positive")
    graph_mask = graph_mask.reshape(-1).to(dtype=torch.bool)
    force_offset = 0
    weighted_direction: list[tuple[torch.Tensor, torch.Tensor]] = []
    weighted_amplitude: list[tuple[torch.Tensor, torch.Tensor]] = []
    weighted_sign: list[tuple[torch.Tensor, torch.Tensor]] = []
    for graph_index in range(node_ptr.numel() - 1):
        start, stop = int(node_ptr[graph_index]), int(node_ptr[graph_index + 1])
        atoms = stop - start
        block_values = 9 * atoms * atoms
        force_target = true_force_constants_flat[force_offset : force_offset + block_values].reshape(atoms, atoms, 3, 3)
        force_offset += block_values
        if not graph_mask[graph_index]:
            continue
        # Eigenvectors are a fixed true-label basis, so the prediction remains
        # differentiable while the diagonalization cannot create a learned
        # eigengauge.  Float64 is required near soft crossings.
        force64 = force_target.to(torch.float64)
        blocks = _near_degenerate_mode_blocks(force64, degeneracy_tolerance)
        predicted_coupling = response._coupling_voigt(prediction[start:stop]).reshape(3 * atoms, 6)
        target_coupling = response._coupling_voigt(target[start:stop]).reshape(3 * atoms, 6)
        charge = true_born[start:stop].reshape(3 * atoms, 3)
        for eigenvalues64, basis64 in blocks:
            basis = basis64.to(dtype=prediction.dtype)
            eigenvalues = eigenvalues64.to(dtype=prediction.dtype)
            projected_charge = basis.transpose(0, 1) @ charge
            # A block-average spectral denominator is invariant under an
            # arbitrary rotation inside a near-degenerate subspace.
            weight = projected_charge.square().sum() / (
                eigenvalues.square().mean() + response.optical_regularization ** 2
            )
            if not torch.isfinite(weight) or bool(weight <= 0):
                continue
            predicted_gamma = basis.transpose(0, 1) @ predicted_coupling
            target_gamma = basis.transpose(0, 1) @ target_coupling
            predicted_norm = torch.linalg.vector_norm(predicted_gamma)
            target_norm = torch.linalg.vector_norm(target_gamma)
            if bool(target_norm > epsilon):
                cosine = (predicted_gamma * target_gamma).sum() / (predicted_norm * target_norm).clamp_min(epsilon)
                weighted_direction.append((weight, 1.0 - cosine))
            log_ratio = torch.log((predicted_norm + epsilon) / (target_norm + epsilon))
            weighted_amplitude.append((weight, torch.nn.functional.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio))))
            # The average signed Green filter yields a block contribution that
            # is invariant under any rotation of the chosen subspace basis.
            signed_filter = (eigenvalues / (eigenvalues.square() + response.optical_regularization ** 2)).mean()
            projector = basis @ basis.transpose(0, 1)
            predicted_contribution = signed_filter * charge.transpose(0, 1) @ projector @ predicted_coupling
            target_contribution = signed_filter * charge.transpose(0, 1) @ projector @ target_coupling
            active = target_contribution.abs() > epsilon
            if bool(active.any()):
                sign_scale = target_contribution[active].abs().mean().clamp_min(epsilon)
                signed_margin = torch.sign(target_contribution[active]) * predicted_contribution[active] / sign_scale
                # A zero-margin hinge is zero whenever the ionic contribution
                # has the correct sign, so it does not keep inflating an
                # already correctly scaled mode after the amplitude term has
                # converged.
                weighted_sign.append((weight, torch.nn.functional.relu(-signed_margin).mean()))
    if force_offset != true_force_constants_flat.numel():
        raise ValueError("Ragged force-constant labels did not match mode-aware mask")
    zero = prediction.sum() * 0.0

    def reduce(values: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if not values:
            return zero
        weights = torch.stack([weight for weight, _ in values])
        losses = torch.stack([loss for _, loss in values])
        return (weights * losses).sum() / weights.sum().clamp_min(epsilon)

    direction, amplitude, sign = reduce(weighted_direction), reduce(weighted_amplitude), reduce(weighted_sign)
    return {
        "direction": direction,
        "amplitude": amplitude,
        "sign": sign,
        "total": direction_weight * direction + amplitude_weight * amplitude + sign_weight * sign,
    }


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
        translation = target_matrix.new_zeros(3 * atoms, 3)
        for axis in range(3):
            translation[axis::3, axis] = atoms ** -0.5
        projector = torch.eye(3 * atoms, dtype=target.dtype, device=target.device) - translation @ translation.transpose(0, 1)
        target_matrix = projector @ target_matrix @ projector
        cleaned = target_matrix.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)
        scale = cleaned.abs().mean().clamp_min(torch.finfo(cleaned.dtype).eps)
        losses.append(torch.nn.functional.smooth_l1_loss(predicted / scale, cleaned / scale))
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
        translation = target.new_zeros(3 * atoms, 3)
        for axis in range(3):
            translation[axis::3, axis] = atoms ** -0.5
        projector = torch.eye(3 * atoms, dtype=target.dtype, device=target.device) - translation @ translation.T
        predicted, target = projector @ predicted @ projector, projector @ target @ projector
        predicted_values = torch.linalg.eigvalsh(predicted)
        target_values = torch.linalg.eigvalsh(target)
        predicted_values = predicted_values[torch.argsort(predicted_values.abs())[3:]].sort().values
        target_values = target_values[torch.argsort(target_values.abs())[3:]].sort().values
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
        scale = target.abs().mean().clamp_min(torch.finfo(target.dtype).eps)
        losses.append(torch.nn.functional.smooth_l1_loss(selected / scale, target / scale))
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
    node_mask = graph_mask[batch_index]
    selected_prediction = prediction[node_mask]
    selected_target = target[node_mask]
    scale = selected_target.abs().mean().clamp_min(
        torch.finfo(selected_target.dtype).eps
    )
    return torch.nn.functional.smooth_l1_loss(
        selected_prediction / scale,
        selected_target / scale,
    )


def sketch_loss(model: PiezoJet, batch, target_voigt: torch.Tensor, piezo_cart: torch.Tensor | None = None) -> torch.Tensor:
    """One Gaussian projection of the physical response energy density."""
    graphs = target_voigt.shape[0]
    field0 = torch.zeros(graphs, 3, device=target_voigt.device, dtype=target_voigt.dtype)
    eta0 = torch.zeros(graphs, 6, device=target_voigt.device, dtype=target_voigt.dtype)
    a, b = torch.randn_like(field0), torch.randn_like(eta0)

    def eta_direction(field: torch.Tensor) -> torch.Tensor:
        _, tangent = jvp(lambda eta: model.potential(batch, field, eta), (eta0,), (b,))
        return tangent

    _, mixed = jvp(eta_direction, (field0,), (a,))
    target = torch.einsum("bi,bij,bj->b", a, target_voigt, b)
    predicted_si = -mixed * model.response.PIEZO_C_PER_M2
    return torch.mean((predicted_si - target).square())


def direct_sketch_loss(prediction: torch.Tensor, target_voigt: torch.Tensor, sketches: int = 1) -> torch.Tensor:
    values = []
    predicted = cartesian_to_piezo_voigt(prediction)
    for _ in range(sketches):
        field = torch.randn(target_voigt.shape[0], 3, device=target_voigt.device, dtype=target_voigt.dtype)
        strain = torch.randn_like(target_voigt[..., 0, :])
        values.append((torch.einsum("bi,bij,bj->b", field, predicted, strain) - torch.einsum("bi,bij,bj->b", field, target_voigt, strain)).square())
    return torch.stack(values).mean()


def _diagnostics(prediction: torch.Tensor, target: torch.Tensor, batch, model: PiezoJet, normalized_scale: torch.Tensor) -> list[dict[str, float | int]]:
    pred_voigt = cartesian_to_piezo_voigt(prediction)
    target_voigt = cartesian_to_piezo_voigt(target)
    diff_voigt = pred_voigt - target_voigt
    pred_irreps = piezo_to_irreps(prediction)
    target_irreps = piezo_to_irreps(target)
    irrep_diff = pred_irreps - target_irreps
    block_slices = (("1o", 0, 6), ("2o", 6, 11), ("3o", 11, 18))
    node_counts = (batch.ptr[1:] - batch.ptr[:-1]).tolist() if hasattr(batch, "ptr") else [batch.num_nodes]
    degree = torch.zeros(batch.num_nodes, device=prediction.device, dtype=torch.long)
    degree.scatter_add_(0, batch.edge_index[1], torch.ones_like(batch.edge_index[1]))
    isolated = int((degree == 0).sum())
    grad_norm = torch.sqrt(sum((parameter.grad.detach().square().sum() for parameter in model.parameters() if parameter.grad is not None), torch.zeros((), device=prediction.device)))
    rows = []
    for index in range(prediction.shape[0]):
        row: dict[str, float | int] = {
            "sample_index": index,
            "unnormalized_tensor_mse": float(diff_voigt[index].square().mean()),
            "frob_error": float(torch.linalg.vector_norm(diff_voigt[index]) / torch.sqrt(torch.tensor(18.0, device=prediction.device))),
            "normalized_frob_error": float(torch.linalg.vector_norm(diff_voigt[index]) / (normalized_scale * torch.sqrt(torch.tensor(18.0, device=prediction.device)))),
            "predicted_tensor_norm": float(torch.linalg.vector_norm(pred_voigt[index])),
            "target_tensor_norm": float(torch.linalg.vector_norm(target_voigt[index])),
            "gradient_norm": float(grad_norm),
            "atom_count": int(node_counts[index]),
            "isolated_nodes_in_batch": isolated,
        }
        for name, start, end in block_slices:
            row[f"irrep_{name}_error"] = float(torch.linalg.vector_norm(irrep_diff[index, start:end]))
        rows.append(row)
    return rows


def _epoch(model, loader, optimizer, loss_name: str, scale: torch.Tensor, bin_weights: torch.Tensor, device: torch.device, full_weight: float, dielectric_weight: float = 0.0, born_weight: float = 0.0, ionic_weight: float = 0.0, force_weight: float = 0.0, internal_strain_weight: float = 0.0, internal_strain_full_weight: float = 0.0, soft_mode_weight: float = 0.0, response_active_strain_weight: float = 0.0, mode_aware_strain_weight: float = 0.0, mode_aware_degeneracy_tolerance: float = 1e-2, mode_aware_epsilon: float = 1e-6, mode_aware_direction_weight: float = 1.0, mode_aware_amplitude_weight: float = 1.0, mode_aware_sign_weight: float = 0.25, collect_diagnostics: bool = False, sketch_implementation: str = "jvp", sketch_count: int = 1) -> tuple[float, float, list[dict[str, float | int]], dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total, count, elapsed, diagnostics = 0.0, 0, 0.0, []
    response_predictions, response_targets = [], []
    component_totals = {
        "piezo_full": 0.0,
        "dielectric": 0.0,
        "born": 0.0,
        "force_constant": 0.0,
        "soft_optical": 0.0,
        "internal_strain": 0.0,
        "internal_strain_full": 0.0,
        "response_active_strain": 0.0,
        "mode_aware_strain": 0.0,
        "ionic_piezo": 0.0,
    }
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            components = model.predict_components(batch)
            prediction = components.tensor
            full = full_loss(prediction, batch.y, scale, bin_weights)
            dielectric_component = dielectric_loss(
                components.dielectric, batch.y_dielectric, batch.dielectric_mask
            )
            born_component = born_loss(
                components.born_charges, batch.y_born, batch.born_mask, batch.batch
            )
            ionic_component = ionic_piezo_loss(
                components.ionic_piezo, batch.y_ionic_piezo, batch.ionic_piezo_mask
            )
            force_component = force_constant_loss(
                components.force_constants_flat, batch.dfpt_force_constants_flat, batch.ptr,
                batch.force_constant_mask,
            )
            soft_component = soft_optical_eigenvalue_loss(
                components.force_constants_flat, batch.dfpt_force_constants_flat, batch.ptr,
                batch.force_constant_mask,
            )
            internal_component = internal_strain_loss(
                components.internal_strain, batch.dfpt_internal_strain_flat,
                batch.dfpt_internal_strain_ions, batch.dfpt_internal_strain_directions,
                batch.dfpt_internal_strain_count, batch.ptr,
            )
            full_internal_component = full_internal_strain_loss(
                components.internal_strain,
                batch.dfpt_internal_strain_full,
                batch.internal_strain_full_mask,
                batch.batch,
            )
            active_strain_component = response_active_internal_strain_loss(
                components.internal_strain,
                batch.y_born,
                batch.dfpt_force_constants_flat,
                batch.y_ionic_piezo,
                batch.ionic_piezo_mask,
                batch.ptr,
                batch.cell.reshape(-1, 3, 3),
                model.response,
            )
            mode_aware_component = prediction.sum() * 0.0
            if mode_aware_strain_weight != 0.0:
                mode_aware_component = mode_aware_internal_strain_terms(
                    components.internal_strain,
                    batch.dfpt_internal_strain_full,
                    batch.y_born,
                    batch.dfpt_force_constants_flat,
                    batch.internal_strain_full_mask,
                    batch.ptr,
                    model.response,
                    mode_aware_degeneracy_tolerance,
                    mode_aware_epsilon,
                    mode_aware_direction_weight,
                    mode_aware_amplitude_weight,
                    mode_aware_sign_weight,
                )["total"]
            auxiliary = (
                dielectric_weight * dielectric_component
                + born_weight * born_component
                + ionic_weight * ionic_component
                + force_weight * force_component
                + soft_mode_weight * soft_component
                + internal_strain_weight * internal_component
                + internal_strain_full_weight * full_internal_component
                + response_active_strain_weight * active_strain_component
                + mode_aware_strain_weight * mode_aware_component
            )
            if loss_name == "full":
                loss = full + auxiliary
            else:
                target_voigt = cartesian_to_piezo_voigt(batch.y)
                sketch = direct_sketch_loss(prediction, target_voigt, sketch_count) if sketch_implementation == "direct" else sketch_loss(model, batch, target_voigt, prediction)
                loss = sketch + auxiliary if loss_name == "sketch" else sketch + full_weight * full + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite optimization loss encountered")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                    raise FloatingPointError("Non-finite parameter gradient encountered")
                optimizer.step()
        if collect_diagnostics:
            diagnostics.extend(_diagnostics(prediction.detach(), batch.y.detach(), batch, model, scale))
        total += float(loss.detach()) * batch.num_graphs
        response_predictions.append(prediction.detach().cpu())
        response_targets.append(batch.y.detach().cpu())
        detached_components = {
            "piezo_full": full,
            "dielectric": dielectric_component,
            "born": born_component,
            "force_constant": force_component,
            "soft_optical": soft_component,
            "internal_strain": internal_component,
            "internal_strain_full": full_internal_component,
            "response_active_strain": active_strain_component,
            "mode_aware_strain": mode_aware_component,
            "ionic_piezo": ionic_component,
        }
        for name, value in detached_components.items():
            component_totals[name] += float(value.detach()) * batch.num_graphs
        count += batch.num_graphs
        elapsed += time.perf_counter() - start
    denominator = max(count, 1)
    component_summary = {
        name: value / denominator for name, value in component_totals.items()
    }
    component_summary["tensor_response_skill_vs_zero"] = float(
        response_tensor_skill(
            torch.cat(response_predictions), torch.cat(response_targets)
        )["tensor_response_skill_vs_zero"]
    )
    return total / denominator, elapsed, diagnostics, component_summary


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
    response_active_strain_weight: float,
    mode_aware_strain_weight: float = 0.0,
    mode_aware_degeneracy_tolerance: float = 1e-2,
    mode_aware_epsilon: float = 1e-6,
    mode_aware_direction_weight: float = 1.0,
    mode_aware_amplitude_weight: float = 1.0,
    mode_aware_sign_weight: float = 0.25,
) -> tuple[float, float, dict[str, float]]:
    """Train direct DFPT factors without the ill-conditioned inverse response path."""
    training = optimizer is not None
    model.train(training)
    total, count, elapsed = 0.0, 0, 0.0
    component_totals = {"born": 0.0, "force_constant": 0.0, "soft_optical": 0.0, "internal_strain": 0.0, "internal_strain_full": 0.0, "response_active_strain": 0.0, "mode_aware_strain": 0.0}
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            factors = model.predict_factors(batch)
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
                ),
            }
            components["mode_aware_strain"] = factors.internal_strain.sum() * 0.0
            if mode_aware_strain_weight != 0.0:
                components["mode_aware_strain"] = mode_aware_internal_strain_terms(
                    factors.internal_strain,
                    batch.dfpt_internal_strain_full,
                    batch.y_born,
                    batch.dfpt_force_constants_flat,
                    batch.internal_strain_full_mask,
                    batch.ptr,
                    model.response,
                    mode_aware_degeneracy_tolerance,
                    mode_aware_epsilon,
                    mode_aware_direction_weight,
                    mode_aware_amplitude_weight,
                    mode_aware_sign_weight,
                )["total"]
            loss = (
                born_weight * components["born"]
                + force_weight * components["force_constant"]
                + soft_mode_weight * components["soft_optical"]
                + internal_strain_weight * components["internal_strain"]
                + internal_strain_full_weight * components["internal_strain_full"]
                + response_active_strain_weight * components["response_active_strain"]
                + mode_aware_strain_weight * components["mode_aware_strain"]
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
        total += float(loss.detach()) * batch.num_graphs
        for name, value in components.items():
            component_totals[name] += float(value.detach()) * batch.num_graphs
        count += batch.num_graphs
        elapsed += time.perf_counter() - start
    denominator = max(count, 1)
    return total / denominator, elapsed, {
        name: value / denominator for name, value in component_totals.items()
    }


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def freeze_factor_stack(model: PiezoJet) -> list[str]:
    """Freeze the factor representation after direct-factor model selection.

    Joint response fine-tuning may then learn electronic and macroscopic
    branches without using them to cancel errors by rewriting Z*, Phi, or
    Lambda.  The physical ionic response remains in every forward pass.
    """
    names = ["encoder", "born_head", "local_polar_mode", "global_context"]
    if model.factor_architecture in {
        "energy", "energy_learned_strain", "energy_learned_star"
    }:
        names.append("energy_factors")
    else:
        names.extend(("force_constants", "internal_strain"))
    for name in names:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(False)
    return names


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
    parser.add_argument("--loss", choices=("full", "sketch", "hybrid"), default="full")
    parser.add_argument("--overfit-32", action="store_true")
    parser.add_argument("--epochs", type=int, help="Override config epochs for a bounded smoke run")
    parser.add_argument("--batch-size", type=int, help="Override config batch size")
    parser.add_argument("--learning-rate", type=float, help="Override config learning rate")
    parser.add_argument("--weight-decay", type=float, help="Override config weight decay")
    parser.add_argument("--early-stopping-patience", type=int, help="Override early stopping patience; set 0 to disable for a controlled diagnostic run")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader workers; set 0 for constrained Windows shared-memory diagnostics")
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
    parser.add_argument("--m2-1", action="store_true", help="Strict 300-epoch 32-sample memorization experiment")
    parser.add_argument("--resume", type=Path, help="Resume a saved checkpoint at its next epoch")
    parser.add_argument(
        "--factor-checkpoint",
        type=Path,
        help="Initialize a fresh joint stage from a selected factor checkpoint without restoring its optimizer/epoch",
    )
    parser.add_argument("--sketch-implementation", choices=("direct", "jvp"), default="jvp")
    parser.add_argument("--sketch-count", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--seed", type=int, help="Override config seed for multi-seed experiments")
    parser.add_argument("--factor-pretrain-epochs", type=int, help="Direct Z*, Phi, Lambda curriculum epochs before joint response training")
    parser.add_argument("--factor-pretrain-learning-rate", type=float, help="Learning rate for direct-factor curriculum")
    parser.add_argument("--factor-pretrain-patience", type=int, help="Early-stopping patience for direct-factor validation loss")
    parser.add_argument(
        "--mode-aware-strain-loss-weight", type=float,
        help="Override the gauge-safe response-subspace loss weight during joint training",
    )
    parser.add_argument(
        "--factor-pretrain-mode-aware-strain-weight", type=float,
        help="Override the gauge-safe response-subspace loss weight during direct-factor training",
    )
    parser.add_argument(
        "--factor-architecture",
        choices=("legacy", "energy", "energy_learned_strain", "energy_learned_star"),
        help="Override the factor architecture for a controlled matched-budget comparison",
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=("trs", "loss"),
        help="Preregister the joint-stage early-stopping and primary-checkpoint metric",
    )
    parser.add_argument("--freeze-factors-during-joint", action="store_true", help="Freeze the selected direct-factor stack during joint response fine-tuning")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.m2_1:
        cfg["epochs"] = 300
        cfg["batch_size"] = 32
        cfg["weight_decay"] = 0.0
        cfg["output_dir"] = "outputs/m2_1"
        args.overfit_32 = True
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        cfg["epochs"] = args.epochs
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
    if args.mode_aware_strain_loss_weight is not None:
        if args.mode_aware_strain_loss_weight < 0:
            raise ValueError("--mode-aware-strain-loss-weight must be non-negative")
        cfg["mode_aware_strain_loss_weight"] = args.mode_aware_strain_loss_weight
    if args.factor_pretrain_mode_aware_strain_weight is not None:
        if args.factor_pretrain_mode_aware_strain_weight < 0:
            raise ValueError("--factor-pretrain-mode-aware-strain-weight must be non-negative")
        cfg["factor_pretrain_mode_aware_strain_weight"] = args.factor_pretrain_mode_aware_strain_weight
    if args.factor_architecture is not None:
        cfg["factor_architecture"] = args.factor_architecture
    if args.checkpoint_selection_metric is not None:
        cfg["checkpoint_selection_metric"] = args.checkpoint_selection_metric
    if args.freeze_factors_during_joint:
        cfg["freeze_factors_during_joint"] = True
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
        text = args.material_ids_file.read_text(encoding="utf-8")
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
    elif args.overfit_32:
        splits["train"] = splits["train"][:32]
        splits["val"] = splits["train"]
    cache_key = graph_cache_key(records, cfg["cutoff"], cfg["max_neighbors"])
    dfpt_dir = cfg.get("jarvis_dfpt_dir")
    strain_completion_dir = cfg.get("jarvis_strain_completion_dir")
    train_set = PiezoDataset(records, splits["train"], cfg["cutoff"], cfg["max_neighbors"], processed_dir=cfg["processed_dir"], cache_key=cache_key, dfpt_dir=dfpt_dir, strain_completion_dir=strain_completion_dir)
    val_set = PiezoDataset(records, splits["val"], cfg["cutoff"], cfg["max_neighbors"], processed_dir=cfg["processed_dir"], cache_key=cache_key, dfpt_dir=dfpt_dir, strain_completion_dir=strain_completion_dir)
    loader_options = {"num_workers": cfg["num_workers"], "pin_memory": device.type == "cuda"}
    if cfg["num_workers"] > 0:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, **loader_options)
    train_ids = set(splits["train"])
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
    model.encoder.load_state_dict(pretrained["encoder"], strict=True)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cfg["loss"] = args.loss
    cfg["pretrained_encoder"] = str(pretraining_path)
    cfg["pretraining_epoch"] = pretrained.get("epoch")
    cfg["git_commit"] = _git_commit()
    cfg["data_commit"] = data_commit
    cfg["runtime_device"] = runtime_device
    factor_rows: list[dict[str, float | int]] = []
    factor_best = float("inf")
    factor_best_epoch = 0
    factor_epochs = int(cfg.get("factor_pretrain_epochs", 0))
    if factor_epochs > 0 and args.resume is None and args.factor_checkpoint is None:
        factor_weights = {
            "born_weight": float(cfg.get("factor_pretrain_born_weight", 1.0)),
            "force_weight": float(cfg.get("factor_pretrain_force_weight", 1.0)),
            "internal_strain_weight": float(cfg.get("factor_pretrain_internal_strain_weight", 5.0)),
            "internal_strain_full_weight": float(
                cfg.get("factor_pretrain_internal_strain_full_weight", 0.0)
            ),
            "soft_mode_weight": float(cfg.get("factor_pretrain_soft_mode_weight", 1.0)),
            "response_active_strain_weight": float(
                cfg.get("factor_pretrain_response_active_strain_weight", 0.0)
            ),
            "mode_aware_strain_weight": float(
                cfg.get("factor_pretrain_mode_aware_strain_weight", 0.0)
            ),
            "mode_aware_degeneracy_tolerance": float(
                cfg.get("mode_aware_degeneracy_tolerance", 1e-2)
            ),
            "mode_aware_epsilon": float(cfg.get("mode_aware_epsilon", 1e-6)),
            "mode_aware_direction_weight": float(
                cfg.get("mode_aware_direction_weight", 1.0)
            ),
            "mode_aware_amplitude_weight": float(
                cfg.get("mode_aware_amplitude_weight", 1.0)
            ),
            "mode_aware_sign_weight": float(cfg.get("mode_aware_sign_weight", 0.25)),
        }
        factor_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.get("factor_pretrain_learning_rate", cfg["learning_rate"])),
            weight_decay=float(cfg["weight_decay"]),
        )
        factor_wait, factor_patience = 0, int(cfg.get("factor_pretrain_patience", 0))
        for factor_epoch in range(1, factor_epochs + 1):
            train_factor, train_factor_seconds, train_factor_components = _factor_epoch(
                model, train_loader, factor_optimizer, device, **factor_weights
            )
            val_factor, val_factor_seconds, val_factor_components = _factor_epoch(
                model, val_loader, None, device, **factor_weights
            )
            factor_row = {
                "epoch": factor_epoch,
                "train_loss": train_factor,
                "val_loss": val_factor,
                "train_seconds": train_factor_seconds,
                "val_seconds": val_factor_seconds,
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
        if args.resume is not None:
            raise ValueError("--factor-checkpoint and --resume are mutually exclusive")
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

    if bool(cfg.get("freeze_factors_during_joint", False)):
        if factor_epochs <= 0 and args.resume is None and args.factor_checkpoint is None:
            raise ValueError("Frozen-factor joint training requires direct factor pretraining")
        cfg["joint_frozen_modules"] = freeze_factor_stack(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
    )
    start_epoch = 1
    resumed_from = None
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        resumed_from = int(saved["epoch"])
        start_epoch = resumed_from + 1
        cfg["resumed_from_epoch"] = resumed_from
        cfg["resumed_from_commit"] = saved.get("config", {}).get("git_commit", "unknown")
    selection_metric = str(cfg.get("checkpoint_selection_metric", "trs"))
    if selection_metric not in {"trs", "loss"}:
        raise ValueError("checkpoint_selection_metric must be 'trs' or 'loss'")
    cfg["primary_checkpoint"] = "trs_best.pt" if selection_metric == "trs" else "loss_best.pt"
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
    all_sample_rows: list[dict[str, float | int]] = []
    diagnostic_rows: list[dict[str, float | int]] = []
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
        response_weights = dict(
            dielectric_weight=float(cfg.get("dielectric_loss_weight", 0.0)),
            born_weight=float(cfg.get("born_loss_weight", 0.0)),
            ionic_weight=float(cfg.get("ionic_piezo_loss_weight", 0.0)),
            force_weight=float(cfg.get("force_constant_loss_weight", 0.0)),
            internal_strain_weight=float(cfg.get("internal_strain_loss_weight", 0.0)),
            internal_strain_full_weight=float(
                cfg.get("internal_strain_full_loss_weight", 0.0)
            ),
            soft_mode_weight=float(cfg.get("soft_mode_loss_weight", 0.0)),
            response_active_strain_weight=float(
                cfg.get("response_active_strain_loss_weight", 0.0)
            ),
            mode_aware_strain_weight=float(
                cfg.get("mode_aware_strain_loss_weight", 0.0)
            ),
            mode_aware_degeneracy_tolerance=float(
                cfg.get("mode_aware_degeneracy_tolerance", 1e-2)
            ),
            mode_aware_epsilon=float(cfg.get("mode_aware_epsilon", 1e-6)),
            mode_aware_direction_weight=float(cfg.get("mode_aware_direction_weight", 1.0)),
            mode_aware_amplitude_weight=float(cfg.get("mode_aware_amplitude_weight", 1.0)),
            mode_aware_sign_weight=float(cfg.get("mode_aware_sign_weight", 0.25)),
            collect_diagnostics=args.m2_1,
            sketch_implementation=args.sketch_implementation,
            sketch_count=args.sketch_count,
        )
        train_value, train_seconds, train_diagnostics, train_components = _epoch(
            model, train_loader, optimizer, args.loss, scale, bin_weights, device,
            cfg["hybrid_full_weight"], **response_weights,
        )
        val_value, val_seconds, val_diagnostics, val_components = _epoch(
            model, val_loader, None, args.loss, scale, bin_weights, device,
            cfg["hybrid_full_weight"], **response_weights,
        )
        row = {"epoch": epoch, "train_loss": train_value, "val_loss": val_value, "train_seconds": train_seconds, "val_seconds": val_seconds}
        row.update({f"train_{name}_loss": value for name, value in train_components.items()})
        row.update({f"val_{name}_loss": value for name, value in val_components.items()})
        rows.append(row)
        if args.m2_1:
            all_sample_rows.extend({"epoch": epoch, "phase": "train", **item} for item in train_diagnostics)
            all_sample_rows.extend({"epoch": epoch, "phase": "eval", **item} for item in val_diagnostics)
            diagnostic_rows.append({"epoch": epoch, "phase": "train", **{key: value for key, value in train_diagnostics[0].items() if key != "sample_index"}})
            diagnostic_rows.append({"epoch": epoch, "phase": "eval", **{key: value for key, value in val_diagnostics[0].items() if key != "sample_index"}})
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
        if not args.m2_1 or epoch == 1 or epoch % 10 == 0 or epoch == int(cfg["epochs"]):
            print(f"epoch={epoch} train={train_value:.6g} val={val_value:.6g}")
        if args.m2_1:
            with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
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
        "loss": args.loss,
        "epochs": int(cfg["epochs"]),
        "epochs_completed": len(rows),
        "metrics_rows": len(rows),
        "optimization_loss": args.loss,
        "memorization_loss": args.loss if args.m2_1 else None,
        "runtime_device": runtime_device,
        "all_finite": True,
    }
    if factor_rows:
        summary["factor_pretraining"] = {
            "epochs_completed": len(factor_rows),
            "best_epoch": factor_best_epoch,
            "best_val_loss": factor_best,
            "objective": "weighted direct-factor losses; see objective_weights",
            "objective_weights": factor_weights,
        }
    if resumed_from is not None:
        summary["resumed_from_epoch"] = resumed_from
        summary["resumed_from_commit"] = cfg.get("resumed_from_commit")
        summary["metrics_coverage"] = [int(rows[0]["epoch"]), int(rows[-1]["epoch"])] if rows else []
    if args.m2_1:
        with (output / "sample_errors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_sample_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_sample_rows)
        with (output / "diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=diagnostic_rows[0].keys())
            writer.writeheader()
            writer.writerows(diagnostic_rows)
        summary["diagnostics"] = ["sample_errors.csv", "diagnostics.csv"]
        summary["interpretation_boundary"] = "Same-cohort memorization only; not validation generalization."
        report = f"""# M2.1 strict memorization test

## Git commit

`{cfg['git_commit']}`

## Data manifest

GMTNet commit `{cfg['data_commit']}`; fixed first 32 material IDs from the existing seed-42 training split.

## Configuration

300 epochs, batch size 32, full tensor loss, weight decay 0, seed {cfg['seed']}, dropout disabled.

## What was implemented

Per-sample Cartesian/Voigt errors, irreps block errors, gradient norms, predicted/target tensor norms, graph atom statistics, and non-finite checks.

## Interpretation boundary

This is a strict same-cohort memorization test. It does not estimate random-split or chemical OOD generalization.
"""
        (output / "report.md").write_text(report, encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
