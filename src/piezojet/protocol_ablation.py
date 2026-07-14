"""Registered 69-label factor-preservation optimization ablation.

This entry point deliberately keeps the frozen strict-completion test panel
out of optimization.  It distinguishes four matched 100-update protocols and
one registered factor-convergence follow-up:

``A``
    100 direct-factor updates only;
``B``
    50 direct-factor updates, restore the validation-selected factor state,
    then 50 joint updates with the factor stack frozen;
``C``
    the same as B, but leave the factor stack trainable in the joint stage;
``D``
    50 alternating direct-factor/joint pairs, with factors trainable.
``E``
    100 direct-factor updates, restore the validation-selected factor state,
    then 50 joint updates with the factor stack frozen.
``F``
    50 direct-factor updates, restore the validation-selected factor state,
    then 50 joint updates in which a conflicting macroscopic-response gradient
    is projected away from the direct-factor gradient on the shared factor
    stack.  Response-only heads remain unconstrained.
``G``
    The same as F, followed by unit-norm matching of the projected response
    gradient to the direct-factor gradient on the shared factor stack.  This
    removes the measured response-gradient scale disparity without selecting a
    loss multiplier from the test panel.

All model selection is by the relevant validation loss.  The test set is
evaluated exactly once after selection, solely to emit a persisted diagnostic.
This is an optimization forensic, not a replacement production benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import yaml
from torch_geometric.loader import DataLoader

from .data import PiezoDataset, create_or_load_splits, graph_cache_key, load_gmtnet_records
from .evaluate_dfpt import FactorAccumulator, FACTOR_FLOORS, ionic_aggregate_metrics, ionic_piezo_from_factors
from .metrics import response_tensor_skill
from .model import PiezoJet, model_from_config
from .tensor_ops import piezo_scale
from .train import (
    _data_commit,
    _git_commit,
    born_loss,
    device_from_config,
    dielectric_loss,
    force_constant_loss,
    freeze_factor_stack,
    full_internal_strain_loss,
    full_loss,
    internal_strain_loss,
    ionic_piezo_loss,
    load_explicit_splits,
    response_active_internal_strain_loss,
    response_bin_weights,
    seed_everything,
    soft_optical_eigenvalue_loss,
)


PROTOCOLS = ("A", "B", "C", "D", "E", "F", "G")
# Preserve the registered feedback-4 ``--protocol all`` behavior.  E--G are
# separate, evidence-driven follow-ups and must be requested explicitly.
LEGACY_ALL_PROTOCOLS = ("A", "B", "C", "D")


def _factor_weights(cfg: dict) -> dict[str, float]:
    return {
        "born": float(cfg.get("factor_pretrain_born_weight", 1.0)),
        "force": float(cfg.get("factor_pretrain_force_weight", 1.0)),
        "internal": float(cfg.get("factor_pretrain_internal_strain_weight", 5.0)),
        "internal_full": float(cfg.get("factor_pretrain_internal_strain_full_weight", 1.0)),
        "soft": float(cfg.get("factor_pretrain_soft_mode_weight", 1.0)),
        "response_active": float(cfg.get("factor_pretrain_response_active_strain_weight", 1.0)),
    }


def _joint_weights(cfg: dict) -> dict[str, float]:
    return {
        "dielectric": float(cfg.get("dielectric_loss_weight", 0.0)),
        "born": float(cfg.get("born_loss_weight", 0.0)),
        "ionic": float(cfg.get("ionic_piezo_loss_weight", 0.0)),
        "force": float(cfg.get("force_constant_loss_weight", 0.0)),
        "internal": float(cfg.get("internal_strain_loss_weight", 0.0)),
        "internal_full": float(cfg.get("internal_strain_full_loss_weight", 0.0)),
        "soft": float(cfg.get("soft_mode_loss_weight", 0.0)),
        "response_active": float(cfg.get("response_active_strain_loss_weight", 0.0)),
    }


def _factor_objective(model: PiezoJet, batch, weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    factors = model.predict_factors(batch)
    components = {
        "born": born_loss(factors.born_charges, batch.y_born, batch.born_mask, batch.batch),
        "force_constant": force_constant_loss(
            factors.force_constants_flat, batch.dfpt_force_constants_flat, batch.ptr,
            batch.force_constant_mask,
        ),
        "soft_optical": soft_optical_eigenvalue_loss(
            factors.force_constants_flat, batch.dfpt_force_constants_flat, batch.ptr,
            batch.force_constant_mask,
        ),
        "internal_strain": internal_strain_loss(
            factors.internal_strain, batch.dfpt_internal_strain_flat,
            batch.dfpt_internal_strain_ions, batch.dfpt_internal_strain_directions,
            batch.dfpt_internal_strain_count, batch.ptr,
        ),
        "internal_strain_full": full_internal_strain_loss(
            factors.internal_strain, batch.dfpt_internal_strain_full,
            batch.internal_strain_full_mask, batch.batch,
        ),
        "response_active_strain": response_active_internal_strain_loss(
            factors.internal_strain, batch.y_born, batch.dfpt_force_constants_flat,
            batch.y_ionic_piezo, batch.ionic_piezo_mask, batch.ptr,
            batch.cell.reshape(-1, 3, 3), model.response,
        ),
    }
    loss = (
        weights["born"] * components["born"]
        + weights["force"] * components["force_constant"]
        + weights["soft"] * components["soft_optical"]
        + weights["internal"] * components["internal_strain"]
        + weights["internal_full"] * components["internal_strain_full"]
        + weights["response_active"] * components["response_active_strain"]
    )
    return loss, components


def _joint_objective(
    model: PiezoJet,
    batch,
    scale: torch.Tensor,
    bin_weights: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    factor, response, components = _joint_objective_terms(model, batch, scale, bin_weights, weights)
    return factor + response, components


def _joint_objective_terms(
    model: PiezoJet,
    batch,
    scale: torch.Tensor,
    bin_weights: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Split the existing joint loss into factor and response task groups.

    The summation is deliberately identical to :func:`_joint_objective`.  This
    lets protocol F change only how conflicting gradients reach the shared
    factor stack, rather than silently changing weights or supervision.
    """
    prediction = model.predict_components(batch)
    components = {
        "piezo_full": full_loss(prediction.tensor, batch.y, scale, bin_weights),
        "dielectric": dielectric_loss(prediction.dielectric, batch.y_dielectric, batch.dielectric_mask),
        "born": born_loss(prediction.born_charges, batch.y_born, batch.born_mask, batch.batch),
        "ionic_piezo": ionic_piezo_loss(
            prediction.ionic_piezo, batch.y_ionic_piezo, batch.ionic_piezo_mask
        ),
        "force_constant": force_constant_loss(
            prediction.force_constants_flat, batch.dfpt_force_constants_flat, batch.ptr,
            batch.force_constant_mask,
        ),
        "soft_optical": soft_optical_eigenvalue_loss(
            prediction.force_constants_flat, batch.dfpt_force_constants_flat, batch.ptr,
            batch.force_constant_mask,
        ),
        "internal_strain": internal_strain_loss(
            prediction.internal_strain, batch.dfpt_internal_strain_flat,
            batch.dfpt_internal_strain_ions, batch.dfpt_internal_strain_directions,
            batch.dfpt_internal_strain_count, batch.ptr,
        ),
        "internal_strain_full": full_internal_strain_loss(
            prediction.internal_strain, batch.dfpt_internal_strain_full,
            batch.internal_strain_full_mask, batch.batch,
        ),
        "response_active_strain": response_active_internal_strain_loss(
            prediction.internal_strain, batch.y_born, batch.dfpt_force_constants_flat,
            batch.y_ionic_piezo, batch.ionic_piezo_mask, batch.ptr,
            batch.cell.reshape(-1, 3, 3), model.response,
        ),
    }
    factor = (
        weights["born"] * components["born"]
        + weights["force"] * components["force_constant"]
        + weights["soft"] * components["soft_optical"]
        + weights["internal"] * components["internal_strain"]
        + weights["internal_full"] * components["internal_strain_full"]
        + weights["response_active"] * components["response_active_strain"]
    )
    response = (
        components["piezo_full"]
        + weights["dielectric"] * components["dielectric"]
        + weights["ionic"] * components["ionic_piezo"]
    )
    return factor, response, components


def factor_stack_parameters(model: PiezoJet) -> list[torch.nn.Parameter]:
    """Return the trainable factor stack protected by protocol F.

    This is the same module set frozen by protocols B/E.  Keeping the direct
    electronic head and macroscopic background out of this list is intentional:
    they have no direct BEC/Phi/Lambda label and may follow the response loss.
    """
    names = ["encoder", "born_head", "local_polar_mode", "global_context"]
    if model.factor_architecture in {"energy", "energy_learned_strain", "energy_learned_star"}:
        names.append("energy_factors")
    else:
        names.extend(("force_constants", "internal_strain"))
    seen: set[int] = set()
    parameters: list[torch.nn.Parameter] = []
    for name in names:
        for parameter in getattr(model, name).parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def factor_protected_gradient_projection(
    factor_loss: torch.Tensor,
    response_loss: torch.Tensor,
    protected_parameters: Iterable[torch.nn.Parameter],
    all_parameters: Iterable[torch.nn.Parameter],
    norm_match: bool = False,
) -> dict[str, float | int | bool]:
    """Populate ``.grad`` with a one-sided factor-protected PCGrad update.

    For the protected factor stack, write ``g_factor + g_response_projected``.
    If the two task gradients have a negative dot product, the response part is
    projected to the hyperplane orthogonal to ``g_factor``.  Therefore the
    macroscopic-response term cannot increase the direct-factor objective to
    first order at that update.  If ``norm_match`` is set, it then scales the
    projected response term to the factor-gradient norm; this is an exact
    per-update normalization, not a selected loss coefficient.  Non-factor
    response heads receive their usual response gradient.  The rule has no
    tunable coefficient or random task ordering, unlike generic multi-task
    gradient-surgery variants.
    """
    protected = [parameter for parameter in protected_parameters if parameter.requires_grad]
    protected_ids = {id(parameter) for parameter in protected}
    all_trainable = [parameter for parameter in all_parameters if parameter.requires_grad]
    unprotected = [parameter for parameter in all_trainable if id(parameter) not in protected_ids]
    if not protected:
        raise ValueError("Factor-protected projection requires a non-empty trainable factor stack")

    factor_grads = torch.autograd.grad(factor_loss, protected, retain_graph=True, allow_unused=True)
    response_grads = torch.autograd.grad(response_loss, protected, retain_graph=True, allow_unused=True)
    response_only_grads = torch.autograd.grad(
        response_loss, unprotected, retain_graph=False, allow_unused=True
    ) if unprotected else ()

    factor_parts = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(protected, factor_grads)
    ]
    response_parts = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(protected, response_grads)
    ]
    factor_vector, response_vector = torch.cat(factor_parts), torch.cat(response_parts)
    factor_norm = torch.linalg.vector_norm(factor_vector)
    response_norm = torch.linalg.vector_norm(response_vector)
    epsilon = torch.finfo(factor_vector.dtype).eps
    dot = torch.dot(factor_vector, response_vector)
    conflict = bool((dot < 0).detach()) and bool((factor_norm > epsilon).detach())
    coefficient = dot / factor_norm.square().clamp_min(epsilon) if conflict else torch.zeros_like(dot)
    projected_response_grads = [
        None if response_gradient is None and factor_gradient is None
        else (
            (torch.zeros_like(parameter) if response_gradient is None else response_gradient)
            - coefficient * (torch.zeros_like(parameter) if factor_gradient is None else factor_gradient)
        )
        for parameter, factor_gradient, response_gradient in zip(protected, factor_grads, response_grads)
    ]
    projected_response_vector = torch.cat([
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(protected, projected_response_grads)
    ])
    projected_response_norm = torch.linalg.vector_norm(projected_response_vector)
    response_scale = (
        factor_norm / projected_response_norm.clamp_min(epsilon)
        if norm_match and bool((projected_response_norm > epsilon).detach())
        else torch.ones_like(factor_norm)
    )
    for parameter, factor_gradient, response_gradient in zip(
        protected, factor_grads, projected_response_grads
    ):
        factor_value = torch.zeros_like(parameter) if factor_gradient is None else factor_gradient
        response_value = torch.zeros_like(parameter) if response_gradient is None else response_gradient
        parameter.grad = (factor_value + response_scale * response_value).detach()
    for parameter, gradient in zip(unprotected, response_only_grads):
        parameter.grad = (torch.zeros_like(parameter) if gradient is None else gradient).detach()

    cosine = dot / (factor_norm * response_norm).clamp_min(epsilon)
    removed = torch.linalg.vector_norm(response_vector - projected_response_vector)
    return {
        "factor_protected_parameters": sum(parameter.numel() for parameter in protected),
        "factor_gradient_norm": float(factor_norm.detach()),
        "response_gradient_norm": float(response_norm.detach()),
        "factor_response_gradient_cosine_before_projection": float(cosine.detach()),
        "response_gradient_norm_after_projection": float(projected_response_norm.detach()),
        "response_gradient_scale_after_projection": float(response_scale.detach()),
        "factor_protected_norm_matching": norm_match,
        "removed_response_gradient_fraction": float((removed / response_norm.clamp_min(epsilon)).detach()),
        "response_gradient_conflict_projected": conflict,
    }


def gradient_conflict_metrics(
    lambda_loss: torch.Tensor,
    piezo_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
) -> dict[str, float | int | None]:
    """Return encoder-gradient norms and cosine without changing ``.grad``.

    ``None`` means that the requested parameter stack is intentionally frozen,
    rather than that a zero gradient was observed.  This distinction is
    essential for protocol B.
    """
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        return {
            "shared_encoder_trainable_parameters": 0,
            "lambda_gradient_norm": None,
            "piezo_gradient_norm": None,
            "gradient_cosine": None,
        }
    lambda_grads = torch.autograd.grad(lambda_loss, trainable, retain_graph=True, allow_unused=True)
    piezo_grads = torch.autograd.grad(piezo_loss, trainable, retain_graph=True, allow_unused=True)
    lambda_parts, piezo_parts = [], []
    for lambda_grad, piezo_grad in zip(lambda_grads, piezo_grads):
        if lambda_grad is None and piezo_grad is None:
            continue
        lambda_parts.append(
            torch.zeros_like(piezo_grad).reshape(-1) if lambda_grad is None else lambda_grad.reshape(-1)
        )
        piezo_parts.append(
            torch.zeros_like(lambda_grad).reshape(-1) if piezo_grad is None else piezo_grad.reshape(-1)
        )
    if not lambda_parts:
        return {
            "shared_encoder_trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "lambda_gradient_norm": 0.0,
            "piezo_gradient_norm": 0.0,
            "gradient_cosine": 0.0,
        }
    lambda_vector, piezo_vector = torch.cat(lambda_parts), torch.cat(piezo_parts)
    lambda_norm, piezo_norm = torch.linalg.vector_norm(lambda_vector), torch.linalg.vector_norm(piezo_vector)
    cosine = torch.dot(lambda_vector, piezo_vector) / (lambda_norm * piezo_norm).clamp_min(
        torch.finfo(lambda_vector.dtype).eps
    )
    return {
        "shared_encoder_trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "lambda_gradient_norm": float(lambda_norm.detach()),
        "piezo_gradient_norm": float(piezo_norm.detach()),
        "gradient_cosine": float(cosine.detach()),
    }


def _diagnostic_gradients(
    model: PiezoJet,
    batch,
    scale: torch.Tensor,
    bin_weights: torch.Tensor,
) -> dict[str, float | int | None]:
    """Measure Lambda-vs-macroscopic-piezo conflict on the shared encoder."""
    components = model.predict_components(batch)
    lambda_loss = full_internal_strain_loss(
        components.internal_strain, batch.dfpt_internal_strain_full,
        batch.internal_strain_full_mask, batch.batch,
    )
    piezo_loss = full_loss(components.tensor, batch.y, scale, bin_weights)
    return gradient_conflict_metrics(lambda_loss, piezo_loss, model.encoder.parameters())


def _mean_components(total: dict[str, float], count: int) -> dict[str, float]:
    return {name: value / max(count, 1) for name, value in total.items()}


def _run_factor_update(
    model: PiezoJet,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    weights: dict[str, float],
    diagnostic: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[float, dict[str, float], dict[str, float | int | None] | None]:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    component_totals: dict[str, float] = {}
    gradient_row = None
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        with torch.set_grad_enabled(training):
            loss, components = _factor_objective(model, batch, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite direct-factor loss")
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                if diagnostic is not None:
                    gradient_row = _diagnostic_gradients(model, batch, *diagnostic)
                loss.backward()
                if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                    raise FloatingPointError("Non-finite direct-factor gradient")
                optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        for name, value in components.items():
            component_totals[name] = component_totals.get(name, 0.0) + float(value.detach()) * batch.num_graphs
        count += batch.num_graphs
    return total / max(count, 1), _mean_components(component_totals, count), gradient_row


def _run_joint_update(
    model: PiezoJet,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scale: torch.Tensor,
    bin_weights: torch.Tensor,
    weights: dict[str, float],
    diagnostic: bool = False,
    factor_protected: bool = False,
    factor_norm_matching: bool = False,
) -> tuple[float, dict[str, float], dict[str, float | int | None] | None]:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    component_totals: dict[str, float] = {}
    gradient_row = None
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        with torch.set_grad_enabled(training):
            factor_loss, response_loss, components = _joint_objective_terms(
                model, batch, scale, bin_weights, weights
            )
            loss = factor_loss + response_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite joint loss")
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                if diagnostic:
                    gradient_row = gradient_conflict_metrics(
                        components["internal_strain_full"], components["piezo_full"], model.encoder.parameters()
                    )
                if factor_protected:
                    projection = factor_protected_gradient_projection(
                        factor_loss, response_loss, factor_stack_parameters(model), model.parameters(),
                        norm_match=factor_norm_matching,
                    )
                    if gradient_row is None:
                        gradient_row = projection
                    else:
                        gradient_row.update(projection)
                else:
                    loss.backward()
                if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                    raise FloatingPointError("Non-finite joint gradient")
                optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        for name, value in components.items():
            component_totals[name] = component_totals.get(name, 0.0) + float(value.detach()) * batch.num_graphs
        count += batch.num_graphs
    return total / max(count, 1), _mean_components(component_totals, count), gradient_row


def _quality_metrics(model: PiezoJet, loader, device: torch.device) -> dict[str, float]:
    """Validation/test factor quality with true Z* and Phi for Lambda's ionic map."""
    model.eval()
    lambda_accumulator = FactorAccumulator("internal_strain_full")
    oracle_ionic_predictions, oracle_ionic_targets = [], []
    total_predictions, total_targets = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            components = model.predict_components(batch)
            force_offset = 0
            cells = batch.cell.reshape(-1, 3, 3)
            for graph_index in range(batch.num_graphs):
                start, stop = int(batch.ptr[graph_index]), int(batch.ptr[graph_index + 1])
                atoms, block_values = stop - start, 9 * (stop - start) ** 2
                if not bool(batch.internal_strain_full_mask[graph_index]):
                    raise ValueError("This diagnostic requires a strict full-Lambda label for every panel material")
                force_target = batch.dfpt_force_constants_flat[force_offset : force_offset + block_values].reshape(atoms, atoms, 3, 3)
                force_offset += block_values
                predicted_lambda = components.internal_strain[start:stop]
                lambda_accumulator.add(predicted_lambda, batch.dfpt_internal_strain_full[start:stop])
                volume = torch.linalg.det(cells[graph_index]).abs()
                oracle_ionic = ionic_piezo_from_factors(
                    model.response, batch.y_born[start:stop], force_target, predicted_lambda,
                    volume, "regularized",
                )
                oracle_ionic_predictions.append(oracle_ionic.detach().cpu())
                oracle_ionic_targets.append(batch.y_ionic_piezo[graph_index].detach().cpu())
            total_predictions.append(components.tensor.detach().cpu())
            total_targets.append(batch.y.detach().cpu())
    lambda_metrics = lambda_accumulator.summary()
    ionic_metrics = ionic_aggregate_metrics(oracle_ionic_predictions, oracle_ionic_targets)
    total_skill = response_tensor_skill(torch.cat(total_predictions), torch.cat(total_targets))
    return {
        "full_lambda_cosine": float(lambda_metrics["macro_material_directional_cosine"]),
        "full_lambda_amplitude": float(lambda_metrics["macro_material_stabilized_amplitude_ratio"]),
        "oracle_ionic_cosine_macro_material": float(ionic_metrics["ionic_cosine_macro_material"]),
        "oracle_ionic_cosine_micro_components": float(ionic_metrics["ionic_cosine_micro_components"]),
        "oracle_ionic_cosine_active_only": float(ionic_metrics["ionic_cosine_active_only"]),
        "oracle_ionic_amplitude_ratio_macro": float(ionic_metrics["ionic_amplitude_ratio_macro"]),
        "oracle_ionic_skill_vs_zero_macro": float(ionic_metrics["ionic_mae_skill_vs_zero_macro"]),
        "total_trs": float(total_skill["tensor_response_skill_vs_zero"]),
    }


def _checkpoint(model: PiezoJet, cfg: dict, scale: torch.Tensor, update: int, protocol: str, stage: str) -> dict:
    return {
        "model": model.state_dict(),
        "config": cfg,
        "piezo_scale": float(scale),
        "epoch": update,
        "stage": stage,
        "protocol": protocol,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _restore(path: Path, model: PiezoJet, device: torch.device) -> None:
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])


def _run_one(
    cfg: dict,
    protocol: str,
    output: Path,
    train_loader,
    val_loader,
    test_loader,
    model: PiezoJet,
    device: torch.device,
    scale: torch.Tensor,
    bin_weights: torch.Tensor,
    factor_updates: int,
    joint_updates: int,
    diagnostics_every: int,
) -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol}")
    output.mkdir(parents=True, exist_ok=True)
    factor_weights, joint_weights = _factor_weights(cfg), _joint_weights(cfg)
    factor_optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("factor_pretrain_learning_rate", cfg["learning_rate"])),
        weight_decay=float(cfg["weight_decay"]),
    )
    rows: list[dict] = []
    factor_best, factor_best_update = float("inf"), 0
    joint_best, joint_best_update = float("inf"), 0
    global_update = 0

    def factor_step(stage_update: int) -> None:
        nonlocal factor_best, factor_best_update, global_update
        global_update += 1
        diagnostic = (scale, bin_weights) if global_update % diagnostics_every == 0 else None
        started = time.perf_counter()
        train_loss, train_components, gradients = _run_factor_update(
            model, train_loader, factor_optimizer, device, factor_weights, diagnostic
        )
        val_loss, val_components, _ = _run_factor_update(
            model, val_loader, None, device, factor_weights
        )
        row = {
            "global_update": global_update, "stage_update": stage_update, "phase": "factor",
            "train_loss": train_loss, "val_loss": val_loss, "seconds": time.perf_counter() - started,
        }
        row.update({f"train_{key}_loss": value for key, value in train_components.items()})
        row.update({f"val_{key}_loss": value for key, value in val_components.items()})
        if gradients is not None:
            row.update(gradients)
        if global_update % diagnostics_every == 0:
            row.update({f"val_{key}": value for key, value in _quality_metrics(model, val_loader, device).items()})
        rows.append(row)
        state = _checkpoint(model, cfg, scale, global_update, protocol, "direct_factor_pretraining")
        torch.save(state, output / "factor_last.pt")
        if val_loss < factor_best:
            factor_best, factor_best_update = val_loss, global_update
            torch.save(state, output / "factor_best.pt")

    def joint_step(stage_update: int) -> None:
        nonlocal joint_best, joint_best_update, global_update
        global_update += 1
        diagnostic = global_update % diagnostics_every == 0
        started = time.perf_counter()
        train_loss, train_components, gradients = _run_joint_update(
            model, train_loader, joint_optimizer, device, scale, bin_weights, joint_weights, diagnostic,
            factor_protected=protocol in {"F", "G"},
            factor_norm_matching=protocol == "G",
        )
        val_loss, val_components, _ = _run_joint_update(
            model, val_loader, None, device, scale, bin_weights, joint_weights
        )
        row = {
            "global_update": global_update, "stage_update": stage_update, "phase": "joint",
            "train_loss": train_loss, "val_loss": val_loss, "seconds": time.perf_counter() - started,
        }
        row.update({f"train_{key}_loss": value for key, value in train_components.items()})
        row.update({f"val_{key}_loss": value for key, value in val_components.items()})
        if gradients is not None:
            row.update(gradients)
        if global_update % diagnostics_every == 0:
            row.update({f"val_{key}": value for key, value in _quality_metrics(model, val_loader, device).items()})
        rows.append(row)
        state = _checkpoint(model, cfg, scale, global_update, protocol, "joint")
        torch.save(state, output / "last.pt")
        if val_loss < joint_best:
            joint_best, joint_best_update = val_loss, global_update
            torch.save(state, output / "loss_best.pt")

    # ``joint_optimizer`` is deliberately defined before nested ``joint_step``.
    joint_optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    if protocol == "A":
        for update in range(1, factor_updates + joint_updates + 1):
            factor_step(update)
        selected_path = output / "factor_best.pt"
        selected_stage = "factor_validation_loss"
    elif protocol in {"B", "C", "E", "F", "G"}:
        for update in range(1, factor_updates + 1):
            factor_step(update)
        _restore(output / "factor_best.pt", model, device)
        if protocol in {"B", "E"}:
            frozen_modules = freeze_factor_stack(model)
            # Recreate the optimizer after changing ``requires_grad`` so no
            # frozen tensor remains in the joint optimizer.
            joint_optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]),
            )
        else:
            frozen_modules = []
        for update in range(1, joint_updates + 1):
            joint_step(update)
        selected_path = output / "loss_best.pt"
        selected_stage = "joint_validation_loss"
    else:
        frozen_modules = []
        if factor_updates != joint_updates:
            raise ValueError("Protocol D requires equal factor and joint update counts")
        for update in range(1, factor_updates + 1):
            factor_step(update)
            joint_step(update)
        selected_path = output / "loss_best.pt"
        selected_stage = "joint_validation_loss"

    selected = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    test_metrics = _quality_metrics(model, test_loader, device)
    result = {
        "schema": 1,
        "protocol": protocol,
        "definition": {
            "A": "100 direct-factor updates only",
            "B": "50 factor updates, validation-selected restore, 50 frozen-factor joint updates",
            "C": "50 factor updates, validation-selected restore, 50 unfrozen-factor joint updates",
            "D": "50 alternating direct-factor/joint update pairs with factors trainable",
            "E": f"{factor_updates} factor updates, validation-selected restore, {joint_updates} frozen-factor joint updates",
            "F": (
                f"{factor_updates} factor updates, validation-selected restore, {joint_updates} joint updates "
                "with factor-protected gradient projection on the shared factor stack"
            ),
            "G": (
                f"{factor_updates} factor updates, validation-selected restore, {joint_updates} joint updates "
                "with factor-protected projection and unit-norm response-gradient matching on the shared factor stack"
            ),
        }[protocol],
        "factor_updates": factor_updates,
        "joint_updates": joint_updates,
        "total_updates": global_update,
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_stage": selected_stage,
        "selected_checkpoint_update": int(selected["epoch"]),
        "factor_best_validation_loss": factor_best,
        "factor_best_update": factor_best_update,
        "joint_best_validation_loss": joint_best if joint_best < float("inf") else None,
        "joint_best_update": joint_best_update if joint_best_update else None,
        "frozen_modules": frozen_modules if protocol in {"B", "E"} else [],
        "gradient_surgery": (
            "one-sided factor-protected projection: conflicting response gradients are projected "
            "orthogonal to the direct-factor gradient only on the factor stack frozen by B/E"
            if protocol in {"F", "G"} else None
        ),
        "gradient_norm_matching": protocol == "G",
        "selection_rule": "Validation loss only; the frozen test panel was evaluated after selection and never used to choose a checkpoint, protocol, or hyperparameter.",
        "test_diagnostic": test_metrics,
        "gradient_diagnostic": "Every requested interval: shared-encoder norms of grad(Lambda-full) and grad(macroscopic piezo full), plus cosine. Null values mean the stack was frozen.",
        "all_finite": True,
    }
    _write_rows(output / "trajectory.csv", rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _prepare(cfg: dict, splits_file: Path, device: torch.device):
    records = load_gmtnet_records(cfg["data_root"])
    create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
    known_ids = {str(record["JARVIS_ID"]) for record in records}
    splits = load_explicit_splits(splits_file, known_ids)
    cache_key = graph_cache_key(records, cfg["cutoff"], cfg["max_neighbors"])
    shared = dict(
        cutoff=cfg["cutoff"], max_neighbors=cfg["max_neighbors"], processed_dir=cfg["processed_dir"],
        cache_key=cache_key, dfpt_dir=cfg.get("jarvis_dfpt_dir"),
        strain_completion_dir=cfg.get("jarvis_strain_completion_dir"),
    )
    datasets = {name: PiezoDataset(records, material_ids, **shared) for name, material_ids in splits.items()}
    loader_options = {"num_workers": int(cfg["num_workers"]), "pin_memory": device.type == "cuda"}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=int(cfg["batch_size"]), shuffle=True, **loader_options),
        "val": DataLoader(datasets["val"], batch_size=int(cfg["batch_size"]), shuffle=False, **loader_options),
        "test": DataLoader(datasets["test"], batch_size=int(cfg["batch_size"]), shuffle=False, **loader_options),
    }
    first_target = torch.cat([datasets["train"][index].y_voigt for index in range(len(datasets["train"]))])
    scale = piezo_scale(first_target).to(device)
    bin_weights = response_bin_weights(
        torch.stack([datasets["train"][index].y.squeeze(0) for index in range(len(datasets["train"]))])
    ).to(device)
    return records, splits, loaders, scale, bin_weights


def _load_pretrained_encoder(model: PiezoJet, cfg: dict, device: torch.device) -> dict:
    path = Path(cfg["pretrained_encoder"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing required structural pretraining checkpoint: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("architecture") != "cartesian_local_environment_v1" or "encoder" not in payload:
        raise ValueError("Pretraining checkpoint is not a current Cartesian local-environment encoder")
    model.encoder.load_state_dict(payload["encoder"], strict=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--jarvis-dfpt-dir", type=Path, help="Override the immutable source cache location for an explicitly registered cohort.")
    parser.add_argument("--strict-completion-dir", type=Path, help="Override the strict-completion cache location for an explicitly registered cohort.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/optimization_ablation_v1"))
    parser.add_argument(
        "--device",
        help="Optional runtime-device override (for example cpu); persisted checkpoints retain the declared run device.",
    )
    parser.add_argument("--protocol", choices=("all", *PROTOCOLS), default="all")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--factor-updates", type=int,
        help="Override direct-factor updates. Defaults to 50 (A--D) or 100 (E).",
    )
    parser.add_argument(
        "--joint-updates", type=int,
        help="Override joint updates. Defaults to 50 for every protocol.",
    )
    parser.add_argument("--diagnostics-every", type=int, default=5)
    args = parser.parse_args()
    if (
        (args.factor_updates is not None and args.factor_updates < 1)
        or (args.joint_updates is not None and args.joint_updates < 1)
        or args.diagnostics_every < 1
    ):
        raise ValueError("Explicit update counts and --diagnostics-every must be positive")
    if not args.splits_file.is_file():
        raise FileNotFoundError(args.splits_file)
    base_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.jarvis_dfpt_dir is not None:
        base_cfg["jarvis_dfpt_dir"] = str(args.jarvis_dfpt_dir)
    if args.strict_completion_dir is not None:
        base_cfg["jarvis_strain_completion_dir"] = str(args.strict_completion_dir)
    protocols = LEGACY_ALL_PROTOCOLS if args.protocol == "all" else (args.protocol,)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must be a non-empty list of unique integers")
    all_results = []
    for seed in seeds:
        cfg = dict(base_cfg)
        cfg["seed"] = seed
        if args.device is not None:
            cfg["device"] = args.device
        cfg["splits_file"] = str(args.splits_file)
        cfg["ablation_protocol"] = "registered_factor_preservation_v1"
        cfg["ablation_diagnostics_every"] = args.diagnostics_every
        seed_everything(seed)
        device = device_from_config(cfg["device"])
        _, splits, loaders, scale, bin_weights = _prepare(cfg, args.splits_file, device)
        cfg["restricted_split_counts"] = {name: len(values) for name, values in splits.items()}
        cfg["git_commit"] = _git_commit()
        cfg["data_commit"] = _data_commit(cfg["data_root"])
        cfg["runtime_device"] = str(device)
        for protocol in protocols:
            factor_updates = args.factor_updates if args.factor_updates is not None else (100 if protocol == "E" else 50)
            joint_updates = args.joint_updates if args.joint_updates is not None else 50
            cfg["ablation_factor_updates"] = factor_updates
            cfg["ablation_joint_updates"] = joint_updates
            # Re-seeding makes every protocol begin from the same initialized
            # model for a given seed; only its declared update schedule differs.
            seed_everything(seed)
            model = model_from_config(cfg).to(device)
            pretrained = _load_pretrained_encoder(model, cfg, device)
            cfg["pretraining_epoch"] = pretrained.get("epoch")
            output = args.output_root / f"seed{seed}" / f"protocol_{protocol}"
            result = _run_one(
                cfg, protocol, output, loaders["train"], loaders["val"], loaders["test"],
                model, device, scale, bin_weights, factor_updates, joint_updates,
                args.diagnostics_every,
            )
            result["seed"] = seed
            all_results.append(result)
            print(json.dumps({"seed": seed, "protocol": protocol, **result["test_diagnostic"]}))
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "purpose": "69-label frozen-panel factor-preservation optimization forensic",
        "splits_file": str(args.splits_file),
        "protocols": list(protocols),
        "seeds": seeds,
        "selection_rule": "Validation loss only. Test metrics are post-selection diagnostics.",
        "results": all_results,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
