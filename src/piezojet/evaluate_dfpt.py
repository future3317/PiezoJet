"""Physical-unit evaluation on a formula-disjoint JARVIS DFPT subset.

Unlike :mod:`piezojet.evaluate`, this entry point evaluates the variable-size
atom-coordinate factors used by the relaxed response.  It never pads phonon
coordinates and scores only the internal-strain blocks actually printed by
VASP.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from .data import PiezoDataset, create_or_load_splits, graph_cache_key, load_gmtnet_records
from .metrics import material_bootstrap_confidence_interval, response_tensor_skill
from .model import AtomCoordinateResponsePotential, model_from_config
from .tensor_ops import (
    PIEZO_IRREP_SLICES,
    cartesian_to_piezo_voigt,
    electronic_irrep_decomposition,
    piezo_to_irreps,
    piezo_voigt_to_cartesian,
)
from .projectors import translation_projector
from .train import device_from_config, load_explicit_splits, restrict_splits_to_material_ids


FACTOR_FLOORS = {
    "born_charge": 0.1,  # elementary charge
    "force_constant": 0.01,  # eV / Angstrom^2
    "internal_strain": 0.01,  # eV / Angstrom
    "internal_strain_full": 0.01,  # eV / Angstrom
    "displacement_response": 0.001,  # Angstrom
    "ionic_piezo": 0.05,  # C / m^2
    "factorized_ionic_piezo": 0.05,  # C / m^2
    "electronic_piezo": 0.05,  # C / m^2
    "total_piezo": 0.05,  # C / m^2
    "dfpt_total_piezo": 0.05,  # C / m^2
    "dfpt_branch_sum_piezo": 0.05,  # C / m^2
    "dielectric": 0.1,  # relative permittivity
    "ionic_dielectric": 0.1,  # relative permittivity contribution
    "macro_elastic": 1.0,  # GPa
}

FACTOR_UNITS = {
    "born_charge": "e",
    "force_constant": "eV/Angstrom^2",
    "internal_strain": "eV/Angstrom",
    "internal_strain_full": "eV/Angstrom",
    "displacement_response": "Angstrom",
    "ionic_piezo": "C/m^2",
    "factorized_ionic_piezo": "C/m^2",
    "electronic_piezo": "C/m^2",
    "total_piezo": "C/m^2",
    "dfpt_total_piezo": "C/m^2",
    "dfpt_branch_sum_piezo": "C/m^2",
    "dielectric": "relative",
    "ionic_dielectric": "relative",
    "macro_elastic": "GPa",
}


def clean_force_constant_target(target_flat: torch.Tensor, atoms: int) -> torch.Tensor:
    """Apply the same symmetry and translational projection as training."""
    blocks = target_flat.reshape(atoms, atoms, 3, 3)
    matrix = AtomCoordinateResponsePotential._matrix_from_blocks(blocks)
    matrix = 0.5 * (matrix + matrix.T)
    projector, _ = translation_projector(atoms, matrix)
    cleaned = projector @ matrix @ projector
    return AtomCoordinateResponsePotential._blocks_from_matrix(cleaned, atoms)


def force_constant_matrix(blocks: torch.Tensor) -> torch.Tensor:
    return AtomCoordinateResponsePotential._matrix_from_blocks(blocks)


def optical_eigensystem(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenpairs in the exact reduced optical subspace, sorted by energy."""
    matrix = force_constant_matrix(blocks).to(torch.float64)
    atoms = blocks.shape[0]
    basis = AtomCoordinateResponsePotential._optical_basis(atoms, matrix)
    if basis.shape[1] == 0:
        return matrix.new_empty(0), matrix.new_empty(matrix.shape[0], 0)
    values, reduced_vectors = torch.linalg.eigh(basis.T @ matrix @ basis)
    return values, basis @ reduced_vectors


def spectrum_regularization_regions(
    eigenvalues: torch.Tensor,
    delta: float,
) -> dict[str, float | int]:
    """Count optical modes in the three preregistered resolvent regions."""
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    absolute = eigenvalues.detach().abs().to(torch.float64)
    total = int(absolute.numel())
    below = int((absolute < delta).sum())
    transition = int(((absolute >= delta) & (absolute < 3.0 * delta)).sum())
    outside = total - below - transition
    denominator = max(total, 1)
    return {
        "mode_count": total,
        "below_delta_count": below,
        "delta_to_3delta_count": transition,
        "above_3delta_count": outside,
        "below_delta_fraction": below / denominator if total else float("nan"),
        "delta_to_3delta_fraction": transition / denominator if total else float("nan"),
        "above_3delta_fraction": outside / denominator if total else float("nan"),
    }


def low_rank_displacement_oracle(
    displacement: torch.Tensor,
    born: torch.Tensor,
    ranks: tuple[int, ...] = (1, 2, 4, 6),
) -> dict[str, object]:
    """SVD oracle for matrix rank, not for a number of physical phonon modes."""
    if displacement.ndim != 2 or displacement.shape[1] != 6:
        raise ValueError("displacement must have shape (3N, 6)")
    charge = born.reshape(-1, 3).to(torch.float64)
    target = displacement.to(torch.float64)
    if charge.shape[0] != target.shape[0]:
        raise ValueError("Born-charge and displacement coordinate counts differ")
    left, singular, right = torch.linalg.svd(target, full_matrices=False)
    target_norm = torch.linalg.vector_norm(target).clamp_min(1e-30)
    target_response = charge.T @ target
    response_norm = torch.linalg.vector_norm(target_response).clamp_min(1e-30)
    per_rank: dict[str, dict[str, float | int]] = {}
    for requested in ranks:
        if requested < 1:
            raise ValueError("oracle ranks must be positive")
        retained = min(requested, singular.numel())
        reconstruction = (
            left[:, :retained] * singular[:retained].unsqueeze(0)
        ) @ right[:retained]
        response = charge.T @ reconstruction
        per_rank[str(requested)] = {
            "retained_rank": retained,
            "displacement_relative_frobenius_error": float(
                torch.linalg.vector_norm(reconstruction - target) / target_norm
            ),
            "response_relative_frobenius_error": float(
                torch.linalg.vector_norm(response - target_response) / response_norm
            ),
            "singular_energy_fraction": float(
                singular[:retained].square().sum()
                / singular.square().sum().clamp_min(1e-30)
            ),
        }
    return {
        "algebraic_rank_upper_bound": 6,
        "numerical_rank": int(torch.linalg.matrix_rank(target)),
        "singular_values": [float(value) for value in singular],
        "per_rank": per_rank,
        "interpretation": (
            "rank(U_eta)<=6 is a right-hand-side matrix-rank fact; it does not imply "
            "that six physical phonon eigenmodes dominate"
        ),
    }


def _column_basis(matrix: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return a numerical column-space basis with the matrix-rank tolerance."""
    value = matrix.to(torch.float64)
    left, singular, _ = torch.linalg.svd(value, full_matrices=False)
    if singular.numel() == 0:
        return left[:, :0], 0
    tolerance = max(value.shape) * torch.finfo(value.dtype).eps * singular.max()
    rank = int((singular > tolerance).sum())
    return left[:, :rank], rank


def response_active_alignment_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_born: torch.Tensor,
) -> dict[str, float | int]:
    """Gauge-invariant displacement-subspace and true-charge alignment metrics.

    This deliberately compares projectors and the cross-covariance ``Z^T U``;
    it does not pair individual phonon eigenvectors or assign physical meaning
    to the six right-hand-side directions.
    """
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 6:
        raise ValueError("prediction and target must share shape (3N, 6)")
    predicted = prediction.to(torch.float64)
    expected = target.to(torch.float64)
    charge = target_born.reshape(-1, 3).to(torch.float64)
    if charge.shape[0] != expected.shape[0]:
        raise ValueError("Born-charge and displacement coordinate counts differ")

    predicted_basis, predicted_rank = _column_basis(predicted)
    target_basis, target_rank = _column_basis(expected)
    charge_basis, charge_rank = _column_basis(charge)
    common_rank = min(predicted_rank, target_rank)
    if common_rank:
        principal_cosines = torch.linalg.svdvals(target_basis.T @ predicted_basis)
        projector_overlap = principal_cosines.square().sum() / common_rank
        minimum_principal_cosine = principal_cosines.min()
    else:
        projector_overlap = expected.new_tensor(float("nan"))
        minimum_principal_cosine = expected.new_tensor(float("nan"))

    target_norm = torch.linalg.vector_norm(expected).clamp_min(1e-30)
    predicted_norm = torch.linalg.vector_norm(predicted).clamp_min(1e-30)
    target_active = charge_basis @ (charge_basis.T @ expected)
    predicted_active = charge_basis @ (charge_basis.T @ predicted)
    target_active_norm = torch.linalg.vector_norm(target_active)
    predicted_active_norm = torch.linalg.vector_norm(predicted_active)
    active_cosine = torch.sum(target_active * predicted_active) / (
        target_active_norm * predicted_active_norm
    ).clamp_min(1e-30)

    target_cross_covariance = charge.T @ expected
    predicted_cross_covariance = charge.T @ predicted
    target_cross_norm = torch.linalg.vector_norm(target_cross_covariance)
    predicted_cross_norm = torch.linalg.vector_norm(predicted_cross_covariance)
    cross_cosine = torch.sum(target_cross_covariance * predicted_cross_covariance) / (
        target_cross_norm * predicted_cross_norm
    ).clamp_min(1e-30)
    return {
        "target_displacement_rank": target_rank,
        "predicted_displacement_rank": predicted_rank,
        "true_charge_coordinate_rank": charge_rank,
        "displacement_subspace_projector_overlap": float(projector_overlap),
        "displacement_subspace_minimum_principal_cosine": float(
            minimum_principal_cosine
        ),
        "target_true_charge_active_energy_fraction": float(
            target_active_norm.square() / target_norm.square()
        ),
        "predicted_true_charge_active_energy_fraction": float(
            predicted_active_norm.square() / predicted_norm.square()
        ),
        "true_charge_active_projected_directional_cosine": float(active_cosine),
        "true_charge_cross_covariance_directional_cosine": float(cross_cosine),
        "true_charge_cross_covariance_amplitude_ratio": float(
            predicted_cross_norm / target_cross_norm.clamp_min(1e-30)
        ),
    }


def response_weighted_log_stiffness_bias(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_born: torch.Tensor,
    target_coupling: torch.Tensor,
    delta: float,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compare predicted stiffness on true modes, weighted by true response activity.

    The predicted Rayleigh quotient is evaluated in the true optical basis, so
    no predicted eigenvector pairing is required.  The weight is the product
    of true mode-effective charge, strain coupling, and signed-resolvent
    magnitude.  A positive log bias indicates a response-relevant spectrum
    that is systematically harder than the target.
    """
    if delta <= 0.0 or epsilon <= 0.0:
        raise ValueError("delta and epsilon must be positive")
    true_values, true_vectors = optical_eigensystem(target)
    if true_values.numel() == 0:
        return {
            "mean_log_abs_stiffness_bias": float("nan"),
            "response_weighted_mean_log_abs_stiffness_bias": float("nan"),
            "response_weight_sum": 0.0,
        }
    predicted_matrix = force_constant_matrix(prediction).to(torch.float64)
    predicted_rayleigh = torch.einsum(
        "ik,ij,jk->k", true_vectors, predicted_matrix, true_vectors
    )
    log_bias = torch.log(predicted_rayleigh.abs() + epsilon) - torch.log(
        true_values.abs() + epsilon
    )
    charge = target_born.reshape(-1, 3).to(torch.float64)
    coupling = target_coupling.reshape(target.shape[0] * 3, 6).to(torch.float64)
    charge_strength = torch.linalg.vector_norm(true_vectors.T @ charge, dim=1)
    coupling_strength = torch.linalg.vector_norm(true_vectors.T @ coupling, dim=1)
    filter_magnitude = true_values.abs() / (true_values.square() + delta**2)
    weights = charge_strength * coupling_strength * filter_magnitude
    weighted = (weights * log_bias).sum() / weights.sum().clamp_min(epsilon)
    return {
        "mean_log_abs_stiffness_bias": float(log_bias.mean()),
        "response_weighted_mean_log_abs_stiffness_bias": float(weighted),
        "response_weight_sum": float(weights.sum()),
    }


def soft_mode_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    predicted_born: torch.Tensor,
    target_born: torch.Tensor,
    mode_count: int = 3,
) -> dict[str, float | int]:
    """Response-relevant low-optical-mode diagnostics without mode padding."""
    pred_values, pred_vectors = optical_eigensystem(prediction)
    true_values, true_vectors = optical_eigensystem(target)
    count = min(mode_count, pred_values.numel(), true_values.numel())
    if count == 0:
        return {
            "soft_mode_count": 0,
            "minimum_true_optical_eigenvalue": float("nan"),
            "minimum_predicted_optical_eigenvalue": float("nan"),
            "lowest_optical_eigenvalue_mae": float("nan"),
            "soft_mode_sign_accuracy": float("nan"),
            "soft_mode_subspace_overlap": float("nan"),
            "mode_effective_charge_norm_mae": float("nan"),
        }
    pred_values, true_values = pred_values[:count], true_values[:count]
    pred_soft, true_soft = pred_vectors[:, :count], true_vectors[:, :count]
    overlap = (true_soft.T @ pred_soft).square().sum() / count
    target_charge = target_born.reshape(-1, 3).to(torch.float64)
    predicted_charge = predicted_born.reshape(-1, 3).to(torch.float64)
    # Project both charges on the target soft coordinates.  Comparing norms
    # removes the arbitrary sign of each target eigenvector.
    true_mode_charge = torch.linalg.vector_norm(true_soft.T @ target_charge, dim=-1)
    pred_mode_charge = torch.linalg.vector_norm(true_soft.T @ predicted_charge, dim=-1)
    return {
        "soft_mode_count": count,
        "minimum_true_optical_eigenvalue": float(true_values.min()),
        "minimum_predicted_optical_eigenvalue": float(pred_values.min()),
        "lowest_optical_eigenvalue_mae": float((pred_values - true_values).abs().mean()),
        "soft_mode_sign_accuracy": float((torch.sign(pred_values) == torch.sign(true_values)).to(torch.float64).mean()),
        "soft_mode_subspace_overlap": float(overlap),
        "mode_effective_charge_norm_mae": float((pred_mode_charge - true_mode_charge).abs().mean()),
    }


def replace_printed_internal_strain(
    prediction: torch.Tensor,
    target_flat: torch.Tensor,
    ions: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Replace only genuinely observed Lambda blocks; never invent the rest."""
    output = prediction.clone()
    target = target_flat.reshape(-1, 3, 3)
    output[ions, directions] = 0.5 * (target + target.transpose(-1, -2))
    return output


def ionic_piezo_from_factors(
    response: AtomCoordinateResponsePotential,
    born: torch.Tensor,
    force_constants: torch.Tensor,
    internal_strain: torch.Tensor,
    volume: torch.Tensor | float,
    solve_policy: str = "regularized",
    regularization: float | None = None,
) -> torch.Tensor:
    """Compute the ionic tensor for an explicit oracle factor combination."""
    coupling = response._coupling_voigt(internal_strain).reshape(-1, 6)
    charge = born.reshape(-1, 3)
    volume_tensor = torch.as_tensor(volume, dtype=born.dtype, device=born.device)
    relaxed_coupling = response.apply_optical_operator(
        force_constants, coupling, solve_policy, regularization
    )
    piezo_voigt = response.PIEZO_C_PER_M2 * charge.T @ relaxed_coupling / volume_tensor
    return piezo_voigt_to_cartesian(piezo_voigt)


def ionic_piezo_from_displacement_matrix(
    response: AtomCoordinateResponsePotential,
    born: torch.Tensor,
    displacement: torch.Tensor,
    volume: torch.Tensor | float,
) -> torch.Tensor:
    """Contract ``Z*`` with a ``[3N,6]`` displacement-response matrix."""
    charge = born.reshape(-1, 3)
    if displacement.shape != (charge.shape[0], 6):
        raise ValueError("Born charge and displacement matrix dimensions differ")
    volume_tensor = torch.as_tensor(volume, dtype=born.dtype, device=born.device)
    piezo_voigt = response.PIEZO_C_PER_M2 * charge.transpose(0, 1) @ displacement / volume_tensor
    return piezo_voigt_to_cartesian(piezo_voigt)


def selected_internal_strain(
    prediction: torch.Tensor,
    target_flat: torch.Tensor,
    ions: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select only source-observed ion/direction blocks and symmetrize labels."""
    target = target_flat.reshape(-1, 3, 3)
    if target.shape[0] != ions.numel() or ions.numel() != directions.numel():
        raise ValueError("Internal-strain block metadata is inconsistent")
    selected = prediction[ions, directions]
    return selected, 0.5 * (target + target.transpose(-1, -2))


def pair_metrics(prediction: torch.Tensor, target: torch.Tensor, floor: float) -> dict[str, float | int]:
    """Physical component errors and a zero-predictor comparison for one array."""
    prediction = prediction.detach().to(torch.float64).reshape(-1)
    target = target.detach().to(torch.float64).reshape(-1)
    if prediction.shape != target.shape or target.numel() == 0:
        raise ValueError("Metric arrays must be equally shaped and non-empty")
    residual = prediction - target
    mae = residual.abs().mean()
    zero_mae = target.abs().mean()
    residual_norm = torch.linalg.vector_norm(residual)
    target_norm = torch.linalg.vector_norm(target)
    prediction_norm = torch.linalg.vector_norm(prediction)
    denominator = target_norm.clamp_min(float(floor) * target.numel() ** 0.5)
    return {
        "components": target.numel(),
        "component_mae": float(mae),
        "component_rmse": float(residual.square().mean().sqrt()),
        "frobenius_error": float(residual_norm),
        "stabilized_relative_frobenius_error": float(residual_norm / denominator),
        "zero_component_mae": float(zero_mae),
        "mae_skill_vs_zero": float(1.0 - mae / zero_mae.clamp_min(torch.finfo(torch.float64).eps)),
        "stabilized_amplitude_ratio": float(prediction_norm / denominator),
        "directional_cosine": float(
            torch.dot(prediction, target)
            / (prediction_norm * target_norm).clamp_min(torch.finfo(torch.float64).eps)
        ),
    }


def electronic_irrep_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    floor: float = FACTOR_FLOORS["electronic_piezo"],
) -> dict[str, dict[str, float | int]]:
    """Resolve electronic-piezo error into 2x l=1, l=2, and l=3 blocks."""
    predicted = electronic_irrep_decomposition(prediction)
    expected = electronic_irrep_decomposition(target)
    return {
        name: pair_metrics(predicted[name], expected[name], floor)
        for name in PIEZO_IRREP_SLICES
    }


def electronic_basis_oracle(
    basis_tensors: torch.Tensor,
    target: torch.Tensor,
    floor: float = FACTOR_FLOORS["electronic_piezo"],
) -> dict[str, object]:
    """Least-squares projection onto the current electronic output span.

    ``basis_tensors`` contains the unrestricted atom/channel/family Cartesian
    bases plus the reciprocal rank-three operator.  Solving in the 18D
    orthonormal irrep coordinates avoids double-counting Cartesian shear
    entries and makes the reported rank and residual convention-independent.
    """
    if basis_tensors.ndim != 4 or basis_tensors.shape[-3:] != (3, 3, 3):
        raise ValueError("Electronic basis tensors must have shape [K,3,3,3]")
    design = piezo_to_irreps(basis_tensors).to(torch.float64).transpose(0, 1)
    expected = piezo_to_irreps(target).to(torch.float64).reshape(-1)
    left, singular, _ = torch.linalg.svd(design, full_matrices=False)
    if singular.numel():
        tolerance = max(design.shape) * torch.finfo(design.dtype).eps * singular.max()
        rank = int((singular > tolerance).sum())
    else:
        rank = 0
    fitted = left[:, :rank] @ (left[:, :rank].transpose(0, 1) @ expected)
    residual = fitted - expected
    target_norm = torch.linalg.vector_norm(expected)
    fitted_norm = torch.linalg.vector_norm(fitted)
    residual_norm = torch.linalg.vector_norm(residual)
    denominator = target_norm.clamp_min(float(floor) * expected.numel() ** 0.5)
    per_irrep: dict[str, dict[str, float]] = {}
    for name, block in PIEZO_IRREP_SLICES.items():
        block_target = expected[block]
        block_fitted = fitted[block]
        block_norm = torch.linalg.vector_norm(block_target)
        block_floor = float(floor) * block_target.numel() ** 0.5
        per_irrep[name] = {
            "target_norm": float(block_norm),
            "relative_residual": float(
                torch.linalg.vector_norm(block_fitted - block_target)
                / block_norm.clamp_min(block_floor)
            ),
        }
    maximum_cosine = fitted_norm / target_norm.clamp_min(1e-30)
    return {
        "basis_count": int(basis_tensors.shape[0]),
        "rank_in_18d_irrep_space": rank,
        "target_norm": float(target_norm),
        "minimum_residual_norm": float(residual_norm),
        "minimum_stabilized_relative_residual": float(residual_norm / denominator),
        "theoretical_maximum_cosine": float(maximum_cosine.clamp(0.0, 1.0)),
        "per_irrep": per_irrep,
    }


def macro_material_tensor_metrics(
    predictions: Iterable[torch.Tensor],
    targets: Iterable[torch.Tensor],
    floor: float,
) -> dict[str, float | int]:
    """Material-balanced tensor metrics with an explicit physical floor."""
    rows = [pair_metrics(prediction, target, floor) for prediction, target in zip(predictions, targets)]
    if not rows:
        raise ValueError("Material-balanced metrics require at least one material")
    macro_mae = _mean_rows(rows, "component_mae")
    macro_zero_mae = _mean_rows(rows, "zero_component_mae")
    return {
        "materials": len(rows),
        "cosine_macro_material": _mean_rows(rows, "directional_cosine"),
        "amplitude_ratio_macro": _mean_rows(rows, "stabilized_amplitude_ratio"),
        # Average the errors first, then form a one-zero-predictor comparison.
        # Averaging per-material skills would diverge on a legitimately
        # zero-response material because its zero baseline is exactly zero.
        "mae_skill_vs_zero_macro": 1.0 - macro_mae / max(macro_zero_mae, torch.finfo(torch.float64).eps),
        "amplitude_denominator_floor_per_component": float(floor),
    }


def _active_piezo_norm(target: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return a non-duplicated piezo norm for active-panel registration.

    Cartesian piezo tensors store both ``e_ijk`` and ``e_ikj``.  They have 27
    storage entries but only 18 independent physical coefficients, so an
    active threshold based on the stored Frobenius norm would accidentally
    double-count shear.  The aggregate cosine still uses the evaluator's
    Cartesian representation; this helper is only the registered inclusion
    gate for active-material reporting.
    """
    if target.shape[-3:] == (3, 3, 3) and torch.allclose(target, target.transpose(-1, -2), atol=1e-6, rtol=1e-6):
        voigt = cartesian_to_piezo_voigt(target)
        return torch.linalg.vector_norm(voigt), voigt.numel()
    return torch.linalg.vector_norm(target), target.numel()


def ionic_aggregate_metrics(
    predictions: Iterable[torch.Tensor],
    targets: Iterable[torch.Tensor],
    floor: float = FACTOR_FLOORS["ionic_piezo"],
) -> dict[str, float | int | str]:
    """Canonical ionic metrics resolving macro/micro aggregation ambiguity.

    ``ionic_cosine_macro_material`` gives every material one vote; the legacy
    ``directional_cosine`` equals the all-component micro cosine.  Active-only
    metrics retain only materials whose true Frobenius norm exceeds the same
    per-component resolution floor used by stabilized amplitudes, multiplied
    by ``sqrt(number_of_independent_components)``.  For a symmetric Cartesian
    piezo tensor that is ``sqrt(18)``, not the 27 duplicated storage entries.
    This makes the threshold dimensional and auditable rather than a post-hoc
    response-bin choice.
    """
    prediction_rows = [value.detach().cpu() for value in predictions]
    target_rows = [value.detach().cpu() for value in targets]
    if len(prediction_rows) != len(target_rows) or not prediction_rows:
        raise ValueError("Ionic aggregation requires equally many non-empty predictions and targets")
    if any(prediction.shape != target.shape for prediction, target in zip(prediction_rows, target_rows)):
        raise ValueError("Ionic aggregation requires matching prediction and target shapes")
    macro = macro_material_tensor_metrics(prediction_rows, target_rows, floor)
    micro = pair_metrics(torch.stack(prediction_rows), torch.stack(target_rows), floor)
    _, components = _active_piezo_norm(target_rows[0])
    active_threshold = float(floor) * components ** 0.5
    active = [
        (prediction, target) for prediction, target in zip(prediction_rows, target_rows)
        if float(_active_piezo_norm(target)[0]) > active_threshold
    ]
    active_macro = macro_material_tensor_metrics(
        [prediction for prediction, _ in active], [target for _, target in active], floor
    ) if active else None
    active_micro = pair_metrics(
        torch.stack([prediction for prediction, _ in active]),
        torch.stack([target for _, target in active]), floor,
    ) if active else None
    return {
        # Four canonical names for cross-evaluator comparisons.
        "ionic_cosine_macro_material": float(macro["cosine_macro_material"]),
        "ionic_cosine_micro_components": float(micro["directional_cosine"]),
        "ionic_cosine_active_only": (
            float(active_macro["cosine_macro_material"]) if active_macro is not None else float("nan")
        ),
        "ionic_amplitude_ratio_macro": float(macro["amplitude_ratio_macro"]),
        "ionic_mae_skill_vs_zero_macro": float(macro["mae_skill_vs_zero_macro"]),
        # Additional names make the active-panel definition and old fields
        # directly inspectable without reimplementing a metric downstream.
        "ionic_cosine_active_only_micro_components": (
            float(active_micro["directional_cosine"]) if active_micro is not None else float("nan")
        ),
        "ionic_active_materials": len(active),
        "ionic_materials": len(target_rows),
        "ionic_active_norm_threshold_c_per_m2": active_threshold,
        "ionic_amplitude_denominator_floor_per_component_c_per_m2": float(floor),
        "aggregation_contract": (
            "macro_material=mean(per-material cosine); micro_components=single cosine after concatenation; "
            "active_only=macro material cosine for true Frobenius norm above the registered threshold"
        ),
        # Retain evaluator v2 field names as explicitly micro aliases so older
        # readers cannot mistake them for macro material statistics.
        "directional_cosine": float(micro["directional_cosine"]),
        "stabilized_amplitude_ratio": float(micro["stabilized_amplitude_ratio"]),
        "mae_skill_vs_zero": float(micro["mae_skill_vs_zero"]),
    }


def response_decomposition_metrics(
    electronic_predictions: Iterable[torch.Tensor],
    ionic_predictions: Iterable[torch.Tensor],
    total_targets: Iterable[torch.Tensor],
    ionic_targets: Iterable[torch.Tensor],
    floor: float = FACTOR_FLOORS["ionic_piezo"],
) -> dict[str, float | int]:
    """Audit whether total-response fitting relies on branch cancellation."""
    electronic_predictions = [value.detach().cpu() for value in electronic_predictions]
    ionic_predictions = [value.detach().cpu() for value in ionic_predictions]
    total_targets = [value.detach().cpu() for value in total_targets]
    ionic_targets = [value.detach().cpu() for value in ionic_targets]
    if not electronic_predictions or not (
        len(electronic_predictions) == len(ionic_predictions) == len(total_targets) == len(ionic_targets)
    ):
        raise ValueError("Response decomposition requires equally many non-empty material rows")
    electronic_targets = [total - ionic for total, ionic in zip(total_targets, ionic_targets)]
    electronic = macro_material_tensor_metrics(electronic_predictions, electronic_targets, floor)
    ionic = macro_material_tensor_metrics(ionic_predictions, ionic_targets, floor)
    total_floor = float(floor) * total_targets[0].numel() ** 0.5
    electronic_over_total, ionic_over_total, total_over_total, cancellation = [], [], [], []
    for electronic_prediction, ionic_prediction, total_target in zip(
        electronic_predictions, ionic_predictions, total_targets
    ):
        denominator = torch.linalg.vector_norm(total_target).clamp_min(total_floor)
        electronic_norm = torch.linalg.vector_norm(electronic_prediction)
        ionic_norm = torch.linalg.vector_norm(ionic_prediction)
        total_norm = torch.linalg.vector_norm(electronic_prediction + ionic_prediction)
        electronic_over_total.append(float(electronic_norm / denominator))
        ionic_over_total.append(float(ionic_norm / denominator))
        total_over_total.append(float(total_norm / denominator))
        cancellation.append(float(total_norm / (electronic_norm + ionic_norm + 1e-12)))
    return {
        "materials": len(total_targets),
        "electronic_cosine_macro_material": float(electronic["cosine_macro_material"]),
        "ionic_cosine_macro_material": float(ionic["cosine_macro_material"]),
        "predicted_electronic_norm_over_true_total_macro": sum(electronic_over_total) / len(electronic_over_total),
        "predicted_ionic_norm_over_true_total_macro": sum(ionic_over_total) / len(ionic_over_total),
        "predicted_total_norm_over_true_total_macro": sum(total_over_total) / len(total_over_total),
        "predicted_cancellation_ratio_macro": sum(cancellation) / len(cancellation),
        "total_norm_denominator_floor_c_per_m2": total_floor,
    }


def _oracle_operator_policy(name: str, model_policy: str) -> str:
    """Name the operator used by an oracle experiment without inference.

    Oracle output is consumed by several reports, so this explicit mapping is
    preferable to asking a downstream reader to recover the policy from a
    historical experiment name.  Direct displacement and factorized predictions
    are labelled separately so no inverse-based oracle is confused with the
    maintained ``Z^T U_eta`` forward path.
    """
    if name == "direct_pred_z_pred_u_regularized":
        return "direct_displacement:no_inverse_in_forward"
    if name == "factorized_pred_z_pred_phi_pred_lambda_regularized":
        return f"model_configured:{model_policy}"
    if name.endswith("_regularized"):
        return "regularized"
    if name.endswith("_exact_true_stable"):
        return "exact_true_dfpt_stable_diagnostic"
    raise ValueError(f"Unknown oracle experiment name: {name}")


def _ionic_metric_bundle(
    predictions: Iterable[torch.Tensor],
    targets: Iterable[torch.Tensor],
    floor: float,
) -> dict[str, float | int | str]:
    """Emit canonical ionic aggregation plus explicit legacy micro aliases."""
    prediction_rows = list(predictions)
    target_rows = list(targets)
    canonical = ionic_aggregate_metrics(prediction_rows, target_rows, floor)
    # ``pair_metrics`` is preserved for readers of schema <=2.  Its cosine and
    # amplitude keys are duplicated by the documented micro aliases in the
    # canonical block, never used as the canonical material-balanced result.
    legacy_micro = pair_metrics(torch.stack(prediction_rows), torch.stack(target_rows), floor)
    return {
        **legacy_micro,
        **canonical,
        "legacy_metric_contract": "directional_cosine and stabilized_amplitude_ratio are micro-component aliases; use ionic_* names for canonical comparisons",
    }


@dataclass
class FactorAccumulator:
    name: str
    predictions: list[torch.Tensor] = field(default_factory=list)
    targets: list[torch.Tensor] = field(default_factory=list)
    material_metrics: list[dict[str, float | int]] = field(default_factory=list)

    def add(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
        prediction = prediction.detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        metrics = pair_metrics(prediction, target, FACTOR_FLOORS[self.name])
        self.predictions.append(prediction)
        self.targets.append(target)
        self.material_metrics.append(metrics)
        return metrics

    def summary(self) -> dict[str, float | int | str]:
        if not self.predictions:
            raise ValueError(f"No observations collected for {self.name}")
        micro = pair_metrics(
            torch.cat(self.predictions), torch.cat(self.targets), FACTOR_FLOORS[self.name]
        )
        macro_keys = (
            "component_mae",
            "component_rmse",
            "stabilized_relative_frobenius_error",
            "zero_component_mae",
            "stabilized_amplitude_ratio",
            "directional_cosine",
        )
        macro_mae = sum(float(row["component_mae"]) for row in self.material_metrics) / len(self.material_metrics)
        macro_zero_mae = sum(float(row["zero_component_mae"]) for row in self.material_metrics) / len(self.material_metrics)
        return {
            "unit": FACTOR_UNITS[self.name],
            "materials": len(self.material_metrics),
            **{f"micro_{key}": value for key, value in micro.items()},
            **{
                f"macro_material_{key}": sum(float(row[key]) for row in self.material_metrics)
                / len(self.material_metrics)
                for key in macro_keys
            },
            "macro_material_mae_skill_vs_zero": 1.0 - macro_mae / max(
                macro_zero_mae, torch.finfo(torch.float64).eps
            ),
        }


def _read_material_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    ids = [str(value) for value in values]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Material-ID file must contain a non-empty list of unique IDs")
    return ids


def _mean_rows(rows: Iterable[dict[str, float | int]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def _finite_mean_rows(rows: Iterable[dict[str, float | int]], key: str) -> tuple[float, int]:
    values = [float(row[key]) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return (sum(finite) / len(finite) if finite else float("nan"), len(finite))


def _finite_pearson_rows(
    rows: Iterable[dict[str, float | int]], key_x: str, key_y: str
) -> tuple[float, int]:
    pairs = [
        (float(row[key_x]), float(row[key_y]))
        for row in rows
        if key_x in row and key_y in row
    ]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan"), len(pairs)
    x = torch.tensor([pair[0] for pair in pairs], dtype=torch.float64)
    y = torch.tensor([pair[1] for pair in pairs], dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) == 0.0:
        return float("nan"), len(pairs)
    return float(torch.dot(x, y) / denominator), len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--material-ids-file", type=Path)
    parser.add_argument(
        "--material-ids-split", choices=("global", "same"), default="global",
        help="Use global formula-disjoint membership (default) or same-ID diagnostic membership only.",
    )
    parser.add_argument("--splits-file", type=Path, help="Frozen explicit split JSON used for this evaluation")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device",
        help="Optional execution-device override (for example cpu); it does not alter checkpoint physics or selection.",
    )
    parser.add_argument("--soft-mode-count", type=int, default=3)
    parser.add_argument(
        "--delta-grid",
        default="1e-4,3e-4,1e-3,3e-3,1e-2",
        help="Comma-separated signed-Green regularization scales in eV/Angstrom^2",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20270715)
    args = parser.parse_args()
    if args.soft_mode_count <= 0:
        raise ValueError("--soft-mode-count must be positive")
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be positive")
    delta_grid = [float(value) for value in args.delta_grid.split(",")]
    if not delta_grid or any(value <= 0 for value in delta_grid):
        raise ValueError("--delta-grid must contain positive values")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    checkpoint_device = str(cfg["device"])
    device = device_from_config(str(args.device or checkpoint_device))
    records = load_gmtnet_records(cfg["data_root"])
    if args.splits_file is not None and args.material_ids_file is not None:
        raise ValueError("--splits-file and --material-ids-file are mutually exclusive")
    split_path = args.splits_file or Path(str(cfg.get("splits_file", "")))
    if split_path.is_file():
        splits = load_explicit_splits(split_path, {str(record["JARVIS_ID"]) for record in records})
    else:
        global_splits = create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
        ids_path = args.material_ids_file or Path(str(cfg.get("material_ids_file", "")))
        if not ids_path.is_file():
            raise FileNotFoundError("A persisted audited material-ID file or frozen split file is required for DFPT evaluation")
        selected_ids = _read_material_ids(ids_path)
        splits = restrict_splits_to_material_ids(
            global_splits, selected_ids, args.material_ids_split
        )
    split_ids = splits[args.split]
    if not split_ids:
        raise ValueError(f"The audited DFPT {args.split} split is empty")

    cache_key = graph_cache_key(records, float(cfg["cutoff"]), int(cfg["max_neighbors"]))
    dataset = PiezoDataset(
        records,
        split_ids,
        float(cfg["cutoff"]),
        int(cfg["max_neighbors"]),
        processed_dir=cfg["processed_dir"],
        cache_key=cache_key,
        dfpt_dir=cfg.get("jarvis_dfpt_dir"),
        strain_completion_dir=cfg.get("jarvis_strain_completion_dir"),
        elastic_targets_path=cfg.get("elastic_targets_path"),
    )
    model = model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    factor_names = (
        "born_charge",
        "force_constant",
        "internal_strain",
        "internal_strain_full",
        "displacement_response",
        "ionic_piezo",
        "factorized_ionic_piezo",
        "electronic_piezo",
        "total_piezo",
        "dfpt_total_piezo",
        "dfpt_branch_sum_piezo",
        "dielectric",
        "ionic_dielectric",
        "macro_elastic",
    )
    accumulators = {name: FactorAccumulator(name) for name in factor_names}
    rows: list[dict[str, float | int | str]] = []
    total_predictions, total_targets = [], []
    electronic_predictions, ionic_predictions, factorized_ionic_predictions = [], [], []
    electronic_targets, dfpt_total_targets, ionic_targets = [], [], []
    electronic_irrep_predictions: dict[str, list[torch.Tensor]] = {
        name: [] for name in PIEZO_IRREP_SLICES
    }
    electronic_irrep_targets: dict[str, list[torch.Tensor]] = {
        name: [] for name in PIEZO_IRREP_SLICES
    }
    electronic_basis_oracles: list[dict[str, object]] = []
    total_substitution_values: dict[str, list[torch.Tensor]] = {
        "true_electronic_predicted_ionic": [],
        "predicted_electronic_true_ionic": [],
        "predicted_electronic_predicted_ionic": [],
    }
    oracle_names = (
        "direct_pred_z_pred_u_regularized",
        "factorized_pred_z_pred_phi_pred_lambda_regularized",
        "true_z_true_phi_pred_lambda_regularized",
        "pred_z_true_phi_pred_lambda_regularized",
        "true_z_pred_phi_pred_lambda_regularized",
        "true_z_true_phi_observed_lambda_regularized",
        "true_z_true_phi_pred_lambda_exact_true_stable",
    )
    oracle_values: dict[str, dict[str, list[torch.Tensor]]] = {
        stratum: {name: [] for name in oracle_names}
        for stratum in ("all", "stable", "soft_positive", "unstable")
    }
    oracle_targets: dict[str, list[torch.Tensor]] = {
        stratum: [] for stratum in ("all", "stable", "soft_positive", "unstable")
    }
    delta_values: dict[str, dict[float, list[torch.Tensor]]] = {
        stratum: {delta: [] for delta in delta_grid}
        for stratum in ("all", "stable", "soft_positive", "unstable")
    }
    delta_targets: dict[str, list[torch.Tensor]] = {
        stratum: [] for stratum in ("all", "stable", "soft_positive", "unstable")
    }
    completed_oracle_values: list[torch.Tensor] = []
    completed_oracle_targets: list[torch.Tensor] = []
    strict_substitution_values: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "true_z_true_phi_true_lambda_regularized",
            "pred_z_true_phi_true_lambda_regularized",
            "true_z_pred_phi_true_lambda_regularized",
            "pred_z_pred_phi_true_lambda_regularized",
        )
    }
    strict_direct_u_substitution_values: dict[str, list[torch.Tensor]] = {
        "true_z_predicted_u": [],
        "predicted_z_true_u": [],
        "predicted_z_predicted_u": [],
    }
    strict_displacement_targets: list[torch.Tensor] = []
    strict_low_rank_oracles: list[dict[str, object]] = []
    strict_alignment_rows: list[dict[str, float | int]] = []

    with torch.inference_mode():
        for index in range(len(dataset)):
            graph = dataset[index].clone()
            material_id = str(dataset.records[index]["JARVIS_ID"])
            if not bool(graph.force_constant_mask) or not bool(graph.ionic_piezo_mask) or not bool(graph.born_mask.all()):
                raise ValueError(f"Missing verified DFPT factors for {material_id}")
            if not bool(graph.dfpt_branch_mask):
                raise ValueError(f"Missing same-OUTCAR electronic/total branch labels for {material_id}")
            if not bool(graph.dielectric_mask):
                raise ValueError(f"Missing same-record dielectric target for {material_id}")
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
            graph.ptr = torch.tensor([0, graph.num_nodes], dtype=torch.long)
            graph = graph.to(device)
            components = model.predict_components(graph)

            force_target = clean_force_constant_target(graph.dfpt_force_constants_flat, graph.num_nodes)
            force_prediction = components.force_constants_flat.reshape(graph.num_nodes, graph.num_nodes, 3, 3)
            internal_prediction, internal_target = selected_internal_strain(
                components.internal_strain,
                graph.dfpt_internal_strain_flat,
                graph.dfpt_internal_strain_ions,
                graph.dfpt_internal_strain_directions,
            )
            total_target = graph.y.squeeze(0)
            ionic_target = graph.y_ionic_piezo.squeeze(0)
            dfpt_total_target = graph.y_dfpt_total_piezo.squeeze(0)
            electronic_target = graph.y_electronic_piezo.squeeze(0)
            electronic_prediction = components.electronic_piezo.squeeze(0)
            irrep_prediction = electronic_irrep_decomposition(electronic_prediction)
            irrep_target = electronic_irrep_decomposition(electronic_target)
            irrep_metrics = electronic_irrep_metrics(
                electronic_prediction, electronic_target
            )
            basis_oracle = electronic_basis_oracle(
                model.electronic_output_basis(graph)[0], electronic_target
            )
            electronic_basis_oracles.append(basis_oracle)
            for name in PIEZO_IRREP_SLICES:
                electronic_irrep_predictions[name].append(
                    irrep_prediction[name].detach().cpu()
                )
                electronic_irrep_targets[name].append(
                    irrep_target[name].detach().cpu()
                )
            values = {
                "born_charge": (components.born_charges, graph.y_born),
                "force_constant": (force_prediction, force_target),
                "internal_strain": (internal_prediction, internal_target),
                "ionic_piezo": (components.ionic_piezo.squeeze(0), ionic_target),
                "factorized_ionic_piezo": (
                    components.factorized_ionic_piezo.squeeze(0), ionic_target
                ),
                "electronic_piezo": (electronic_prediction, electronic_target),
                "total_piezo": (components.tensor.squeeze(0), total_target),
                "dfpt_total_piezo": (components.tensor.squeeze(0), dfpt_total_target),
                "dfpt_branch_sum_piezo": (
                    components.physical_tensor.squeeze(0),
                    dfpt_total_target,
                ),
                "dielectric": (components.macro_dielectric.squeeze(0), graph.y_dielectric.squeeze(0)),
            }
            if bool(graph.dfpt_ionic_dielectric_mask):
                values["ionic_dielectric"] = (
                    components.ionic_dielectric.squeeze(0),
                    graph.y_dfpt_ionic_dielectric.squeeze(0),
                )
            if bool(graph.elastic_mask):
                values["macro_elastic"] = (
                    components.macro_elastic.squeeze(0),
                    graph.y_elastic_gpa.squeeze(0),
                )
            row: dict[str, float | int | str] = {
                "material_id": material_id,
                "atoms": graph.num_nodes,
                "printed_internal_strain_blocks": int(graph.dfpt_internal_strain_count.item()),
                "symmetry_completed_internal_strain": int(
                    bool(graph.internal_strain_full_mask)
                ),
                "ionic_parameterization": model.ionic_parameterization,
            }
            for irrep_name, metrics in irrep_metrics.items():
                for key, value in metrics.items():
                    row[f"electronic_{irrep_name}_{key}"] = value
            row["electronic_basis_rank_18d"] = int(
                basis_oracle["rank_in_18d_irrep_space"]
            )
            row["electronic_basis_minimum_relative_residual"] = float(
                basis_oracle["minimum_stabilized_relative_residual"]
            )
            row["electronic_basis_theoretical_maximum_cosine"] = float(
                basis_oracle["theoretical_maximum_cosine"]
            )
            true_born = graph.y_born
            predicted_lambda = components.internal_strain
            observed_lambda = replace_printed_internal_strain(
                predicted_lambda,
                graph.dfpt_internal_strain_flat,
                graph.dfpt_internal_strain_ions,
                graph.dfpt_internal_strain_directions,
            )
            volume = torch.linalg.det(graph.cell.reshape(-1, 3, 3)[0]).abs()
            true_optical, _ = optical_eigensystem(force_target)
            minimum_true = float(true_optical.min()) if true_optical.numel() else float("inf")
            if minimum_true > model.response.optical_stability_cutoff:
                stratum = "stable"
            elif minimum_true > 0.0:
                stratum = "soft_positive"
            else:
                stratum = "unstable"
            row["stability_stratum"] = stratum
            mode_row = soft_mode_metrics(
                force_prediction,
                force_target,
                components.born_charges,
                true_born,
                args.soft_mode_count,
            )
            row.update(mode_row)
            predicted_optical, _ = optical_eigensystem(force_prediction)
            for spectrum_name, spectrum in (
                ("true", true_optical), ("predicted", predicted_optical)
            ):
                for eigen_index in range(min(3, spectrum.numel())):
                    row[f"{spectrum_name}_optical_eigenvalue_{eigen_index}"] = float(
                        spectrum[eigen_index]
                    )
                regions = spectrum_regularization_regions(
                    spectrum, model.response.optical_regularization
                )
                for key, value in regions.items():
                    row[f"{spectrum_name}_spectrum_{key}"] = value
                row[f"{spectrum_name}_near_regularization_fraction"] = (
                    float(regions["below_delta_fraction"])
                    + float(regions["delta_to_3delta_fraction"])
                )

            oracle = {
                "direct_pred_z_pred_u_regularized": components.ionic_piezo.squeeze(0),
                "factorized_pred_z_pred_phi_pred_lambda_regularized": (
                    components.factorized_ionic_piezo.squeeze(0)
                ),
                "true_z_true_phi_pred_lambda_regularized": ionic_piezo_from_factors(
                    model.response, true_born, force_target, predicted_lambda, volume, "regularized"
                ),
                "pred_z_true_phi_pred_lambda_regularized": ionic_piezo_from_factors(
                    model.response, components.born_charges, force_target, predicted_lambda, volume, "regularized"
                ),
                "true_z_pred_phi_pred_lambda_regularized": ionic_piezo_from_factors(
                    model.response, true_born, force_prediction, predicted_lambda, volume, "regularized"
                ),
                "true_z_true_phi_observed_lambda_regularized": ionic_piezo_from_factors(
                    model.response, true_born, force_target, observed_lambda, volume, "regularized"
                ),
            }
            if stratum == "stable":
                oracle["true_z_true_phi_pred_lambda_exact_true_stable"] = ionic_piezo_from_factors(
                    model.response, true_born, force_target, predicted_lambda, volume, "exact"
                )
            else:
                oracle["true_z_true_phi_pred_lambda_exact_true_stable"] = torch.full_like(
                    ionic_target, torch.nan
                )

            true_operator = model.response.optical_operator(force_target, "regularized")
            predicted_operator = model.response.optical_operator(force_prediction, "regularized")
            coupling = model.response._coupling_voigt(predicted_lambda).reshape(3 * graph.num_nodes, 6)
            response_weighted_phi = (
                model.response.PIEZO_C_PER_M2
                * true_born.reshape(-1, 3).T
                @ (predicted_operator - true_operator)
                @ coupling
                / volume
            )
            row["response_weighted_force_constant_mae_c_per_m2"] = float(response_weighted_phi.abs().mean())
            for name, value in oracle.items():
                metric = pair_metrics(value, ionic_target, FACTOR_FLOORS["ionic_piezo"])
                row[f"oracle_{name}_mae"] = metric["component_mae"]
                for target_stratum in ("all", stratum):
                    oracle_values[target_stratum][name].append(value.detach().cpu())
            for target_stratum in ("all", stratum):
                oracle_targets[target_stratum].append(ionic_target.detach().cpu())

            for delta in delta_grid:
                value = ionic_piezo_from_factors(
                    model.response,
                    true_born,
                    force_target,
                    predicted_lambda,
                    volume,
                    "regularized",
                    delta,
                )
                for target_stratum in ("all", stratum):
                    delta_values[target_stratum][delta].append(value.detach().cpu())
            for target_stratum in ("all", stratum):
                delta_targets[target_stratum].append(ionic_target.detach().cpu())
            for name, (prediction, target) in values.items():
                metrics = accumulators[name].add(prediction, target)
                for key, value in metrics.items():
                    row[f"{name}_{key}"] = value
            if bool(graph.internal_strain_full_mask):
                completed_target = graph.dfpt_internal_strain_full
                completed_metrics = accumulators["internal_strain_full"].add(
                    predicted_lambda, completed_target
                )
                for key, value in completed_metrics.items():
                    row[f"internal_strain_full_{key}"] = value
                true_coupling = model.response._coupling_voigt(completed_target).reshape(
                    3 * graph.num_nodes, 6
                )
                stiffness_bias = response_weighted_log_stiffness_bias(
                    force_prediction,
                    force_target,
                    true_born,
                    true_coupling,
                    model.response.optical_regularization,
                )
                for key, value in stiffness_bias.items():
                    row[f"stiffness_{key}"] = value
                true_displacement = model.response.apply_optical_operator(
                    force_target, true_coupling, "regularized"
                )
                predicted_displacement = model.response._coupling_voigt(
                    components.displacement_response
                ).reshape(3 * graph.num_nodes, 6)
                direct_u_substitutions = {
                    "true_z_predicted_u": ionic_piezo_from_displacement_matrix(
                        model.response, true_born, predicted_displacement, volume
                    ),
                    "predicted_z_true_u": ionic_piezo_from_displacement_matrix(
                        model.response,
                        components.born_charges,
                        true_displacement,
                        volume,
                    ),
                    "predicted_z_predicted_u": components.ionic_piezo.squeeze(0),
                }
                for name, value in direct_u_substitutions.items():
                    strict_direct_u_substitution_values[name].append(
                        value.detach().cpu()
                    )
                    metrics = pair_metrics(
                        value, ionic_target, FACTOR_FLOORS["ionic_piezo"]
                    )
                    for key, metric_value in metrics.items():
                        row[f"strict_direct_u_{name}_{key}"] = metric_value
                displacement_metrics = accumulators["displacement_response"].add(
                    predicted_displacement, true_displacement
                )
                for key, value in displacement_metrics.items():
                    row[f"displacement_response_{key}"] = value
                strict_displacement_targets.append(true_displacement.detach().cpu())
                low_rank_oracle = low_rank_displacement_oracle(
                    true_displacement, true_born
                )
                strict_low_rank_oracles.append(low_rank_oracle)
                row["displacement_response_numerical_rank"] = int(
                    low_rank_oracle["numerical_rank"]
                )
                for rank, metrics in low_rank_oracle["per_rank"].items():
                    for key, value in metrics.items():
                        row[f"displacement_rank{rank}_{key}"] = value
                alignment = response_active_alignment_metrics(
                    predicted_displacement, true_displacement, true_born
                )
                strict_alignment_rows.append(alignment)
                for key, value in alignment.items():
                    row[f"response_active_alignment_{key}"] = value

                strict_values = {
                    "true_z_true_phi_true_lambda_regularized": ionic_piezo_from_factors(
                        model.response, true_born, force_target, completed_target, volume, "regularized"
                    ),
                    "pred_z_true_phi_true_lambda_regularized": ionic_piezo_from_factors(
                        model.response, components.born_charges, force_target, completed_target, volume, "regularized"
                    ),
                    "true_z_pred_phi_true_lambda_regularized": ionic_piezo_from_factors(
                        model.response, true_born, force_prediction, completed_target, volume, "regularized"
                    ),
                    "pred_z_pred_phi_true_lambda_regularized": ionic_piezo_from_factors(
                        model.response, components.born_charges, force_prediction, completed_target, volume, "regularized"
                    ),
                }
                for name, value in strict_values.items():
                    strict_substitution_values[name].append(value.detach().cpu())
                    row[f"strict_oracle_{name}_mae"] = pair_metrics(
                        value, ionic_target, FACTOR_FLOORS["ionic_piezo"]
                    )["component_mae"]
                completed_oracle_values.append(
                    strict_values["true_z_true_phi_true_lambda_regularized"].detach().cpu()
                )
                completed_oracle_targets.append(ionic_target.detach().cpu())
            rows.append(row)
            total_substitutions = {
                "true_electronic_predicted_ionic": (
                    electronic_target + components.ionic_piezo.squeeze(0)
                ),
                "predicted_electronic_true_ionic": (
                    electronic_prediction + ionic_target
                ),
                "predicted_electronic_predicted_ionic": (
                    electronic_prediction + components.ionic_piezo.squeeze(0)
                ),
            }
            for name, value in total_substitutions.items():
                total_substitution_values[name].append(value.detach().cpu())
            total_predictions.append(components.tensor.cpu())
            total_targets.append(graph.y.cpu())
            electronic_predictions.append(components.electronic_piezo.cpu())
            electronic_targets.append(graph.y_electronic_piezo.cpu())
            ionic_predictions.append(components.ionic_piezo.cpu())
            factorized_ionic_predictions.append(components.factorized_ionic_piezo.cpu())
            dfpt_total_targets.append(graph.y_dfpt_total_piezo.cpu())
            ionic_targets.append(graph.y_ionic_piezo.cpu())

    total_prediction = torch.cat(total_predictions)
    total_target = torch.cat(total_targets)
    electronic_prediction = torch.cat(electronic_predictions)
    oracle_summary: dict[str, object] = {}
    delta_summary: dict[str, object] = {}
    for stratum in ("all", "stable", "soft_positive", "unstable"):
        if not oracle_targets[stratum]:
            oracle_summary[stratum] = {"materials": 0}
            delta_summary[stratum] = {"materials": 0}
            continue
        target_rows = oracle_targets[stratum]
        oracle_summary[stratum] = {
            "materials": len(target_rows),
            "canonical_operator_policy": "regularized",
            "canonical_experiment": "true_z_true_phi_pred_lambda_regularized",
            "experiments": {
                name: {
                    **_ionic_metric_bundle(values, target_rows, FACTOR_FLOORS["ionic_piezo"]),
                    "operator_policy": _oracle_operator_policy(name, model.response.optical_solve_policy),
                }
                for name, values in oracle_values[stratum].items()
                if values and torch.isfinite(torch.stack(values)).all()
            },
        }
        canonical_values = oracle_values[stratum]["true_z_true_phi_pred_lambda_regularized"]
        oracle_summary[stratum]["canonical_metrics"] = _ionic_metric_bundle(
            canonical_values, target_rows, FACTOR_FLOORS["ionic_piezo"]
        )
        delta_target_rows = delta_targets[stratum]
        delta_summary[stratum] = {
            "materials": len(delta_target_rows),
            "operator_policy": "regularized",
            "regularized_true_z_true_phi_pred_lambda": {
                str(delta): {
                    **_ionic_metric_bundle(
                        delta_values[stratum][delta], delta_target_rows, FACTOR_FLOORS["ionic_piezo"]
                    ),
                    "regularization_eV_per_A2": delta,
                }
                for delta in delta_grid
            },
        }

    total_response_ci = material_bootstrap_confidence_interval(
        total_predictions,
        total_targets,
        lambda prediction, target: response_tensor_skill(
            torch.stack(prediction), torch.stack(target)
        )["tensor_response_skill_vs_zero"],
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    ionic_macro_ci = material_bootstrap_confidence_interval(
        ionic_predictions,
        ionic_targets,
        lambda prediction, target: ionic_aggregate_metrics(
            prediction, target, FACTOR_FLOORS["ionic_piezo"]
        )["ionic_cosine_macro_material"],
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    factorized_ionic_macro_ci = material_bootstrap_confidence_interval(
        factorized_ionic_predictions,
        ionic_targets,
        lambda prediction, target: ionic_aggregate_metrics(
            prediction, target, FACTOR_FLOORS["ionic_piezo"]
        )["ionic_cosine_macro_material"],
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    low_rank_summary: dict[str, object] = {
        "materials": len(strict_low_rank_oracles),
        "algebraic_rank_upper_bound": 6,
        "interpretation": (
            "rank(U_eta)<=6 is a matrix-rank fact from six strain right-hand sides, "
            "not evidence that six physical phonon eigenmodes dominate"
        ),
        "numerical_rank_counts": {
            str(rank): sum(
                int(item["numerical_rank"]) == rank for item in strict_low_rank_oracles
            )
            for rank in range(7)
        },
        "per_rank": {
            str(rank): {
                key: sum(
                    float(item["per_rank"][str(rank)][key])
                    for item in strict_low_rank_oracles
                ) / max(len(strict_low_rank_oracles), 1)
                for key in (
                    "displacement_relative_frobenius_error",
                    "response_relative_frobenius_error",
                    "singular_energy_fraction",
                )
            }
            for rank in (1, 2, 4, 6)
        },
    }
    alignment_keys = (
        "displacement_subspace_projector_overlap",
        "displacement_subspace_minimum_principal_cosine",
        "target_true_charge_active_energy_fraction",
        "predicted_true_charge_active_energy_fraction",
        "true_charge_active_projected_directional_cosine",
        "true_charge_cross_covariance_directional_cosine",
        "true_charge_cross_covariance_amplitude_ratio",
    )
    alignment_means = {
        key: _finite_mean_rows(strict_alignment_rows, key)[0] for key in alignment_keys
    }
    alignment_valid_counts = {
        key: _finite_mean_rows(strict_alignment_rows, key)[1] for key in alignment_keys
    }
    alignment_summary: dict[str, object] = {
        "materials": len(strict_alignment_rows),
        "comparison": "predicted direct-U versus true regularized U using true BEC",
        "gauge_policy": (
            "column-space projectors and Z^T U cross-covariance; no individual "
            "phonon-eigenvector matching"
        ),
        "target_displacement_rank_counts": {
            str(rank): sum(
                int(row["target_displacement_rank"]) == rank
                for row in strict_alignment_rows
            )
            for rank in range(7)
        },
        "predicted_displacement_rank_counts": {
            str(rank): sum(
                int(row["predicted_displacement_rank"]) == rank
                for row in strict_alignment_rows
            )
            for rank in range(7)
        },
        "finite_materials_per_metric": alignment_valid_counts,
        "mean": alignment_means,
    }
    spectrum_region_summary: dict[str, dict[str, float | int]] = {}
    for spectrum_name in ("true", "predicted"):
        counts = {
            region: sum(
                int(row[f"{spectrum_name}_spectrum_{region}_count"])
                for row in rows
            )
            for region in ("below_delta", "delta_to_3delta", "above_3delta")
        }
        mode_count = sum(counts.values())
        spectrum_region_summary[spectrum_name] = {
            "mode_count": mode_count,
            **{f"{region}_count": count for region, count in counts.items()},
            **{
                f"{region}_fraction": count / max(mode_count, 1)
                for region, count in counts.items()
            },
        }
    stiffness_amplitude_correlation, stiffness_correlation_materials = (
        _finite_pearson_rows(
            rows,
            "stiffness_response_weighted_mean_log_abs_stiffness_bias",
            "ionic_piezo_stabilized_amplitude_ratio",
        )
    )
    electronic_irrep_summary = {
        name: macro_material_tensor_metrics(
            electronic_irrep_predictions[name],
            electronic_irrep_targets[name],
            FACTOR_FLOORS["electronic_piezo"],
        )
        for name in PIEZO_IRREP_SLICES
    }
    basis_oracle_summary: dict[str, object] = {
        "materials": len(electronic_basis_oracles),
        "basis_definition": (
            "unrestricted per-atom/channel/family Cartesian readout bases plus "
            "the reciprocal rank-three operator; coefficient-network constraints omitted"
        ),
        "rank_counts_in_18d_irrep_space": {
            str(rank): sum(
                int(row["rank_in_18d_irrep_space"]) == rank
                for row in electronic_basis_oracles
            )
            for rank in range(19)
        },
        "mean_minimum_stabilized_relative_residual": sum(
            float(row["minimum_stabilized_relative_residual"])
            for row in electronic_basis_oracles
        ) / max(len(electronic_basis_oracles), 1),
        "mean_theoretical_maximum_cosine": sum(
            float(row["theoretical_maximum_cosine"])
            for row in electronic_basis_oracles
        ) / max(len(electronic_basis_oracles), 1),
        "per_irrep_mean_relative_residual": {
            name: sum(
                float(row["per_irrep"][name]["relative_residual"])
                for row in electronic_basis_oracles
            ) / max(len(electronic_basis_oracles), 1)
            for name in PIEZO_IRREP_SLICES
        },
    }
    total_substitution_summary = {
        name: macro_material_tensor_metrics(
            values, dfpt_total_targets, FACTOR_FLOORS["total_piezo"]
        )
        for name, values in total_substitution_values.items()
    }
    strict_direct_u_substitution_summary = {
        name: _ionic_metric_bundle(
            values, completed_oracle_targets, FACTOR_FLOORS["ionic_piezo"]
        )
        for name, values in strict_direct_u_substitution_values.items()
        if values
    }
    summary: dict[str, object] = {
        "schema": 7,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_declared_device": checkpoint_device,
        "runtime_device": str(device),
        "split": args.split,
        "formula_disjoint": args.material_ids_file is None or args.material_ids_split == "global",
        "material_ids_split_mode": args.material_ids_split if args.material_ids_file is not None else "explicit_frozen_split",
        "material_count": len(dataset),
        "material_ids": [str(record["JARVIS_ID"]) for record in dataset.records],
        "factor_denominator_floors": FACTOR_FLOORS,
        "factors": {
            name: accumulator.summary()
            for name, accumulator in accumulators.items()
            if accumulator.predictions
        },
        "total_response_skill": response_tensor_skill(total_prediction, total_target),
        "total_response_skill_bootstrap_95": total_response_ci,
        "electronic_only_response_skill_vs_gmtnet_total": response_tensor_skill(electronic_prediction, total_target),
        "electronic_irrep_decomposition": electronic_irrep_summary,
        "electronic_output_basis_oracle": basis_oracle_summary,
        "same_outcar_total_substitution_grid": total_substitution_summary,
        "zero_response_skill": response_tensor_skill(torch.zeros_like(total_target), total_target),
        "ionic_response_aggregation": _ionic_metric_bundle(
            ionic_predictions, ionic_targets, FACTOR_FLOORS["ionic_piezo"]
        ),
        "ionic_macro_cosine_bootstrap_95": {
            "direct_displacement_ionic": ionic_macro_ci,
            "factorized_phi_lambda_diagnostic": factorized_ionic_macro_ci,
        },
        "jarvis_outcar_branch_decomposition": {
            "direct_displacement_ionic": response_decomposition_metrics(
                electronic_predictions, ionic_predictions, dfpt_total_targets, ionic_targets
            ),
            "factorized_phi_lambda_diagnostic": response_decomposition_metrics(
                electronic_predictions, factorized_ionic_predictions, dfpt_total_targets, ionic_targets
            ),
            "source_contract": "electronic, ionic, and total targets are all from the same JARVIS OUTCAR",
        },
        "coverage": {
            "atoms": sum(int(row["atoms"]) for row in rows),
            "printed_internal_strain_blocks": sum(int(row["printed_internal_strain_blocks"]) for row in rows),
            "mean_printed_blocks_per_material": _mean_rows(rows, "printed_internal_strain_blocks"),
            "strict_symmetry_completed_materials": sum(
                int(row["symmetry_completed_internal_strain"]) for row in rows
            ),
        },
        "stability": {
            "cutoff_eV_per_A2": model.response.optical_stability_cutoff,
            "stable_materials": sum(row["stability_stratum"] == "stable" for row in rows),
            "soft_positive_materials": sum(row["stability_stratum"] == "soft_positive" for row in rows),
            "unstable_materials": sum(row["stability_stratum"] == "unstable" for row in rows),
            "mean_true_fraction_with_abs_lambda_below_3delta": _mean_rows(
                rows, "true_near_regularization_fraction"
            ),
            "mean_predicted_fraction_with_abs_lambda_below_3delta": _mean_rows(
                rows, "predicted_near_regularization_fraction"
            ),
            "resolvent_spectrum_regions": spectrum_region_summary,
            "response_weighted_log_stiffness_bias": {
                "materials": _finite_mean_rows(
                    rows,
                    "stiffness_response_weighted_mean_log_abs_stiffness_bias",
                )[1],
                "mean": _finite_mean_rows(
                    rows,
                    "stiffness_response_weighted_mean_log_abs_stiffness_bias",
                )[0],
                "pearson_correlation_with_direct_ionic_amplitude_ratio": (
                    stiffness_amplitude_correlation
                ),
                "correlation_materials": stiffness_correlation_materials,
                "interpretation": (
                    "positive bias means predicted Rayleigh stiffness is harder on "
                    "true BEC/strain-coupled modes; correlation is diagnostic only"
                ),
            },
        },
        "soft_mode_metrics": {
            key: _mean_rows(rows, key)
            for key in (
                "lowest_optical_eigenvalue_mae",
                "soft_mode_sign_accuracy",
                "soft_mode_subspace_overlap",
                "mode_effective_charge_norm_mae",
                "response_weighted_force_constant_mae_c_per_m2",
            )
        },
        "oracle_factor_replacement": oracle_summary,
        "strict_symmetry_completed_lambda_oracle": (
            {
                "materials": len(completed_oracle_targets),
                "operator_policy": "regularized",
                "metrics": _ionic_metric_bundle(
                    completed_oracle_values, completed_oracle_targets, FACTOR_FLOORS["ionic_piezo"]
                ),
                "substitution_grid": {
                    name: _ionic_metric_bundle(
                        values, completed_oracle_targets, FACTOR_FLOORS["ionic_piezo"]
                    )
                    for name, values in strict_substitution_values.items()
                },
                "direct_u_z_substitution_grid": strict_direct_u_substitution_summary,
                "displacement_response_target": (
                    accumulators["displacement_response"].summary()
                    if strict_displacement_targets
                    else {"materials": 0}
                ),
                "low_rank_displacement_oracle": low_rank_summary,
                "response_active_alignment": alignment_summary,
            }
            if completed_oracle_targets else {"materials": 0}
        ),
        "delta_sensitivity": delta_summary,
        "unavailable_oracles": {
            "true_lambda_all_materials": (
                "A complete Lambda is unavailable for the full split. Strict space-group plus "
                "acoustic-null completion is reported only for samples whose printed blocks "
                "uniquely determine the invariant tensor and pass redundant-block and ionic "
                "closure gates; all other samples remain masked."
            ),
            "modewise_strain_coupling_target_all_materials": "Requires complete true Lambda for every material.",
            "true_factor_exact_upper_bound_all_materials": "Requires complete true Lambda for every material.",
        },
        "resampling_material_rows": [
            {
                "material_id": str(dataset.records[index]["JARVIS_ID"]),
                "total_prediction": total_predictions[index].reshape(-1).tolist(),
                "total_target": total_targets[index].reshape(-1).tolist(),
                "ionic_prediction": ionic_predictions[index].reshape(-1).tolist(),
                "ionic_target": ionic_targets[index].reshape(-1).tolist(),
                "factorized_ionic_prediction": factorized_ionic_predictions[index]
                .reshape(-1)
                .tolist(),
            }
            for index in range(len(dataset))
        ],
        "resampling_contract": (
            "complete materials are the resampling unit; rows are serialized only "
            "for post-evaluation hierarchical seed/material intervals and never "
            "for checkpoint selection"
        ),
    }
    if args.material_ids_file is not None and args.material_ids_split == "same":
        summary["interpretation_boundary"] = (
            "Same-ID diagnostic evaluation only: training and evaluation membership may overlap; "
            "this output cannot support a formula-disjoint or frozen-panel performance claim."
        )
    output = args.output or args.checkpoint.parent / f"dfpt_factor_evaluation_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
