"""Certified identifiability of atom-resolved internal strain.

The full internal-strain tensor is represented in the orthonormal
space-group/acoustic basis ``vec(Lambda)=B c``.  Macroscopic ionic piezoelectric
data then observe only ``M_macro c``.  Printed OUTCAR blocks add rows of ``B``.

Ranks are algebraic statements.  Condition numbers are reported only after an
explicit, unit-invariant family scaling; concatenating C/m2 and eV/Angstrom
observations without such a scaling would make the certificate depend on the
choice of units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .model import AtomCoordinateResponsePotential
from .strain_completion import (
    _observed_system,
    cartesian_space_group_operations,
    invariant_basis,
    vector_to_internal_tensor,
)
from .tensor_ops import source_voigt_to_canonical


RANK_RELATIVE_TOLERANCE = 1e-8
RANK_ABSOLUTE_TOLERANCE = 1e-10
MACRO_RMS_FLOOR_C_PER_M2 = 0.05
PRINTED_RMS_FLOOR = 1e-6


@dataclass(frozen=True)
class IdentificationSystem:
    """Linear observations for one material in invariant coordinates."""

    basis: torch.Tensor
    macro_matrix: torch.Tensor
    macro_target: torch.Tensor
    printed_matrix: torch.Tensor
    printed_target: torch.Tensor
    printed_blocks: tuple[torch.Tensor, ...]
    macro_scale: float
    printed_scale: float

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])

    def joint_matrix(self, printed_rows: torch.Tensor | None = None) -> torch.Tensor:
        printed = self.printed_matrix if printed_rows is None else self.printed_matrix[printed_rows]
        return torch.cat(
            (self.macro_matrix / self.macro_scale, printed / self.printed_scale),
            dim=0,
        )

    def joint_target(self, printed_rows: torch.Tensor | None = None) -> torch.Tensor:
        printed = self.printed_target if printed_rows is None else self.printed_target[printed_rows]
        return torch.cat(
            (self.macro_target / self.macro_scale, printed / self.printed_scale),
            dim=0,
        )


def _family_rms(target: torch.Tensor, floor: float) -> float:
    if target.numel() == 0:
        return float(floor)
    rms = torch.linalg.vector_norm(target) / target.numel() ** 0.5
    return max(float(rms), float(floor))


def linear_map_metrics(
    matrix: torch.Tensor,
    domain_dimension: int | None = None,
    *,
    relative_tolerance: float = RANK_RELATIVE_TOLERANCE,
    absolute_tolerance: float = RANK_ABSOLUTE_TOLERANCE,
) -> dict[str, float | int | bool | None]:
    """Return rank/nullity and full-column conditioning of a linear map."""
    if matrix.ndim != 2:
        raise ValueError("Identification matrix must be rank two")
    dimensions = int(matrix.shape[1] if domain_dimension is None else domain_dimension)
    if dimensions != matrix.shape[1]:
        raise ValueError("domain_dimension must equal the matrix column count")
    if dimensions == 0:
        return {
            "rank": 0,
            "nullity": 0,
            "full_column_rank": True,
            "sigma_max": None,
            "sigma_min_nonzero": None,
            "sigma_min_full": None,
            "condition_number_full": 1.0,
            "rank_threshold": absolute_tolerance,
        }
    singular = torch.linalg.svdvals(matrix.to(torch.float64))
    sigma_max = float(singular[0]) if singular.numel() else 0.0
    threshold = max(float(absolute_tolerance), float(relative_tolerance) * sigma_max)
    rank = int((singular > threshold).sum())
    full = rank == dimensions and singular.numel() >= dimensions
    sigma_min_nonzero = float(singular[rank - 1]) if rank else None
    sigma_min_full = float(singular[dimensions - 1]) if full else None
    condition = sigma_max / sigma_min_full if full and sigma_min_full else None
    return {
        "rank": rank,
        "nullity": dimensions - rank,
        "full_column_rank": full,
        "sigma_max": sigma_max,
        "sigma_min_nonzero": sigma_min_nonzero,
        "sigma_min_full": sigma_min_full,
        "condition_number_full": condition,
        "rank_threshold": threshold,
    }


def observable_null_projectors(
    matrix: torch.Tensor,
    *,
    relative_tolerance: float = RANK_RELATIVE_TOLERANCE,
    absolute_tolerance: float = RANK_ABSOLUTE_TOLERANCE,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Exact orthogonal projectors onto row-space and null-space in c-space."""
    dimensions = int(matrix.shape[1])
    if dimensions == 0:
        empty = matrix.new_zeros((0, 0))
        return empty, empty, 0
    _, singular, right_h = torch.linalg.svd(matrix.to(torch.float64), full_matrices=True)
    sigma_max = float(singular[0]) if singular.numel() else 0.0
    threshold = max(float(absolute_tolerance), float(relative_tolerance) * sigma_max)
    rank = int((singular > threshold).sum())
    right = right_h.transpose(0, 1)
    observable = right[:, :rank] @ right[:, :rank].transpose(0, 1)
    null = right[:, rank:] @ right[:, rank:].transpose(0, 1)
    return observable.to(matrix.dtype), null.to(matrix.dtype), rank


def macro_response_matrix(
    *,
    basis: torch.Tensor,
    born_charges: torch.Tensor,
    force_constants: torch.Tensor,
    volume: torch.Tensor | float,
    regularization: float = 1e-3,
) -> torch.Tensor:
    """Map invariant Lambda coefficients to canonical 3x6 ionic response.

    All six strain directions for all basis vectors are propagated in one
    reduced optical solve.  The returned matrix has shape ``[18, d]``.
    """
    atoms = int(born_charges.shape[0])
    if basis.shape[0] != 18 * atoms:
        raise ValueError("Invariant basis and Born-charge atom count differ")
    dimensions = int(basis.shape[1])
    if dimensions == 0:
        return basis.new_zeros((18, 0))
    response = AtomCoordinateResponsePotential(
        optical_solve_policy="regularized",
        optical_regularization=regularization,
    )
    tensors = vector_to_internal_tensor(basis.transpose(0, 1), atoms)
    couplings = response._coupling_voigt(tensors).reshape(dimensions, 3 * atoms, 6)
    rhs = couplings.permute(1, 0, 2).reshape(3 * atoms, dimensions * 6)
    relaxed = response.apply_optical_operator(
        force_constants.to(torch.float64),
        rhs.to(torch.float64),
        solve_policy="regularized",
        regularization=regularization,
    ).reshape(3 * atoms, dimensions, 6).permute(1, 0, 2)
    charge = born_charges.to(torch.float64).reshape(3 * atoms, 3)
    volume_tensor = torch.as_tensor(volume, dtype=torch.float64).abs().clamp_min(1e-30)
    response_by_basis = (
        response.PIEZO_C_PER_M2
        * torch.einsum("ai,dav->div", charge, relaxed)
        / volume_tensor
    )
    return response_by_basis.reshape(dimensions, 18).transpose(0, 1)


def build_identification_system(
    record: dict[str, Any],
    payload: dict[str, Any],
    *,
    symprec: float = 1e-5,
    rank_tolerance: float = 1e-7,
    regularization: float = 1e-3,
    macro_rms_floor: float = MACRO_RMS_FLOOR_C_PER_M2,
    printed_rms_floor: float = PRINTED_RMS_FLOOR,
) -> IdentificationSystem:
    """Build macro and printed-block observations without completing Lambda."""
    atoms = len(record["atoms"]["elements"])
    operations = cartesian_space_group_operations(record, symprec=symprec)
    basis = invariant_basis(atoms, operations, rank_tolerance=rank_tolerance)
    indices, printed_target, blocks = _observed_system(
        payload["internal_strain_tensors"],
        payload["internal_strain_ions"],
        payload["internal_strain_directions"],
        atoms,
    )
    printed_matrix = basis[indices]
    cell = torch.as_tensor(record["atoms"]["lattice_mat"], dtype=torch.float64)
    macro_matrix = macro_response_matrix(
        basis=basis,
        born_charges=payload["born_charges"],
        force_constants=payload["force_constants"],
        volume=torch.linalg.det(cell),
        regularization=regularization,
    )
    macro_target = source_voigt_to_canonical(
        payload["ionic_piezo_source"].to(torch.float64)
    ).reshape(-1)
    return IdentificationSystem(
        basis=basis,
        macro_matrix=macro_matrix,
        macro_target=macro_target,
        printed_matrix=printed_matrix,
        printed_target=printed_target,
        printed_blocks=tuple(blocks),
        macro_scale=_family_rms(macro_target, macro_rms_floor),
        printed_scale=_family_rms(printed_target, printed_rms_floor),
    )


def identification_certificate(system: IdentificationSystem) -> dict[str, Any]:
    """JSON-serializable per-material identifiability certificate."""
    macro = linear_map_metrics(system.macro_matrix, system.dimension)
    printed = linear_map_metrics(system.printed_matrix / system.printed_scale, system.dimension)
    joint = linear_map_metrics(system.joint_matrix(), system.dimension)
    return {
        "identifiable_dimension": system.dimension,
        "rank_macro": macro["rank"],
        "null_dimension_macro": macro["nullity"],
        "rank_printed": printed["rank"],
        "null_dimension_printed": printed["nullity"],
        "rank_joint": joint["rank"],
        "null_dimension_joint": joint["nullity"],
        "macro_full_identifiable": macro["full_column_rank"],
        "printed_full_identifiable": printed["full_column_rank"],
        "joint_full_identifiable": joint["full_column_rank"],
        "sigma_min_joint_scaled": joint["sigma_min_full"],
        "condition_joint_scaled": joint["condition_number_full"],
        "macro_observation_scale_c_per_m2": system.macro_scale,
        "printed_observation_scale": system.printed_scale,
        "printed_blocks": len(system.printed_blocks),
        "printed_scalar_components": int(system.printed_target.numel()),
        "rank_policy": {
            "relative_tolerance": RANK_RELATIVE_TOLERANCE,
            "absolute_tolerance": RANK_ABSOLUTE_TOLERANCE,
        },
        "conditioning_policy": (
            "macro and printed families divided by per-material target RMS, "
            f"floored at {MACRO_RMS_FLOOR_C_PER_M2} C/m2 and {PRINTED_RMS_FLOOR}"
        ),
    }
