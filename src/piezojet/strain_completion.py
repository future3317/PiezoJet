"""Strict symmetry completion audit for atom-resolved strain-force tensors.

The public JARVIS OUTCAR files print only symmetry-inequivalent ion/direction
blocks.  This module does not fill missing entries heuristically.  It builds
the exact linear representation of the crystal space group on the full
``[N, 3, 3, 3]`` tensor, restricts that representation to its invariant
subspace, and accepts a completion only when the printed blocks uniquely
determine every invariant degree of freedom.  Redundant blocks and the
reported ionic piezoelectric tensor provide independent closure checks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import linear_sum_assignment

from .data import load_gmtnet_records
from .evaluate_dfpt import ionic_piezo_from_factors
from .jarvis_dfpt import JarvisDFPTCache
from .model import AtomCoordinateResponsePotential
from .tensor_ops import piezo_voigt_to_cartesian, source_voigt_to_canonical


VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
COMPLETION_SCHEMA = 2


@dataclass(frozen=True)
class CartesianSymmetryOperation:
    rotation: torch.Tensor
    permutation: torch.Tensor
    mapping_error_angstrom: float


def _structure(record: dict[str, Any]) -> Structure:
    atoms = record["atoms"]
    return Structure(
        atoms["lattice_mat"], atoms["elements"], atoms["coords"],
        coords_are_cartesian=False,
    )


def cartesian_space_group_operations(
    record: dict[str, Any],
    symprec: float = 1e-5,
) -> list[CartesianSymmetryOperation]:
    """Return Cartesian rotations and bijective atom permutations."""
    structure = _structure(record)
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    operations = analyzer.get_symmetry_operations(cartesian=True)
    if not operations:
        raise ValueError("No space-group operations returned by pymatgen")
    cartesian = np.asarray(structure.cart_coords, dtype=float)
    fractional = np.asarray(structure.frac_coords, dtype=float)
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    inverse_lattice = np.linalg.inv(lattice)
    species = np.asarray([str(site.specie) for site in structure])
    output: list[CartesianSymmetryOperation] = []
    for operation in operations:
        transformed_cartesian = np.stack(
            [operation.operate(position) for position in cartesian]
        )
        transformed_fractional = transformed_cartesian @ inverse_lattice
        permutation = np.full(len(structure), -1, dtype=np.int64)
        maximum_error = 0.0
        for symbol in np.unique(species):
            sources = np.flatnonzero(species == symbol)
            targets = np.flatnonzero(species == symbol)
            delta = (
                transformed_fractional[sources, None, :]
                - fractional[targets][None, :, :]
            )
            delta -= np.round(delta)
            distances = np.linalg.norm(delta @ lattice, axis=-1)
            source_assignment, target_assignment = linear_sum_assignment(distances)
            permutation[sources[source_assignment]] = targets[target_assignment]
            maximum_error = max(
                maximum_error,
                float(distances[source_assignment, target_assignment].max(initial=0.0)),
            )
        if (permutation < 0).any() or len(np.unique(permutation)) != len(structure):
            raise ValueError("Space-group operation did not produce a bijective atom map")
        output.append(
            CartesianSymmetryOperation(
                rotation=torch.as_tensor(
                    np.asarray(operation.rotation_matrix), dtype=torch.float64
                ),
                permutation=torch.as_tensor(permutation, dtype=torch.long),
                mapping_error_angstrom=maximum_error,
            )
        )
    return output


def internal_tensor_to_vector(tensor: torch.Tensor) -> torch.Tensor:
    """Pack symmetric strain indices without engineering-shear rescaling."""
    values = torch.stack([tensor[..., i, j] for i, j in VOIGT_PAIRS], dim=-1)
    return values.reshape(*values.shape[:-3], -1)


def vector_to_internal_tensor(vector: torch.Tensor, atoms: int) -> torch.Tensor:
    leading = vector.shape[:-1]
    packed = vector.reshape(*leading, atoms, 3, 6)
    tensor = vector.new_zeros(*leading, atoms, 3, 3, 3)
    for index, (i, j) in enumerate(VOIGT_PAIRS):
        tensor[..., i, j] = packed[..., index]
        tensor[..., j, i] = packed[..., index]
    return tensor


def transform_internal_tensor(
    tensor: torch.Tensor,
    operation: CartesianSymmetryOperation,
) -> torch.Tensor:
    """Apply one space-group operation to Lambda[kappa,alpha,j,k]."""
    rotation = operation.rotation.to(dtype=tensor.dtype, device=tensor.device)
    rotated = torch.einsum(
        "ab,jc,kd,...nbcd->...najk",
        rotation,
        rotation,
        rotation,
        tensor,
    )
    output = torch.empty_like(rotated)
    output[..., operation.permutation.to(device=tensor.device), :, :, :] = rotated
    return output


def invariant_basis(
    atoms: int,
    operations: list[CartesianSymmetryOperation],
    rank_tolerance: float = 1e-7,
) -> torch.Tensor:
    """Orthonormal basis for the Reynolds-projected invariant subspace."""
    dimensions = 18 * atoms
    basis_vectors = torch.eye(dimensions, dtype=torch.float64)
    basis_tensors = vector_to_internal_tensor(basis_vectors, atoms)
    reynolds = torch.zeros(dimensions, dimensions, dtype=torch.float64)
    for operation in operations:
        transformed = transform_internal_tensor(basis_tensors, operation)
        # Each row is the transformed image of one input basis vector.
        reynolds += internal_tensor_to_vector(transformed).transpose(0, 1)
    reynolds /= len(operations)
    left, singular, _ = torch.linalg.svd(reynolds, full_matrices=False)
    rank = int((singular > rank_tolerance).sum())
    symmetry_basis = left[:, :rank]
    # Physical internal forces cannot contain a net translation for any
    # strain component: sum_kappa Lambda[kappa, alpha, mu] = 0.  Intersect the
    # space-group invariant range with this acoustic-null subspace before
    # asking whether printed blocks uniquely determine a tensor.
    acoustic = torch.zeros(18, dimensions, dtype=torch.float64)
    for atom in range(atoms):
        for direction in range(3):
            for component in range(6):
                acoustic[direction * 6 + component, ((atom * 3 + direction) * 6 + component)] = 1.0
    constrained = acoustic @ symmetry_basis
    _, constrained_singular, right = torch.linalg.svd(
        constrained, full_matrices=True
    )
    constrained_rank = int((constrained_singular > rank_tolerance).sum())
    null_coordinates = right[constrained_rank:].transpose(0, 1)
    return symmetry_basis @ null_coordinates


def _observed_system(
    tensors: torch.Tensor,
    ions: torch.Tensor,
    directions: torch.Tensor,
    atoms: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    tensors = 0.5 * (tensors.to(torch.float64) + tensors.to(torch.float64).transpose(-1, -2))
    packed = torch.stack([tensors[..., i, j] for i, j in VOIGT_PAIRS], dim=-1)
    block_indices, indices = [], []
    for ion, direction in zip(ions.tolist(), directions.tolist()):
        current = torch.tensor(
            [((int(ion) * 3 + int(direction)) * 6 + component) for component in range(6)],
            dtype=torch.long,
        )
        block_indices.append(current)
        indices.extend(current.tolist())
    return torch.tensor(indices, dtype=torch.long), packed.reshape(-1), block_indices


def complete_internal_strain(
    record: dict[str, Any],
    payload: dict[str, Any],
    symprec: float = 1e-5,
    mapping_tolerance_angstrom: float = 2e-5,
    rank_tolerance: float = 1e-7,
    fit_relative_tolerance: float = 5e-3,
    cross_validation_relative_tolerance: float = 5e-2,
    ionic_relative_tolerance: float = 5e-2,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Complete one tensor only after uniqueness and closure gates pass."""
    atoms = len(record["atoms"]["elements"])
    operations = cartesian_space_group_operations(record, symprec)
    maximum_mapping_error = max(op.mapping_error_angstrom for op in operations)
    basis = invariant_basis(atoms, operations, rank_tolerance)
    indices, values, blocks = _observed_system(
        payload["internal_strain_tensors"],
        payload["internal_strain_ions"],
        payload["internal_strain_directions"],
        atoms,
    )
    observed_basis = basis[indices]
    observed_rank = int(torch.linalg.matrix_rank(observed_basis, tol=rank_tolerance))
    invariant_dimensions = basis.shape[1]
    unique = observed_rank == invariant_dimensions
    fit_relative = float("inf")
    leave_one_out_relative: list[float] = []
    ionic_relative = float("inf")
    ionic_mae = float("inf")
    completed: torch.Tensor | None = None
    if unique:
        coefficients = torch.linalg.lstsq(observed_basis, values).solution
        vector = basis @ coefficients
        fitted = observed_basis @ coefficients
        fit_relative = float(
            torch.linalg.vector_norm(fitted - values)
            / torch.linalg.vector_norm(values).clamp_min(torch.finfo(values.dtype).eps)
        )
        all_rows = torch.arange(indices.numel())
        for held in blocks:
            held_mask = torch.isin(indices, held)
            kept = all_rows[~held_mask]
            kept_basis = observed_basis[kept]
            if int(torch.linalg.matrix_rank(kept_basis, tol=rank_tolerance)) != invariant_dimensions:
                continue
            held_coefficients = torch.linalg.lstsq(kept_basis, values[kept]).solution
            held_prediction = observed_basis[held_mask] @ held_coefficients
            held_target = values[held_mask]
            leave_one_out_relative.append(
                float(
                    torch.linalg.vector_norm(held_prediction - held_target)
                    / torch.linalg.vector_norm(held_target).clamp_min(
                        torch.finfo(held_target.dtype).eps
                    )
                )
            )
        completed = vector_to_internal_tensor(vector, atoms).to(torch.float32)
        response = AtomCoordinateResponsePotential(
            optical_solve_policy="regularized", optical_regularization=1e-3
        )
        cell = torch.as_tensor(record["atoms"]["lattice_mat"], dtype=torch.float32)
        predicted_ionic = ionic_piezo_from_factors(
            response,
            payload["born_charges"],
            payload["force_constants"],
            completed,
            torch.linalg.det(cell).abs(),
            solve_policy="regularized",
        )
        target_ionic = piezo_voigt_to_cartesian(
            source_voigt_to_canonical(payload["ionic_piezo_source"])
        )
        ionic_relative = float(
            torch.linalg.vector_norm(predicted_ionic - target_ionic)
            / torch.linalg.vector_norm(target_ionic).clamp_min(0.05)
        )
        ionic_mae = float((predicted_ionic - target_ionic).abs().mean())
    redundant_validation_available = bool(leave_one_out_relative)
    cross_validation_max = (
        max(leave_one_out_relative) if leave_one_out_relative else None
    )
    accepted = bool(
        unique
        and maximum_mapping_error <= mapping_tolerance_angstrom
        and fit_relative <= fit_relative_tolerance
        and (
            not redundant_validation_available
            or cross_validation_max <= cross_validation_relative_tolerance
        )
        and ionic_relative <= ionic_relative_tolerance
    )
    audit = {
        "jid": str(record["JARVIS_ID"]),
        "atoms": atoms,
        "space_group_operations": len(operations),
        "maximum_mapping_error_angstrom": maximum_mapping_error,
        "invariant_dimensions": invariant_dimensions,
        "observed_scalar_components": int(indices.numel()),
        "observed_rank": observed_rank,
        "uniquely_determined": unique,
        "fit_relative_residual": fit_relative,
        "redundant_validation_blocks": len(leave_one_out_relative),
        "leave_one_block_out_relative_max": cross_validation_max,
        "ionic_closure_relative_error": ionic_relative,
        "ionic_closure_mae_c_per_m2": ionic_mae,
        "accepted": accepted,
    }
    return (completed if accepted else None), audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict symmetry completion of JARVIS strain-force labels")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--dfpt-dir", type=Path, default=Path("data/processed/jarvis_dfpt_v1"))
    parser.add_argument("--material-ids-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/jarvis_strain_completion_v2"))
    args = parser.parse_args()
    records = load_gmtnet_records(args.data_root)
    if args.material_ids_file is not None:
        parsed = json.loads(args.material_ids_file.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            parsed = parsed.get("material_ids")
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("Material-ID JSON must be a non-empty list or contain a material_ids list")
        selected = {str(value) for value in parsed}
        records = [record for record in records if str(record["JARVIS_ID"]) in selected]
    cache = JarvisDFPTCache(args.dfpt_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, accepted = [], 0
    for position, record in enumerate(records, start=1):
        jid = str(record["JARVIS_ID"])
        payload = cache.load(jid)
        if payload is None:
            continue
        try:
            completed, audit = complete_internal_strain(record, payload)
        except Exception as error:
            completed, audit = None, {"jid": jid, "accepted": False, "error": str(error)}
        rows.append(audit)
        if completed is not None:
            accepted += 1
            torch.save(
                {
                    "schema": COMPLETION_SCHEMA,
                    "jid": jid,
                    "internal_strain_full": completed,
                    "audit": audit,
                },
                args.output_dir / f"{jid}.pt",
            )
        print(f"[{position}/{len(records)}] {jid}: accepted={audit.get('accepted', False)}")
    manifest = {
        "schema": COMPLETION_SCHEMA,
        "requested": len(records),
        "audited": len(rows),
        "accepted": accepted,
        "acceptance_rate": accepted / max(len(rows), 1),
        "policy": (
            "unique space-group and acoustic-null reconstruction; strict atom mapping; "
            "redundant-block validation when available; ionic-response closure"
        ),
        "rows": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
