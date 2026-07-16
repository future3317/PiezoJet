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
from .evaluate_dfpt import ionic_piezo_from_factors, optical_eigensystem
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


def observed_identification_metrics(
    observed_basis: torch.Tensor,
    *,
    rank_tolerance: float,
) -> dict[str, float | int | None]:
    """Expose numerical identification quality, not just a binary rank gate.

    The invariant basis is orthonormal, so the singular values of ``M B``
    directly quantify how errors in printed blocks amplify in invariant
    coordinates.  No arbitrary conditioning cutoff is silently imposed here:
    consumers may register one prospectively, while the raw quantities remain
    available for every accepted and rejected record.
    """
    singular = torch.linalg.svdvals(observed_basis)
    rank = int((singular > rank_tolerance).sum())
    dimensions = int(observed_basis.shape[1])
    if dimensions == 0:
        # The symmetry+ASR invariant space is {0}. Its only tensor is uniquely
        # identified without a nonzero singular spectrum; the inverse map has
        # zero norm and condition number one by the empty-identity convention.
        return {
            "identification_rank": 0,
            "sigma_min_mb": None,
            "sigma_max_mb": None,
            "condition_number_mb": 1.0,
            "pseudoinverse_operator_norm_mb": 0.0,
        }
    if rank < dimensions or singular.numel() < dimensions:
        return {
            "identification_rank": rank,
            "sigma_min_mb": None,
            "sigma_max_mb": float(singular.max()) if singular.numel() else None,
            "condition_number_mb": None,
            "pseudoinverse_operator_norm_mb": None,
        }
    sigma_min = singular[dimensions - 1]
    sigma_max = singular[0]
    return {
        "identification_rank": rank,
        "sigma_min_mb": float(sigma_min),
        "sigma_max_mb": float(sigma_max),
        "condition_number_mb": float(sigma_max / sigma_min),
        "pseudoinverse_operator_norm_mb": float(sigma_min.reciprocal()),
    }


def printed_block_rounding_bootstrap(
    *,
    basis: torch.Tensor,
    observed_basis: torch.Tensor,
    values: torch.Tensor,
    halfwidths: torch.Tensor | None,
    samples: int,
    seed: int,
) -> dict[str, float | int | str | None]:
    """Propagate OUTCAR's displayed rounding interval through completion.

    This is not an inferred DFPT error bar: each printed scalar is sampled
    uniformly in its own half-last-digit interval.  It therefore gives a
    reproducible *rounding sensitivity* only when raw schema-4 payloads retain
    those intervals.  Historical caches honestly report that this information
    is unavailable.
    """
    if samples < 0:
        raise ValueError("bootstrap samples must be non-negative")
    if samples == 0:
        return {"status": "not_requested", "samples": 0}
    if halfwidths is None:
        return {
            "status": "unavailable_missing_outcar_rounding_intervals",
            "samples": samples,
        }
    if halfwidths.shape != values.shape:
        raise ValueError("OUTCAR rounding halfwidths must align with observed scalar values")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perturbation = (2.0 * torch.rand(values.numel(), samples, generator=generator, dtype=values.dtype) - 1.0)
    sampled_values = values[:, None] + halfwidths[:, None] * perturbation
    sampled_coefficients = torch.linalg.lstsq(observed_basis, sampled_values).solution
    sampled_vectors = basis @ sampled_coefficients
    reference = basis @ torch.linalg.lstsq(observed_basis, values).solution
    relative = torch.linalg.vector_norm(sampled_vectors - reference[:, None], dim=0) / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    return {
        "status": "outcar_rounding_uniform",
        "samples": samples,
        "seed": seed,
        "maximum_input_halfwidth": float(halfwidths.max()),
        "lambda_relative_std": float(relative.std(unbiased=False)),
        "lambda_relative_p95": float(torch.quantile(relative, 0.95)),
    }


def complete_internal_strain(
    record: dict[str, Any],
    payload: dict[str, Any],
    symprec: float = 1e-5,
    mapping_tolerance_angstrom: float = 2e-5,
    rank_tolerance: float = 1e-7,
    fit_relative_tolerance: float = 5e-3,
    cross_validation_relative_tolerance: float = 5e-2,
    ionic_relative_tolerance: float = 5e-2,
    optical_stability_cutoff: float = 1e-4,
    max_condition_number: float | None = None,
    rounding_bootstrap_samples: int = 0,
    rounding_bootstrap_seed: int = 0,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Complete one tensor only after uniqueness and closure gates pass."""
    atoms = len(record["atoms"]["elements"])
    parse_audit = payload.get("internal_strain_parse_audit")
    source_block_parse_complete = (
        bool(parse_audit.get("complete_observed_block_parse", False))
        if isinstance(parse_audit, dict)
        else int(payload["internal_strain_tensors"].shape[0]) > 0
    )
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
    identification = observed_identification_metrics(
        observed_basis, rank_tolerance=rank_tolerance
    )
    observed_rank = int(identification["identification_rank"])
    invariant_dimensions = basis.shape[1]
    unique = observed_rank == invariant_dimensions
    fit_relative = float("inf")
    leave_one_out_relative: list[float] = []
    ionic_relative = float("inf")
    ionic_mae = float("inf")
    stable_exact_ionic_relative: float | None = None
    stable_exact_ionic_mae: float | None = None
    stability_stratum: str | None = None
    minimum_optical_eigenvalue: float | None = None
    regularized_closure_by_delta: dict[str, float] = {}
    rounding_bootstrap: dict[str, Any] = {"status": "not_requested", "samples": 0}
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
            optical_solve_policy="regularized", optical_regularization=1e-3,
            optical_stability_cutoff=optical_stability_cutoff,
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
        # All stable structures must additionally close under the physical
        # stationary inverse.  The regularized signed operator remains only a
        # formal diagnostic for soft/unstable spectra and is never relabelled
        # as a static equilibrium response.
        optical_values, _ = optical_eigensystem(payload["force_constants"])
        minimum_optical_eigenvalue = (
            float(optical_values.min()) if optical_values.numel() else float("inf")
        )
        if optical_values.numel() == 0 or minimum_optical_eigenvalue > optical_stability_cutoff:
            stability_stratum = "stable_exact"
            exact_ionic = ionic_piezo_from_factors(
                response,
                payload["born_charges"],
                payload["force_constants"],
                completed,
                torch.linalg.det(cell).abs(),
                solve_policy="exact",
            )
            stable_exact_ionic_relative = float(
                torch.linalg.vector_norm(exact_ionic - target_ionic)
                / torch.linalg.vector_norm(target_ionic).clamp_min(0.05)
            )
            stable_exact_ionic_mae = float((exact_ionic - target_ionic).abs().mean())
        elif minimum_optical_eigenvalue > 0.0:
            stability_stratum = "soft_positive_signed_diagnostic"
        else:
            stability_stratum = "unstable_signed_diagnostic"
        for delta in (1e-4, 1e-3, 1e-2):
            delta_prediction = ionic_piezo_from_factors(
                response,
                payload["born_charges"],
                payload["force_constants"],
                completed,
                torch.linalg.det(cell).abs(),
                solve_policy="regularized",
                regularization=delta,
            )
            regularized_closure_by_delta[f"{delta:.0e}"] = float(
                torch.linalg.vector_norm(delta_prediction - target_ionic)
                / torch.linalg.vector_norm(target_ionic).clamp_min(0.05)
            )
        halfwidths = None
        if "internal_strain_rounding_halfwidth" in payload:
            _, halfwidths, _ = _observed_system(
                payload["internal_strain_rounding_halfwidth"],
                payload["internal_strain_ions"],
                payload["internal_strain_directions"],
                atoms,
            )
        rounding_bootstrap = printed_block_rounding_bootstrap(
            basis=basis,
            observed_basis=observed_basis,
            values=values,
            halfwidths=halfwidths,
            samples=rounding_bootstrap_samples,
            seed=rounding_bootstrap_seed,
        )
    redundant_validation_available = bool(leave_one_out_relative)
    cross_validation_max = (
        max(leave_one_out_relative) if leave_one_out_relative else None
    )
    accepted = bool(
        source_block_parse_complete
        and unique
        and maximum_mapping_error <= mapping_tolerance_angstrom
        and fit_relative <= fit_relative_tolerance
        and (
            not redundant_validation_available
            or cross_validation_max <= cross_validation_relative_tolerance
        )
        and ionic_relative <= ionic_relative_tolerance
        and (
            stable_exact_ionic_relative is None
            or stable_exact_ionic_relative <= ionic_relative_tolerance
        )
        and (
            max_condition_number is None
            or (
                identification["condition_number_mb"] is not None
                and float(identification["condition_number_mb"]) <= max_condition_number
            )
        )
    )
    audit = {
        "jid": str(record["JARVIS_ID"]),
        "atoms": atoms,
        "space_group_operations": len(operations),
        "maximum_mapping_error_angstrom": maximum_mapping_error,
        "invariant_dimensions": invariant_dimensions,
        "observed_scalar_components": int(indices.numel()),
        "source_internal_strain_block_parse_complete": source_block_parse_complete,
        "source_internal_strain_parse_audit": parse_audit,
        "observed_rank": observed_rank,
        **identification,
        "uniquely_determined": unique,
        "fit_relative_residual": fit_relative,
        "redundant_validation_blocks": len(leave_one_out_relative),
        "leave_one_block_out_relative_max": cross_validation_max,
        "ionic_closure_relative_error": ionic_relative,
        "ionic_closure_mae_c_per_m2": ionic_mae,
        "minimum_optical_eigenvalue_eV_per_A2": minimum_optical_eigenvalue,
        "stability_stratum": stability_stratum,
        "stable_exact_ionic_closure_relative_error": stable_exact_ionic_relative,
        "stable_exact_ionic_closure_mae_c_per_m2": stable_exact_ionic_mae,
        "regularized_ionic_closure_relative_by_delta": regularized_closure_by_delta,
        "max_condition_number_gate": max_condition_number,
        "rounding_bootstrap": rounding_bootstrap,
        "accepted": accepted,
    }
    return (completed if accepted else None), audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict symmetry completion of JARVIS strain-force labels")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dfpt-dir", type=Path, required=True)
    parser.add_argument("--material-ids-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-condition-number",
        type=float,
        help="Optional prospective M@B conditioning gate; omitted preserves the audited rank-only criterion while recording conditioning.",
    )
    parser.add_argument(
        "--rounding-bootstrap-samples",
        type=int,
        default=0,
        help="Propagate schema-4 OUTCAR displayed rounding intervals; 0 records no bootstrap rather than inventing source noise.",
    )
    parser.add_argument("--rounding-bootstrap-seed", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--finalize-shards",
        action="store_true",
        help="Merge completed shard manifests and verify the accepted payload set.",
    )
    args = parser.parse_args()
    if args.max_condition_number is not None and args.max_condition_number <= 1.0:
        raise ValueError("--max-condition-number must exceed one")
    if args.rounding_bootstrap_samples < 0:
        raise ValueError("--rounding-bootstrap-samples must be non-negative")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")
    records = load_gmtnet_records(args.data_root)
    if args.material_ids_file is not None:
        parsed = json.loads(args.material_ids_file.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            parsed = parsed.get("material_ids")
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("Material-ID JSON must be a non-empty list or contain a material_ids list")
        selected = {str(value) for value in parsed}
        records = [record for record in records if str(record["JARVIS_ID"]) in selected]
    source_requested = len(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_shards:
        if args.shard_count == 1:
            raise ValueError("--finalize-shards requires --shard-count greater than one")
        manifests = []
        for shard_index in range(args.shard_count):
            path = args.output_dir / f"manifest.shard_{shard_index}_of_{args.shard_count}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completion shard manifest: {path}")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("schema") != COMPLETION_SCHEMA or manifest.get("shard") != {
                "index": shard_index,
                "count": args.shard_count,
                "source_requested": source_requested,
            }:
                raise ValueError(f"Incompatible completion shard manifest: {path}")
            manifests.append(manifest)
        rows_by_jid = {}
        for manifest in manifests:
            for row in manifest["rows"]:
                jid = str(row["jid"])
                if jid in rows_by_jid:
                    raise ValueError(f"Duplicate material across completion shards: {jid}")
                rows_by_jid[jid] = row
        ordered_rows = [
            rows_by_jid[jid]
            for record in records
            if (jid := str(record["JARVIS_ID"])) in rows_by_jid
        ]
        accepted_ids = {
            str(row["jid"]) for row in ordered_rows if bool(row.get("accepted", False))
        }
        actual_ids = {path.stem for path in args.output_dir.glob("JVASP-*.pt")}
        if actual_ids != accepted_ids:
            missing = sorted(accepted_ids - actual_ids)
            extra = sorted(actual_ids - accepted_ids)
            raise ValueError(
                "Completion payload set does not match merged accepted rows: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        template = manifests[0]
        merged = {
            "schema": COMPLETION_SCHEMA,
            "requested": source_requested,
            "audited": len(ordered_rows),
            "accepted": len(accepted_ids),
            "acceptance_rate": len(accepted_ids) / max(len(ordered_rows), 1),
            "policy": template["policy"],
            "conditioning_gate": template["conditioning_gate"],
            "rounding_bootstrap": template["rounding_bootstrap"],
            "rows": ordered_rows,
            "finalized_from_shards": args.shard_count,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(merged, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"finalized audited={len(ordered_rows)} accepted={len(accepted_ids)} "
            f"from {args.shard_count} shards"
        )
        return
    if args.shard_count > 1:
        records = [
            record
            for position, record in enumerate(records)
            if position % args.shard_count == args.shard_index
        ]
    cache = JarvisDFPTCache(args.dfpt_dir)
    rows, accepted = [], 0
    for position, record in enumerate(records, start=1):
        jid = str(record["JARVIS_ID"])
        payload = cache.load(jid)
        if payload is None:
            continue
        try:
            completed, audit = complete_internal_strain(
                record,
                payload,
                max_condition_number=args.max_condition_number,
                rounding_bootstrap_samples=args.rounding_bootstrap_samples,
                rounding_bootstrap_seed=args.rounding_bootstrap_seed,
            )
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
            "redundant-block validation when available; regularized ionic-response closure; "
            "stable structures additionally require exact-inverse ionic closure; "
            "all records persist M@B conditioning and signed-operator delta sensitivity"
        ),
        "conditioning_gate": args.max_condition_number,
        "rounding_bootstrap": {
            "samples": args.rounding_bootstrap_samples,
            "seed": args.rounding_bootstrap_seed,
            "source": "OUTCAR displayed final-digit intervals when schema-4 cache payloads provide them",
        },
        "rows": rows,
    }
    if args.shard_count > 1:
        manifest["shard"] = {
            "index": args.shard_index,
            "count": args.shard_count,
            "source_requested": source_requested,
        }
        manifest_path = args.output_dir / (
            f"manifest.shard_{args.shard_index}_of_{args.shard_count}.json"
        )
    else:
        manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
