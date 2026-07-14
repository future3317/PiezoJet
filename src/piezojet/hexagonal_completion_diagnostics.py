"""Minimal reproducer for non-orthogonal strict-completion diagnostics.

This module deliberately leaves the strict-completion gate untouched.  It
separates three questions which can otherwise be conflated for hexagonal
materials: whether the invariant linear algebra recovers a synthetic tensor,
whether spglib/pymatgen fractional operations agree with their Cartesian
representations for the row-vector lattice convention, and whether the raw
printed DFPT blocks fail under a particular operation class.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .completion_forensics import _fit_and_pairwise
from .data import load_gmtnet_records
from .jarvis_dfpt import JarvisDFPTCache
from .strain_completion import _observed_system, _structure, cartesian_space_group_operations, invariant_basis


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-15))


def _close_mod_one(actual: np.ndarray, expected: np.ndarray, tolerance: float = 2e-8) -> bool:
    difference = actual - expected
    return bool(np.max(np.abs(difference - np.round(difference))) <= tolerance)


def _operation_class(rotation: np.ndarray) -> str:
    determinant = float(np.linalg.det(rotation))
    trace = float(np.trace(rotation))
    if determinant > 0.0:
        return "identity" if np.allclose(rotation, np.eye(3), atol=1e-8) else "proper_rotation"
    # An improper 3D orthogonal operation with a +1 eigenvalue is a mirror
    # (possibly combined with a lattice translation); otherwise it is a
    # rotoinversion/inversion branch.
    eigenvalues = np.linalg.eigvals(rotation)
    return "mirror" if np.min(np.abs(eigenvalues - 1.0)) < 1e-7 else "rotoinversion"


def fractional_cartesian_checks(record: dict[str, Any], symprec: float = 1e-5) -> dict[str, Any]:
    """Verify fractional/Cartesian operations using the code's row-vector cell.

    ``lattice`` contains lattice vectors as rows, hence ``x_row=f_row @ L``.
    For a fractional column-vector operation ``f'=W f``, the corresponding
    Cartesian column-vector rotation must be ``R=L.T @ W @ inv(L.T)``.
    """
    structure = _structure(record)
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    fractional_operations = analyzer.get_symmetry_operations(cartesian=False)
    cartesian_operations = analyzer.get_symmetry_operations(cartesian=True)
    if len(fractional_operations) != len(cartesian_operations):
        raise ValueError("Fractional and Cartesian symmetry-operation counts differ")
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    inverse_lattice = np.linalg.inv(lattice)
    rng = np.random.default_rng(17)
    fractional_vectors = rng.normal(size=(7, 3))
    metrics: list[dict[str, Any]] = []
    affine = []
    for index, (fractional, cartesian) in enumerate(zip(fractional_operations, cartesian_operations)):
        w = np.asarray(fractional.rotation_matrix, dtype=float)
        translation = np.asarray(fractional.translation_vector, dtype=float)
        rotation = np.asarray(cartesian.rotation_matrix, dtype=float)
        expected_rotation = lattice.T @ w @ np.linalg.inv(lattice.T)
        cartesian_vectors = fractional_vectors @ lattice
        via_fractional = (fractional_vectors @ w.T) @ lattice
        via_cartesian = cartesian_vectors @ rotation.T
        recovered_fractional = via_cartesian @ inverse_lattice
        metrics.append({
            "operation_index": index,
            "determinant": float(np.linalg.det(rotation)),
            "trace": float(np.trace(rotation)),
            "class": _operation_class(rotation),
            "cartesian_orthogonality_error": _relative_error(rotation.T @ rotation, np.eye(3)),
            "determinant_distance_to_pm_one": float(abs(abs(np.linalg.det(rotation)) - 1.0)),
            "fractional_to_cartesian_rotation_error": _relative_error(rotation, expected_rotation),
            "vector_round_trip_error": _relative_error(via_cartesian, via_fractional),
            "cartesian_back_to_fractional_error": _relative_error(recovered_fractional, fractional_vectors @ w.T),
        })
        affine.append((w, translation))
    # Test closure in fractional affine form.  Apply g1 then g2: W2 W1,
    # t2 + W2 t1.  This avoids falsely judging a screw/glide operation by its
    # Cartesian rotation alone.
    closure_failures = 0
    maximum_closure_error = 0.0
    for first_w, first_t in affine:
        for second_w, second_t in affine:
            expected_w = second_w @ first_w
            expected_t = second_t + second_w @ first_t
            matches = [
                np.allclose(w, expected_w, atol=2e-8, rtol=0.0) and _close_mod_one(t, expected_t)
                for w, t in affine
            ]
            if not any(matches):
                closure_failures += 1
                rotation_error = min(np.max(np.abs(w - expected_w)) for w, _ in affine)
                translation_error = min(np.max(np.abs((t - expected_t) - np.round(t - expected_t))) for _, t in affine)
                maximum_closure_error = max(maximum_closure_error, float(max(rotation_error, translation_error)))
    return {
        "operations": len(metrics),
        "maximum_cartesian_orthogonality_error": max(item["cartesian_orthogonality_error"] for item in metrics),
        "maximum_determinant_distance_to_pm_one": max(item["determinant_distance_to_pm_one"] for item in metrics),
        "maximum_fractional_to_cartesian_rotation_error": max(item["fractional_to_cartesian_rotation_error"] for item in metrics),
        "maximum_vector_round_trip_error": max(item["vector_round_trip_error"] for item in metrics),
        "maximum_cartesian_back_to_fractional_error": max(item["cartesian_back_to_fractional_error"] for item in metrics),
        "affine_group_closure_failures": closure_failures,
        "maximum_affine_closure_error": maximum_closure_error,
        "operation_metrics": metrics,
    }


def synthetic_recovery(record: dict[str, Any], payload: dict[str, Any], seed: int = 17) -> dict[str, Any]:
    """Recover a known invariant tensor through the real printed-block mask."""
    atoms = len(record["atoms"]["elements"])
    basis = invariant_basis(atoms, cartesian_space_group_operations(record))
    indices, _, _ = _observed_system(
        payload["internal_strain_tensors"], payload["internal_strain_ions"],
        payload["internal_strain_directions"], atoms,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coefficients = torch.randn(basis.shape[1], generator=generator, dtype=torch.float64)
    target = basis @ coefficients
    observed_basis = basis[indices]
    recovered_coefficients = torch.linalg.lstsq(observed_basis, target[indices]).solution
    recovered = basis @ recovered_coefficients
    cosine = torch.nn.functional.cosine_similarity(recovered, target, dim=0)
    return {
        "invariant_dimensions": int(basis.shape[1]),
        "observed_rank": int(torch.linalg.matrix_rank(observed_basis, tol=1e-7)),
        "relative_recovery_error": float(torch.linalg.vector_norm(recovered - target) / torch.linalg.vector_norm(target).clamp_min(1e-15)),
        "cosine": float(cosine),
        "observed_fit_relative_error": float(torch.linalg.vector_norm(observed_basis @ recovered_coefficients - target[indices]) / torch.linalg.vector_norm(target[indices]).clamp_min(1e-15)),
    }


def raw_operation_classes(record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Attach operation classes to directly comparable raw-block residuals."""
    analysis = _fit_and_pairwise(record, payload, symprec=1e-5)
    operations = cartesian_space_group_operations(record)
    classes: dict[str, list[float]] = defaultdict(list)
    rows = []
    for row in analysis["pairwise_operations"]:
        operation = operations[int(row["operation_index"])]
        rotation = operation.rotation.detach().cpu().numpy()
        annotated = dict(row)
        annotated.update({
            "determinant": float(np.linalg.det(rotation)),
            "trace": float(np.trace(rotation)),
            "class": _operation_class(rotation),
        })
        classes[annotated["class"]].append(float(annotated["relative_residual"]))
        rows.append(annotated)
    summary = {
        name: {
            "operations": len(values), "maximum_relative_residual": max(values),
            "median_relative_residual": float(np.median(values)),
        }
        for name, values in classes.items()
    }
    return {"fit_relative_residual": analysis["fit_relative_residual"], "operation_class_summary": summary, "operations": rows}


def _space_group(record: dict[str, Any]) -> tuple[int, str]:
    analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
    return int(analyzer.get_space_group_number()), str(analyzer.get_space_group_symbol())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--dfpt-dir", type=Path, default=Path("data/processed/jarvis_dfpt_v1"))
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-space-group", type=int, default=9)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    ids = [str(value) for value in cohort["material_ids"]]
    records = {str(item["JARVIS_ID"]): item for item in load_gmtnet_records(args.data_root)}
    cache = JarvisDFPTCache(args.dfpt_dir)
    selected: list[tuple[str, int, str]] = []
    counts: dict[int, int] = defaultdict(int)
    # No.187 is the reproducer.  No.186/189 are non-orthogonal hexagonal
    # controls already present in the challenge cohort; the first cubic
    # member provides a convention sanity check without becoming the control.
    wanted = {187, 186, 189}
    for jid in ids:
        number, symbol = _space_group(records[jid])
        if number in wanted and counts[number] < args.max_per_space_group:
            selected.append((jid, number, symbol))
            counts[number] += 1
    results = []
    for jid, number, symbol in selected:
        payload = cache.load(jid)
        if payload is None:
            continue
        record = records[jid]
        result = {
            "jid": jid, "space_group_number": number, "space_group_symbol": symbol,
            "atoms": len(record["atoms"]["elements"]),
            "synthetic_recovery": synthetic_recovery(record, payload),
            "coordinate_checks": fractional_cartesian_checks(record),
            "raw_operation_classes": raw_operation_classes(record, payload),
        }
        results.append(result)
        print(f"{jid}: SG {number}, synthetic cosine={result['synthetic_recovery']['cosine']:.8f}")
    summary = {
        "materials": len(results),
        "by_space_group": {
            str(number): {
                "materials": sum(item["space_group_number"] == number for item in results),
                "worst_synthetic_relative_error": max((item["synthetic_recovery"]["relative_recovery_error"] for item in results if item["space_group_number"] == number), default=None),
                "worst_coordinate_round_trip_error": max((item["coordinate_checks"]["maximum_vector_round_trip_error"] for item in results if item["space_group_number"] == number), default=None),
                "maximum_raw_pairwise_residual": max((max((row["relative_residual"] for row in item["raw_operation_classes"]["operations"]), default=0.0) for item in results if item["space_group_number"] == number), default=None),
            }
            for number in sorted({item["space_group_number"] for item in results})
        },
    }
    report = {
        "schema": 1,
        "scope": "diagnostic only; strict completion thresholds and source labels are unchanged",
        "row_vector_lattice_convention": "x_row = f_row @ lattice",
        "summary": summary,
        "materials_detail": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hexagonal_diagnostics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Non-orthogonal strict-completion diagnostics", "",
        "This audit does not relax strict completion or assert that any source record is wrong.", "",
        "## Results", "",
    ]
    for number, item in summary["by_space_group"].items():
        lines.append(
            f"- SG {number}: n={item['materials']}; worst synthetic recovery error={item['worst_synthetic_relative_error']:.3e}; "
            f"worst coordinate round-trip error={item['worst_coordinate_round_trip_error']:.3e}; "
            f"maximum raw pairwise residual={item['maximum_raw_pairwise_residual']:.6g}."
        )
    lines.extend(["", "Detailed per-operation determinant, class, coordinate checks, and raw residuals are in `hexagonal_diagnostics.json`."])
    (args.output_dir / "hexagonal_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "hexagonal_diagnostics.md")


if __name__ == "__main__":
    main()
