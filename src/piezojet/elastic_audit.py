"""Strict source audit and target preparation for JARVIS/GMTNet elasticity.

Raw source tensors remain in kbar.  Only records that pass point-group,
stiffness/compliance, and source-derived-modulus checks receive a GPa target.
The result is intentionally an availability-masked auxiliary cohort, never a
filtered replacement for the full piezoelectric macro corpus.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import torch
from jarvis.db.figshare import data as figshare_data
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .elastic_dielectric_ops import (
    elastic_cartesian_to_voigt,
    elastic_kbar_to_gpa,
    elastic_voigt_to_cartesian,
    rotate_elastic,
    voigt_bulk_shear_moduli,
    voigt_compliance_from_stiffness,
)


def _structure(record: dict[str, Any]) -> Structure:
    atoms = record["atoms"]
    return Structure(atoms["lattice_mat"], atoms["elements"], atoms["coords"], coords_are_cartesian=False)


def _point_group(record: dict[str, Any]) -> tuple[SpacegroupAnalyzer, list[torch.Tensor]]:
    analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
    operations = [torch.tensor(operation.rotation_matrix, dtype=torch.float64) for operation in analyzer.get_point_group_operations(cartesian=True)]
    if not operations:
        raise ValueError("No Cartesian point-group operations")
    return analyzer, operations


def _reynolds_residual(stiffness: torch.Tensor, operations: list[torch.Tensor]) -> float:
    projected = torch.stack([rotate_elastic(stiffness, operation) for operation in operations]).mean(dim=0)
    return float(torch.linalg.vector_norm(stiffness - projected) / torch.linalg.vector_norm(stiffness).clamp_min(1e-30))


def _source_modulus_error(moduli: dict[str, torch.Tensor], source: dict[str, Any]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for source_key, calculated_key, output_key in (
        ("bulk_modulus_kv", "bulk_voigt_gpa", "bulk_voigt_source_absolute_error_gpa"),
        ("shear_modulus_gv", "shear_voigt_gpa", "shear_voigt_source_absolute_error_gpa"),
    ):
        source_value = source.get(source_key)
        output[output_key] = None if source_value is None else abs(float(moduli[calculated_key]) - float(source_value))
    return output


def _standard_constraint_residual(voigt: torch.Tensor, crystal_system: str, point_group: str) -> dict[str, float | str] | None:
    """Explicit conventional-axis zero/equivalence checks for high-symmetry reps."""
    c = voigt
    if crystal_system == "cubic":
        checks = [
            c[0, 0] - c[1, 1], c[0, 0] - c[2, 2], c[0, 1] - c[0, 2], c[0, 1] - c[1, 2],
            c[3, 3] - c[4, 4], c[3, 3] - c[5, 5], c[0, 3], c[0, 4], c[0, 5], c[1, 3], c[1, 4], c[1, 5],
            c[2, 3], c[2, 4], c[2, 5], c[3, 4], c[3, 5], c[4, 5],
        ]
        return {"convention": "cubic: C11=C22=C33; C12=C13=C23; C44=C55=C66; listed couplings zero", "maximum_constraint_abs_gpa": float(torch.stack(checks).abs().max())}
    if crystal_system == "hexagonal" and point_group in {"6/mmm", "6mm", "-6m2", "6/m", "622", "-6"}:
        checks = [
            c[0, 0] - c[1, 1], c[0, 2] - c[1, 2], c[3, 3] - c[4, 4],
            c[5, 5] - 0.5 * (c[0, 0] - c[0, 1]), c[0, 3], c[0, 4], c[1, 3], c[1, 4], c[2, 3], c[2, 4], c[3, 5], c[4, 5],
        ]
        return {"convention": "hexagonal conventional axis: C11=C22, C13=C23, C44=C55, C66=(C11-C12)/2; listed couplings zero", "maximum_constraint_abs_gpa": float(torch.stack(checks).abs().max())}
    if crystal_system == "tetragonal" and point_group == "4/mmm":
        checks = [
            c[0, 0] - c[1, 1], c[0, 2] - c[1, 2], c[3, 3] - c[4, 4],
            c[0, 3], c[0, 4], c[0, 5], c[1, 3], c[1, 4], c[1, 5], c[2, 3], c[2, 4], c[2, 5], c[3, 5], c[4, 5],
        ]
        return {"convention": "tetragonal 4/mmm: C11=C22, C13=C23, C44=C55; listed couplings zero", "maximum_constraint_abs_gpa": float(torch.stack(checks).abs().max())}
    return None


def audit_elastic(
    *, data_root: Path, dft3d_cache_dir: Path, output_dir: Path,
    symmetry_tolerance: float = 1e-5, modulus_absolute_tolerance_gpa: float = 0.1,
    modulus_relative_tolerance: float = 0.01,
) -> dict[str, Any]:
    root = data_root / "data"
    with (root / "jarvis_diele_piezo.pkl").open("rb") as handle:
        piezo_ids = {str(row["JARVIS_ID"]) for row in pickle.load(handle)}
    with (root / "jarvis_elastic.pkl").open("rb") as handle:
        elastic_rows = pickle.load(handle)
    dft = {
        str(row["jid"]): row
        for row in figshare_data("dft_3d", store_dir=str(dft3d_cache_dir))
    }
    rows: list[dict[str, Any]] = []
    accepted: dict[str, torch.Tensor] = {}
    representative_candidates: dict[str, list[dict[str, Any]]] = {"cubic": [], "hexagonal": [], "tetragonal": []}
    for source in elastic_rows:
        jid = str(source["JARVIS_ID"])
        if jid not in piezo_ids:
            continue
        row: dict[str, Any] = {"jid": jid, "accepted": False, "rejection_reasons": []}
        try:
            raw_kbar = torch.tensor(source["elastic_total_kbar"], dtype=torch.float64)
            if raw_kbar.shape != (6, 6) or not torch.isfinite(raw_kbar).all():
                raise ValueError("invalid/nonfinite 6x6 elastic_total_kbar")
            if not torch.allclose(raw_kbar, raw_kbar.T, atol=1e-8, rtol=1e-8):
                raise ValueError("source Voigt stiffness is not major-symmetric")
            raw_gpa = elastic_kbar_to_gpa(raw_kbar)
            raw_cartesian = elastic_voigt_to_cartesian(raw_gpa)
            restored = elastic_cartesian_to_voigt(raw_cartesian)
            row["raw_voigt_cartesian_roundtrip_max_abs_gpa"] = float((restored - raw_gpa).abs().max())
            analyzer, operations = _point_group(source)
            cartesian = torch.stack([rotate_elastic(raw_cartesian, operation) for operation in operations]).mean(dim=0)
            stiffness_gpa = elastic_cartesian_to_voigt(cartesian)
            row.update({
                "crystal_system": str(analyzer.get_crystal_system()), "point_group": str(analyzer.get_point_group_symbol()),
                "space_group_number": int(analyzer.get_space_group_number()), "point_group_operations": len(operations),
                "raw_point_group_reynolds_relative_residual": _reynolds_residual(raw_cartesian, operations),
                "target_point_group_reynolds_relative_residual": _reynolds_residual(cartesian, operations),
            })
            compliance = voigt_compliance_from_stiffness(stiffness_gpa)
            row["stiffness_compliance_inverse_max_abs"] = float((stiffness_gpa @ compliance - torch.eye(6, dtype=torch.float64)).abs().max())
            moduli = voigt_bulk_shear_moduli(stiffness_gpa)
            row.update({name: float(value) for name, value in moduli.items()})
            modulus_error = _source_modulus_error(moduli, dft.get(jid, {}))
            row.update(modulus_error)
            for name, source_key in (("bulk_voigt_source_absolute_error_gpa", "bulk_modulus_kv"), ("shear_voigt_source_absolute_error_gpa", "shear_modulus_gv")):
                error = row[name]
                source_value = dft.get(jid, {}).get(source_key)
                if error is None or error > max(modulus_absolute_tolerance_gpa, modulus_relative_tolerance * abs(float(source_value))):
                    row["rejection_reasons"].append(f"{name} exceeds declared source-rounding tolerance")
            if row["target_point_group_reynolds_relative_residual"] > symmetry_tolerance:
                row["rejection_reasons"].append("Reynolds-projected target violates point-group tolerance")
            if row["raw_voigt_cartesian_roundtrip_max_abs_gpa"] > 1e-10:
                row["rejection_reasons"].append("Voigt/Cartesian round trip failed")
            if row["stiffness_compliance_inverse_max_abs"] > 1e-8:
                row["rejection_reasons"].append("stiffness/compliance inverse check failed")
            row["accepted"] = not row["rejection_reasons"]
            if row["accepted"]:
                accepted[jid] = stiffness_gpa.to(torch.float32)
                if row["crystal_system"] in representative_candidates:
                    constraint = _standard_constraint_residual(stiffness_gpa, row["crystal_system"], row["point_group"])
                    if constraint is not None:
                        row["standard_constraint"] = constraint
                        representative_candidates[row["crystal_system"]].append(row)
        except Exception as error:
            row["rejection_reasons"].append(str(error))
        rows.append(row)
    representatives = {
        system: min(values, key=lambda value: value["raw_point_group_reynolds_relative_residual"])
        for system, values in representative_candidates.items() if values
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": 1, "unit": "GPa", "source_field": "elastic_total_kbar", "targets": accepted}, output_dir / "accepted_targets_gpa.pt")
    summary = {
        "schema": 1,
        "source": str(root / "jarvis_elastic.pkl"), "source_field": "elastic_total_kbar", "source_unit": "kbar", "target_unit": "GPa",
        "unit_conversion": "GPa = kbar / 10 exactly once at ingestion",
        "target_construction": "full Cartesian point-group Reynolds projection of the raw GPa stiffness; raw source and projection residual remain in audit rows",
        "input_piezo_elastic_intersection": len(rows),
        "accepted": len(accepted), "rejected": len(rows) - len(accepted),
        "acceptance_policy": {"point_group_reynolds_relative_tolerance": symmetry_tolerance, "source_modulus_absolute_tolerance_gpa": modulus_absolute_tolerance_gpa, "source_modulus_relative_tolerance": modulus_relative_tolerance},
        "training_policy": "Accepted targets are availability-masked elasticity auxiliaries; all macro piezo records remain in total-tensor training.",
        "representative_symmetry_checks": representatives,
        "rows": rows,
    }
    (output_dir / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dft3d-cache-dir",
        type=Path,
        required=True,
        help="Dedicated jarvis-tools dft_3d cache; never reuse the raw_files cache directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--symmetry-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    result = audit_elastic(
        data_root=args.data_root,
        dft3d_cache_dir=args.dft3d_cache_dir,
        output_dir=args.output_dir,
        symmetry_tolerance=args.symmetry_tolerance,
    )
    print(json.dumps({key: result[key] for key in ("input_piezo_elastic_intersection", "accepted", "rejected", "representative_symmetry_checks")}, indent=2))


if __name__ == "__main__":
    main()
