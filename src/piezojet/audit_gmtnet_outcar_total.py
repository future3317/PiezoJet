"""Audit consistency between GMTNet total labels and same-ID OUTCAR totals.

The all-material GMTNet objective consumes a Reynolds-projected total tensor,
whereas the DFPT branch targets are direct OUTCAR tensors.  This module makes
that distinction explicit: it reports both raw source agreement and the
agreement of the *actual training total* with the OUTCAR total.  It does not
rewrite either source.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .data import SymmetryTargetCache, _raw_cartesian_target, load_gmtnet_records, source_voigt_to_canonical
from .jarvis_dfpt import JarvisDFPTCache
from .tensor_ops import piezo_voigt_to_cartesian


DEFAULT_ABSOLUTE_TOLERANCE_C_PER_M2 = 0.05
DEFAULT_RELATIVE_TOLERANCE = 0.05
_RELATIVE_FLOOR_C_PER_M2 = DEFAULT_ABSOLUTE_TOLERANCE_C_PER_M2 * (18.0 ** 0.5)


def total_labels_are_consistent(
    absolute_difference_c_per_m2: float,
    relative_difference: float,
    *,
    absolute_tolerance_c_per_m2: float = DEFAULT_ABSOLUTE_TOLERANCE_C_PER_M2,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> bool:
    """Return the registered double-threshold consistency decision.

    Requiring both thresholds avoids classifying harmless rounding differences
    on small tensors as a source conflict, while catching a materially different
    projected training target.
    """
    return not (
        absolute_difference_c_per_m2 > absolute_tolerance_c_per_m2
        and relative_difference > relative_tolerance
    )


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p95": float(torch.quantile(tensor, 0.95)),
        "maximum": float(tensor.max()),
    }


def _split_lookup(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = payload.get("splits", payload)
    lookup: dict[str, str] = {}
    for name, material_ids in splits.items():
        if isinstance(material_ids, list):
            for material_id in material_ids:
                lookup[str(material_id)] = str(name)
    return lookup


def audit_gmtnet_outcar_total_consistency(
    records: list[dict[str, Any]],
    *,
    dfpt_dir: str | Path,
    processed_dir: str | Path,
    global_splits_file: str | Path | None = None,
    absolute_tolerance_c_per_m2: float = DEFAULT_ABSOLUTE_TOLERANCE_C_PER_M2,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare raw and actual-training GMTNet totals against OUTCAR totals."""
    if absolute_tolerance_c_per_m2 < 0 or relative_tolerance < 0:
        raise ValueError("Consistency tolerances must be non-negative")
    cache = JarvisDFPTCache(dfpt_dir)
    symmetry_targets = SymmetryTargetCache(processed_dir)
    split_lookup = _split_lookup(global_splits_file)
    rows: list[dict[str, Any]] = []
    for record in records:
        material_id = str(record["JARVIS_ID"])
        payload = cache.load(material_id)
        if payload is None:
            continue
        raw_gmtnet = _raw_cartesian_target(record).to(dtype=torch.float64)
        training_gmtnet = symmetry_targets.get(record).to(dtype=torch.float64)
        outcar_total = piezo_voigt_to_cartesian(
            source_voigt_to_canonical(payload["total_piezo_source"])
        ).to(dtype=torch.float64)
        raw_difference = torch.linalg.vector_norm(raw_gmtnet - outcar_total)
        training_difference = torch.linalg.vector_norm(training_gmtnet - outcar_total)
        outcar_norm = torch.linalg.vector_norm(outcar_total)
        scale = outcar_norm.clamp_min(_RELATIVE_FLOOR_C_PER_M2)
        raw_relative = raw_difference / scale
        training_relative = training_difference / scale
        rows.append(
            {
                "material_id": material_id,
                "global_split": split_lookup.get(material_id, "not_in_global_split"),
                "dfpt_schema": int(payload.get("schema", -1)),
                "raw_gmtnet_outcar_frobenius_c_per_m2": float(raw_difference),
                "raw_gmtnet_outcar_relative": float(raw_relative),
                "training_gmtnet_outcar_frobenius_c_per_m2": float(training_difference),
                "training_gmtnet_outcar_relative": float(training_relative),
                "outcar_total_frobenius_c_per_m2": float(outcar_norm),
                "training_total_consistent": total_labels_are_consistent(
                    float(training_difference),
                    float(training_relative),
                    absolute_tolerance_c_per_m2=absolute_tolerance_c_per_m2,
                    relative_tolerance=relative_tolerance,
                ),
            }
        )
    inconsistent = [row for row in rows if not row["training_total_consistent"]]
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({str(row["global_split"]) for row in rows}):
        members = [row for row in rows if row["global_split"] == split]
        by_split[split] = {
            "materials": len(members),
            "inconsistent_training_totals": sum(not bool(row["training_total_consistent"]) for row in members),
            "training_difference_c_per_m2": _summary(
                float(row["training_gmtnet_outcar_frobenius_c_per_m2"]) for row in members
            ),
        }
    report = {
        "schema": 1,
        "purpose": "GMTNet total versus same-ID OUTCAR total consistency audit",
        "dfpt_directory": str(dfpt_dir),
        "raw_label_policy": "GMTNet raw total and OUTCAR total are compared after the same source-Voigt to canonical-Cartesian conversion.",
        "training_label_policy": "GMTNet total is Reynolds-projected; OUTCAR total remains a source tensor for branch supervision.",
        "relative_denominator_floor_c_per_m2": _RELATIVE_FLOOR_C_PER_M2,
        "consistency_gate": {
            "absolute_tolerance_c_per_m2": absolute_tolerance_c_per_m2,
            "relative_tolerance": relative_tolerance,
            "rule": "mask DFPT macro ionic/electronic/branch response supervision only if both tolerances are exceeded; retain GMTNet total supervision",
        },
        "materials_with_same_id_dfpt": len(rows),
        "raw_gmtnet_minus_outcar": {
            "frobenius_c_per_m2": _summary(
                float(row["raw_gmtnet_outcar_frobenius_c_per_m2"]) for row in rows
            ),
            "relative": _summary(float(row["raw_gmtnet_outcar_relative"]) for row in rows),
        },
        "training_gmtnet_minus_outcar": {
            "frobenius_c_per_m2": _summary(
                float(row["training_gmtnet_outcar_frobenius_c_per_m2"]) for row in rows
            ),
            "relative": _summary(float(row["training_gmtnet_outcar_relative"]) for row in rows),
            "inconsistent_materials": len(inconsistent),
            "inconsistent_ids": [str(row["material_id"]) for row in inconsistent],
        },
        "global_split_breakdown": by_split,
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dfpt-dir", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--global-splits-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE_C_PER_M2)
    parser.add_argument("--relative-tolerance", type=float, default=DEFAULT_RELATIVE_TOLERANCE)
    args = parser.parse_args()
    report, rows = audit_gmtnet_outcar_total_consistency(
        load_gmtnet_records(args.data_root),
        dfpt_dir=args.dfpt_dir,
        processed_dir=args.processed_dir,
        global_splits_file=args.global_splits_file,
        absolute_tolerance_c_per_m2=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_material.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["material_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
