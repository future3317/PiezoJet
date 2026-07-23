"""Read-only stratified audit for electrostatic-generator checkpoints.

The evaluator consumes the per-material metrics already persisted in an
immutable development checkpoint.  It never constructs a model, reads a
frozen panel, or changes selection.  JARVIS dft_3d metadata is used only for
descriptive strata (formula, atom count, space group/crystal system); it is
not a response-label source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def _summary(rows: list[dict[str, Any]], *, task: str) -> dict[str, Any]:
    if task == "electronic":
        rel = "stabilized_relative_frobenius_error"
        target = "target_norm_c_per_m2"
        pred = "prediction_norm_c_per_m2"
        amp = "stabilized_amplitude_ratio"
        cosine = "cosine"
    elif task == "born":
        rel = "stabilized_relative_frobenius_error"
        target = "target_norm_e"
        pred = "prediction_norm_e"
        amp = None
        cosine = "cosine"
    elif task == "dielectric":
        rel = "stabilized_relative_frobenius_error"
        target = "target_norm"
        pred = None
        amp = None
        cosine = None
    else:  # pragma: no cover - guarded by callers
        raise ValueError(task)
    result: dict[str, Any] = {
        "materials": len(rows),
        "mean_stabilized_relative_error": _mean(row.get(rel, math.nan) for row in rows),
        "mean_target_norm": _mean(row.get(target, math.nan) for row in rows),
    }
    if pred is not None:
        result["mean_prediction_norm"] = _mean(row.get(pred, math.nan) for row in rows)
    if amp is not None:
        result["mean_amplitude_ratio"] = _mean(row.get(amp, math.nan) for row in rows)
    if cosine is not None:
        result["mean_cosine"] = _mean(row.get(cosine, math.nan) for row in rows)
    if task == "electronic":
        active_rows = [row for row in rows if bool(row.get("active", False))]
        result["active_materials"] = len(active_rows)
        result["active_mean_stabilized_relative_error"] = _mean(
            row.get(rel, math.nan) for row in active_rows
        )
        result["active_mean_cosine"] = _mean(row.get(cosine, math.nan) for row in active_rows)
        result["active_mean_amplitude_ratio"] = _mean(
            row.get(amp, math.nan) for row in active_rows
        )
        result["active_mean_target_norm"] = _mean(
            row.get(target, math.nan) for row in active_rows
        )
        result["active_mean_prediction_norm"] = _mean(
            row.get(pred, math.nan) for row in active_rows
        )
    return result


def _bin_target_norm(value: float) -> str:
    if value <= 1.0e-12:
        return "zero"
    if value < 0.1:
        return "(0,0.1)"
    if value < 0.5:
        return "[0.1,0.5)"
    if value < 1.0:
        return "[0.5,1)"
    return "[1,inf)"


def _bin_atoms(value: int) -> str:
    if value <= 2:
        return "[1,2]"
    if value <= 8:
        return "[3,8]"
    if value <= 16:
        return "[9,16]"
    if value <= 32:
        return "[17,32]"
    return "[33,inf)"


def _load_dft3d(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected one JSON member in {path}, found {names}")
        records = json.loads(archive.read(names[0]))
    result = {}
    for record in records:
        jid = str(record.get("jid", ""))
        if jid:
            result[jid] = {
                "formula": record.get("formula"),
                "crystal_system": record.get("crys"),
                "space_group": record.get("spg_symbol", record.get("spg")),
                "spg_number": record.get("spg_number"),
                "atoms": record.get("nat"),
            }
    return result


def build_report(
    checkpoint_path: str | Path,
    folds_path: str | Path,
    fold: int,
    dft3d_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    folds_path = Path(folds_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = payload.get("checkpoint_provenance", {})
    if provenance.get("frozen_validation_test_labels_read", False):
        raise ValueError("Refusing stratified audit: checkpoint reports frozen labels were read")
    contract = payload.get("training_contract", {})
    development_ids = [str(value) for value in contract.get("development_ids", [])]
    if not development_ids:
        raise ValueError("Checkpoint does not contain development_ids")
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    fold_records = folds["folds"][int(fold)]
    expected_ids = {str(value) for value in fold_records["development"]}
    if set(development_ids) != expected_ids or len(development_ids) != len(expected_ids):
        raise ValueError("Checkpoint development IDs do not match the requested fold")
    metrics = payload["development_metrics"]
    dft3d = _load_dft3d(Path(dft3d_path) if dft3d_path else None)
    electronic = metrics["electronic"]["per_material"]
    born = metrics["born"]["per_material"]
    dielectric = metrics["dielectric"]["per_material"]
    if not (len(electronic) == len(born) == len(dielectric) == len(development_ids)):
        raise ValueError("Per-material metric lengths do not match development IDs")

    by_stratum: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    joined: list[dict[str, Any]] = []
    for jid, e_row, b_row, d_row in zip(development_ids, electronic, born, dielectric):
        meta = dft3d.get(jid, {})
        atoms = int(b_row.get("atoms", meta.get("atoms") or 0))
        e_norm = float(e_row.get("target_norm_c_per_m2", 0.0))
        item = {
            "jid": jid,
            "formula": meta.get("formula"),
            "crystal_system": meta.get("crystal_system") or "unknown",
            "space_group": meta.get("space_group"),
            "atoms": atoms,
            "electronic": e_row,
            "born": b_row,
            "dielectric": d_row,
        }
        joined.append(item)
        by_stratum["electronic_target_norm"][_bin_target_norm(e_norm)].append(item)
        by_stratum["atom_count"][_bin_atoms(atoms)].append(item)
        by_stratum["crystal_system"][str(item["crystal_system"])].append(item)

    strata: dict[str, dict[str, Any]] = {}
    for dimension, groups in by_stratum.items():
        strata[dimension] = {}
        for label, items in sorted(groups.items()):
            strata[dimension][label] = {
                "materials": len(items),
                "electronic": _summary([item["electronic"] for item in items], task="electronic"),
                "born": _summary([item["born"] for item in items], task="born"),
                "dielectric": _summary([item["dielectric"] for item in items], task="dielectric"),
            }

    return {
        "schema": 1,
        "diagnostic": "electrostatic_stratified_development_audit",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "fold": int(fold),
        "fold_source": str(folds_path),
        "fold_source_sha256": _sha256(folds_path),
        "dft3d_metadata_source": str(dft3d_path) if dft3d_path else None,
        "dft3d_metadata_sha256": _sha256(Path(dft3d_path)) if dft3d_path else None,
        "development_materials": len(joined),
        "metadata_coverage": sum(bool(dft3d.get(jid)) for jid in development_ids),
        "frozen_validation_test_labels_read": False,
        "overall": {
            "electronic": _summary(electronic, task="electronic"),
            "born": _summary(born, task="born"),
            "dielectric": _summary(dielectric, task="dielectric"),
            "per_irrep": metrics["electronic"].get("per_irrep"),
        },
        "strata": strata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--dft3d-metadata", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.checkpoint, args.folds, args.fold, args.dft3d_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
