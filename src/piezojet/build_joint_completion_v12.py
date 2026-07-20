"""Calibrate and build symmetry-restricted joint-identifiable Lambda labels.

The macroscopic ionic tensor is allowed to participate in reconstruction, so
it is never reused as independent validation.  Acceptance requires prediction
of at least one held-out printed OUTCAR block plus condition thresholds
calibrated only on reduced-formula-safe strict training materials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .data import load_gmtnet_records
from .identifiability import (
    IdentificationSystem,
    build_identification_system,
    identification_certificate,
    linear_map_metrics,
)
from .jarvis_dfpt import JarvisDFPTCache
from .strain_completion import vector_to_internal_tensor


SCHEMA = 3
HELDOUT_RELATIVE_LIMIT = 5e-2
PRINTED_FIT_RELATIVE_LIMIT = 5e-3
MACRO_FIT_RELATIVE_LIMIT = 5e-2


def _read_ids(path: Path, key: str | None = None) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if key is not None:
        payload = payload[key]
    if isinstance(payload, dict) and "material_ids" in payload:
        payload = payload["material_ids"]
    if not isinstance(payload, list):
        raise ValueError(f"Could not read material IDs from {path}")
    return [str(value) for value in payload]


def _relative(error: torch.Tensor, target: torch.Tensor, floor: float) -> float:
    denominator = torch.linalg.vector_norm(target).clamp_min(
        floor * max(target.numel(), 1) ** 0.5
    )
    return float(torch.linalg.vector_norm(error) / denominator)


def _solve(
    system: IdentificationSystem,
    printed_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    matrix = system.joint_matrix(printed_rows)
    target = system.joint_target(printed_rows)
    metrics = linear_map_metrics(matrix, system.dimension)
    if not bool(metrics["full_column_rank"]):
        raise ValueError("Joint observation system is not full column rank")
    coefficients = torch.linalg.lstsq(matrix, target).solution
    return coefficients, metrics


def _fit_metrics(system: IdentificationSystem, coefficients: torch.Tensor) -> dict[str, float]:
    macro_error = system.macro_matrix @ coefficients - system.macro_target
    printed_error = system.printed_matrix @ coefficients - system.printed_target
    return {
        "macro_fit_relative": _relative(macro_error, system.macro_target, 0.05),
        "printed_fit_relative": _relative(printed_error, system.printed_target, 1e-6),
    }


def leave_one_printed_block_out(system: IdentificationSystem) -> list[dict[str, Any]]:
    """Predict each block not used in a macro+remaining-block solve."""
    rows = torch.arange(system.printed_target.numel())
    output = []
    for block_index in range(len(system.printed_blocks)):
        held = torch.arange(6 * block_index, 6 * (block_index + 1))
        kept = rows[~torch.isin(rows, held)]
        matrix = system.joint_matrix(kept)
        metrics = linear_map_metrics(matrix, system.dimension)
        if not bool(metrics["full_column_rank"]):
            continue
        coefficients = torch.linalg.lstsq(matrix, system.joint_target(kept)).solution
        prediction = system.printed_matrix[held] @ coefficients
        target = system.printed_target[held]
        output.append({
            "block_index": block_index,
            "relative_error": _relative(prediction - target, target, 1e-6),
            "condition_joint_scaled": metrics["condition_number_full"],
            "sigma_min_joint_scaled": metrics["sigma_min_full"],
        })
    return output


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot calibrate a threshold from an empty panel")
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q))


def _calibrate(
    *,
    strict_ids: list[str],
    records: dict[str, dict[str, Any]],
    cache: JarvisDFPTCache,
    strict_completion_dir: Path,
    progress_interval: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, good_conditions, good_sigmas = [], [], []
    for position, jid in enumerate(strict_ids, start=1):
        payload = cache.load(jid)
        completion_path = strict_completion_dir / f"{jid}.pt"
        if payload is None or not completion_path.is_file():
            rows.append({"jid": jid, "error": "strict payload or completion missing"})
            continue
        system = build_identification_system(records[jid], payload)
        folds = leave_one_printed_block_out(system)
        valid = [fold for fold in folds if fold["relative_error"] <= HELDOUT_RELATIVE_LIMIT]
        good_conditions.extend(float(fold["condition_joint_scaled"]) for fold in valid)
        good_sigmas.extend(float(fold["sigma_min_joint_scaled"]) for fold in valid)
        coefficients, _ = _solve(system)
        reconstructed = vector_to_internal_tensor(
            system.basis @ coefficients, len(records[jid]["atoms"]["elements"])
        )
        true = torch.load(completion_path, map_location="cpu", weights_only=False)[
            "internal_strain_full"
        ].to(torch.float64)
        rows.append({
            "jid": jid,
            "identifiable_dimension": system.dimension,
            "heldout_identifiable_blocks": len(folds),
            "heldout_passing_blocks": len(valid),
            "heldout_relative_max": max((fold["relative_error"] for fold in folds), default=None),
            "heldout_relative_median": (
                _quantile([fold["relative_error"] for fold in folds], 0.5) if folds else None
            ),
            "joint_reconstruction_relative_to_strict": _relative(
                reconstructed - true, true, 1e-6
            ),
            **_fit_metrics(system, coefficients),
        })
        if position == 1 or position % progress_interval == 0 or position == len(strict_ids):
            print(
                f"[calibration {position}/{len(strict_ids)}] {jid} heldout={len(folds)}",
                flush=True,
            )
    thresholds = {
        "heldout_relative_max": HELDOUT_RELATIVE_LIMIT,
        "printed_fit_relative_max": PRINTED_FIT_RELATIVE_LIMIT,
        "macro_fit_relative_max": MACRO_FIT_RELATIVE_LIMIT,
        "condition_joint_scaled_max": _quantile(good_conditions, 0.95),
        "sigma_min_joint_scaled_min": _quantile(good_sigmas, 0.05),
        "calibration_good_heldout_blocks": len(good_conditions),
        "condition_quantile": 0.95,
        "sigma_min_quantile": 0.05,
    }
    return thresholds, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dfpt-dir", type=Path, required=True)
    parser.add_argument("--strict-completion-dir", type=Path, required=True)
    parser.add_argument(
        "--strict-split",
        type=Path,
        default=Path("data/processed/strict_completion_benchmark_train_v11_reduced_formula_safe.json"),
    )
    parser.add_argument("--census-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"v12 output directory must be fresh: {args.output_dir}")
    if args.progress_interval < 1:
        raise ValueError("--progress-interval must be positive")
    split = json.loads(args.strict_split.read_text(encoding="utf-8-sig"))["splits"]
    strict_ids = [str(value) for value in split["train"]]
    forbidden = {str(value) for value in split["val"] + split["test"]}
    census = json.loads(args.census_summary.read_text(encoding="utf-8"))
    candidates = [
        str(value) for value in census["joint_increment_material_ids"]
        if str(value) not in forbidden and str(value) not in set(strict_ids)
    ]
    records = {str(record["JARVIS_ID"]): record for record in load_gmtnet_records(args.data_root)}
    cache = JarvisDFPTCache(args.dfpt_dir)
    thresholds, calibration_rows = _calibrate(
        strict_ids=strict_ids,
        records=records,
        cache=cache,
        strict_completion_dir=args.strict_completion_dir,
        progress_interval=args.progress_interval,
    )
    args.output_dir.mkdir(parents=True)
    accepted, candidate_rows = [], []
    for position, jid in enumerate(candidates, start=1):
        payload = cache.load(jid)
        if payload is None:
            candidate_rows.append({"jid": jid, "accepted": False, "error": "DFPT payload missing"})
            continue
        try:
            system = build_identification_system(records[jid], payload)
            certificate = identification_certificate(system)
            coefficients, metrics = _solve(system)
            fit = _fit_metrics(system, coefficients)
            folds = leave_one_printed_block_out(system)
            heldout_max = max((fold["relative_error"] for fold in folds), default=None)
            parse_audit = payload.get("internal_strain_parse_audit")
            parse_complete = (
                bool(parse_audit.get("complete_observed_block_parse", False))
                if isinstance(parse_audit, dict) else len(system.printed_blocks) > 0
            )
            accepted_flag = bool(
                parse_complete
                and folds
                and heldout_max is not None
                and heldout_max <= thresholds["heldout_relative_max"]
                and fit["printed_fit_relative"] <= thresholds["printed_fit_relative_max"]
                and fit["macro_fit_relative"] <= thresholds["macro_fit_relative_max"]
                and float(metrics["condition_number_full"]) <= thresholds["condition_joint_scaled_max"]
                and float(metrics["sigma_min_full"]) >= thresholds["sigma_min_joint_scaled_min"]
            )
            row = {
                "jid": jid,
                "lambda_label_type": "joint_identifiable",
                **certificate,
                **fit,
                "source_internal_strain_block_parse_complete": parse_complete,
                "independent_validation": "leave_one_printed_block_out",
                "heldout_identifiable_blocks": len(folds),
                "heldout_relative_max": heldout_max,
                "heldout_relative_median": (
                    _quantile([fold["relative_error"] for fold in folds], 0.5) if folds else None
                ),
                "thresholds": thresholds,
                "accepted": accepted_flag,
            }
            candidate_rows.append(row)
            if accepted_flag:
                atoms = len(records[jid]["atoms"]["elements"])
                completed = vector_to_internal_tensor(system.basis @ coefficients, atoms).to(torch.float32)
                torch.save({
                    "schema": SCHEMA,
                    "jid": jid,
                    "lambda_label_type": "joint_identifiable",
                    "internal_strain_full": completed,
                    "identifiability": row,
                    "audit": {"accepted": True, **row},
                }, args.output_dir / f"{jid}.pt")
                accepted.append(jid)
            print(f"[candidate {position}/{len(candidates)}] {jid} accepted={accepted_flag}", flush=True)
        except Exception as error:
            candidate_rows.append({"jid": jid, "accepted": False, "error": str(error)})
    manifest = {
        "schema": SCHEMA,
        "name": "strain_completion_v12_joint_identifiable",
        "scope": "development-train-only; frozen validation/test labels unread",
        "strict_calibration_materials": len(strict_ids),
        "algebraic_joint_increment_candidates": len(candidates),
        "accepted": len(accepted),
        "accepted_material_ids": accepted,
        "thresholds": thresholds,
        "independent_validation_policy": (
            "macro ionic response participates in fitting and is never counted as validation; "
            "at least one unused printed block must be predicted"
        ),
        "calibration_rows": calibration_rows,
        "candidate_rows": candidate_rows,
        "frozen_validation_test_labels_read": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "strict_calibration_materials": len(strict_ids),
        "candidates": len(candidates),
        "accepted": len(accepted),
        "thresholds": thresholds,
    }, indent=2))


if __name__ == "__main__":
    main()
