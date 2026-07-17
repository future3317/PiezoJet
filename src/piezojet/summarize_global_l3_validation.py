"""Summarize the validation-only global-l3 replication and matched control.

This utility reads only run-local ``metrics.csv`` and ``summary.json`` files.
It never opens a frozen-test prediction or evaluator artifact.  Checkpoints are
selected independently within each seed by minimum validation loss, exactly as
declared by the training drivers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PHYSICAL_COLUMNS = {
    "validation_loss": "val_loss",
    "total_trs": "val_tensor_response_skill_vs_zero_loss",
    "direct_u_loss": "val_displacement_response_loss",
    "ionic_loss": "val_ionic_piezo_loss",
    "first_order_consistency_loss": "val_displacement_first_order_consistency_loss",
    "electronic_loss": "val_electronic_piezo_loss",
    "branch_sum_closure": "val_branch_sum_loss",
}
DIRECT_COLUMNS = {
    "validation_loss": "val_loss",
    "total_trs": "val_tensor_response_skill_vs_zero",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Metrics file has no rows: {path}")
    return rows


def _selected(run_dir: Path, columns: dict[str, str]) -> dict[str, float | int | str]:
    metrics_path = run_dir / "metrics.csv"
    rows = _rows(metrics_path)
    selected = min(rows, key=lambda row: float(row["val_loss"]))
    result: dict[str, float | int | str] = {
        "run_dir": run_dir.as_posix(),
        "selected_epoch": int(float(selected["epoch"])),
        "selection_rule": "minimum validation loss",
    }
    for output_name, column in columns.items():
        if column not in selected or selected[column] == "":
            raise ValueError(f"Selected row in {metrics_path} lacks {column}")
        value = float(selected[column])
        if not math.isfinite(value):
            raise ValueError(f"Selected {column} is non-finite in {metrics_path}")
        result[output_name] = value
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        declared_epoch = summary.get("loss_best_epoch", summary.get("selected_epoch"))
        if declared_epoch is not None and int(declared_epoch) != result["selected_epoch"]:
            raise ValueError(
                f"Run summary and metrics disagree on selected epoch: {run_dir}"
            )
    return result


def _mean_sd(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "seeds": len(values),
    }


def _aggregate(rows: dict[int, dict[str, float | int | str]], names: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    return {
        name: _mean_sd([float(rows[seed][name]) for seed in sorted(rows)])
        for name in names
    }


def summarize(
    replication_root: Path,
    seed42_run: Path,
    seeds: list[int],
    *,
    direct_root: Path | None = None,
    training_source_manifest: Path | None = None,
    direct_training_source_manifest: Path | None = None,
) -> dict[str, object]:
    physical: dict[int, dict[str, float | int | str]] = {}
    for seed in seeds:
        run_dir = seed42_run if seed == 42 else replication_root / f"factorized_seed{seed}"
        physical[seed] = _selected(run_dir, PHYSICAL_COLUMNS)
    physical_names = tuple(PHYSICAL_COLUMNS)
    result: dict[str, object] = {
        "schema": 1,
        "status": "completed_validation_only",
        "selection": "minimum run-local validation loss independently per seed",
        "split": "train1603; formula-disjoint val10; frozen test20 unread",
        "seeds": seeds,
        "physical_individual": {str(seed): physical[seed] for seed in seeds},
        "physical_mean_sample_sd": _aggregate(physical, physical_names),
        "validation_gate": {
            "three_finite_seeds": len(physical) == 3,
            "positive_mean_total_trs": statistics.mean(
                float(row["total_trs"]) for row in physical.values()
            ) > 0.0,
            "interpretation": (
                "Validation-only replication gate; it authorizes a matched direct "
                "validation control, not frozen-test evaluation or production promotion."
            ),
        },
    }
    if training_source_manifest is not None:
        result["training_source_manifest"] = training_source_manifest.as_posix()
        result["training_source_manifest_payload"] = json.loads(
            training_source_manifest.read_text(encoding="utf-8-sig")
        )
    if direct_root is not None:
        direct = {
            seed: _selected(direct_root / f"direct_seed{seed}", DIRECT_COLUMNS)
            for seed in seeds
        }
        differences = {
            seed: float(physical[seed]["total_trs"]) - float(direct[seed]["total_trs"])
            for seed in seeds
        }
        result["direct_individual"] = {str(seed): direct[seed] for seed in seeds}
        result["direct_mean_sample_sd"] = _aggregate(direct, tuple(DIRECT_COLUMNS))
        result["paired_physical_minus_direct_total_trs"] = {
            "individual": {str(seed): differences[seed] for seed in seeds},
            **_mean_sd(list(differences.values())),
            "interpretation": (
                "Validation-only paired difference. Positive favors the isolated macro "
                "tower used alongside the physical model; it does not identify ionic factors."
            ),
        }
    if direct_training_source_manifest is not None:
        result["direct_training_source_manifest"] = (
            direct_training_source_manifest.as_posix()
        )
        result["direct_training_source_manifest_payload"] = json.loads(
            direct_training_source_manifest.read_text(encoding="utf-8-sig")
        )
    return result


def markdown_report(result: dict[str, object]) -> str:
    physical = result["physical_individual"]
    lines = [
        "# Global-l3 validation-only replication",
        "",
        "Frozen test20 was not read. Each seed is selected by its own minimum val10 loss.",
        "",
        "| Seed | Epoch | Val loss | Total TRS | direct-U | Ionic | First-order | Branch closure |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in result["seeds"]:
        row = physical[str(seed)]
        lines.append(
            f"| {seed} | {row['selected_epoch']} | {row['validation_loss']:.6f} | "
            f"{row['total_trs']:.6f} | {row['direct_u_loss']:.6f} | "
            f"{row['ionic_loss']:.6f} | {row['first_order_consistency_loss']:.6f} | "
            f"{row['branch_sum_closure']:.6f} |"
        )
    aggregate = result["physical_mean_sample_sd"]
    lines.extend(
        [
            "",
            "## Three-seed aggregate",
            "",
            f"- Total TRS: {aggregate['total_trs']['mean']:.6f} +/- {aggregate['total_trs']['sample_sd']:.6f}.",
            f"- Val loss: {aggregate['validation_loss']['mean']:.6f} +/- {aggregate['validation_loss']['sample_sd']:.6f}.",
            f"- direct-U loss: {aggregate['direct_u_loss']['mean']:.6f} +/- {aggregate['direct_u_loss']['sample_sd']:.6f}.",
            f"- Ionic loss: {aggregate['ionic_loss']['mean']:.6f} +/- {aggregate['ionic_loss']['sample_sd']:.6f}.",
        ]
    )
    if "direct_individual" in result:
        lines.extend(
            [
                "",
                "## Matched direct validation control",
                "",
                "| Seed | Direct epoch | Direct val loss | Direct TRS | Physical - direct TRS |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        direct = result["direct_individual"]
        differences = result["paired_physical_minus_direct_total_trs"]["individual"]
        for seed in result["seeds"]:
            row = direct[str(seed)]
            lines.append(
                f"| {seed} | {row['selected_epoch']} | {row['validation_loss']:.6f} | "
                f"{row['total_trs']:.6f} | {differences[str(seed)]:.6f} |"
            )
    lines.extend(
        [
            "",
            "These are post-freeze validation diagnostics, not test or production-performance claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--seed42-run", type=Path, required=True)
    parser.add_argument("--seeds", default="42,7,1729")
    parser.add_argument("--direct-root", type=Path)
    parser.add_argument("--training-source-manifest", type=Path)
    parser.add_argument("--direct-training-source-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.replication_root,
        args.seed42_run,
        [int(value) for value in args.seeds.split(",")],
        direct_root=args.direct_root,
        training_source_manifest=args.training_source_manifest,
        direct_training_source_manifest=args.direct_training_source_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
