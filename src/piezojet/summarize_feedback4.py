"""Aggregate the registered fourth-feedback optimization forensics.

This script only reads persisted artifacts.  It does not train, evaluate a
checkpoint anew, or use test outputs to select a protocol or subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev


ABLATON_RUN = re.compile(r"seed(\d+).+protocol_([ABCD])")


def _summary(values: list[float]) -> dict[str, float | int]:
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0, "count": len(values)}


def _factor_values(payload: dict) -> dict[str, float]:
    factor = payload["factors"]["internal_strain_full"]
    oracle = payload["oracle_factor_replacement"]["all"]["experiments"]["true_z_true_phi_pred_lambda_auto"]
    return {
        "full_lambda_cosine": float(factor["macro_material_directional_cosine"]),
        "full_lambda_amplitude": float(factor["macro_material_stabilized_amplitude_ratio"]),
        "oracle_ionic_cosine": float(oracle["directional_cosine"]),
        "oracle_ionic_amplitude": float(oracle["stabilized_amplitude_ratio"]),
        "predicted_ionic_skill_vs_zero": float(payload["factors"]["ionic_piezo"]["macro_material_mae_skill_vs_zero"]),
        "total_trs": float(payload["total_response_skill"]["tensor_response_skill_vs_zero"]),
    }


def _read_ablation(root: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    for path in sorted(root.glob("seed*/protocol_*/summary.json")):
        match = ABLATON_RUN.search(str(path.parent))
        if match is None:
            raise ValueError(f"Could not parse seed/protocol from {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"seed": int(match.group(1)), "protocol": match.group(2), **payload["test_diagnostic"]})
    if len(rows) != 12:
        raise ValueError(f"Expected 12 registered ablation rows, found {len(rows)}")
    metrics = tuple(key for key in rows[0] if key not in {"seed", "protocol"})
    aggregate = {
        protocol: {metric: _summary([float(row[metric]) for row in rows if row["protocol"] == protocol]) for metric in metrics}
        for protocol in "ABCD"
    }
    return rows, aggregate


def _trajectory(root: Path) -> dict[str, float]:
    values = {}
    for protocol in ("B", "C"):
        rows = list(csv.DictReader((root / "seed42" / f"protocol_{protocol}" / "trajectory.csv").open(encoding="utf-8")))
        for update in (50, 100):
            row = next(item for item in rows if int(item["global_update"]) == update)
            values[f"seed42_{protocol}_val_lambda_cosine_update{update}"] = float(row["val_full_lambda_cosine"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", type=Path, default=Path("outputs/optimization_ablation_v1"))
    parser.add_argument("--resampling-root", type=Path, default=Path("outputs/stratified_subset_resampling_v1/runs"))
    parser.add_argument("--mode-aware", type=Path, default=Path("outputs/mode_aware_smoke_v1/seed42/dfpt_test.json"))
    parser.add_argument("--baseline69", type=Path, default=Path("outputs/strict_learning_curve_v1/factor_n69_seed42/dfpt_test.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feedback4_execution_v1/report"))
    args = parser.parse_args()
    ablation_rows, ablation = _read_ablation(args.ablation_root)
    resampling_rows = []
    for path in sorted(args.resampling_root.glob("subset*_seed42/dfpt_test.json")):
        resampling_rows.append({"subset": path.parent.name, **_factor_values(json.loads(path.read_text(encoding="utf-8")))})
    if len(resampling_rows) != 5:
        raise ValueError(f"Expected five resampled 35-label diagnostics, found {len(resampling_rows)}")
    mode = _factor_values(json.loads(args.mode_aware.read_text(encoding="utf-8")))
    baseline = _factor_values(json.loads(args.baseline69.read_text(encoding="utf-8")))
    report = {
        "schema": 1,
        "scope": "post-freeze diagnostic; no test result selected a protocol, subset, checkpoint, or hyperparameter",
        "protocol_definitions": {
            "A": "100 factor-only updates",
            "B": "50 factor updates, validation-selected restore, 50 frozen-factor joint updates",
            "C": "50 factor updates, validation-selected restore, 50 unfrozen-factor joint updates",
            "D": "50 alternating factor/joint pairs with factors trainable",
        },
        "ablation_runs": ablation_rows,
        "ablation_aggregate": ablation,
        "seed42_validation_trajectory": _trajectory(args.ablation_root),
        "resampled_35_single_seed": resampling_rows,
        "resampled_35_lambda_cosine": _summary([row["full_lambda_cosine"] for row in resampling_rows]),
        "mode_aware_single_seed": {"baseline_factor_n69_seed42": baseline, "mode_aware": mode},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "feedback4_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "ablation_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)
    with (args.output_dir / "resampled35_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(resampling_rows[0]))
        writer.writeheader()
        writer.writerows(resampling_rows)
    def fmt(protocol: str, metric: str) -> str:
        item = ablation[protocol][metric]
        return f"{item['mean']:.3f} +/- {item['std']:.3f}"
    trajectory = report["seed42_validation_trajectory"]
    lines = [
        "# Fourth-feedback registered forensics", "",
        "All results use the frozen 69/10/20 formula-disjoint strict-completion panel. "
        "Validation loss selected checkpoints; the test panel was read only after selection.", "",
        "## Fixed-coverage optimization ablation", "",
        "| Protocol | Full Lambda cosine | Full Lambda amplitude | Oracle ionic cosine | Oracle ionic skill vs zero | Total TRS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for protocol in "ABCD":
        lines.append(
            f"| {protocol} | {fmt(protocol, 'full_lambda_cosine')} | {fmt(protocol, 'full_lambda_amplitude')} | "
            f"{fmt(protocol, 'oracle_ionic_cosine')} | {fmt(protocol, 'oracle_ionic_skill_vs_zero')} | {fmt(protocol, 'total_trs')} |"
        )
    lines.extend([
        "", "A is the factor-learning control; its total response is not a trained response model. "
        "B and C share the first 50 factor updates, so their difference isolates whether joint optimization is allowed to rewrite the factor stack.",
        f"For seed 42, validation full-Lambda cosine is {trajectory['seed42_B_val_lambda_cosine_update50']:.3f} after factor pretraining; "
        f"after 50 joint updates it is {trajectory['seed42_B_val_lambda_cosine_update100']:.3f} with frozen factors (B) and "
        f"{trajectory['seed42_C_val_lambda_cosine_update100']:.3f} with unfrozen factors (C).",
        "", "## 35-label composition diagnostic", "",
        "| Subset | Full Lambda cosine | Full Lambda amplitude | Oracle ionic cosine | Total TRS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in resampling_rows:
        lines.append(
            f"| {row['subset']} | {row['full_lambda_cosine']:.3f} | {row['full_lambda_amplitude']:.3f} | "
            f"{row['oracle_ionic_cosine']:.3f} | {row['total_trs']:.3f} |"
        )
    resampled = report["resampled_35_lambda_cosine"]
    lines.extend([
        "", f"The five one-seed, response-bin-matched subsets span full-Lambda cosine "
        f"{resampled['mean']:.3f} +/- {resampled['std']:.3f} (range "
        f"{min(row['full_lambda_cosine'] for row in resampling_rows):.3f}--{max(row['full_lambda_cosine'] for row in resampling_rows):.3f}). "
        "They quantify subset-composition uncertainty; they are not a multi-seed learning curve.",
        "", "## Mode-aware single-seed smoke", "",
        "| Run | Full Lambda cosine | Oracle ionic cosine | Predicted ionic skill vs zero | Total TRS |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| baseline factor n69 seed42 | {baseline['full_lambda_cosine']:.3f} | {baseline['oracle_ionic_cosine']:.3f} | {baseline['predicted_ionic_skill_vs_zero']:.3f} | {baseline['total_trs']:.3f} |",
        f"| mode-aware factor n69 seed42 | {mode['full_lambda_cosine']:.3f} | {mode['oracle_ionic_cosine']:.3f} | {mode['predicted_ionic_skill_vs_zero']:.3f} | {mode['total_trs']:.3f} |",
        "", "The mode-aware row is one seed and a smoke test only.  It validates the implementation but does not establish an improvement or justify changing the production default.",
    ])
    (args.output_dir / "feedback4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "feedback4_report.md")


if __name__ == "__main__":
    main()
