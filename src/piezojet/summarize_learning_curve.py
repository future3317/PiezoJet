"""Aggregate registered strict-Lambda learning-curve evaluations.

Reads only persisted test JSON files.  It never re-evaluates checkpoints or
selects a checkpoint from the test panel, making the report safe to regenerate
after all registered seeds finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


RUN = re.compile(r"^(production|factor)_n(\d+)_seed(\d+)$")


METRICS = {
    "lambda_cosine": ("factors", "internal_strain_full", "macro_material_directional_cosine"),
    "lambda_amplitude": ("factors", "internal_strain_full", "macro_material_stabilized_amplitude_ratio"),
    "lambda_mae": ("factors", "internal_strain_full", "macro_material_component_mae"),
    "ionic_cosine": ("factors", "ionic_piezo", "macro_material_directional_cosine"),
    "ionic_amplitude": ("factors", "ionic_piezo", "macro_material_stabilized_amplitude_ratio"),
    "ionic_skill_zero": ("factors", "ionic_piezo", "macro_material_mae_skill_vs_zero"),
    "total_trs": ("total_response_skill", "tensor_response_skill_vs_zero"),
    "soft_sign_accuracy": ("soft_mode_metrics", "soft_mode_sign_accuracy"),
    "soft_subspace_overlap": ("soft_mode_metrics", "soft_mode_subspace_overlap"),
    "true_factor_lambda_cosine": ("oracle_factor_replacement", "all", "experiments", "true_z_true_phi_pred_lambda_auto", "directional_cosine"),
    "true_factor_lambda_amplitude": ("oracle_factor_replacement", "all", "experiments", "true_z_true_phi_pred_lambda_auto", "stabilized_amplitude_ratio"),
}


def _get(value: dict, path: tuple[str, ...]) -> float:
    for key in path:
        value = value[key]
    return float(value)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0, "seeds": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/strict_learning_curve_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/strict_learning_curve_v1/report"))
    args = parser.parse_args()
    rows = []
    for directory in sorted(args.root.iterdir()):
        match = RUN.match(directory.name)
        path = directory / "dfpt_test.json"
        if match is None or not path.is_file():
            continue
        protocol, size, seed = match.groups()
        result = json.loads(path.read_text(encoding="utf-8"))
        row = {"protocol": protocol, "train_materials": int(size), "seed": int(seed), "checkpoint_epoch": int(result["checkpoint_epoch"])}
        for name, keys in METRICS.items():
            row[name] = _get(result, keys)
        rows.append(row)
    expected = {(protocol, size, seed) for protocol in ("production", "factor") for size in (19, 35, 69) for seed in (42, 43, 44)}
    present = {(row["protocol"], row["train_materials"], row["seed"]) for row in rows}
    missing = sorted(expected - present)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["protocol"], row["train_materials"]].append(row)
    aggregate = {
        f"{protocol}_n{size}": {
            "protocol": protocol, "train_materials": size,
            **{metric: _summary([row[metric] for row in group]) for metric in METRICS},
        }
        for (protocol, size), group in sorted(grouped.items())
    }
    report = {
        "schema": 1,
        "protocol": {
            "test_panel": "frozen 20-material formula-disjoint strict-completion benchmark",
            "production": "100 fixed updates; validation-loss checkpoint selection",
            "factor": "50 direct-factor updates + 50 frozen-factor joint updates; validation-loss checkpoint selection",
            "selection_rule": "No test result selected a checkpoint, hyperparameter, or subset.",
        },
        "runs": rows, "aggregate": aggregate, "missing_expected_anchor_runs": missing,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "learning_curve.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "learning_curve_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Strict-Lambda nested learning curve", "",
        "Frozen 20-material formula-disjoint test panel; checkpoint selection uses validation loss only.", "",
        "## Registered results", "",
        "Rows with three seeds are the anchor comparisons; the N=23 and N=50 rows are seed-42 phase-1 diagnostics only.", "",
        "| Protocol | Train labels | Seeds | Full Lambda cosine | Full Lambda amplitude | True-factor ionic cosine | True-factor ionic amplitude | Ionic skill vs zero | Total TRS | Soft sign accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, value in aggregate.items():
        def fmt(name: str) -> str:
            item = value[name]
            return f"{item['mean']:.3f} +/- {item['std']:.3f}"
        lines.append(
            f"| {value['protocol']} | {value['train_materials']} | {value['lambda_cosine']['seeds']} | {fmt('lambda_cosine')} | {fmt('lambda_amplitude')} | "
            f"{fmt('true_factor_lambda_cosine')} | {fmt('true_factor_lambda_amplitude')} | {fmt('ionic_skill_zero')} | {fmt('total_trs')} | {fmt('soft_sign_accuracy')} |"
        )
    lines.extend([
        "", "## Interpretation boundary", "",
        "These are learning-curve diagnostics, not replacements for the earlier production tables. "
        "They establish whether certified full-Lambda coverage and optimization protocol change held-out factor learning on a fixed panel; "
        "they do not support a population-level accuracy claim or test-guided hyperparameter selection.",
    ])
    if missing:
        lines.extend(["", f"Missing registered anchor runs: {missing}"])
    (args.output_dir / "learning_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "learning_curve.md")


if __name__ == "__main__":
    main()
