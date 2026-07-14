"""Summarize canonical P0/P2 diagnostics without selecting on the test panel."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


RUN = re.compile(r"^([A-G])_seed(\d+)$")


def _summary(values: list[float]) -> dict[str, float | int]:
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0, "count": len(values)}


def _row(path: Path) -> dict[str, float | int | str]:
    match = RUN.match(path.stem)
    if match is None:
        raise ValueError(f"Expected a canonical protocol file named A_seed42.json, got {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    factor = payload["factors"]["internal_strain_full"]
    oracle = payload["oracle_factor_replacement"]["all"]
    if oracle["canonical_operator_policy"] != "regularized":
        raise ValueError(f"{path} does not declare the production regularized canonical oracle")
    canonical = oracle["canonical_metrics"]
    ionic = payload["ionic_response_aggregation"]
    decomposition = payload["response_decomposition"]
    return {
        "protocol": match.group(1),
        "seed": int(match.group(2)),
        "checkpoint": str(payload["checkpoint"]),
        "full_lambda_cosine_macro_material": float(factor["macro_material_directional_cosine"]),
        "full_lambda_amplitude_ratio_macro": float(factor["macro_material_stabilized_amplitude_ratio"]),
        "oracle_ionic_cosine_macro_material": float(canonical["ionic_cosine_macro_material"]),
        "oracle_ionic_cosine_micro_components": float(canonical["ionic_cosine_micro_components"]),
        "oracle_ionic_cosine_active_only": float(canonical["ionic_cosine_active_only"]),
        "oracle_ionic_amplitude_ratio_macro": float(canonical["ionic_amplitude_ratio_macro"]),
        "predicted_ionic_cosine_macro_material": float(ionic["ionic_cosine_macro_material"]),
        "predicted_ionic_skill_vs_zero_macro": float(ionic["ionic_mae_skill_vs_zero_macro"]),
        "total_trs": float(payload["total_response_skill"]["tensor_response_skill_vs_zero"]),
        "electronic_cosine_macro_material": float(decomposition["electronic_cosine_macro_material"]),
        "predicted_electronic_norm_over_true_total_macro": float(decomposition["predicted_electronic_norm_over_true_total_macro"]),
        "predicted_ionic_norm_over_true_total_macro": float(decomposition["predicted_ionic_norm_over_true_total_macro"]),
        "predicted_total_norm_over_true_total_macro": float(decomposition["predicted_total_norm_over_true_total_macro"]),
        "predicted_cancellation_ratio_macro": float(decomposition["predicted_cancellation_ratio_macro"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Directory of canonical protocol JSON files; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feedback5_execution_v1/report"))
    parser.add_argument(
        "--title",
        default="Fifth-feedback canonical diagnostics",
        help="Markdown report title. This changes presentation only, never the metrics or checkpoint selection.",
    )
    parser.add_argument(
        "--scope",
        default="post-selection, frozen-test diagnostic only; no test value chose a checkpoint, protocol, hyperparameter, or data candidate",
        help="Machine-readable scope statement recorded in the JSON report.",
    )
    args = parser.parse_args()
    rows = []
    for root in args.input:
        rows.extend(_row(path) for path in sorted(root.glob("[A-G]_seed*.json")))
    if not rows:
        raise ValueError("No canonical protocol evaluations found")
    identities = {(row["protocol"], row["seed"]) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("Duplicate protocol/seed canonical evaluation")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda value: (str(value["protocol"]), int(value["seed"]))):
        grouped[str(row["protocol"])].append(row)
    metric_names = [key for key in rows[0] if key not in {"protocol", "seed", "checkpoint"}]
    aggregate = {
        protocol: {name: _summary([float(row[name]) for row in values]) for name in metric_names}
        for protocol, values in grouped.items()
    }
    report = {
        "schema": 1,
        "scope": args.scope,
        "canonical_contract": {
            "oracle_operator_policy": "regularized",
            "macro": "mean of per-material values",
            "micro": "one component-concatenated value retained only for aggregation audit",
            "active": "true ionic independent-Voigt norm above 0.05 C/m^2 times sqrt(18)",
        },
        "runs": rows,
        "aggregate": aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "feedback5_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "canonical_protocol_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda value: (str(value["protocol"]), int(value["seed"]))))

    def fmt(protocol: str, metric: str) -> str:
        item = aggregate[protocol][metric]
        return f"{item['mean']:.3f} +/- {item['std']:.3f}"

    lines = [
        f"# {args.title}", "",
        "All rows are post-selection evaluations on the frozen 20-material formula-disjoint test panel. "
        "The regularized true-Z*/true-Phi/predicted-Lambda oracle is the canonical ionic comparison. "
        "The micro component cosine is displayed only to expose aggregation sensitivity; it is not interchangeable with the material-balanced cosine.", "",
        "| Protocol | Full Lambda macro cosine | Oracle ionic macro cosine | Oracle ionic micro cosine | Oracle active-only macro cosine | Predicted ionic macro cosine | Total TRS | Cancellation ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for protocol in sorted(aggregate):
        lines.append(
            f"| {protocol} | {fmt(protocol, 'full_lambda_cosine_macro_material')} | "
            f"{fmt(protocol, 'oracle_ionic_cosine_macro_material')} | {fmt(protocol, 'oracle_ionic_cosine_micro_components')} | "
            f"{fmt(protocol, 'oracle_ionic_cosine_active_only')} | {fmt(protocol, 'predicted_ionic_cosine_macro_material')} | "
            f"{fmt(protocol, 'total_trs')} | {fmt(protocol, 'predicted_cancellation_ratio_macro')} |"
        )
    lines.extend([
        "", "The response-decomposition table is retained in the CSV/JSON with electronic and ionic norm-over-total ratios. "
        "A lower cancellation ratio means more cancellation between predicted branches; it is a diagnostic, not a correctness score.",
    ])
    (args.output_dir / "feedback5_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "feedback5_report.md")


if __name__ == "__main__":
    main()
