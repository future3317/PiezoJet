"""Summarize matched 1/8/32 same-ID operator-learning capacity runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FACTORS = (
    "born_charge", "force_constant", "internal_strain_full",
    "displacement_response", "ionic_piezo", "direct_u_ionic_piezo",
    "dielectric", "ionic_dielectric", "macro_elastic",
)


def _factor_row(payload: dict[str, Any], name: str) -> dict[str, float] | None:
    value = payload.get("factors", {}).get(name)
    if value is None:
        return None
    return {
        "relative_frobenius_error": float(value["macro_material_stabilized_relative_frobenius_error"]),
        "cosine": float(value["macro_material_directional_cosine"]),
        "amplitude_ratio": float(value["macro_material_stabilized_amplitude_ratio"]),
        "component_mae": float(value["macro_material_component_mae"]),
    }


def summarize(root: Path) -> dict[str, Any]:
    probes: dict[str, Any] = {}
    for count in (1, 8, 32):
        rows = {}
        for variant in ("baseline", "operator"):
            path = root / variant / f"samples{count}" / "overfit_dfpt_train.json"
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete capacity run: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows[variant] = {
                "artifact": str(path),
                "factors": {
                    name: value for name in FACTORS
                    if (value := _factor_row(payload, name)) is not None
                },
            }
        deltas = {}
        common = set(rows["baseline"]["factors"]) & set(rows["operator"]["factors"])
        for factor in sorted(common):
            baseline = rows["baseline"]["factors"][factor]
            operator = rows["operator"]["factors"][factor]
            deltas[factor] = {
                "relative_error_reduction_fraction": (
                    (baseline["relative_frobenius_error"] - operator["relative_frobenius_error"])
                    / max(baseline["relative_frobenius_error"], 1e-30)
                ),
                "cosine_change": operator["cosine"] - baseline["cosine"],
                "amplitude_ratio_change": operator["amplitude_ratio"] - baseline["amplitude_ratio"],
            }
        probes[f"samples{count}"] = {**rows, "paired_summary_delta": deltas}
    return {
        "schema": 1,
        "diagnostic": "matched_same_id_operator_learning_capacity",
        "selection": "explicit strict-train 1/8/32 IDs; no frozen validation/test read",
        "interpretation": (
            "Capacity/optimization evidence only. Positive same-ID deltas do not establish "
            "formula-disjoint predictive validity or authorize a production promotion."
        ),
        "probes": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Operator-learning same-ID capacity", "",
        "This report reads only explicit strict-train 1/8/32 cohorts. It is not a generalization result.", "",
        "| Samples | Factor | Baseline rel. error | Operator rel. error | Reduction | Cosine change |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for probe_name, probe in payload["probes"].items():
        count = probe_name.removeprefix("samples")
        for factor, delta in probe["paired_summary_delta"].items():
            baseline = probe["baseline"]["factors"][factor]
            operator = probe["operator"]["factors"][factor]
            lines.append(
                f"| {count} | {factor} | {baseline['relative_frobenius_error']:.5f} | "
                f"{operator['relative_frobenius_error']:.5f} | "
                f"{delta['relative_error_reduction_fraction']:.1%} | {delta['cosine_change']:+.5f} |"
            )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
