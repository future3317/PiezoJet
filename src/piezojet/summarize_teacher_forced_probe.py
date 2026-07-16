"""Summarize the explicitly noninductive teacher-forced capacity probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_PATHS = {
    "born_charge_cosine": ("factors", "born_charge", "macro_material_directional_cosine"),
    "force_constant_cosine": ("factors", "force_constant", "macro_material_directional_cosine"),
    "completed_lambda_cosine": ("factors", "internal_strain_full", "macro_material_directional_cosine"),
    "regularized_displacement_cosine": (
        "strict_symmetry_completed_lambda_oracle",
        "displacement_response_target",
        "macro_material_directional_cosine",
    ),
    "true_bec_ionic_cosine": ("ionic_response_aggregation", "ionic_cosine_macro_material"),
}


def _value(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return float(value)


def summarize(root: Path, threshold: float = 0.99) -> dict[str, Any]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    probes: dict[str, dict[str, Any]] = {}
    for label, expected_count in (("samples1", 1), ("samples8", 8), ("samples32", 32)):
        path = root / label / "overfit_dfpt_train.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if bool(payload.get("formula_disjoint", True)):
            raise ValueError(f"{path} is not explicitly marked as same-ID diagnostic")
        metrics = {name: _value(payload, metric_path) for name, metric_path in METRIC_PATHS.items()}
        material_count = int(payload["material_count"])
        if material_count != expected_count:
            raise ValueError(
                f"{path} reports {material_count} materials; expected {expected_count} for {label}"
            )
        probes[label] = {
            "path": str(path),
            "material_count": material_count,
            "metrics": metrics,
            "all_metrics_above_threshold": all(value >= threshold for value in metrics.values()),
            "interpretation_boundary": payload.get("interpretation_boundary"),
        }
    return {
        "schema": 1,
        "threshold": threshold,
        "probes": probes,
        "capacity_probe_passes": all(item["all_metrics_above_threshold"] for item in probes.values()),
        "interpretation": (
            "This is a same-ID capacity/optimization falsification test. A pass does not establish "
            "formula-disjoint generalization; a failure motivates optimization or model-class diagnosis "
            "before any data-expansion decision."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.99)
    args = parser.parse_args()
    result = summarize(args.root, args.threshold)
    output = args.output or args.root / "capacity_probe_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
