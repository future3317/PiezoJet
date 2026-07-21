"""Summarize and plot the matched Stage-A N=800 architecture adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "A0 independent": "stage_a_n800_fold0_a0_independent_irreps_seed42",
    "A1 shared": "stage_a_n800_fold0_a1_electromechanical_jet_seed42",
    "A1.5 adapters": (
        "stage_a_n800_fold0_a15_soft_shared_electromechanical_jet_seed42"
    ),
}
COLORS = {
    "A0 independent": "#315F87",
    "A1 shared": "#A55272",
    "A1.5 adapters": "#C58B24",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _curve_record(run_dir: Path, label: str) -> dict[str, Any]:
    curve = _load(run_dir / "training_curve.json")
    if not curve:
        raise ValueError(f"Empty training curve: {run_dir}")
    summary_path = run_dir / "summary.json"
    summary = _load(summary_path) if summary_path.exists() else None
    progress = _load(run_dir / "progress.json")
    if summary is not None:
        selected_update = int(summary["selected_update"])
        selected = next(row for row in curve if row["update"] == selected_update)
        status = str(summary["status"])
        parameter_count = int(summary["parameter_count"])
        runtime_seconds = float(summary["runtime"]["seconds"])
        peak_mib = float(summary["runtime"]["peak_allocated_mib"])
    else:
        selected_update = int(progress["best_update"])
        selected = next(row for row in curve if row["update"] == selected_update)
        status = str(progress["status"])
        parameter_count = None
        runtime_seconds = None
        peak_mib = None
    return {
        "label": label,
        "run_directory": run_dir.as_posix(),
        "status": status,
        "complete": summary is not None and status.startswith("complete"),
        "completed_update": int(progress["completed_update"]),
        "selected_update": selected_update,
        "selection_score": float(selected["development_selection_score"]),
        "electronic_stabilized_relative": float(
            selected["development_components"]["electronic_stabilized_relative"]
        ),
        "born_stabilized_relative": float(
            selected["development_components"]["born_stabilized_relative"]
        ),
        "dielectric_stabilized_relative": float(
            selected["development_components"][
                "electronic_dielectric_stabilized_relative"
            ]
        ),
        "electronic_active_cosine": float(
            selected["development_guardrails"]["electronic_active_cosine"]
        ),
        "electronic_active_amplitude_ratio": float(
            selected["development_guardrails"][
                "electronic_active_amplitude_ratio"
            ]
        ),
        "born_nonzero_cosine": float(
            selected["development_guardrails"]["bec_nonzero_cosine"]
        ),
        "generalization_score_gap": float(selected["generalization_score_gap"]),
        "parameter_count": parameter_count,
        "runtime_seconds": runtime_seconds,
        "peak_allocated_mib": peak_mib,
        "frozen_validation_test_labels_read": False,
        "curve": curve,
    }


def _write_summary(records: list[dict[str, Any]], output: Path) -> None:
    compact = []
    for record in records:
        compact.append({key: value for key, value in record.items() if key != "curve"})
    payload = {
        "schema": 1,
        "protocol": (
            "matched N=800 fold0 seed42 Stage-A development-only architecture "
            "adjudication"
        ),
        "selection": "electrostatic_stabilized_v2; lower is better",
        "result_scope": (
            "single-seed formula-disjoint development evidence; A1.5 is an "
            "explicitly interrupted partial trajectory"
        ),
        "frozen_validation_test_labels_read": False,
        "records": compact,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _plot(records: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.8,
            "legend.fontsize": 7.2,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))

    ax = axes[0]
    for record in records:
        curve = record["curve"]
        updates = [int(row["update"]) for row in curve]
        scores = [float(row["development_selection_score"]) for row in curve]
        linestyle = "-" if record["complete"] else "--"
        marker = "o" if record["complete"] else "s"
        ax.plot(
            updates,
            scores,
            color=COLORS[record["label"]],
            linestyle=linestyle,
            marker=marker,
            markersize=3.2,
            linewidth=1.35,
            label=record["label"],
        )
        ax.annotate(
            f'{record["selection_score"]:.3f}',
            (record["selected_update"], record["selection_score"]),
            xytext=(3, -9 if record["label"] == "A1 shared" else 5),
            textcoords="offset points",
            color=COLORS[record["label"]],
            fontsize=7.1,
        )
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel(r"Development score $S_{\rm dev}$ (lower is better)")
    ax.set_title("(a) Formula-disjoint development trajectory", loc="left")
    ax.set_xlim(35, 515)
    ax.grid(axis="y", color="#D8DDE1", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.03,
        "A1.5 stopped at update 350",
        transform=ax.transAxes,
        fontsize=7.0,
        color=COLORS["A1.5 adapters"],
    )

    ax = axes[1]
    tasks = [r"$e^{\rm el}$", r"$Z^*$", r"$\epsilon_r^{\rm el}$"]
    keys = [
        "electronic_stabilized_relative",
        "born_stabilized_relative",
        "dielectric_stabilized_relative",
    ]
    x = np.arange(len(tasks))
    width = 0.24
    offsets = [-width, 0.0, width]
    for record, offset in zip(records, offsets, strict=True):
        values = [record[key] for key in keys]
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=COLORS[record["label"]],
            alpha=0.82 if record["complete"] else 0.42,
            edgecolor=COLORS[record["label"]],
            hatch=None if record["complete"] else "///",
            linewidth=0.8,
            label=record["label"],
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.3,
                rotation=90,
            )
    ax.set_xticks(x, tasks)
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("Stabilized material-relative error")
    ax.set_title("(b) Selected task decomposition", loc="left")
    ax.grid(axis="y", color="#D8DDE1", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.02,
        0.97,
        "Hatched bars: best observed partial A1.5 checkpoint",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
    )

    fig.tight_layout(w_pad=1.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    records = [
        _curve_record(args.cohort / directory, label)
        for label, directory in RUNS.items()
    ]
    _write_summary(records, args.summary)
    _plot(records, args.output)


if __name__ == "__main__":
    main()
