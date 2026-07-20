"""Render direct-U capacity and matched validation-boundary evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#315F87"
ROSE = "#A55272"
INK = "#24313A"
GRAY = "#7A858D"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capacity = json.loads(args.capacity.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    metrics = capacity["metrics"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.15, 3.05))

    labels = [
        "$U$ relative error",
        "$U$ cosine",
        "Active ionic cosine",
        "Active ionic amplitude",
    ]
    values = [
        metrics["u_relative_frobenius_error"],
        metrics["u_cosine"],
        metrics["active_true_bec_ionic_cosine"],
        metrics["active_true_bec_ionic_amplitude_ratio"],
    ]
    acceptable = [(0.0, 0.2), (0.95, 1.05), (0.95, 1.05), (0.8, 1.2)]
    y = np.arange(len(labels))
    for index, (low, high) in enumerate(acceptable):
        left.plot([low, high], [index, index], color="#C7D5DF", linewidth=8,
                  solid_capstyle="round", zorder=1)
        left.plot([low, high], [index, index], color=BLUE, linewidth=1.0,
                  solid_capstyle="round", zorder=2)
    left.scatter(values, y, s=42, color=ROSE, edgecolor=INK, linewidth=0.5, zorder=3)
    for value, row in zip(values, y, strict=True):
        left.text(value + 0.025, row, f"{value:.3f}", va="center", fontsize=7, color=INK)
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_xlim(-0.02, 1.27)
    left.set_xlabel("Metric value; blue interval is the preregistered gate")
    left.set_title("(a) Same-ID direct-$U$ capacity (N=32)", loc="left", weight="bold")
    left.grid(axis="x", color="#D9DEE2", linewidth=0.5)
    left.set_axisbelow(True)

    seeds = ["42", "7", "1729"]
    physical = [validation["physical_individual"][seed]["total_trs"] for seed in seeds]
    direct = [validation["direct_individual"][seed]["total_trs"] for seed in seeds]
    for seed, first, second in zip(seeds, physical, direct, strict=True):
        right.plot([0, 1], [first, second], color="#9DA8AF", linewidth=1.0, zorder=1)
        right.scatter([0], [first], color=ROSE, edgecolor=INK, linewidth=0.4, s=35, zorder=2)
        right.scatter([1], [second], color=BLUE, edgecolor=INK, linewidth=0.4, s=35, zorder=2)
        right.text(1.05, second, f"seed {seed}", fontsize=6.6, va="center", color=GRAY)
    means = [np.mean(physical), np.mean(direct)]
    sds = [np.std(physical, ddof=1), np.std(direct, ddof=1)]
    right.errorbar(
        [0, 1], means, yerr=sds, fmt="D", color=INK, markerfacecolor="white",
        markersize=5, capsize=3, linewidth=1.2, zorder=4, label="mean $\pm$ sample SD",
    )
    right.set_xticks([0, 1], ["Physical macro", "Matched direct"])
    right.set_xlim(-0.35, 1.55)
    right.set_ylim(0.15, 0.52)
    right.set_ylabel("Historical val10 tensor-response skill vs zero")
    right.set_title("(b) Legacy split total-response diagnostic", loc="left", weight="bold")
    right.grid(axis="y", color="#D9DEE2", linewidth=0.5)
    right.set_axisbelow(True)
    right.legend(frameon=False, loc="lower right", fontsize=6.7)

    fig.tight_layout(w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
