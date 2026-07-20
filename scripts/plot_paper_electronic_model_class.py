"""Render the samples32 electronic model-class capacity diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#315F87"
GOLD = "#C58B24"
ROSE = "#A55272"
INK = "#24313A"
GRAY = "#7A858D"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cartesian", type=Path, required=True)
    parser.add_argument("--global-irrep", type=Path, required=True)
    parser.add_argument("--first-order", type=Path, required=True)
    parser.add_argument("--nonlinear", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = ["Cartesian head", r"Global irreps $\ell\leq3$", "First-order jet", r"Nonlinear $\Delta P$"]
    payloads = [
        _load(args.cartesian), _load(args.global_irrep),
        _load(args.first_order), _load(args.nonlinear),
    ]
    metrics = [payload["metrics"] for payload in payloads]
    active_error = [float(metric["mean_active_relative_frobenius_error"]) for metric in metrics]
    active_cosine = [float(metric["mean_active_cosine"]) for metric in metrics]
    irreps = ["l1_copy0", "l1_copy1", "l2", "l3"]
    irrep_labels = [r"$\ell=1_a$", r"$\ell=1_b$", r"$\ell=2$", r"$\ell=3$"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 6.4,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.15, 3.0), gridspec_kw={"width_ratios": [0.92, 1.18]})

    y = np.arange(len(labels))
    colors = [ROSE, GOLD, BLUE, "#6F7F48"]
    left.scatter(active_error, y, s=45, color=colors, edgecolor=INK, linewidth=0.5, zorder=3)
    for value, cosine, row in zip(active_error, active_cosine, y, strict=True):
        left.text(value * 1.12, row, f"{value:.3f}  (cos {cosine:.3f})", va="center", fontsize=6.4, color=INK)
    left.set_xscale("log")
    left.set_xlim(0.025, 1.05)
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_xlabel("Active relative Frobenius error (log scale)")
    left.set_title("(a) Full electronic tensor", loc="left", weight="bold")
    left.grid(axis="x", color="#D9DEE2", linewidth=0.5, which="both")
    left.set_axisbelow(True)

    x = np.arange(len(irreps))
    offsets = np.linspace(-0.24, 0.24, len(payloads))
    markers = ["o", "s", "D", "^"]
    for label, metric, color, marker, offset in zip(labels, metrics, colors, markers, offsets, strict=True):
        values = [float(metric["per_irrep"][name]["mean_stabilized_relative_error"]) for name in irreps]
        right.scatter(
            x + offset, values, s=28, marker=marker, color=color,
            edgecolor=INK, linewidth=0.4, label=label, zorder=3,
        )
        right.plot(x + offset, values, color=color, linewidth=0.65, alpha=0.75, zorder=2)
    right.set_yscale("log")
    right.set_ylim(6e-5, 1.0)
    right.set_xticks(x, irrep_labels)
    right.set_ylabel("Stabilized relative error (log scale)")
    right.set_title("(b) Irrep-resolved representation floor", loc="left", weight="bold")
    right.grid(axis="y", color="#D9DEE2", linewidth=0.5, which="both")
    right.set_axisbelow(True)
    handles, legend_labels = right.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, frameon=False, loc="lower center",
        bbox_to_anchor=(0.67, -0.005), ncol=2,
        handletextpad=0.4, columnspacing=0.8,
    )
    right.annotate(
        r"Cartesian failure is concentrated in $\ell=3$",
        xy=(3 + offsets[0], float(metrics[0]["per_irrep"]["l3"]["mean_stabilized_relative_error"])),
        xytext=(1.15, 0.16), textcoords="data", fontsize=6.2, color=GRAY,
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.8},
    )

    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
