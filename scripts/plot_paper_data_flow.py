"""Render the audited PiezoJet data and label-routing overview."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#24313A"
BLUE = "#315F87"
GOLD = "#C58B24"
ROSE = "#A55272"
GRAY = "#7A858D"


def node(ax, x, y, w, h, title, body, face, edge=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=face, edgecolor=edge, linewidth=1.0,
    ))
    ax.text(x + 0.05, y + h - 0.09, title, ha="left", va="top",
            fontsize=8.1, fontweight="bold", color=INK)
    ax.text(x + 0.05, y + h - 0.31, body, ha="left", va="top",
            fontsize=6.7, color=INK, linespacing=1.18)


def arrow(ax, start, end, color=INK, style="-"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=1.15, color=color, linestyle=style, shrinkA=2, shrinkB=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    node(ax, 0.2, 4.65, 1.75, 1.15, "JARVIS DFPT raw", "4,998 requested archives\nSHA256-indexed", "#F1F3F4")
    node(ax, 2.35, 4.65, 1.85, 1.15, "Parsed payloads", "4,995 schema-4 records\nsource cells retained", "#EAF1F7", BLUE)
    node(ax, 2.35, 3.05, 1.85, 0.95, "Quarantine", "3 raw ZIPs\nno fabricated labels", "#F7ECEC", "#9A5B5B")

    node(ax, 4.75, 4.7, 2.1, 1.05, "Electrostatic-ready", "4,995 pass finite/shape gates\n$Z^*,e^{el},\epsilon^{el},\Phi$", "#EAF1F7", BLUE)
    node(ax, 7.35, 4.7, 2.3, 1.05, "Development folds", "4,939 reduced-formula-safe\n988 / 989 / 987 / 988 / 987", "#E5F0F7", BLUE)

    node(ax, 4.75, 2.75, 2.1, 1.05, "Strict $\Lambda$ audit", "4,995 parsed records\nunchanged completion gates", "#FBF2DF", GOLD)
    node(ax, 7.35, 2.75, 2.3, 1.05, "Strict-complete", "1,638 accepted\nselection-biased, not uniform", "#FBF2DF", GOLD)
    node(ax, 7.35, 0.85, 2.3, 1.25, "Frozen-safe split", "train 1,595  |  val 10  |  test 20\n13 held-out-formula exclusions\nval/test share reduced formula HNaO", "#F8EAF0", ROSE)

    node(ax, 0.2, 1.55, 1.75, 1.15, "GMTNet piezo", "4,998 total tensors\nbenchmark-pinned structures", "#F8EAF0", ROSE)
    node(ax, 2.35, 1.55, 1.85, 1.15, "Macro train pool", "4,944 formula-safe totals\nphysical factors masked", "#F8EAF0", ROSE)
    node(ax, 0.2, 0.15, 4.0, 0.8, "Official dft_3d auxiliary", "75,993 unique JIDs; metadata/structures only, never response-label replacement", "#F1F3F4", GRAY)

    arrow(ax, (1.95, 5.22), (2.35, 5.22), BLUE)
    arrow(ax, (3.25, 4.65), (3.25, 4.0), "#9A5B5B")
    arrow(ax, (4.2, 5.22), (4.75, 5.22), BLUE)
    arrow(ax, (6.85, 5.22), (7.35, 5.22), BLUE)
    arrow(ax, (4.2, 4.92), (4.75, 3.3), GOLD)
    arrow(ax, (6.85, 3.3), (7.35, 3.3), GOLD)
    arrow(ax, (8.5, 2.75), (8.5, 2.1), ROSE)
    arrow(ax, (1.95, 2.12), (2.35, 2.12), ROSE)

    ax.text(5.8, 0.25, "Frozen test20 response labels remain unread during development",
            fontsize=7.2, color="#4C5963", ha="left")
    fig.tight_layout(pad=0.15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
