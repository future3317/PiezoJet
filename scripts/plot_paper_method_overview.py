"""Render the paper's vector overview of maintained PiezoJet pathways."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#24313A"
BLUE = "#315F87"
GOLD = "#C58B24"
ROSE = "#A55272"
PALE_BLUE = "#EAF1F7"
PALE_GOLD = "#FBF2DF"
PALE_ROSE = "#F8EAF0"
PALE_GRAY = "#F1F3F4"


def box(ax, xy, width, height, title, body, face, edge=INK, linewidth=1.0):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + 0.04, xy[1] + height - 0.09, title,
        ha="left", va="top", fontsize=8.2, fontweight="bold", color=INK,
    )
    ax.text(
        xy[0] + 0.04, xy[1] + height - 0.31, body,
        ha="left", va="top", fontsize=6.8, color=INK, linespacing=1.2,
    )
    return patch


def arrow(ax, start, end, color=INK, linestyle="-", width=1.2, zorder=2):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=width, color=color, linestyle=linestyle,
        shrinkA=2, shrinkB=2, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    box(
        ax, (0.15, 2.55), 1.25, 1.15,
        "Structure $x$", r"$\{Z_\kappa,r_\kappa,L\}$" + "\ncomplete-shell PBC graph",
        PALE_GRAY,
    )
    box(
        ax, (1.85, 4.45), 2.25, 1.35,
        "Mechanical factor",
        r"$\frac{1}{2}u^T\Phi u-u^T\Lambda\eta$" + "\n" + r"independent $\Phi,\Lambda$",
        PALE_GOLD, GOLD,
    )
    box(
        ax, (1.85, 2.55), 2.25, 1.35,
        "Electrostatic jet",
        r"$Z^*,\ e^{\rm el},\ \epsilon_r^{\rm el}$" + "\nA1 shared / A1.5 adapters",
        PALE_BLUE, BLUE,
    )
    box(
        ax, (1.85, 0.65), 2.25, 1.35,
        "Direct-$U$ coordinate",
        r"global-$\ell=3$ $U_{\eta,\delta}$" + "\ntranslation-free atom head",
        PALE_ROSE, ROSE,
    )
    box(
        ax, (4.7, 4.55), 2.15, 1.15,
        "$\Lambda$ identifiability",
        r"$M_{\rm macro},M_{\rm joint}$" + "\n" + r"$P_{\rm obs}$ / $P_{\rm null}$ label routing",
        PALE_GRAY,
    )
    box(
        ax, (4.7, 3.0), 2.15, 1.05,
        "Factorized diagnostic",
        r"$U_{\Phi\Lambda}=D_\delta(\Phi)\Lambda$" + "\n" + r"first-order $U/V$ residual",
        PALE_GOLD, GOLD,
    )
    box(
        ax, (4.7, 1.25), 2.15, 1.05,
        "Ionic contraction",
        r"$e_U^{\rm ion}=\frac{c_e}{\Omega}Z^{*T}U_{\eta,\delta}$",
        PALE_ROSE, ROSE,
    )
    box(
        ax, (7.45, 2.15), 2.25, 1.35,
        "Production response",
        r"$\widehat e^{\rm phys}=\widehat e^{\rm el}+\widehat e_U^{\rm ion}$" + "\ndirect-$U$ ionic path",
        "#E8F3F0", "#2D7465", linewidth=1.4,
    )
    box(
        ax, (7.45, 0.35), 2.25, 1.05,
        "Isolated macro",
        "total-only supervision\nno gradient to physical branches",
        PALE_GRAY, "#7A858D",
    )

    arrow(ax, (1.4, 3.3), (1.85, 5.1), GOLD)
    arrow(ax, (1.4, 3.15), (1.85, 3.25), BLUE)
    arrow(ax, (1.4, 2.95), (1.85, 1.35), ROSE)
    arrow(ax, (1.4, 2.75), (7.45, 0.88), "#7A858D", linestyle=":")
    arrow(ax, (4.1, 5.1), (4.7, 5.1), GOLD)
    arrow(ax, (4.1, 4.82), (4.7, 3.62), GOLD, linestyle="--")
    arrow(ax, (4.1, 3.15), (5.25, 2.3), BLUE)
    arrow(ax, (4.1, 1.35), (4.7, 1.78), ROSE)
    arrow(ax, (6.85, 1.78), (7.45, 2.65), ROSE)
    arrow(ax, (6.3, 3.0), (7.45, 3.15), GOLD, linestyle="--")
    arrow(ax, (4.1, 3.35), (7.45, 3.05), BLUE)

    ax.text(5.45, 2.48, "$Z^*$", color=BLUE, fontsize=7.5, ha="center")
    ax.text(6.95, 3.15, "diagnostic only", color=GOLD, fontsize=7, ha="center")
    ax.text(1.9, 0.18, "Solid: production", color=INK, fontsize=6.6)
    ax.text(4.35, 0.18, "Dashed: diagnostic", color=GOLD, fontsize=6.6)
    ax.text(7.0, 0.18, "Dotted: isolated", color="#7A858D", fontsize=6.6)

    fig.tight_layout(pad=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
