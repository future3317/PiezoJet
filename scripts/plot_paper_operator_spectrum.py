"""Render the signed-resolvent policy and first-order solve used by PiezoJet."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


BLUE = "#315F87"
GOLD = "#C58B24"
INK = "#24313A"
GRAY = "#7A858D"


def _box(axis: plt.Axes, xy: tuple[float, float], width: float, height: float,
         title: str, body: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=f"{color}18", edgecolor=color, linewidth=1.1,
        transform=axis.transAxes,
    )
    axis.add_patch(patch)
    axis.text(xy[0] + 0.03, xy[1] + height - 0.08, title,
              transform=axis.transAxes, fontsize=7.5, weight="bold", color=INK, va="top")
    axis.text(xy[0] + 0.03, xy[1] + height - 0.17, body,
              transform=axis.transAxes, fontsize=6.8, color=INK, va="top", linespacing=1.35)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.15, 2.8))

    x = np.linspace(-5.0, 5.0, 1601)
    regularized = x / (x * x + 1.0)
    left.axvspan(-1, 1, color="#E9EEF1", zorder=0)
    left.plot(x, regularized, color=BLUE, linewidth=1.7,
              label=r"production: $\delta\mathcal{D}_\delta$")
    positive = np.linspace(0.22, 5.0, 900)
    exact = np.clip(1.0 / positive, -1.4, 1.4)
    left.plot(positive, exact, color=GOLD, linestyle="--", linewidth=1.3,
              label=r"true-stable diagnostic: $\delta\mathcal{O}_0$")
    left.axhline(0, color=INK, linewidth=0.6)
    left.axvline(0, color=INK, linewidth=0.6)
    left.axvline(-1, color=GRAY, linewidth=0.6, linestyle=":")
    left.axvline(1, color=GRAY, linewidth=0.6, linestyle=":")
    left.text(0, -0.72, r"soft window $|\lambda|<\delta$", ha="center", fontsize=6.2, color=GRAY)
    left.set_xlim(-5, 5)
    left.set_ylim(-0.78, 1.42)
    left.set_xlabel(r"Normalized optical eigenvalue $\lambda/\delta$")
    left.set_ylabel("Normalized response")
    left.set_title("(a) One continuous signed spectral filter", loc="left", weight="bold")
    left.grid(color="#D9DEE2", linewidth=0.45)
    left.set_axisbelow(True)
    left.legend(frameon=False, loc="upper left", fontsize=6.4)

    right.axis("off")
    right.set_title("(b) Production solve and diagnostic boundary", loc="left", weight="bold")
    _box(
        right, (0.02, 0.57), 0.43, 0.30, "First-order U/V solve",
        r"$\Phi U-\delta V=\Lambda$" "\n" r"$\Phi V+\delta U=0$",
        BLUE,
    )
    _box(
        right, (0.55, 0.57), 0.42, 0.30, "Production response",
        r"$U_{\eta,\delta}=\mathcal{D}_\delta(\Phi)\Lambda$" "\n"
        "all true spectral strata",
        BLUE,
    )
    _box(
        right, (0.55, 0.10), 0.42, 0.28, "Stable exact diagnostic",
        r"$U_{\eta,\rm stat}=\mathcal{O}_0(\Phi)\Lambda$" "\n"
        "true-DFPT stable only",
        GOLD,
    )
    right.annotate(
        "", xy=(0.55, 0.72), xytext=(0.45, 0.72), xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.3},
    )
    right.annotate(
        "explicit diagnostic; never predicted-spectrum switching",
        xy=(0.76, 0.40), xytext=(0.50, 0.49), xycoords="axes fraction",
        ha="center", fontsize=6.2, color=GOLD,
        arrowprops={"arrowstyle": "-|>", "color": GOLD, "lw": 1.0, "linestyle": "--"},
    )
    right.text(
        0.02, 0.02, "Translations are projected out; no dense inverse is materialized.",
        transform=right.transAxes, fontsize=6.4, color=GRAY,
    )

    fig.tight_layout(w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
