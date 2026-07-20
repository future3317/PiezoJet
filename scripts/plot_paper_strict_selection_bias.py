"""Render strict-Lambda completion selection-bias panels for the paper."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#315F87"
ROSE = "#A55272"
INK = "#24313A"
GRID = "#D9DEE2"


def _group(rows: list[dict[str, object]], key: str) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        value = str(row[key])
        counts[value][0] += 1
        counts[value][1] += int(bool(row["strict_complete"]))
    return {name: (values[0], values[1]) for name, values in counts.items()}


def _rate(entry: tuple[int, int]) -> float:
    total, accepted = entry
    return 100.0 * accepted / total


def _annotate_bars(axis: plt.Axes, bars: object, entries: list[tuple[int, int]]) -> None:
    for bar, (total, accepted) in zip(bars, entries, strict=True):
        axis.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{accepted}/{total}",
            va="center",
            fontsize=6.3,
            color=INK,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = payload["rows"]
    overall = 100.0 * sum(bool(row["strict_complete"]) for row in rows) / 4995.0

    crystal = _group(rows, "crystal_system")
    atoms = _group(rows, "atom_count_bin")
    response = _group(rows, "gmtnet_response_bin")
    crystal_order = [
        "cubic", "hexagonal", "tetragonal", "orthorhombic",
        "trigonal", "monoclinic", "triclinic",
    ]
    atom_order = ["1-2", "3-4", "5-8", "9-16", "17+"]
    response_order = ["0", "1", "2", "3", "4"]
    response_labels = ["exact 0", "(0,.05)", "[.05,.5)", "[.5,1)", "1+"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(7.15, 2.95),
        gridspec_kw={"width_ratios": [1.28, 0.90, 1.0]},
    )

    crystal_entries = [crystal[name] for name in crystal_order]
    y = np.arange(len(crystal_order))
    colors = [ROSE if name == "trigonal" else BLUE for name in crystal_order]
    bars = axes[0].barh(
        y, [_rate(entry) for entry in crystal_entries], color=colors,
        edgecolor=INK, linewidth=0.35,
    )
    bars[crystal_order.index("trigonal")].set_hatch("//")
    axes[0].set_yticks(y, [name.capitalize() for name in crystal_order])
    axes[0].invert_yaxis()
    axes[0].set_title("(a) Crystal system", loc="left", weight="bold")
    axes[0].set_xlabel("Strict-complete rate (%)")
    _annotate_bars(axes[0], bars, crystal_entries)

    atom_entries = [atoms[name] for name in atom_order]
    x = np.arange(len(atom_order))
    colors = [ROSE if name == "1-2" else BLUE for name in atom_order]
    bars = axes[1].bar(
        x, [_rate(entry) for entry in atom_entries], color=colors,
        edgecolor=INK, linewidth=0.35,
    )
    bars[0].set_hatch("//")
    axes[1].set_xticks(x, atom_order)
    axes[1].set_title("(b) Atom count", loc="left", weight="bold")
    axes[1].set_xlabel("Atoms in response cell")
    for bar, (total, accepted) in zip(bars, atom_entries, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{accepted}/{total}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=5.8,
            color=INK,
        )

    response_entries = [response[name] for name in response_order]
    x = np.arange(len(response_order))
    bars = axes[2].bar(
        x, [_rate(entry) for entry in response_entries], color=BLUE,
        edgecolor=INK, linewidth=0.35,
    )
    axes[2].set_xticks(x, response_labels, rotation=28, ha="right")
    axes[2].set_title("(c) Total-response norm", loc="left", weight="bold")
    axes[2].set_xlabel(r"$\|e^{\rm total}\|_F$ (C/m$^2$)")
    for bar, (total, accepted) in zip(bars, response_entries, strict=True):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"n={total}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=5.7,
            color=INK,
        )

    for axis in axes:
        axis.set_axisbelow(True)
    axes[0].axvline(overall, color=INK, linestyle="--", linewidth=0.8)
    axes[0].set_xlim(0, 66)
    axes[0].grid(axis="x", color=GRID, linewidth=0.5)
    axes[0].text(overall + 1.0, 6.65, "overall 32.8%", ha="left", fontsize=6.2, color=INK)
    for axis in axes[1:]:
        axis.axhline(overall, color=INK, linestyle="--", linewidth=0.8)
        axis.set_ylim(0, 63)
        axis.grid(axis="y", color=GRID, linewidth=0.5)
        axis.text(4.42, overall + 1.0, "overall", ha="right", fontsize=6.0, color=INK)

    fig.tight_layout(w_pad=1.7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
