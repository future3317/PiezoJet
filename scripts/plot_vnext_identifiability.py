"""Render the vNext identifiability census figure used by the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "macro": "#315F87",
    "printed": "#C58B24",
    "joint": "#A55272",
}
LABELS = {
    "macro": "Macro ionic only",
    "printed": "Printed blocks only",
    "joint": "Macro + printed",
}


def _fractions(rows: list[dict[str, object]], key: str) -> list[float]:
    return [100.0 * float(row[key]) / int(row["materials"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    groups = summary["group_statistics"]

    crystal_order = [
        "cubic", "hexagonal", "tetragonal", "orthorhombic",
        "trigonal", "monoclinic", "triclinic",
    ]
    crystal_by_name = {row["crystal_system"]: row for row in groups["crystal_system"]}
    crystal_rows = [crystal_by_name[name] for name in crystal_order]
    atom_order = ["1-2", "3-4", "5-8", "9-16", "17+"]
    atom_by_name = {row["atom_count_bin"]: row for row in groups["atom_count_bin"]}
    atom_rows = [atom_by_name[name] for name in atom_order]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.15, 3.15), gridspec_kw={"width_ratios": [1.28, 1.0]}
    )

    y = np.arange(len(crystal_rows))
    height = 0.22
    for offset, (series, key, hatch) in zip(
        (-height, 0.0, height),
        (
            ("macro", "macro_full_identifiable", None),
            ("printed", "printed_full_identifiable", "//"),
            ("joint", "joint_full_identifiable", ".."),
        ),
        strict=True,
    ):
        left.barh(
            y + offset,
            _fractions(crystal_rows, key),
            height=height,
            color=COLORS[series],
            edgecolor="#26333D",
            linewidth=0.35,
            hatch=hatch,
            label=LABELS[series],
        )
    left.set_yticks(
        y,
        [
            f"{row['crystal_system'].capitalize()}  (n={row['materials']})"
            for row in crystal_rows
        ],
    )
    left.invert_yaxis()
    left.set_xlim(0, 103)
    left.set_xlabel("Full-rank certificates (%)")
    left.set_title("(a) Certificate rate by crystal system", loc="left", weight="bold")
    left.grid(axis="x", color="#D9DEE2", linewidth=0.5)
    left.set_axisbelow(True)
    x = np.arange(len(atom_rows))
    for series, key, marker, linestyle in (
        ("macro", "macro_full_identifiable", "o", "-"),
        ("printed", "printed_full_identifiable", "s", "--"),
        ("joint", "joint_full_identifiable", "^", ":"),
    ):
        right.plot(
            x,
            _fractions(atom_rows, key),
            marker=marker,
            markersize=4.5,
            markeredgecolor="#26333D",
            markeredgewidth=0.4,
            linewidth=1.4,
            linestyle=linestyle,
            color=COLORS[series],
            label=LABELS[series],
        )
    right.set_xticks(
        x,
        [f"{row['atom_count_bin']}\n(n={row['materials']})" for row in atom_rows],
    )
    right.set_ylim(0, 103)
    right.set_ylabel("Full-rank certificates (%)")
    right.set_xlabel("Atoms in primitive response cell")
    right.set_title("(b) Certificate rate by cell size", loc="left", weight="bold")
    right.grid(axis="y", color="#D9DEE2", linewidth=0.5)
    right.set_axisbelow(True)

    handles, labels = right.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
