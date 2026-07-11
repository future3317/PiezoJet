"""Persistent composition and coarse prototype OOD splits without data leakage."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import spglib
from pymatgen.core import Element

from .data import create_or_load_splits, load_gmtnet_records


def reduced_formula(record) -> str:
    counts = Counter(record["atoms"]["elements"])
    divisor = math.gcd(*counts.values())
    return "".join(f"{element}{counts[element] // divisor if counts[element] // divisor != 1 else ''}" for element in sorted(counts))


def chemical_system(record) -> str:
    return "-".join(sorted(set(record["atoms"]["elements"])))


def prototype_key(record) -> str:
    atoms = record["atoms"]
    cell = (atoms["lattice_mat"], atoms["coords"], [Element(item).Z for item in atoms["elements"]])
    symmetry = spglib.get_symmetry_dataset(cell, symprec=1e-5)
    number = int(symmetry["number"]) if symmetry is not None else 0
    return f"{reduced_formula(record)}|sg{number}|n{len(atoms['elements'])}"


def grouped_split(records, key_fn, seed: int = 42) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(str(record["JARVIS_ID"]))
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    targets = {"train": int(len(records) * 0.8), "val": int(len(records) * 0.1), "test": len(records) - int(len(records) * 0.8) - int(len(records) * 0.1)}
    splits = {"train": [], "val": [], "test": []}
    for _, ids in ordered:
        destination = min(splits, key=lambda name: (max(targets[name] - len(splits[name]), 0), name))
        splits[destination].extend(sorted(ids))
    return splits


def validate_no_overlap(splits: dict[str, list[str]], records, key_fn) -> dict:
    by_id = {str(record["JARVIS_ID"]): record for record in records}
    grouped = {name: {key_fn(by_id[item]) for item in ids} for name, ids in splits.items()}
    return {f"{left}_{right}": sorted(grouped[left] & grouped[right]) for left, right in (("train", "val"), ("train", "test"), ("val", "test"))}


def write_split(records, output: Path, name: str, key_fn) -> None:
    splits = grouped_split(records, key_fn)
    output.mkdir(parents=True, exist_ok=True)
    (output / "split.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    overlap = validate_no_overlap(splits, records, key_fn)
    (output / "overlap_check.json").write_text(json.dumps(overlap, indent=2) + "\n", encoding="utf-8")
    counts = {split: len(ids) for split, ids in splits.items()}
    (output / "statistics.json").write_text(json.dumps({"counts": counts, "groups": len({key_fn(record) for record in records})}, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(f"# {name} OOD split\n\nGroups are assigned deterministically and never cross train/val/test.\n", encoding="utf-8")
    if any(overlap.values()):
        raise RuntimeError(f"OOD overlap detected for {name}: {overlap}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_gmtnet_records(args.data_root)
    write_split(records, args.output / "formula", "formula", reduced_formula)
    write_split(records, args.output / "chemical_system", "chemical-system", chemical_system)
    write_split(records, args.output / "prototype", "prototype", prototype_key)
    print(f"OOD splits written to {args.output}")


if __name__ == "__main__":
    main()
