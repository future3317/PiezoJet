"""Freeze a formula-disjoint strict-completion benchmark split.

The panel is created once from the current audited cache and is intentionally
not regenerated when later completion batches arrive.  It is therefore a
proper regression/generalization panel rather than a moving subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .completion_forensics import _negative_optical_modes
from .data import _raw_cartesian_target, formula, load_gmtnet_records, response_norm_bin
from .jarvis_dfpt import JarvisDFPTCache
from .strain_completion import _structure


def _stable_key(seed: int, value: str) -> int:
    return int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:16], 16)


def _features(record: dict, payload: dict) -> dict:
    analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
    operations = analyzer.get_symmetry_operations(cartesian=True)
    projector = torch.tensor([operation.rotation_matrix for operation in operations], dtype=torch.float64).mean(dim=0)
    return {
        "formula": formula(record), "response_bin": response_norm_bin(_raw_cartesian_target(record)),
        "crystal_system": str(analyzer.get_crystal_system()),
        "polar": bool(torch.linalg.matrix_rank(projector, tol=1e-7) > 0),
        "negative_optical_mode": _negative_optical_modes(payload),
        "atoms": len(record["atoms"]["elements"]),
    }


def _select_groups(groups: dict[str, list[dict]], count: int, seed: int) -> list[str]:
    """Greedily cover response/system/polar/soft strata without formula leakage."""
    selected: list[str] = []
    selected_materials: list[dict] = []
    remaining = set(groups)
    while len(selected_materials) < count:
        fitting = [key for key in remaining if len(selected_materials) + len(groups[key]) <= count]
        if not fitting:
            break
        response_counts = Counter(item["response_bin"] for item in selected_materials)
        systems = Counter(item["crystal_system"] for item in selected_materials)
        polar = Counter(item["polar"] for item in selected_materials)
        soft = Counter(item["negative_optical_mode"] for item in selected_materials)
        def score(key: str) -> tuple:
            members = groups[key]
            diversity = sum(
                1.0 / (1 + response_counts[item["response_bin"]])
                + 0.35 / (1 + systems[item["crystal_system"]])
                + 0.15 / (1 + polar[item["polar"]])
                + 0.15 / (1 + soft[item["negative_optical_mode"]])
                for item in members
            )
            # Do not let a repeated formula group consume the entire panel.
            return (diversity / len(members), -len(members), -_stable_key(seed, key))
        choice = max(fitting, key=score)
        selected.append(choice)
        selected_materials.extend(groups[choice])
        remaining.remove(choice)
    if len(selected_materials) != count:
        raise RuntimeError(f"Could not exactly construct a {count}-material formula-disjoint panel")
    return selected


def _describe(items: list[dict]) -> dict:
    return {
        "materials": len(items), "formulas": len({item["formula"] for item in items}),
        "response_bins": dict(Counter(str(item["response_bin"]) for item in items)),
        "crystal_systems": dict(Counter(item["crystal_system"] for item in items)),
        "polar": sum(bool(item["polar"]) for item in items),
        "negative_optical_modes": sum(bool(item["negative_optical_mode"]) for item in items),
        "atom_count_min_max": [min(item["atoms"] for item in items), max(item["atoms"] for item in items)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--dfpt-dir", type=Path, default=Path("data/processed/jarvis_dfpt_v1"))
    parser.add_argument("--completion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/strict_completion_benchmark_v1.json"))
    parser.add_argument("--test-materials", type=int, default=20)
    parser.add_argument("--val-materials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen benchmark: {args.output}")
    manifest = json.loads((args.completion_dir / "manifest.json").read_text(encoding="utf-8"))
    accepted = set(map(str, manifest["material_ids"]))
    records = {str(item["JARVIS_ID"]): item for item in load_gmtnet_records(args.data_root)}
    cache = JarvisDFPTCache(args.dfpt_dir)
    items = []
    for jid in sorted(accepted):
        payload = cache.load(jid)
        if payload is None:
            raise FileNotFoundError(f"Missing DFPT payload for strict completion {jid}")
        item = {"jid": jid, **_features(records[jid], payload)}
        items.append(item)
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item["formula"]].append(item)
    test_groups = _select_groups(groups, args.test_materials, args.seed)
    remaining = {key: value for key, value in groups.items() if key not in test_groups}
    val_groups = _select_groups(remaining, args.val_materials, args.seed + 1)
    test = sorted(item["jid"] for key in test_groups for item in groups[key])
    val = sorted(item["jid"] for key in val_groups for item in groups[key])
    train = sorted(accepted - set(test) - set(val))
    report = {
        "schema": 1,
        "frozen": True,
        "policy": "formula-disjoint, deterministic stratified panel; do not reassign future strict completions into test or validation",
        "seed": args.seed, "source_completion_manifest": str(args.completion_dir / "manifest.json"),
        "splits": {"train": train, "val": val, "test": test},
        "summary": {"train": _describe([item for item in items if item["jid"] in set(train)]), "val": _describe([item for item in items if item["jid"] in set(val)]), "test": _describe([item for item in items if item["jid"] in set(test)])},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
