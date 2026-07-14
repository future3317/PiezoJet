"""Create diverse, response-stratified 35-label subsets of frozen training IDs.

The frozen validation and test panels are copied unchanged.  These subsets are
diagnostics for the *composition* uncertainty hidden by one nested learning
curve, not candidates for changing the benchmark split or selecting a model.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .data import _raw_cartesian_target, is_polar_point_group, load_gmtnet_records, response_norm_bin
from .jarvis_dfpt import JarvisDFPTCache
from .strain_completion import _structure


def apportion(total: int, counts: Counter) -> dict[str, int]:
    """Largest-remainder proportional quotas that exactly sum to ``total``."""
    population = sum(counts.values())
    if total < 1 or population < total:
        raise ValueError("Subset size must lie in [1, population]")
    raw = {key: total * value / population for key, value in counts.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    for key, _ in sorted(raw.items(), key=lambda item: (item[1] - int(item[1]), item[0]), reverse=True)[: total - sum(quotas.values())]:
        quotas[key] += 1
    if sum(quotas.values()) != total:
        raise RuntimeError("Largest-remainder apportionment did not conserve subset size")
    return quotas


def _negative_optical_modes(payload: dict) -> bool:
    values = torch.as_tensor(payload["force_constants"], dtype=torch.float64)
    atoms = values.shape[0]
    matrix = values.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
    eig = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    return bool((eig[3:] < -1e-5).any()) if eig.numel() > 3 else False


def features(record: dict, payload: dict) -> dict:
    analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
    rotations = torch.from_numpy(np.stack([
        operation.rotation_matrix for operation in analyzer.get_symmetry_operations(cartesian=True)
    ])).to(dtype=torch.float64)
    atoms = len(record["atoms"]["elements"])
    return {
        "response_bin": str(response_norm_bin(_raw_cartesian_target(record))),
        "crystal_system": str(analyzer.get_crystal_system()),
        "polar": str(is_polar_point_group(rotations)),
        "negative_optical_mode": str(_negative_optical_modes(payload)),
        "atom_count_bin": str(0 if atoms <= 4 else 1 if atoms <= 8 else 2 if atoms <= 16 else 3),
        "elements": tuple(sorted(set(str(value) for value in record["atoms"]["elements"]))),
    }


def _counts(ids: list[str], feature_map: dict[str, dict], key: str) -> Counter:
    return Counter(feature_map[jid][key] for jid in ids)


def _summary(ids: list[str], feature_map: dict[str, dict]) -> dict:
    return {
        "materials": len(ids),
        "response_bins": dict(sorted(_counts(ids, feature_map, "response_bin").items())),
        "crystal_systems": dict(sorted(_counts(ids, feature_map, "crystal_system").items())),
        "polar": dict(sorted(_counts(ids, feature_map, "polar").items())),
        "negative_optical_modes": dict(sorted(_counts(ids, feature_map, "negative_optical_mode").items())),
        "atom_count_bins": dict(sorted(_counts(ids, feature_map, "atom_count_bin").items())),
        "elements": sorted({element for jid in ids for element in feature_map[jid]["elements"]}),
    }


def _score(ids: list[str], feature_map: dict[str, dict], targets: dict[str, dict[str, float]], all_elements: set[str]) -> float:
    # Response-bin counts are held exactly by construction.  The remaining
    # terms prefer a close, broad match to the parent train population.
    weights = {"crystal_system": 2.0, "polar": 1.0, "negative_optical_mode": 1.0, "atom_count_bin": 1.0}
    value = 0.0
    for key, weight in weights.items():
        observed = _counts(ids, feature_map, key)
        for category, target in targets[key].items():
            value += weight * abs(observed[category] - target) / max(target, 1.0)
    coverage = len({element for jid in ids for element in feature_map[jid]["elements"]}) / max(len(all_elements), 1)
    return value + 0.5 * (1.0 - coverage)


def _candidate(
    rng: random.Random,
    ids_by_response: dict[str, list[str]],
    quotas: dict[str, int],
) -> list[str]:
    chosen = []
    for response, quota in quotas.items():
        if quota > len(ids_by_response[response]):
            raise ValueError(f"Response quota {quota} exceeds population for bin {response}")
        chosen.extend(rng.sample(ids_by_response[response], quota))
    return sorted(chosen)


def choose_subsets(
    train: list[str],
    feature_map: dict[str, dict],
    size: int,
    count: int,
    seed: int,
    candidates: int = 20000,
) -> tuple[list[list[str]], dict]:
    """Return low-discrepancy, mutually non-identical stratified subsets."""
    if count < 1 or candidates < count:
        raise ValueError("Need at least one requested subset and as many candidates")
    response_counts = _counts(train, feature_map, "response_bin")
    quotas = apportion(size, response_counts)
    ids_by_response = {key: [jid for jid in train if feature_map[jid]["response_bin"] == key] for key in quotas}
    targets = {
        key: {category: size * value / len(train) for category, value in _counts(train, feature_map, key).items()}
        for key in ("crystal_system", "polar", "negative_optical_mode", "atom_count_bin")
    }
    all_elements = {element for jid in train for element in feature_map[jid]["elements"]}
    rng = random.Random(seed)
    scored = []
    seen = set()
    for _ in range(candidates):
        subset = _candidate(rng, ids_by_response, quotas)
        key = tuple(subset)
        if key in seen:
            continue
        seen.add(key)
        scored.append((_score(subset, feature_map, targets, all_elements), subset))
    if len(scored) < count:
        raise RuntimeError("Insufficient distinct candidates")
    scored.sort(key=lambda item: (item[0], item[1]))
    selected: list[list[str]] = []
    # The best subset anchors the matching quality.  Each following subset is
    # selected from a small near-optimal pool while preferring a distinct set
    # of materials, so the exercise measures composition uncertainty rather
    # than trivially repeating the nested 35-label prefix.
    pool = scored[: min(len(scored), 2000)]
    while len(selected) < count:
        def candidate_value(item: tuple[float, list[str]]) -> tuple[float, tuple[str, ...]]:
            score, subset = item
            if not selected:
                return score, tuple(subset)
            overlap = max(len(set(subset) & set(previous)) / size for previous in selected)
            return score + 2.0 * overlap, tuple(subset)
        _, choice = min(pool, key=candidate_value)
        selected.append(choice)
        pool = [item for item in pool if item[1] != choice]
        if not pool:
            raise RuntimeError("Candidate pool was exhausted before selecting all subsets")
    metadata = {
        "response_bin_quotas": quotas,
        "parent_targets": targets,
        "candidate_count": len(scored),
        "random_seed": seed,
        "selection": "exact response-bin quotas; minimum parent-stratum discrepancy plus diversity among a near-optimal candidate pool",
    }
    return selected, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/processed/strict_completion_benchmark_v1.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--dfpt-dir", type=Path, default=Path("data/processed/jarvis_dfpt_v1"))
    parser.add_argument("--size", type=int, default=35)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--candidates", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stratified_subset_resampling_v1/splits"))
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    parent = {name: [str(value) for value in benchmark["splits"][name]] for name in ("train", "val", "test")}
    if args.size >= len(parent["train"]):
        raise ValueError("Subset size must be smaller than the frozen train panel")
    records = {str(record["JARVIS_ID"]): record for record in load_gmtnet_records(args.data_root)}
    cache = JarvisDFPTCache(args.dfpt_dir)
    feature_map = {jid: features(records[jid], cache.load(jid)) for jid in parent["train"]}
    subsets, metadata = choose_subsets(parent["train"], feature_map, args.size, args.count, args.seed, args.candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, subset in enumerate(subsets):
        split = {
            "schema": 1,
            "purpose": "composition-resampling diagnostic; not a replacement frozen benchmark",
            "frozen_parent": str(args.benchmark),
            "subset_index": index,
            "size": args.size,
            "splits": {"train": subset, "val": parent["val"], "test": parent["test"]},
            "summary": _summary(subset, feature_map),
            "sampling": metadata,
        }
        path = args.output_dir / f"strict_lambda_n{args.size}_subset{index:02d}.json"
        path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        rows.append({"subset_index": index, "path": str(path), **split["summary"]})
    manifest = {
        "schema": 1,
        "purpose": "five diverse response-stratified 35-label subsets from the frozen 69-label training panel",
        "benchmark": str(args.benchmark),
        "size": args.size,
        "count": args.count,
        "sampling": metadata,
        "parent_train_summary": _summary(parent["train"], feature_map),
        "subsets": rows,
        "test_policy": "Validation and test IDs are copied unchanged. No subset is selected using frozen test outputs.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / "manifest.json")


if __name__ == "__main__":
    main()
