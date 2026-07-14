"""Select a response- and structure-stratified JARVIS DFPT retrieval cohort.

The selector is intentionally independent of model predictions: before enough
complete Lambda labels exist, uncertainty scores would mostly rank arbitrary
out-of-distribution extrapolations.  It instead gives a first raw-DFPT batch
coverage over response strength, atom count, and chemical elements, with a
deliberate high-response allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .data import _raw_cartesian_target, formula, load_gmtnet_records, response_norm_bin
from .jarvis_dfpt import JarvisDFPTCache


RESPONSE_FRACTIONS = {0: 0.05, 1: 0.10, 2: 0.20, 3: 0.25, 4: 0.40}


def _atom_bin(atoms: int) -> int:
    if atoms <= 2:
        return 0
    if atoms <= 4:
        return 1
    if atoms <= 8:
        return 2
    return 3


def _stable_key(record: dict) -> str:
    return hashlib.sha256(str(record["JARVIS_ID"]).encode()).hexdigest()


def select(records: list[dict], cache: JarvisDFPTCache, size: int) -> tuple[list[dict], dict]:
    candidates = [record for record in records if cache.load(str(record["JARVIS_ID"])) is None]
    if len(candidates) < size:
        raise ValueError(f"Requested {size} uncached records but only {len(candidates)} are available")
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in candidates:
        grouped[response_norm_bin(_raw_cartesian_target(record))].append(record)
    selected: list[dict] = []
    used_elements: set[str] = set()
    per_response: dict[int, int] = {}
    for response_bin, fraction in RESPONSE_FRACTIONS.items():
        quota = round(size * fraction)
        if response_bin == 4:
            quota += size - sum(round(size * value) for value in RESPONSE_FRACTIONS.values())
        pool = grouped[response_bin]
        if not pool:
            continue
        atom_counts: dict[int, int] = defaultdict(int)
        while pool and per_response.get(response_bin, 0) < quota:
            # Least-covered size class first; within it choose new chemical
            # coverage, then a deterministic ID hash.  This avoids a random,
            # response-only cohort dominated by one familiar chemistry.
            minimum = min(atom_counts[_atom_bin(len(item["atoms"]["elements"]))] for item in pool)
            eligible = [item for item in pool if atom_counts[_atom_bin(len(item["atoms"]["elements"]))] == minimum]
            eligible.sort(
                key=lambda item: (
                    -len(set(item["atoms"]["elements"]) - used_elements),
                    _stable_key(item),
                )
            )
            choice = eligible[0]
            pool.remove(choice)
            selected.append(choice)
            used_elements.update(choice["atoms"]["elements"])
            atom_counts[_atom_bin(len(choice["atoms"]["elements"]))] += 1
            per_response[response_bin] = per_response.get(response_bin, 0) + 1
    # Sparse response strata can make the requested fractional quotas
    # unattainable. Fill deterministically from the remaining high-response
    # candidates while preserving atom-count and elemental diversity.
    remaining = [item for group in grouped.values() for item in group]
    while len(selected) < size:
        remaining.sort(
            key=lambda item: (
                -response_norm_bin(_raw_cartesian_target(item)),
                -len(set(item["atoms"]["elements"]) - used_elements),
                _stable_key(item),
            )
        )
        choice = remaining.pop(0)
        selected.append(choice)
        used_elements.update(choice["atoms"]["elements"])
        category = response_norm_bin(_raw_cartesian_target(choice))
        per_response[category] = per_response.get(category, 0) + 1
    summary = {
        "requested": size,
        "candidate_uncached": len(candidates),
        "response_bin_counts": {str(key): value for key, value in sorted(per_response.items())},
        "unique_elements": len(used_elements),
        "atom_count_bins": {
            str(index): sum(_atom_bin(len(record["atoms"]["elements"])) == index for record in selected)
            for index in range(4)
        },
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--dfpt-dir", type=Path, default=Path("data/processed/jarvis_dfpt_v1"))
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("outputs/jarvis_dfpt_expansion_v1/cohort.json"))
    args = parser.parse_args()
    if args.size < 1:
        raise ValueError("--size must be positive")
    selected, summary = select(load_gmtnet_records(args.data_root), JarvisDFPTCache(args.dfpt_dir), args.size)
    payload = {
        "schema": 1,
        "policy": "response-bin quotas (high response prioritized), atom-count balance, greedy novel-element coverage",
        "summary": summary,
        "material_ids": [str(record["JARVIS_ID"]) for record in selected],
        "materials": [
            {
                "jid": str(record["JARVIS_ID"]), "formula": formula(record),
                "atoms": len(record["atoms"]["elements"]),
                "response_bin": response_norm_bin(_raw_cartesian_target(record)),
                "elements": sorted(set(record["atoms"]["elements"])),
            }
            for record in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
