"""Build a conservative high-likelihood queue for strict Lambda completion.

The queue is separate from the high-response challenge cohort.  It favors
small cells and high-order space groups, while retaining response-bin coverage
and excluding point groups under unresolved forensic investigation.  It is a
retrieval priority, never a relaxed label-acceptance rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .data import _raw_cartesian_target, formula, load_gmtnet_records, response_norm_bin
from .jarvis_dfpt import JarvisDFPTCache
from .strain_completion import _structure


# Completion-oriented coverage intentionally places less mass on the rarest,
# highest-response bin; that population is preserved in the challenge cohort.
RESPONSE_FRACTIONS = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.15, 4: 0.10}


def _atom_bin(atoms: int) -> int:
    return 0 if atoms <= 2 else 1 if atoms <= 4 else 2 if atoms <= 8 else 3


def _key(record: dict) -> str:
    return hashlib.sha256(str(record["JARVIS_ID"]).encode()).hexdigest()


def _quota(size: int, response_bin: int) -> int:
    values = {key: round(size * fraction) for key, fraction in RESPONSE_FRACTIONS.items()}
    values[0] += size - sum(values.values())
    return values[response_bin]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--dfpt-dir", type=Path, default=Path("data/processed/jarvis_dfpt_v1"))
    parser.add_argument("--size", type=int, default=250)
    parser.add_argument("--candidate-pool", type=int, default=1000)
    parser.add_argument("--exclude-space-groups", default="187", help="Comma-separated unresolved space-group numbers")
    parser.add_argument("--output", type=Path, default=Path("outputs/jarvis_dfpt_expansion_v1/completion_likelihood_cohort.json"))
    args = parser.parse_args()
    if args.size < 1 or args.candidate_pool < args.size:
        raise ValueError("Candidate pool must be at least the requested positive queue size")
    excluded = {int(value) for value in args.exclude_space_groups.split(",") if value.strip()}
    cache = JarvisDFPTCache(args.dfpt_dir)
    candidates = [record for record in load_gmtnet_records(args.data_root) if cache.load(str(record["JARVIS_ID"])) is None]
    if len(candidates) < args.size:
        raise ValueError("Not enough uncached JARVIS records for requested queue")
    by_response: dict[int, list[dict]] = defaultdict(list)
    for record in candidates:
        by_response[response_norm_bin(_raw_cartesian_target(record))].append(record)
    preliminary: list[dict] = []
    for response_bin, records in by_response.items():
        records.sort(key=lambda record: (len(record["atoms"]["elements"]), _key(record)))
        preliminary.extend(records[: max(_quota(args.candidate_pool, response_bin), args.size)])
    # Deduplicate while retaining the small-cell, response-stratified pool.
    preliminary = list({str(record["JARVIS_ID"]): record for record in preliminary}.values())
    featured = []
    for position, record in enumerate(preliminary, start=1):
        analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
        number = int(analyzer.get_space_group_number())
        operations = len(analyzer.get_symmetry_operations(cartesian=True))
        if number not in excluded:
            featured.append((record, number, operations))
        if position % 100 == 0:
            print(f"profiled {position}/{len(preliminary)}")
    selected, used_elements = [], set()
    for response_bin in range(5):
        pool = [item for item in featured if response_norm_bin(_raw_cartesian_target(item[0])) == response_bin]
        atom_counts: dict[int, int] = defaultdict(int)
        for _ in range(min(_quota(args.size, response_bin), len(pool))):
            minimum = min(atom_counts[_atom_bin(len(item[0]["atoms"]["elements"]))] for item in pool)
            eligible = [item for item in pool if atom_counts[_atom_bin(len(item[0]["atoms"]["elements"]))] == minimum]
            eligible.sort(key=lambda item: (
                -item[2], len(item[0]["atoms"]["elements"]),
                -len(set(item[0]["atoms"]["elements"]) - used_elements), _key(item[0]),
            ))
            choice = eligible[0]
            pool.remove(choice)
            selected.append(choice)
            atom_counts[_atom_bin(len(choice[0]["atoms"]["elements"]))] += 1
            used_elements.update(choice[0]["atoms"]["elements"])
    remaining = [item for item in featured if item not in selected]
    while len(selected) < args.size:
        remaining.sort(key=lambda item: (
            -item[2], len(item[0]["atoms"]["elements"]),
            -len(set(item[0]["atoms"]["elements"]) - used_elements), _key(item[0]),
        ))
        choice = remaining.pop(0)
        selected.append(choice)
        used_elements.update(choice[0]["atoms"]["elements"])
    payload = {
        "schema": 1,
        "policy": "small cells and high space-group order, response-bin coverage, novel-element tie-breaker; excluded unresolved space groups",
        "excluded_space_groups": sorted(excluded),
        "summary": {
            "requested": args.size, "candidate_uncached": len(candidates), "profiled": len(preliminary),
            "unique_elements": len(used_elements),
            "response_bin_counts": {str(index): sum(response_norm_bin(_raw_cartesian_target(item[0])) == index for item in selected) for index in range(5)},
            "atom_count_bins": {str(index): sum(_atom_bin(len(item[0]["atoms"]["elements"])) == index for item in selected) for index in range(4)},
        },
        "material_ids": [str(item[0]["JARVIS_ID"]) for item in selected],
        "materials": [
            {
                "jid": str(record["JARVIS_ID"]), "formula": formula(record), "atoms": len(record["atoms"]["elements"]),
                "response_bin": response_norm_bin(_raw_cartesian_target(record)), "space_group_number": number,
                "space_group_operations": operations, "elements": sorted(set(record["atoms"]["elements"])),
            }
            for record, number, operations in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
