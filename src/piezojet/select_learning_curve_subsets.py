"""Create nested, stratified strict-Lambda learning-curve splits.

The validation and test portions are copied verbatim from the frozen benchmark.
Only the training prefix changes, so any metric difference is attributable to a
nested increase in complete Lambda supervision rather than a moving holdout.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .completion_forensics import _negative_optical_modes
from .data import _raw_cartesian_target, load_gmtnet_records, response_norm_bin
from .jarvis_dfpt import JarvisDFPTCache
from .strain_completion import _structure


def _features(record: dict, payload: dict) -> dict:
    analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
    operations = analyzer.get_symmetry_operations(cartesian=True)
    projector = torch.as_tensor([op.rotation_matrix for op in operations], dtype=torch.float64).mean(dim=0)
    atoms = len(record["atoms"]["elements"])
    return {
        "response_bin": response_norm_bin(_raw_cartesian_target(record)),
        "crystal_system": str(analyzer.get_crystal_system()),
        "polar": bool(torch.linalg.matrix_rank(projector, tol=1e-7) > 0),
        "negative_optical_mode": _negative_optical_modes(payload),
        "atom_count_bin": 0 if atoms <= 2 else 1 if atoms <= 4 else 2 if atoms <= 8 else 3,
        "elements": tuple(sorted(set(record["atoms"]["elements"]))),
    }


def _choose_next(candidates: list[str], features: dict[str, dict], selected: list[str], preferred: set[str]) -> str:
    response = Counter(features[jid]["response_bin"] for jid in selected)
    system = Counter(features[jid]["crystal_system"] for jid in selected)
    polar = Counter(features[jid]["polar"] for jid in selected)
    soft = Counter(features[jid]["negative_optical_mode"] for jid in selected)
    atom = Counter(features[jid]["atom_count_bin"] for jid in selected)
    elements = Counter(element for jid in selected for element in features[jid]["elements"])
    def score(jid: str) -> tuple[float, str]:
        item = features[jid]
        value = (
            2.0 / (1 + response[item["response_bin"]])
            + 1.0 / (1 + system[item["crystal_system"]])
            + 0.4 / (1 + polar[item["polar"]])
            + 0.4 / (1 + soft[item["negative_optical_mode"]])
            + 0.3 / (1 + atom[item["atom_count_bin"]])
            + sum(0.1 / (1 + elements[element]) for element in item["elements"])
        )
        # Historical completions are retained where doing so does not defeat
        # the response/system balancing needed for an interpretable curve.
        return value + (0.2 if jid in preferred else 0.0), jid
    return max(candidates, key=score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/processed/strict_completion_benchmark_v1.json"))
    parser.add_argument("--historical-manifest", type=Path, default=Path("data/processed/jarvis_strain_completion_v2/manifest.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--dfpt-dir", type=Path, default=Path("data/processed/jarvis_dfpt_v1"))
    parser.add_argument("--sizes", default="19,23,35,50,69")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/strict_learning_curve_v1/splits"))
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if sizes != sorted(set(sizes)):
        raise ValueError("--sizes must be strictly increasing")
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    train, val, test = (list(benchmark["splits"][name]) for name in ("train", "val", "test"))
    if sizes[-1] != len(train):
        raise ValueError(f"Largest requested size must equal frozen train size {len(train)}")
    historical = json.loads(args.historical_manifest.read_text(encoding="utf-8"))
    preferred = [str(row["jid"]) for row in historical["rows"] if row.get("accepted") and str(row["jid"]) in set(train)]
    records = {str(row["JARVIS_ID"]): row for row in load_gmtnet_records(args.data_root)}
    cache = JarvisDFPTCache(args.dfpt_dir)
    features = {jid: _features(records[jid], cache.load(jid)) for jid in train}
    preferred_set = set(preferred)
    selected: list[str] = []
    while len(selected) < len(train):
        selected.append(_choose_next([jid for jid in train if jid not in set(selected)], features, selected, preferred_set))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for size in sizes:
        subset = selected[:size]
        split = {
            "schema": 1, "frozen_parent": str(args.benchmark), "nested": True,
            "size": size, "splits": {"train": subset, "val": val, "test": test},
            "summary": {
                "historical_prefix_members": sum(jid in preferred for jid in subset),
                "response_bins": dict(Counter(str(features[jid]["response_bin"]) for jid in subset)),
                "crystal_systems": dict(Counter(features[jid]["crystal_system"] for jid in subset)),
                "polar": sum(features[jid]["polar"] for jid in subset),
                "negative_optical_modes": sum(features[jid]["negative_optical_mode"] for jid in subset),
            },
        }
        path = args.output_dir / f"strict_lambda_n{size}.json"
        path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        reports.append({"size": size, "path": str(path), **split["summary"]})
    (args.output_dir / "manifest.json").write_text(json.dumps({"schema": 1, "sizes": sizes, "order": selected, "subsets": reports}, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / "manifest.json")


if __name__ == "__main__":
    main()
