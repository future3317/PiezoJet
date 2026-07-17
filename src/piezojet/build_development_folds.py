"""Build immutable formula-disjoint development folds inside strict train1603.

The frozen validation10/test20 are never members of these folds. Formula
groups are greedily balanced by material count, crystal system, and a
train-only GMTNet total-response norm bin. The output is a development role
map, not a replacement for the canonical production split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import spglib

from .data import load_gmtnet_records
from .ood import reduced_formula
from .project_config import load_project_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crystal_system(record: dict[str, object]) -> str:
    atoms = record["atoms"]
    dataset = spglib.get_symmetry_dataset(
        (atoms["lattice_mat"], atoms["coords"], [
            # Atomic numbers are needed only to distinguish species orbits;
            # stable enumeration by element symbol is sufficient here.
            sorted(set(atoms["elements"])).index(value) + 1
            for value in atoms["elements"]
        ]),
        symprec=1e-5,
    )
    number = int(dataset.number) if dataset is not None else 0
    if number <= 0:
        return "unknown"
    if number <= 2:
        return "triclinic"
    if number <= 15:
        return "monoclinic"
    if number <= 74:
        return "orthorhombic"
    if number <= 142:
        return "tetragonal"
    if number <= 167:
        return "trigonal"
    if number <= 194:
        return "hexagonal"
    return "cubic"


def _response_norm(record: dict[str, object]) -> float:
    return float(np.linalg.norm(np.asarray(record["piezoelectric_C_m2"], dtype=float)))


def assign_formula_groups(
    records: list[dict[str, object]],
    folds: int = 5,
    seed: int = 42,
) -> list[list[str]]:
    """Deterministically balance indivisible formula groups across folds."""
    if folds < 2:
        raise ValueError("At least two development folds are required")
    if len(records) < folds:
        raise ValueError("Fewer records than requested development folds")
    norms = np.asarray([_response_norm(record) for record in records])
    boundaries = np.quantile(norms, np.linspace(0.0, 1.0, 6)[1:-1])
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record, norm in zip(records, norms, strict=True):
        response_bin = int(np.searchsorted(boundaries, norm, side="right"))
        stratum = f"{_crystal_system(record)}|r{response_bin}"
        groups[reduced_formula(record)].append((str(record["JARVIS_ID"]), stratum))

    total_strata = Counter(stratum for values in groups.values() for _, stratum in values)
    target_size = len(records) / folds
    fold_ids: list[list[str]] = [[] for _ in range(folds)]
    fold_strata: list[Counter[str]] = [Counter() for _ in range(folds)]
    # A seeded hash breaks exact objective ties without depending on Python's
    # randomized hash salt; large and composition-diverse groups are placed first.
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            -len({stratum for _, stratum in item[1]}),
            hashlib.sha256(f"{seed}|{item[0]}".encode()).hexdigest(),
        ),
    )
    for _, members in ordered:
        group_hist = Counter(stratum for _, stratum in members)
        scores = []
        for fold_index in range(folds):
            # Compare normalized *load*, not distance to the final target.
            # The latter fills one fold toward its target before touching an
            # empty fold (a rich-get-richer failure for many small groups).
            size_score = (
                len(fold_ids[fold_index]) + len(members)
            ) / target_size
            stratum_score = sum(
                group_hist[name]
                * (fold_strata[fold_index][name] + group_hist[name])
                / (total_strata[name] + 1.0)
                for name in group_hist
            ) / len(members)
            scores.append((size_score + stratum_score, len(fold_ids[fold_index]), fold_index))
        destination = min(scores)[2]
        fold_ids[destination].extend(identifier for identifier, _ in members)
        fold_strata[destination].update(group_hist)
    return [sorted(values) for values in fold_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--strict-split",
        type=Path,
        default=Path("data/processed/strict_completion_benchmark_train_v10_full_public.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/strict_train1603_development_folds_v1.json"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_project_config(args.config)
    split_payload = json.loads(args.strict_split.read_text(encoding="utf-8-sig"))
    source_splits = split_payload["splits"]
    train_ids = [str(value) for value in source_splits["train"]]
    frozen_ids = set(map(str, source_splits["val"] + source_splits["test"]))
    if frozen_ids & set(train_ids):
        raise ValueError("Source strict split itself overlaps frozen IDs")
    record_by_id = {
        str(record["JARVIS_ID"]): record
        for record in load_gmtnet_records(config["data_root"])
    }
    missing = sorted(set(train_ids) - set(record_by_id))
    if missing:
        raise ValueError(f"Strict-train IDs missing from GMTNet records: {missing[:5]}")
    train_records = [record_by_id[identifier] for identifier in train_ids]
    dev_folds = assign_formula_groups(train_records, args.folds, args.seed)
    if any(not values for values in dev_folds):
        raise RuntimeError("Development-fold assignment produced an empty fold")
    formula_by_id = {
        identifier: reduced_formula(record_by_id[identifier]) for identifier in train_ids
    }
    union = set().union(*map(set, dev_folds))
    if union != set(train_ids) or sum(map(len, dev_folds)) != len(train_ids):
        raise RuntimeError("Development folds do not form an exact train1603 partition")
    folds_payload = []
    for index, dev_ids in enumerate(dev_folds):
        dev_set = set(dev_ids)
        fit_ids = sorted(set(train_ids) - dev_set)
        fit_formulas = {formula_by_id[value] for value in fit_ids}
        dev_formulas = {formula_by_id[value] for value in dev_ids}
        if fit_formulas & dev_formulas:
            raise RuntimeError(f"Fold {index} is not formula-disjoint")
        if (set(fit_ids) | dev_set) & frozen_ids:
            raise RuntimeError(f"Fold {index} reads a frozen validation/test ID")
        dev_norms = [_response_norm(record_by_id[value]) for value in dev_ids]
        folds_payload.append(
            {
                "fold": index,
                "train": fit_ids,
                "development": dev_ids,
                "train_materials": len(fit_ids),
                "development_materials": len(dev_ids),
                "train_formulas": len(fit_formulas),
                "development_formulas": len(dev_formulas),
                "development_total_response_norm_mean_c_per_m2": float(np.mean(dev_norms)),
                "development_total_response_norm_median_c_per_m2": float(np.median(dev_norms)),
            }
        )
    payload = {
        "schema": 1,
        "role": "method-development-only; not production validation or test",
        "source_split": str(args.strict_split),
        "source_split_sha256": _sha256(args.strict_split),
        "seed": args.seed,
        "fold_count": args.folds,
        "population": "strict train1603 only",
        "frozen_validation_test_labels_read": False,
        "balancing": "indivisible reduced-formula groups; material count + crystal-system/response-norm strata",
        "folds": folds_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "fold_sizes": [len(v) for v in dev_folds]}, indent=2))


if __name__ == "__main__":
    main()
