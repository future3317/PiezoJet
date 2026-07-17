from collections import Counter

import numpy as np

from piezojet.build_development_folds import assign_formula_groups
from piezojet.ood import reduced_formula


def _record(identifier: str, elements: list[str], scale: float):
    return {
        "JARVIS_ID": identifier,
        "atoms": {
            "lattice_mat": np.eye(3).tolist(),
            "coords": [[0.0, 0.0, 0.0] for _ in elements],
            "elements": elements,
        },
        "piezoelectric_C_m2": (np.eye(3, 6) * scale).tolist(),
    }


def test_development_folds_are_complete_deterministic_and_formula_disjoint():
    records = []
    for index in range(20):
        # Two polymorph IDs per formula group.
        formula = ["H", "O"] if index < 2 else ["H"] * (index // 2 + 1) + ["O"]
        records.append(_record(f"JVASP-{index}", formula, 0.1 + index))
    first = assign_formula_groups(records, folds=5, seed=42)
    second = assign_formula_groups(records, folds=5, seed=42)
    assert first == second
    assert all(first)
    assert Counter(value for fold in first for value in fold) == Counter(
        str(record["JARVIS_ID"]) for record in records
    )
    formula_by_id = {
        str(record["JARVIS_ID"]): reduced_formula(record) for record in records
    }
    seen: dict[str, int] = {}
    for fold_index, identifiers in enumerate(first):
        for identifier in identifiers:
            formula = formula_by_id[identifier]
            assert formula not in seen or seen[formula] == fold_index
            seen[formula] = fold_index
