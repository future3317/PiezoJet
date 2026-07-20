import json

import pytest

from piezojet.build_balanced_electrostatic_subsets import balanced_order
from piezojet.electrostatic_subset import load_response_subset, material_id_sha256


def test_fixed_response_subset_rejects_fold_leakage_and_stale_hash(tmp_path):
    path = tmp_path / "subset.json"
    payload = {
        "schema": 1,
        "fold": 0,
        "materials": 2,
        "material_ids": ["a", "b"],
        "material_id_sha256": material_id_sha256(["a", "b"]),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    ids, loaded = load_response_subset(path, fold=0, allowed_ids=["a", "b", "c"])
    assert ids == ["a", "b"]
    assert loaded["materials"] == 2

    payload["material_ids"] = ["a", "held-out"]
    payload["material_id_sha256"] = material_id_sha256(payload["material_ids"])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-fold-train"):
        load_response_subset(path, fold=0, allowed_ids=["a", "b", "c"])


def test_balanced_order_is_deterministic_nested_and_covers_minority_categories():
    rows = []
    for index in range(20):
        minority = index >= 16
        rows.append({
            "electronic_quantile": 1 if minority else 0,
            "born_quantile": 1 if minority else 0,
            "dielectric_quantile": index % 2,
            "crystal_system": "minority" if minority else "majority",
            "atom_bin": "9-16" if minority else "1-2",
            "elements": ("O", "Xe") if minority else ("O",),
            "reduced_formula": f"F{index}",
        })
    order = balanced_order(rows, 8, seed=42)
    assert order == balanced_order(rows, 8, seed=42)
    assert len(order) == len(set(order)) == 8
    assert any(index >= 16 for index in order[:4])
