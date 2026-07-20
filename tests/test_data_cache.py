import pytest

import piezojet.data as data_module
from piezojet.data import (
    PiezoDataset,
    completion_label_metadata,
    formula,
    load_gmtnet_records,
    record_to_graph,
)


def test_formula_is_reduced_and_independent_of_unit_cell_multiplicity():
    primitive = {"atoms": {"elements": ["Na", "O", "H"]}}
    doubled = {"atoms": {"elements": ["Na", "O", "H"] * 2}}
    assert formula(primitive) == formula(doubled) == "HNaO"


def test_piezo_dataset_graph_cache_preserves_graph():
    records = load_gmtnet_records("data/raw/gmtnet")[:1]
    material_id = str(records[0]["JARVIS_ID"])
    cached = PiezoDataset(records, [material_id], 5.0, 32)
    first = cached[0]
    second = cached[0]
    assert first is second
    assert first.edge_index.shape == second.edge_index.shape


def test_piezo_dataset_reuses_persistent_disk_graph(monkeypatch, tmp_path):
    records = load_gmtnet_records("data/raw/gmtnet")[:1]
    material_id = str(records[0]["JARVIS_ID"])
    first = PiezoDataset(records, [material_id], 5.0, 32, processed_dir=tmp_path)
    graph = first[0]
    assert list((tmp_path / "pbc_graph_cache").rglob("*.pt"))
    second = PiezoDataset(records, [material_id], 5.0, 32, processed_dir=tmp_path)
    monkeypatch.setattr(data_module, "record_to_graph", lambda *_: (_ for _ in ()).throw(AssertionError("disk cache miss")))
    restored = second[0]
    assert restored.material_id == graph.material_id


def test_explicit_none_dielectric_is_a_masked_missing_label():
    record = dict(load_gmtnet_records("data/raw/gmtnet")[0])
    record["dielectric"] = None
    graph = record_to_graph(record, 5.0, 32)
    assert not bool(graph.dielectric_mask)
    assert graph.y_dielectric.shape == (1, 3, 3)


def test_completion_schema_routes_only_certified_full_lambda_labels():
    label_type, certificate = completion_label_metadata({
        "schema": 2,
        "audit": {"accepted": True, "invariant_dimensions": 7},
    })
    assert label_type == "strict_completion"
    assert certificate["invariant_dimensions"] == 7
    label_type, _ = completion_label_metadata({
        "schema": 3,
        "lambda_label_type": "joint_identifiable",
        "identifiability": {"identifiable_dimension": 5},
        "audit": {"accepted": True},
    })
    assert label_type == "joint_identifiable"
    with pytest.raises(ValueError, match="full-Lambda label"):
        completion_label_metadata({
            "schema": 3,
            "lambda_label_type": "macro_only",
            "audit": {"accepted": True},
        })
