import piezojet.data as data_module
from piezojet.data import PiezoDataset, load_gmtnet_records


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
