from piezojet.data import PiezoDataset, load_gmtnet_records


def test_piezo_dataset_graph_cache_preserves_graph():
    records = load_gmtnet_records("data/raw/gmtnet")[:1]
    material_id = str(records[0]["JARVIS_ID"])
    cached = PiezoDataset(records, [material_id], 5.0, 32)
    first = cached[0]
    second = cached[0]
    assert first is second
    assert first.edge_index.shape == second.edge_index.shape
