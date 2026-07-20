import torch

from piezojet.baselines import DirectCartesianPiezoBaseline, E3nnDirectPiezoBaseline
from piezojet.data import load_gmtnet_records, record_to_graph
from piezojet.tensor_ops import rotate_piezo
from tests.data_paths import gmtnet_root


def test_matched_direct_cartesian_baseline_emits_a_strain_symmetric_tensor():
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[0], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = DirectCartesianPiezoBaseline(cutoff=5.0, num_blocks=1)
    output = model(graph)
    assert output.shape == (1, 3, 3, 3)
    assert torch.allclose(output, output.transpose(-1, -2), atol=1e-6, rtol=1e-6)


def test_e3nn_direct_baseline_emits_an_equivariant_strain_symmetric_tensor():
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[0], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = graph.clone()
    rotated.pos = graph.pos @ rotation.T
    rotated.edge_shift = graph.edge_shift @ rotation.T
    model = E3nnDirectPiezoBaseline(cutoff=5.0, lmax=2, num_blocks=1)
    model.eval()
    output, transformed = model(graph), model(rotated)
    assert output.shape == (1, 3, 3, 3)
    assert torch.allclose(output, output.transpose(-1, -2), atol=1e-6, rtol=1e-6)
    assert torch.allclose(transformed, rotate_piezo(output, rotation), atol=2e-4, rtol=2e-4)
