import torch

from piezojet.data import load_gmtnet_records, record_to_graph
from piezojet.model import PiezoJet
from piezojet.tensor_ops import rotate_piezo


def test_collective_polar_spectrum_and_weighted_frames_preserve_rotation_equivariance():
    torch.manual_seed(11)
    record = load_gmtnet_records("data/raw/gmtnet")[10]
    graph = record_to_graph(record, 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = graph.clone()
    rotated.pos = graph.pos @ rotation.T
    rotated.edge_shift = graph.edge_shift @ rotation.T
    rotated.cell = graph.cell @ rotation.T
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    # Force a nonzero frame-refinement contribution; the test therefore checks
    # the complete collective-context path rather than only the direct head.
    with torch.no_grad():
        model.frame_refiner.delta_network[-1].bias.normal_()
        prediction, transformed = model(graph), model(rotated)
    residual = torch.linalg.vector_norm(transformed - rotate_piezo(prediction, rotation)) / torch.linalg.vector_norm(prediction).clamp_min(1e-6)
    assert residual < 1e-4

