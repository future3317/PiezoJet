import torch
from torch_geometric.data import Data

from piezojet.model import PiezoJet, ResponsePotential
from piezojet.tensor_ops import rotate_piezo, rotate_strain, symmetric_matrix_to_voigt


def _graph() -> Data:
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.2, 1.1, 0.4]])
    edge_index = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
    return Data(z=torch.tensor([14, 8, 8]), pos=pos, edge_index=edge_index, edge_shift=torch.zeros(6, 3), batch=torch.zeros(3, dtype=torch.long))


def _rotation() -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    return q


def test_untrained_model_is_rotation_equivariant_and_potential_invariant():
    torch.manual_seed(0)
    graph, rotation = _graph(), _rotation()
    rotated = graph.clone()
    rotated.pos = graph.pos @ rotation.T
    rotated.edge_shift = graph.edge_shift @ rotation.T
    model = PiezoJet(cutoff=5.0, num_blocks=1)
    model.eval()
    prediction, transformed_prediction = model(graph), model(rotated)
    expected = rotate_piezo(prediction, rotation)
    residual = torch.linalg.vector_norm(transformed_prediction - expected) / torch.linalg.vector_norm(prediction).clamp_min(1e-12)
    assert residual < 1e-4
    field, eta6 = torch.randn(1, 3), torch.randn(1, 6)
    strain = rotate_strain(__import__("piezojet.tensor_ops", fromlist=["voigt_to_symmetric_matrix"]).voigt_to_symmetric_matrix(eta6), rotation)
    potential = ResponsePotential()
    assert torch.allclose(potential(prediction, field, eta6), potential(transformed_prediction, field @ rotation.T, symmetric_matrix_to_voigt(strain)), atol=1e-5)


def test_encoder_radial_centers_are_registered_buffer():
    model = PiezoJet()
    assert "radial_centers" in dict(model.encoder.named_buffers())
