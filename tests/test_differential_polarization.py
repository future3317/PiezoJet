import torch

from piezojet.data import load_gmtnet_records, record_to_graph
from piezojet.model import DifferentialPolarizationTower
from piezojet.tensor_ops import rotate_piezo


def _graph(index: int = 10):
    graph = record_to_graph(load_gmtnet_records("data/raw/gmtnet")[index], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    return graph


def _model():
    return DifferentialPolarizationTower(
        embedding_dim=8,
        cutoff=5.0,
        lmax=3,
        num_blocks=1,
        radial_basis=4,
        radial_hidden=16,
        global_context_dim=16,
        spectral_channels=4,
        spectral_shells=3,
        polar_fluctuation_shells=3,
        reciprocal_cutoff=4.0,
        attention_dim=8,
    )


def _rotated(graph, rotation: torch.Tensor):
    transformed = graph.clone()
    transformed.pos = graph.pos @ rotation.T
    transformed.edge_shift = graph.edge_shift @ rotation.T
    transformed.cell = graph.cell @ rotation.T
    return transformed


def test_polarization_increment_has_exact_zero_reference_and_acoustic_charge_sum():
    torch.manual_seed(5)
    graph = _graph()
    model = _model().eval()
    prediction = model.coefficients(graph)
    zero = model.polarization_increment_from_coefficients(
        prediction, torch.zeros_like(graph.pos), torch.zeros(1, 3, 3), graph
    )
    assert torch.equal(zero, torch.zeros_like(zero))
    assert torch.linalg.vector_norm(prediction.born_charges.sum(dim=0)) < 1e-6
    assert torch.allclose(
        prediction.electronic_piezo,
        prediction.electronic_piezo.transpose(-1, -2),
        atol=1e-7,
        rtol=1e-7,
    )


def test_born_and_electronic_tensors_are_exact_jacobians_of_one_increment():
    torch.manual_seed(7)
    graph = _graph()
    model = _model().eval()
    prediction = model.coefficients(graph)
    displacement = torch.zeros_like(graph.pos, requires_grad=True)
    strain = torch.zeros(1, 3, 3, requires_grad=True)
    displacement_jacobian = torch.autograd.functional.jacobian(
        lambda value: model.polarization_increment_from_coefficients(
            prediction, value, torch.zeros_like(strain), graph
        ),
        displacement,
    )
    strain_jacobian = torch.autograd.functional.jacobian(
        lambda value: model.polarization_increment_from_coefficients(
            prediction, torch.zeros_like(displacement), value, graph
        ),
        strain,
    )
    volume = torch.linalg.det(graph.cell).abs()
    expected_displacement = (
        model.PIEZO_C_PER_M2 / volume
        * prediction.born_charges.permute(2, 0, 1)
    )
    assert torch.allclose(
        displacement_jacobian[0], expected_displacement, atol=2e-6, rtol=2e-6
    )
    assert torch.allclose(
        strain_jacobian[0, :, 0], prediction.electronic_piezo[0],
        atol=2e-6, rtol=2e-6,
    )


def test_differential_polarization_coefficients_and_increment_are_o3_equivariant():
    torch.manual_seed(11)
    graph = _graph(17)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = _rotated(graph, rotation)
    model = _model().eval()
    with torch.no_grad():
        prediction = model.coefficients(graph)
        transformed = model.coefficients(rotated)
        displacement = torch.randn_like(graph.pos)
        displacement = displacement - displacement.mean(dim=0)
        raw_strain = torch.randn(3, 3)
        strain = 0.5 * (raw_strain + raw_strain.T)
        increment = model.polarization_increment_from_coefficients(
            prediction, displacement, strain[None], graph
        )
        transformed_increment = model.polarization_increment_from_coefficients(
            transformed,
            displacement @ rotation.T,
            (rotation @ strain @ rotation.T)[None],
            rotated,
        )
    expected_born = torch.einsum(
        "ia,nab,jb->nij", rotation, prediction.born_charges, rotation
    )
    assert torch.allclose(transformed.born_charges, expected_born, atol=3e-5, rtol=3e-5)
    assert torch.allclose(
        transformed.electronic_piezo,
        rotate_piezo(prediction.electronic_piezo, rotation),
        atol=3e-5,
        rtol=3e-5,
    )
    assert torch.allclose(
        transformed_increment, increment @ rotation.T, atol=3e-5, rtol=3e-5
    )
