import pytest
import torch
from torch_geometric.loader import DataLoader

from piezojet.data import load_gmtnet_records, record_to_graph
from piezojet.model import NonlinearDifferentialPolarizationTower
from piezojet.tensor_ops import (
    cartesian_to_piezo_voigt,
    rotate_piezo,
    symmetric_matrix_to_voigt,
)


def _graph(index: int = 10):
    graph = record_to_graph(
        load_gmtnet_records("data/raw/gmtnet")[index], 5.0, 32
    )
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    return graph


def _model(polarization_variable: str = "cartesian"):
    return NonlinearDifferentialPolarizationTower(
        polarization_variable=polarization_variable,
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


def _permuted(graph, permutation: torch.Tensor):
    transformed = graph.clone()
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    transformed.z = graph.z[permutation]
    transformed.pos = graph.pos[permutation]
    transformed.frac = graph.frac[permutation]
    transformed.edge_index = inverse[graph.edge_index]
    transformed.batch = torch.zeros_like(graph.batch)
    return transformed


def test_literal_increment_is_zero_and_coefficients_are_its_jacobians():
    torch.manual_seed(101)
    graph = _graph()
    model = _model().eval()
    zero_u = torch.zeros_like(graph.pos)
    zero_eta = torch.zeros(1, 6)
    increment = model.polarization_increment(graph, zero_u, zero_eta)
    assert torch.equal(increment, torch.zeros_like(increment))
    uniform_translation = torch.randn(1, 3).expand_as(graph.pos)
    translated = model.polarization_increment(
        graph, uniform_translation, zero_eta
    )
    assert torch.equal(translated, torch.zeros_like(translated))

    coefficients = model.coefficients(graph, create_graph=False)
    displacement_jacobian = torch.autograd.functional.jacobian(
        lambda value: model.polarization_increment(graph, value, zero_eta),
        zero_u,
    )[0]
    strain_jacobian = torch.autograd.functional.jacobian(
        lambda value: model.polarization_increment(graph, zero_u, value),
        zero_eta,
    )[0, :, 0]
    volume = torch.linalg.det(graph.cell).abs()
    expected_displacement = (
        model.PIEZO_C_PER_M2 / volume
        * coefficients.born_charges.permute(2, 0, 1)
    )
    assert torch.allclose(
        displacement_jacobian, expected_displacement, atol=3e-6, rtol=3e-6
    )
    assert torch.allclose(
        strain_jacobian,
        cartesian_to_piezo_voigt(coefficients.electronic_piezo)[0],
        atol=3e-6,
        rtol=3e-6,
    )
    assert torch.linalg.vector_norm(coefficients.born_charges.sum(dim=0)) < 2e-6


def test_reduced_polarization_coefficients_match_the_reduced_increment_jacobian():
    torch.manual_seed(102)
    graph = _graph()
    model = _model("reduced").eval()
    zero_u = torch.zeros_like(graph.pos)
    zero_eta = torch.zeros(1, 6)
    coefficients = model.coefficients(graph, create_graph=False)
    strain_jacobian = torch.autograd.functional.jacobian(
        lambda value: model.polarization_increment(graph, zero_u, value),
        zero_eta,
    )[0, :, 0]
    assert torch.equal(
        model.polarization_increment(graph, zero_u, zero_eta),
        torch.zeros(1, 3),
    )
    assert torch.allclose(
        strain_jacobian,
        cartesian_to_piezo_voigt(coefficients.electronic_piezo)[0],
        atol=3e-6,
        rtol=3e-6,
    )


def test_literal_response_coefficients_support_second_order_training():
    torch.manual_seed(103)
    graph = _graph()
    model = _model().train()
    coefficients = model.coefficients(graph, create_graph=True)
    loss = (
        coefficients.born_charges.square().mean()
        + coefficients.electronic_piezo.square().mean()
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("column", (3, 4, 5))
def test_literal_increment_finite_difference_uses_engineering_shear(column: int):
    torch.manual_seed(107)
    graph = _graph()
    model = _model().eval()
    coefficients = model.coefficients(graph, create_graph=False)
    electronic_voigt = cartesian_to_piezo_voigt(
        coefficients.electronic_piezo
    )[0]
    step = 2e-4
    plus = torch.zeros(1, 6)
    minus = torch.zeros(1, 6)
    plus[0, column] = step
    minus[0, column] = -step
    finite_difference = (
        model.polarization_increment(graph, torch.zeros_like(graph.pos), plus)
        - model.polarization_increment(graph, torch.zeros_like(graph.pos), minus)
    )[0] / (2 * step)
    assert torch.allclose(
        finite_difference, electronic_voigt[:, column], atol=2e-4, rtol=2e-3
    )


def test_literal_coefficients_are_o3_covariant():
    torch.manual_seed(109)
    graph = _graph(17)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = _rotated(graph, rotation)
    model = _model().eval()
    prediction = model.coefficients(graph, create_graph=False)
    transformed = model.coefficients(rotated, create_graph=False)
    expected_born = torch.einsum(
        "ia,nab,jb->nij", rotation, prediction.born_charges, rotation
    )
    assert torch.allclose(
        transformed.born_charges, expected_born, atol=8e-5, rtol=8e-5
    )
    assert torch.allclose(
        transformed.electronic_piezo,
        rotate_piezo(prediction.electronic_piezo, rotation),
        atol=8e-5,
        rtol=8e-5,
    )
    displacement = 0.01 * torch.randn_like(graph.pos)
    raw_strain = 0.01 * torch.randn(3, 3)
    strain = 0.5 * (raw_strain + raw_strain.T)
    increment = model.polarization_increment(
        graph, displacement, symmetric_matrix_to_voigt(strain)[None]
    )
    transformed_increment = model.polarization_increment(
        rotated,
        displacement @ rotation.T,
        symmetric_matrix_to_voigt(rotation @ strain @ rotation.T)[None],
    )
    assert torch.allclose(
        transformed_increment, increment @ rotation.T,
        atol=8e-5, rtol=8e-5,
    )


def test_literal_coefficients_are_atom_permutation_and_batch_invariant():
    torch.manual_seed(113)
    graph = _graph()
    permutation = torch.randperm(graph.num_nodes)
    permuted = _permuted(graph, permutation)
    model = _model().eval()
    direct = model.coefficients(graph, create_graph=False)
    reordered = model.coefficients(permuted, create_graph=False)
    assert torch.allclose(
        reordered.born_charges,
        direct.born_charges[permutation],
        atol=3e-5,
        rtol=3e-5,
    )
    assert torch.allclose(
        reordered.electronic_piezo, direct.electronic_piezo,
        atol=3e-5, rtol=3e-5,
    )

    graphs = [_graph(10), _graph(17)]
    for item in graphs:
        del item.batch
    batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False, num_workers=0)))
    batched = model.coefficients(batch, create_graph=False)
    singles = []
    for item in graphs:
        item.batch = torch.zeros(item.num_nodes, dtype=torch.long)
        singles.append(model.coefficients(item, create_graph=False))
    assert torch.allclose(
        batched.electronic_piezo,
        torch.cat([value.electronic_piezo for value in singles]),
        atol=5e-5,
        rtol=5e-5,
    )
    assert torch.allclose(
        batched.born_charges,
        torch.cat([value.born_charges for value in singles]),
        atol=5e-5,
        rtol=5e-5,
    )
