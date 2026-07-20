import math

import pytest
import torch
from torch_geometric.loader import DataLoader

from piezojet.data import load_gmtnet_records, record_to_graph
from piezojet.model import CrystalGlobalContext, PiezoJet
from piezojet.tensor_ops import rotate_piezo
from tests.data_paths import gmtnet_root


def _graph(index: int):
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[index], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    return graph


def _rotated(graph, rotation: torch.Tensor):
    transformed = graph.clone()
    transformed.pos = graph.pos @ rotation.T
    transformed.edge_shift = graph.edge_shift @ rotation.T
    transformed.cell = graph.cell @ rotation.T
    return transformed


def _unimodular_cell_representation(graph, transform: torch.Tensor):
    """Return the same Cartesian crystal in an integral lattice basis."""
    transformed = graph.clone()
    transformed.cell = transform @ graph.cell
    transformed.frac = graph.frac @ torch.linalg.inv(transform)
    # Cartesian positions and PBC edge shifts describe the same infinite
    # structure and therefore intentionally remain unchanged.
    return transformed


def _scalar_physical_shell(cell: torch.Tensor, cutoff: float):
    volume = torch.linalg.det(cell).abs().clamp_min(torch.finfo(cell.dtype).eps)
    basis = (
        2.0
        * torch.pi
        * torch.linalg.inv(cell).transpose(-1, -2)
        * volume.pow(1.0 / 3.0)
    )
    smallest = torch.linalg.svdvals(basis).amin().clamp_min(
        torch.finfo(cell.dtype).eps
    )
    limit = max(1, math.ceil(cutoff / float(smallest)))
    values = torch.arange(-limit, limit + 1, dtype=cell.dtype)
    indices = torch.stack(
        torch.meshgrid(values, values, values, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    reciprocal = indices @ basis
    norm = torch.linalg.vector_norm(reciprocal, dim=-1)
    keep = (norm > torch.finfo(cell.dtype).eps) & (norm <= cutoff)
    return indices[keep], reciprocal[keep] / norm[keep, None], norm[keep]


def test_batched_reciprocal_bounds_match_scalar_physical_shell_reference():
    cells = torch.tensor([
        [[3.1, 0.2, 0.0], [0.0, 4.2, 0.3], [0.1, 0.0, 5.3]],
        [[4.0, 1.1, 0.2], [0.0, 3.4, 0.5], [0.0, 0.0, 6.2]],
    ])
    cutoff = 4.0
    context = CrystalGlobalContext(
        context_dim=8,
        spectral_channels=2,
        spectral_shells=2,
        polar_fluctuation_shells=2,
        reciprocal_cutoff=cutoff,
    )
    volume = torch.linalg.det(cells).abs()
    basis = (
        2.0
        * torch.pi
        * torch.linalg.inv(cells).transpose(-1, -2)
        * volume.pow(1.0 / 3.0)[:, None, None]
    )
    smallest = torch.linalg.svdvals(basis).amin(dim=-1)
    limits = torch.ceil(cutoff / smallest).clamp_min(1).to(torch.long)
    for index, limit in enumerate(limits.tolist()):
        observed = context._physical_shell_from_basis(basis[index], limit)
        expected = _scalar_physical_shell(cells[index], cutoff)
        assert torch.equal(observed[0], expected[0])
        assert torch.allclose(observed[1], expected[1], atol=1e-7, rtol=1e-7)
        assert torch.allclose(observed[2], expected[2], atol=1e-7, rtol=1e-7)


def test_tensorial_collective_operator_preserves_rotation_equivariance():
    torch.manual_seed(11)
    graph = _graph(10)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    with torch.no_grad():
        features, transformed_features = model.encode(graph), model.encode(_rotated(graph, rotation))
        direct = model.electronic_head(features, graph.batch)
        transformed_direct = model.electronic_head(transformed_features, graph.batch)
        _, operator = model.global_context(graph, graph.batch, model.local_polar_mode(features), return_operator=True)
        _, transformed_operator = model.global_context(
            _rotated(graph, rotation), graph.batch, model.local_polar_mode(transformed_features), return_operator=True
        )
        prediction, transformed = direct + operator, transformed_direct + transformed_operator
    # Randomly initialized equivariant heads can have a near-zero norm, for
    # which a relative residual is ill-defined.  The absolute tensor error is
    # the appropriate exact-symmetry check here.
    assert torch.linalg.vector_norm(transformed - rotate_piezo(prediction, rotation)) < 1e-6


@pytest.mark.parametrize(
    "transform",
    [
        ((1.0, 2.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ],
)
def test_collective_context_and_tensor_operator_are_cell_basis_invariant(transform):
    """A shear GL(3,Z) basis change must only relabel reciprocal vectors."""
    torch.manual_seed(17)
    graph = _graph(17)
    transform = torch.tensor(transform)
    equivalent = _unimodular_cell_representation(graph, transform)
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    with torch.no_grad():
        features = model.encode(graph)
        equivalent_features = model.encode(equivalent)
        context, operator = model.global_context(graph, graph.batch, model.local_polar_mode(features), return_operator=True)
        equivalent_context, equivalent_operator = model.global_context(
            equivalent, equivalent.batch, model.local_polar_mode(equivalent_features), return_operator=True
        )
    assert torch.allclose(context, equivalent_context, atol=2e-5, rtol=2e-5)
    assert torch.allclose(operator, equivalent_operator, atol=2e-5, rtol=2e-5)


def test_collective_cross_spectrum_is_origin_invariant():
    torch.manual_seed(19)
    graph = _graph(11)
    shifted = graph.clone()
    shifted.frac = graph.frac + torch.tensor([0.173, -0.241, 0.389])
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    with torch.no_grad():
        features = model.encode(graph)
        shifted_features = model.encode(shifted)
        _, operator = model.global_context(graph, graph.batch, model.local_polar_mode(features), return_operator=True)
        _, shifted_operator = model.global_context(shifted, shifted.batch, model.local_polar_mode(shifted_features), return_operator=True)
    assert torch.allclose(operator, shifted_operator, atol=2e-5, rtol=2e-5)


def test_collective_context_is_invariant_to_fractional_wrapping():
    torch.manual_seed(23)
    graph = _graph(13)
    wrapped = graph.clone()
    wrapped.frac = torch.remainder(graph.frac + torch.tensor([2.0, -3.0, 5.0]), 1.0)
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    with torch.no_grad():
        features = model.encode(graph)
        wrapped_features = model.encode(wrapped)
        context, operator = model.global_context(graph, graph.batch, model.local_polar_mode(features), return_operator=True)
        wrapped_context, wrapped_operator = model.global_context(
            wrapped, wrapped.batch, model.local_polar_mode(wrapped_features), return_operator=True
        )
    assert torch.allclose(context, wrapped_context, atol=2e-5, rtol=2e-5)
    assert torch.allclose(operator, wrapped_operator, atol=2e-5, rtol=2e-5)


def test_vectorized_collective_context_matches_individual_graphs_and_reuses_cache():
    """Padding/batched GEMM must not make the response batch-dependent."""
    torch.manual_seed(29)
    graphs = [_graph(10), _graph(17)]
    for graph in graphs:
        del graph.batch
    batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False, num_workers=0)))
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    with torch.no_grad():
        features = model.encode(batch)
        context, operator = model.global_context(
            batch, batch.batch, model.local_polar_mode(features), return_operator=True
        )
        cached = model.global_context._geometry_cache
        repeated_context, repeated_operator = model.global_context(
            batch, batch.batch, model.local_polar_mode(features), return_operator=True
        )
        assert model.global_context._geometry_cache is cached
        individual = []
        for graph in graphs:
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
            graph_features = model.encode(graph)
            individual.append(
                model.global_context(
                    graph,
                    graph.batch,
                    model.local_polar_mode(graph_features),
                    return_operator=True,
                )
            )
    expected_context = torch.cat([value[0] for value in individual], dim=0)
    expected_operator = torch.cat([value[1] for value in individual], dim=0)
    assert torch.allclose(context, repeated_context, atol=1e-6, rtol=1e-6)
    assert torch.allclose(operator, repeated_operator, atol=1e-6, rtol=1e-6)
    assert torch.allclose(context, expected_context, atol=2e-5, rtol=2e-5)
    assert torch.allclose(operator, expected_operator, atol=2e-5, rtol=2e-5)
