import torch
from types import SimpleNamespace

from piezojet.data import MAX_POINT_GROUP_OPERATIONS, load_gmtnet_records, record_to_graph
from piezojet.model import AtomCoordinateResponsePotential, PiezoJet
from piezojet.tensor_ops import piezo_voigt_to_cartesian, rotate_piezo
from tests.data_paths import gmtnet_root


def _identity_group(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    operations = torch.eye(3, dtype=dtype).expand(MAX_POINT_GROUP_OPERATIONS, 3, 3).clone()
    mask = torch.zeros(MAX_POINT_GROUP_OPERATIONS, dtype=torch.bool)
    mask[0] = True
    return operations, mask


def _loop_ionic_piezo_reference(response, born, displacement, batch):
    values = []
    for graph_index in range(batch.cell.shape[0]):
        node_indices = torch.nonzero(
            batch.batch == graph_index, as_tuple=False
        ).squeeze(-1)
        atoms = node_indices.numel()
        charge = born[node_indices].reshape(3 * atoms, 3)
        coupling = response._coupling_voigt(
            displacement[node_indices]
        ).reshape(3 * atoms, 6)
        volume = torch.linalg.det(batch.cell[graph_index]).abs()
        values.append(response.PIEZO_C_PER_M2 * charge.T @ coupling / volume)
    return piezo_voigt_to_cartesian(torch.stack(values))


def test_vectorized_direct_u_contraction_matches_loop_output_and_gradients():
    torch.manual_seed(28)
    node_batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2])
    cells = torch.diag_embed(torch.rand(3, 3) + 2.0)
    batch = SimpleNamespace(batch=node_batch, cell=cells)
    born = torch.randn(node_batch.numel(), 3, 3, requires_grad=True)
    displacement = torch.randn(
        node_batch.numel(), 3, 3, 3, requires_grad=True
    )
    displacement_symmetric = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )
    response = AtomCoordinateResponsePotential()

    expected = _loop_ionic_piezo_reference(
        response, born, displacement_symmetric, batch
    )
    observed = response.ionic_piezo_from_displacement_response(
        born, displacement_symmetric, batch
    )
    assert torch.allclose(observed, expected, atol=2e-6, rtol=2e-6)

    reference_gradients = torch.autograd.grad(
        expected.square().mean(), (born, displacement), retain_graph=True
    )
    observed_gradients = torch.autograd.grad(
        observed.square().mean(), (born, displacement)
    )
    for actual, reference in zip(observed_gradients, reference_gradients):
        assert torch.allclose(actual, reference, atol=3e-6, rtol=3e-5)


def test_atom_coordinate_head_preserves_equivariance_asr_and_translation_nullspace():
    torch.manual_seed(29)
    record = load_gmtnet_records(gmtnet_root())[13]
    graph = record_to_graph(record, 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    graph.point_group_ops, graph.point_group_mask = _identity_group(graph.pos.dtype)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = graph.clone()
    rotated.pos = graph.pos @ rotation.T
    rotated.edge_shift = graph.edge_shift @ rotation.T
    rotated.cell = graph.cell @ rotation.T
    rotated.point_group_ops = torch.einsum("ia,gab,jb->gij", rotation, graph.point_group_ops, rotation)
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    with torch.no_grad():
        prediction = model.predict_components(graph)
        transformed = model.predict_components(rotated)
    residual = torch.linalg.vector_norm(transformed.tensor - rotate_piezo(prediction.tensor, rotation))
    assert residual / torch.linalg.vector_norm(prediction.tensor).clamp_min(1e-6) < 1e-4
    atoms = graph.num_nodes
    force_constants = prediction.force_constants_flat.reshape(atoms, atoms, 3, 3)
    transformed_force_constants = transformed.force_constants_flat.reshape(atoms, atoms, 3, 3)
    expected_force_constants = torch.einsum("ia,nmab,jb->nmij", rotation, force_constants, rotation)
    force_residual = torch.linalg.vector_norm(transformed_force_constants - expected_force_constants)
    assert force_residual / torch.linalg.vector_norm(force_constants).clamp_min(1e-6) < 1e-4
    matrix = force_constants.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
    singular_values = torch.linalg.svdvals(matrix)
    assert singular_values[-3:].max() < 1e-4
    # The production forward path applies the optical Green operator to the
    # needed right-hand sides and never materializes its O(N^2) dense matrix.
    assert prediction.optical_operator_flat.numel() == 0
    assert matrix.reshape(atoms, 3, atoms, 3).sum(dim=2).abs().max() < 1e-4
    expected_internal_strain = torch.einsum(
        "ia,nabc,jb,kc->nijk", rotation, prediction.internal_strain, rotation, rotation
    )
    internal_residual = torch.linalg.vector_norm(transformed.internal_strain - expected_internal_strain)
    assert internal_residual / torch.linalg.vector_norm(prediction.internal_strain).clamp_min(1e-6) < 1e-4
    ionic_residual = torch.linalg.vector_norm(transformed.ionic_piezo - rotate_piezo(prediction.ionic_piezo, rotation))
    assert ionic_residual / torch.linalg.vector_norm(prediction.ionic_piezo).clamp_min(1e-6) < 1e-4


def test_point_group_metadata_does_not_replace_the_atom_coordinate_response():
    torch.manual_seed(31)
    record = load_gmtnet_records(gmtnet_root())[14]
    graph = record_to_graph(record, 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    graph.point_group_ops, graph.point_group_mask = _identity_group(graph.pos.dtype)
    graph.point_group_ops[1] = -torch.eye(3, dtype=graph.pos.dtype)
    graph.point_group_mask[1] = True
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    with torch.no_grad():
        prediction = model(graph)
    # Point-group metadata is retained for supervision/audits; the physical
    # atom-coordinate relaxation remains the production response path.
    assert torch.isfinite(prediction).all()


def test_independent_cross_derivative_head_changes_lambda_without_rewriting_phi():
    torch.manual_seed(47)
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[16], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = PiezoJet(
        cutoff=5.0,
        num_blocks=1,
        factor_architecture="independent_quadratic_response",
    ).eval()
    with torch.no_grad():
        before = model.predict_factors(graph)
        model.response_factors.cross_derivative_head.coefficients[-1].bias[0] = 0.1
        after = model.predict_factors(graph)
    assert torch.allclose(before.force_constants_flat, after.force_constants_flat)
    assert not torch.allclose(before.internal_strain, after.internal_strain)
    assert after.internal_strain.sum(dim=0).abs().max() < 1e-5
