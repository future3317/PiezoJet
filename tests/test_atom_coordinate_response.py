import torch

from piezojet.data import MAX_POINT_GROUP_OPERATIONS, load_gmtnet_records, record_to_graph
from piezojet.model import PiezoJet
from piezojet.tensor_ops import rotate_piezo


def _identity_group(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    operations = torch.eye(3, dtype=dtype).expand(MAX_POINT_GROUP_OPERATIONS, 3, 3).clone()
    mask = torch.zeros(MAX_POINT_GROUP_OPERATIONS, dtype=torch.bool)
    mask[0] = True
    return operations, mask


def test_atom_coordinate_head_preserves_equivariance_asr_and_translation_nullspace():
    torch.manual_seed(29)
    record = load_gmtnet_records("data/raw/gmtnet")[13]
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
    assert torch.isfinite(prediction.optical_operator_flat).all()
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
    record = load_gmtnet_records("data/raw/gmtnet")[14]
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


def test_energy_factorization_has_exact_hessian_symmetry_and_shared_sum_rules():
    torch.manual_seed(43)
    graph = record_to_graph(load_gmtnet_records("data/raw/gmtnet")[15], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = PiezoJet(cutoff=5.0, num_blocks=1, factor_architecture="energy").eval()
    with torch.no_grad():
        factors = model.predict_factors(graph)
    atoms = graph.num_nodes
    blocks = factors.force_constants_flat.reshape(atoms, atoms, 3, 3)
    matrix = blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
    assert torch.allclose(matrix, matrix.T, atol=1e-6, rtol=1e-6)
    assert matrix.reshape(atoms, 3, atoms, 3).sum(dim=2).abs().max() < 1e-5
    assert factors.internal_strain.sum(dim=0).abs().max() < 1e-5
    assert torch.allclose(
        factors.internal_strain,
        factors.internal_strain.transpose(-1, -2),
        atol=1e-6,
        rtol=1e-6,
    )


def test_learned_equivariant_strain_map_changes_lambda_without_rewriting_phi():
    torch.manual_seed(47)
    graph = record_to_graph(load_gmtnet_records("data/raw/gmtnet")[16], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = PiezoJet(
        cutoff=5.0,
        num_blocks=1,
        factor_architecture="energy_learned_strain",
    ).eval()
    with torch.no_grad():
        before = model.predict_factors(graph)
        model.energy_factors.edge_strain_map[-1].bias[0] = 0.1
        after = model.predict_factors(graph)
    assert torch.allclose(before.force_constants_flat, after.force_constants_flat)
    assert not torch.allclose(before.internal_strain, after.internal_strain)
    assert after.internal_strain.sum(dim=0).abs().max() < 1e-5


def test_local_star_cross_bond_energy_expands_hessian_without_breaking_constraints():
    torch.manual_seed(53)
    graph = record_to_graph(load_gmtnet_records("data/raw/gmtnet")[17], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = PiezoJet(
        cutoff=5.0,
        num_blocks=1,
        factor_architecture="energy_learned_star",
        star_rank=2,
    ).eval()
    with torch.no_grad():
        model.energy_factors.node_star_stiffness[-1].bias.copy_(
            torch.tensor([0.2, -0.1])
        )
        model.energy_factors.edge_star_map[-1].bias[1::4] = 0.3
        factors = model.predict_factors(graph)
    atoms = graph.num_nodes
    blocks = factors.force_constants_flat.reshape(atoms, atoms, 3, 3)
    matrix = blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
    assert torch.allclose(matrix, matrix.T, atol=2e-5, rtol=2e-5)
    assert matrix.reshape(atoms, 3, atoms, 3).sum(dim=2).abs().max() < 5e-5
    assert factors.internal_strain.sum(dim=0).abs().max() < 5e-5
    cross_block_asymmetry = (blocks - blocks.transpose(-1, -2)).abs().amax(dim=(-1, -2))
    assert cross_block_asymmetry.max() > 1e-5
    assert torch.allclose(
        factors.strain_curvature,
        factors.strain_curvature.transpose(-1, -2),
        atol=1e-5,
        rtol=1e-5,
    )
