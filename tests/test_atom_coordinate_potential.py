import torch
from torch.func import jvp

from piezojet.data import load_gmtnet_records, record_to_graph
from piezojet.model import AtomCoordinateResponsePotential, PiezoJet
from piezojet.tensor_ops import cartesian_to_piezo_voigt


def test_response_energy_density_mixed_derivative_matches_predicted_piezo():
    torch.manual_seed(37)
    graph = record_to_graph(load_gmtnet_records("data/raw/gmtnet")[19], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    field0 = torch.zeros(1, 3)
    eta0 = torch.zeros(1, 6)
    field_direction = torch.randn_like(field0)
    strain_direction = torch.randn_like(eta0)
    with torch.no_grad():
        direct = cartesian_to_piezo_voigt(model(graph))
    _, mixed = jvp(
        lambda field: jvp(lambda eta: model.potential(graph, field, eta), (eta0,), (strain_direction,))[1],
        (field0,),
        (field_direction,),
    )
    expected = -torch.einsum("bi,bij,bj->b", field_direction, direct, strain_direction)
    expected = expected / model.response.PIEZO_C_PER_M2
    assert torch.allclose(mixed, expected, atol=1e-5, rtol=1e-5)


def test_optical_displacement_stationarity_translation_removal_and_response_shapes():
    torch.manual_seed(41)
    graph = record_to_graph(load_gmtnet_records("data/raw/gmtnet")[23], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    field, eta6 = torch.randn(1, 3), torch.randn(1, 6)
    with torch.no_grad():
        components = model.predict_components(graph)
        atoms = graph.num_nodes
        force_blocks = components.force_constants_flat.reshape(atoms, atoms, 3, 3)
        force_matrix = force_blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
        inverse_blocks = components.optical_operator_flat.reshape(atoms, atoms, 3, 3)
        inverse = inverse_blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
        coupling = model.response._coupling_voigt(components.internal_strain).reshape(3 * atoms, 6)
        charge = components.born_charges.reshape(3 * atoms, 3)
        generalized_force = coupling @ eta6[0] + charge @ field[0]
        displacement = inverse @ generalized_force
        translation = displacement.reshape(atoms, 3).sum(dim=0)
        translation_basis = force_matrix.new_zeros(3 * atoms, 3)
        for axis in range(3):
            translation_basis[axis::3, axis] = atoms ** -0.5
        projector = torch.eye(3 * atoms) - translation_basis @ translation_basis.T
        normal = (
            force_matrix @ force_matrix
            + model.response.optical_regularization ** 2 * projector
            + translation_basis @ translation_basis.T
        )
        residual = normal @ displacement - force_matrix @ generalized_force
    assert torch.linalg.vector_norm(residual) < 2e-3
    assert torch.linalg.vector_norm(translation) / torch.linalg.vector_norm(displacement).clamp_min(1e-8) < 1e-6
    assert components.dielectric.shape == (1, 3, 3)
    assert components.elastic.shape == (1, 6, 6)


def test_signed_regularized_green_retains_negative_modes_and_has_soft_mode_gradient():
    response = AtomCoordinateResponsePotential(optical_regularization=0.1)
    relative = torch.zeros(3, 6)
    for axis in range(3):
        relative[axis, axis] = 2.0 ** -0.5
        relative[axis, axis + 3] = -(2.0 ** -0.5)
    eigenvalues = torch.tensor([2.0, -3.0, 1e-6], requires_grad=True)
    matrix = torch.einsum("a,ai,aj->ij", eigenvalues, relative, relative)
    blocks = matrix.reshape(2, 3, 2, 3).permute(0, 2, 1, 3)
    inverse_matrix = response.signed_regularized_optical_green(blocks)
    expected = eigenvalues / (eigenvalues.square() + 0.1 ** 2)
    observed = torch.einsum("ai,ij,aj->a", relative, inverse_matrix, relative)
    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-5)
    observed[-1].backward()
    assert torch.isfinite(eigenvalues.grad).all()
    assert eigenvalues.grad[-1].abs() > 0


def test_stable_auto_policy_uses_exact_stationary_optical_inverse():
    response = AtomCoordinateResponsePotential(
        optical_regularization=0.1,
        optical_stability_cutoff=1e-5,
        optical_solve_policy="auto",
    )
    relative = torch.zeros(3, 6)
    for axis in range(3):
        relative[axis, axis] = 2.0 ** -0.5
        relative[axis, axis + 3] = -(2.0 ** -0.5)
    eigenvalues = torch.tensor([2.0, 3.0, 4.0])
    matrix = torch.einsum("a,ai,aj->ij", eigenvalues, relative, relative)
    blocks = matrix.reshape(2, 3, 2, 3).permute(0, 2, 1, 3)
    inverse = response.optical_operator(blocks)
    observed = torch.einsum("ai,ij,aj->a", relative, inverse, relative)
    assert torch.allclose(observed, eigenvalues.reciprocal(), atol=1e-6, rtol=1e-6)
    projector = relative.T @ relative
    assert torch.allclose(matrix @ inverse, projector, atol=1e-6, rtol=1e-6)


def test_response_generator_converts_all_si_blocks_to_one_energy_density_unit():
    response = AtomCoordinateResponsePotential()
    piezo = torch.ones(1, 3, 3, 3) * response.PIEZO_C_PER_M2
    elastic = torch.eye(6).unsqueeze(0) * response.EV_PER_A3_TO_GPA
    dielectric = torch.eye(3).unsqueeze(0) * response.DIELECTRIC_RELATIVE
    field = torch.tensor([[1.0, 0.0, 0.0]])
    eta = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    energy_density = response(piezo, elastic, dielectric, field, eta)
    # -E e eta + 1/2 eta C eta - 1/2 E epsilon E = -1 here.
    assert torch.allclose(energy_density, torch.tensor([-1.0]))


def test_elastic_response_counts_shared_strain_curvature_once():
    response = AtomCoordinateResponsePotential(optical_solve_policy="regularized")
    atoms = 2
    relative = torch.zeros(3, 3 * atoms)
    for axis in range(3):
        relative[axis, axis] = 2.0 ** -0.5
        relative[axis, axis + 3] = -(2.0 ** -0.5)
    matrix = torch.einsum("a,ai,aj->ij", torch.tensor([2.0, 3.0, 4.0]), relative, relative)
    blocks = matrix.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)
    internal = torch.randn(atoms, 3, 3, 3)
    internal = 0.5 * (internal + internal.transpose(-1, -2))
    internal = internal - internal.mean(dim=0, keepdim=True)
    curvature = torch.eye(6).unsqueeze(0) * 0.7
    elastic_direct = torch.eye(6).unsqueeze(0) * 5.0
    batch = type("Batch", (), {})()
    batch.batch = torch.zeros(atoms, dtype=torch.long)
    batch.cell = torch.diag(torch.tensor([2.0, 2.0, 2.5])).unsqueeze(0)
    _, _, elastic, _, shared = response.responses(
        torch.zeros(1, 3, 3, 3),
        torch.zeros(atoms, 3, 3),
        internal,
        blocks.reshape(-1),
        curvature,
        batch,
        elastic_direct,
        torch.eye(3).unsqueeze(0),
    )
    volume = torch.linalg.det(batch.cell[0])
    operator = response.optical_operator(blocks, "regularized")
    coupling = response._coupling_voigt(internal).reshape(3 * atoms, 6)
    softening = response.EV_PER_A3_TO_GPA * coupling.T @ operator @ coupling / volume
    expected_shared = response.EV_PER_A3_TO_GPA * curvature[0] / volume
    assert torch.allclose(shared[0], expected_shared)
    assert torch.allclose(elastic[0], elastic_direct[0] + expected_shared - softening)
