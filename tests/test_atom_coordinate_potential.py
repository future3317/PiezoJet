import torch
from torch.func import jvp

from piezojet.data import load_gmtnet_records, record_to_graph
from piezojet.model import AtomCoordinateResponsePotential, PiezoJet
from tests.data_paths import gmtnet_root


def test_independent_phi_lambda_coefficients_share_one_scalar_energy():
    torch.manual_seed(38)
    response = AtomCoordinateResponsePotential()
    atoms = 3
    raw = torch.randn(3 * atoms, 3 * atoms)
    phi = 0.5 * (raw + raw.T)
    blocks = phi.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)
    internal = torch.randn(atoms, 3, 3, 3)
    internal = 0.5 * (internal + internal.transpose(-1, -2))
    internal = internal - internal.mean(dim=0, keepdim=True)
    curvature = torch.randn(6, 6)
    curvature = 0.5 * (curvature + curvature.T)
    u0 = torch.zeros(atoms, 3)
    eta0 = torch.zeros(6)
    u_direction = torch.randn_like(u0)
    eta_direction = torch.randn_like(eta0)
    _, mixed = jvp(
        lambda eta: jvp(
            lambda u: response.internal_quadratic_energy(
                blocks, internal, curvature, u, eta
            ),
            (u0,),
            (u_direction,),
        )[1],
        (eta0,),
        (eta_direction,),
    )
    coupling = response._coupling_voigt(internal).reshape(3 * atoms, 6)
    expected = -torch.einsum("i,ij,j->", u_direction.reshape(-1), coupling, eta_direction)
    assert torch.allclose(mixed, expected.to(mixed), atol=1e-6, rtol=1e-6)


def test_optical_displacement_stationarity_translation_removal_and_response_shapes():
    torch.manual_seed(41)
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[23], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    field, eta6 = torch.randn(1, 3), torch.randn(1, 6)
    with torch.no_grad():
        components = model.predict_components(graph)
        atoms = graph.num_nodes
        force_blocks = components.force_constants_flat.reshape(atoms, atoms, 3, 3)
        force_matrix = force_blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
        coupling = model.response._coupling_voigt(components.internal_strain).reshape(3 * atoms, 6)
        charge = components.born_charges.reshape(3 * atoms, 3)
        generalized_force = coupling @ eta6[0] + charge @ field[0]
        displacement = model.response.apply_optical_operator(
            force_blocks, generalized_force.unsqueeze(-1), solve_policy="regularized"
        ).squeeze(-1)
        translation = displacement.reshape(atoms, 3).sum(dim=0)
        translation_basis = force_matrix.new_zeros(3 * atoms, 3)
        for axis in range(3):
            translation_basis[axis::3, axis] = atoms ** -0.5
        projector = torch.eye(3 * atoms) - translation_basis @ translation_basis.T
        residual = (
            force_matrix @ force_matrix
            + model.response.optical_regularization ** 2 * projector
        ) @ displacement - force_matrix @ projector @ generalized_force
    assert torch.linalg.vector_norm(residual) < 2e-3
    assert torch.linalg.vector_norm(translation) / torch.linalg.vector_norm(displacement).clamp_min(1e-8) < 1e-6
    assert components.dielectric.shape == (1, 3, 3)
    assert components.ionic_dielectric.shape == (1, 3, 3)
    assert components.elastic.shape == (1, 6, 6)
    assert components.elastic_softening.shape == (1, 6, 6)


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


def test_default_optical_policy_is_tikhonov_not_predicted_spectrum_switch():
    response = AtomCoordinateResponsePotential()
    assert response.optical_solve_policy == "tikhonov"
    model = PiezoJet(cutoff=5.0, num_blocks=1)
    assert model.response.optical_solve_policy == "tikhonov"
    assert model.ionic_parameterization == "isolated_global_octupole_displacement"
    assert not hasattr(model, "observable_lift_policy")


def test_total_only_macro_tower_has_no_gradient_route_to_physical_factors():
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[5], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = PiezoJet(cutoff=5.0, num_blocks=1)
    model.predict_macro_total(graph).square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.macro_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.response_factors.parameters())
    assert all(parameter.grad is None for parameter in model.displacement_response_head.parameters())


def test_direct_u_coordinate_head_has_no_gradient_route_to_factor_or_macro_towers():
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[6], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = PiezoJet(cutoff=5.0, num_blocks=1)
    model.predict_displacement_response(graph).square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.displacement_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.response_factors.parameters())
    assert all(parameter.grad is None for parameter in model.macro_encoder.parameters())


def test_multistream_pruned_forward_preserves_active_physical_outputs(monkeypatch):
    torch.manual_seed(39)
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[7], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = PiezoJet(cutoff=5.0, num_blocks=1).eval()
    with torch.no_grad():
        reference = model.predict_components(graph)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("inactive response tower was evaluated")

    monkeypatch.setattr(model, "predict_macro_responses", forbidden)
    monkeypatch.setattr(model.response, "responses", forbidden)
    with torch.no_grad():
        pruned = model.predict_components(
            graph,
            compute_macro_response=False,
            compute_factorized_response=False,
        )
    for name in (
        "electronic_piezo",
        "direct_u_ionic_piezo",
        "displacement_response",
        "born_charges",
        "force_constants_flat",
        "internal_strain",
    ):
        assert torch.allclose(
            getattr(pruned, name), getattr(reference, name),
            atol=2e-6, rtol=2e-6,
        )
    # With compute_factorized_response=False the factor/Schur ionic path is
    # disabled, so physical_tensor and ionic_piezo collapse to the electronic
    # contribution only.
    assert torch.allclose(pruned.physical_tensor, pruned.electronic_piezo)
    assert torch.count_nonzero(pruned.ionic_piezo) == 0
    assert torch.count_nonzero(pruned.tensor) == 0


def test_pruned_strict_objective_keeps_inactive_towers_out_of_autograd():
    torch.manual_seed(40)
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[8], 5.0, 32)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = PiezoJet(cutoff=5.0, num_blocks=1)
    components = model.predict_components(
        graph,
        compute_macro_response=False,
        compute_factorized_response=False,
    )
    strict_like_loss = (
        components.displacement_response.square().mean()
        + components.internal_strain.square().mean()
    )
    strict_like_loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.displacement_response_head.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.response_factors.cross_derivative_head.parameters()
    )
    for inactive in (
        model.macro_encoder,
        model.macro_total_head,
        model.electronic_head,
        model.background,
        model.born_head,
    ):
        assert all(parameter.grad is None for parameter in inactive.parameters())


def test_explicit_stable_dfpt_diagnostic_uses_exact_stationary_optical_inverse():
    response = AtomCoordinateResponsePotential(
        optical_regularization=0.1,
        optical_stability_cutoff=1e-5,
        optical_solve_policy="exact",
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


def test_elastic_response_counts_shared_strain_curvature_once():
    # Keep the independently recomputed optical solve away from an accidental
    # cancellation-sensitive random draw; this is an algebraic identity test,
    # not a random stress test.
    torch.manual_seed(0)
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
    _, _, elastic, _, shared, _, _ = response.responses(
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
    coupling = response._coupling_voigt(internal).reshape(3 * atoms, 6)
    softening = response.EV_PER_A3_TO_GPA * coupling.T @ response.apply_optical_operator(
        blocks, coupling, "regularized"
    ) / volume
    expected_shared = response.EV_PER_A3_TO_GPA * curvature[0] / volume
    assert torch.allclose(shared[0], expected_shared)
    assert torch.allclose(elastic[0], elastic_direct[0] + expected_shared - softening)
