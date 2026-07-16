import torch

from piezojet.model import AtomCoordinateResponsePotential
from piezojet.operator_losses import (
    born_charge_probe_loss,
    born_oracle_piezo_loss,
    internal_strain_probe_loss,
    ionic_elastic_response_loss,
    low_mode_operator_action_losses,
    mixed_force_constant_probe_loss,
    phi_oracle_normal_equation_loss,
)
from piezojet.tensor_ops import piezo_voigt_to_cartesian


def _system(dtype=torch.float64):
    atoms = 2
    translation = torch.zeros(3 * atoms, 3, dtype=dtype)
    for axis in range(3):
        translation[axis::3, axis] = atoms ** -0.5
    projector = torch.eye(3 * atoms, dtype=dtype) - translation @ translation.T
    # Three non-degenerate optical values keep the test response well scaled.
    vectors = torch.linalg.qr(projector).Q[:, :3]
    matrix = torch.einsum("a,ia,ja->ij", torch.tensor([0.4, 1.3, 2.1], dtype=dtype), vectors, vectors)
    blocks = AtomCoordinateResponsePotential._blocks_from_matrix(matrix, atoms)
    born = torch.randn(atoms, 3, 3, dtype=dtype)
    born = born - born.mean(dim=0, keepdim=True)
    internal = torch.randn(atoms, 3, 3, 3, dtype=dtype)
    internal = 0.5 * (internal + internal.transpose(-1, -2))
    internal = internal - internal.mean(dim=0, keepdim=True)
    return blocks, born, internal


def test_low_mode_action_and_leak_are_zero_for_true_operator():
    blocks, _, _ = _system()
    flat = blocks.reshape(-1)
    action, leak = low_mode_operator_action_losses(
        flat, flat, torch.tensor([0, 2]), torch.tensor([True]), mode_count=2
    )
    assert action < 1e-20
    assert leak < 1e-20


def test_low_mode_leak_detects_coupling_outside_true_low_subspace():
    blocks, _, _ = _system()
    matrix = AtomCoordinateResponsePotential._matrix_from_blocks(blocks)
    basis = AtomCoordinateResponsePotential._optical_basis(2, matrix)
    values, reduced = torch.linalg.eigh(basis.T @ matrix @ basis)
    modes = basis @ reduced
    order = torch.argsort(values.abs())
    low, hard = modes[:, order[0]], modes[:, order[-1]]
    perturbed = matrix + 0.2 * (low[:, None] @ hard[None, :] + hard[:, None] @ low[None, :])
    predicted = AtomCoordinateResponsePotential._blocks_from_matrix(perturbed, 2).reshape(-1)
    action, leak = low_mode_operator_action_losses(
        predicted, blocks.reshape(-1), torch.tensor([0, 2]), torch.tensor([True]), mode_count=1
    )
    assert action > 0
    assert leak > 0


def test_soft_mode_action_uses_material_spectral_floor_for_bounded_gradient():
    blocks, _, _ = _system()
    matrix = AtomCoordinateResponsePotential._matrix_from_blocks(blocks)
    basis = AtomCoordinateResponsePotential._optical_basis(2, matrix)
    _, reduced = torch.linalg.eigh(basis.T @ matrix @ basis)
    modes = basis @ reduced
    true_matrix = torch.einsum(
        "a,ia,ja->ij",
        torch.tensor([1e-12, 1.0, 2.0], dtype=matrix.dtype),
        modes,
        modes,
    )
    predicted_matrix = true_matrix + 0.01 * modes[:, :1] @ modes[:, :1].T
    predicted = AtomCoordinateResponsePotential._blocks_from_matrix(
        predicted_matrix, 2
    ).reshape(-1).requires_grad_()
    target = AtomCoordinateResponsePotential._blocks_from_matrix(true_matrix, 2).reshape(-1)
    action, _ = low_mode_operator_action_losses(
        predicted, target, torch.tensor([0, 2]), torch.tensor([True]), mode_count=1
    )
    action.backward()
    assert torch.isfinite(action)
    assert torch.isfinite(predicted.grad).all()
    assert torch.linalg.vector_norm(predicted.grad) < 1e4


def test_direct_operator_probes_are_exact_for_true_factors_and_route_gradients():
    blocks, born, internal = _system()
    response = AtomCoordinateResponsePotential(optical_regularization=0.1)
    predicted_phi = blocks.reshape(-1).clone().requires_grad_()
    predicted_lambda = internal.clone().requires_grad_()
    predicted_born = born.clone().requires_grad_()
    node_ptr = torch.tensor([0, 2])
    graph_mask = torch.tensor([True])
    batch = torch.zeros(2, dtype=torch.long)
    phi = mixed_force_constant_probe_loss(
        predicted_phi, blocks.reshape(-1), internal, node_ptr, graph_mask, graph_mask,
        response, material_ids=["synthetic"],
    )
    lam = internal_strain_probe_loss(
        predicted_lambda, internal, graph_mask, batch, material_ids=["synthetic"]
    )
    bec = born_charge_probe_loss(
        predicted_born, born, torch.ones(2, dtype=torch.bool), batch,
        material_ids=["synthetic"],
    )
    total = phi + lam + bec
    total.backward()
    assert total < 1e-20
    assert predicted_phi.grad is not None
    assert predicted_lambda.grad is not None
    assert predicted_born.grad is not None


def test_isolated_born_and_phi_oracles_close_on_true_factors():
    blocks, born, internal = _system()
    response = AtomCoordinateResponsePotential(optical_regularization=0.1)
    coupling = response._coupling_voigt(internal).reshape(6, 6)
    u_true = response.apply_optical_operator(blocks, coupling, solve_policy="regularized")
    cell = torch.diag(torch.tensor([3.0, 3.2, 3.4], dtype=torch.float64)).unsqueeze(0)
    volume = torch.linalg.det(cell[0])
    ionic_voigt = response.PIEZO_C_PER_M2 * born.reshape(6, 3).T @ u_true / volume
    ionic = piezo_voigt_to_cartesian(ionic_voigt).unsqueeze(0)
    node_ptr = torch.tensor([0, 2])
    mask = torch.tensor([True])

    predicted_born = born.clone().requires_grad_()
    z_loss = born_oracle_piezo_loss(
        predicted_born, blocks.reshape(-1), internal, ionic, mask, mask,
        node_ptr, cell, response,
    )
    z_loss.backward()
    assert z_loss < 1e-20
    assert predicted_born.grad is not None

    predicted_phi = blocks.reshape(-1).clone().requires_grad_()
    phi_loss = phi_oracle_normal_equation_loss(
        predicted_phi, blocks.reshape(-1), internal, mask, mask, node_ptr, response
    )
    phi_loss.backward()
    assert phi_loss < 1e-12
    assert predicted_phi.grad is not None

    softening = response.EV_PER_A3_TO_GPA * coupling.T @ u_true / volume
    predicted_softening = softening.unsqueeze(0).clone().requires_grad_()
    elastic_loss = ionic_elastic_response_loss(
        predicted_softening, blocks.reshape(-1), internal, mask, mask,
        node_ptr, cell, response,
    )
    elastic_loss.backward()
    assert elastic_loss < 1e-20
    assert predicted_softening.grad is not None
