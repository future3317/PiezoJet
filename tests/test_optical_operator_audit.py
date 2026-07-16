import torch

from piezojet.model import AtomCoordinateResponsePotential


def _blocks_from_reduced(reduced: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    atoms = 3
    reference = torch.zeros(3 * atoms, 3 * atoms, dtype=reduced.dtype)
    basis = AtomCoordinateResponsePotential._optical_basis(atoms, reference)
    matrix = basis @ reduced @ basis.T
    blocks = matrix.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)
    return blocks, basis


def _explicit_signed_resolvent(
    basis: torch.Tensor,
    reduced: torch.Tensor,
    rhs: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(reduced)
    weights = values / (values.square() + delta**2)
    return basis @ vectors @ torch.diag(weights) @ vectors.T @ basis.T @ rhs


def test_regularized_operator_matches_eigendecomposition_and_optical_basis_rotation():
    torch.manual_seed(701)
    dtype = torch.float64
    delta = 0.07
    raw = torch.randn(6, 6, dtype=dtype)
    reduced = 0.5 * (raw + raw.T) + torch.diag(
        torch.tensor([-0.3, 1e-10, 1e-10, 0.8, 1.2, 2.0], dtype=dtype)
    )
    blocks, basis = _blocks_from_reduced(reduced)
    rhs = torch.randn(9, 4, dtype=dtype)
    response = AtomCoordinateResponsePotential(
        optical_regularization=delta, optical_solve_policy="regularized"
    )
    actual = response.apply_optical_operator(blocks, rhs)
    expected = _explicit_signed_resolvent(basis, reduced, rhs, delta)
    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-10)

    rotation, _ = torch.linalg.qr(torch.randn(6, 6, dtype=dtype))
    rotated_basis = basis @ rotation
    rotated_reduced = rotation.T @ reduced @ rotation
    rotated_expected = _explicit_signed_resolvent(
        rotated_basis, rotated_reduced, rhs, delta
    )
    relative = torch.linalg.vector_norm(rotated_expected - expected) / torch.linalg.vector_norm(
        expected
    )
    assert relative < 1e-10


def test_regularized_operator_vjp_matches_central_finite_difference_near_degeneracy():
    torch.manual_seed(702)
    dtype = torch.float64
    delta = 0.11
    base = torch.diag(torch.tensor([-0.2, -0.2 + 1e-9, 0.0, 0.0, 0.7, 1.4], dtype=dtype))
    direction_raw = torch.randn(6, 6, dtype=dtype)
    direction = 0.5 * (direction_raw + direction_raw.T)
    direction = direction / torch.linalg.vector_norm(direction)
    rhs = torch.randn(9, 3, dtype=dtype)
    probe = torch.randn(9, 3, dtype=dtype)
    response = AtomCoordinateResponsePotential(
        optical_regularization=delta, optical_solve_policy="regularized"
    )

    reduced = base.clone().requires_grad_()
    blocks, _ = _blocks_from_reduced(reduced)
    scalar = (response.apply_optical_operator(blocks, rhs) * probe).sum()
    (gradient,) = torch.autograd.grad(scalar, reduced)
    autograd_directional = (gradient * direction).sum()

    step = 1e-6
    plus_blocks, _ = _blocks_from_reduced(base + step * direction)
    minus_blocks, _ = _blocks_from_reduced(base - step * direction)
    plus = (response.apply_optical_operator(plus_blocks, rhs) * probe).sum()
    minus = (response.apply_optical_operator(minus_blocks, rhs) * probe).sum()
    finite_difference = (plus - minus) / (2 * step)
    relative = (autograd_directional - finite_difference).abs() / finite_difference.abs().clamp_min(
        1e-10
    )
    assert torch.isfinite(gradient).all()
    assert relative < 1e-4


def test_regularized_operator_is_finite_for_repeated_and_zero_optical_modes():
    dtype = torch.float64
    reduced = torch.diag(torch.tensor([-1.0, -1.0, 0.0, 0.0, 2.0, 2.0], dtype=dtype))
    blocks, _ = _blocks_from_reduced(reduced)
    rhs = torch.arange(27, dtype=dtype).reshape(9, 3)
    response = AtomCoordinateResponsePotential(
        optical_regularization=1e-3, optical_solve_policy="regularized"
    )
    result = response.apply_optical_operator(blocks, rhs)
    assert torch.isfinite(result).all()
