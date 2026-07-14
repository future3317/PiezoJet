import pytest
import torch

from piezojet.model import AtomCoordinateResponsePotential
from piezojet.train import _near_degenerate_mode_blocks, mode_aware_internal_strain_terms


def _two_atom_blocks(eigenvalues: torch.Tensor) -> torch.Tensor:
    relative = torch.zeros(3, 6, dtype=eigenvalues.dtype)
    for axis in range(3):
        relative[axis, axis] = 2.0 ** -0.5
        relative[axis, axis + 3] = -(2.0 ** -0.5)
    matrix = torch.einsum("a,ai,aj->ij", eigenvalues, relative, relative)
    return matrix.reshape(2, 3, 2, 3).permute(0, 2, 1, 3)


def _coupling() -> torch.Tensor:
    value = torch.randn(2, 3, 3, 3)
    value = 0.5 * (value + value.transpose(-1, -2))
    return value - value.mean(dim=0, keepdim=True)


def _terms(prediction: torch.Tensor, target: torch.Tensor, force: torch.Tensor):
    born = torch.randn(2, 3, 3)
    born = born - born.mean(dim=0, keepdim=True)
    response = AtomCoordinateResponsePotential(optical_regularization=1e-3)
    return mode_aware_internal_strain_terms(
        prediction, target, born, force.reshape(-1), torch.tensor([True]),
        torch.tensor([0, 2]), response, degeneracy_tolerance=1e-4,
    )


def test_mode_aware_loss_is_zero_for_exact_complete_lambda():
    torch.manual_seed(4)
    target = _coupling()
    terms = _terms(target, target, _two_atom_blocks(torch.tensor([1.0, 2.0, 3.0])))
    assert float(terms["direction"]) == pytest.approx(0.0, abs=1e-6)
    assert float(terms["amplitude"]) == pytest.approx(0.0, abs=1e-6)
    assert float(terms["sign"]) == pytest.approx(0.0, abs=1e-6)
    assert float(terms["total"]) == pytest.approx(0.0, abs=1e-6)


def test_mode_aware_loss_has_finite_gradient_through_prediction_at_soft_modes():
    torch.manual_seed(5)
    target = _coupling()
    prediction = torch.zeros_like(target, requires_grad=True)
    terms = _terms(prediction, target, _two_atom_blocks(torch.tensor([-1e-4, 0.2, 2.0])))
    terms["total"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_near_degenerate_block_uses_a_projector_not_an_eigenvector_gauge():
    blocks = _near_degenerate_mode_blocks(_two_atom_blocks(torch.tensor([1.0, 1.00001, 3.0])).double(), 1e-3)
    _, basis = blocks[0]
    rotation, _ = torch.linalg.qr(torch.randn(basis.shape[1], basis.shape[1], dtype=basis.dtype))
    assert basis.shape[1] == 2
    assert torch.allclose(basis @ basis.T, (basis @ rotation) @ (basis @ rotation).T, atol=1e-10)
