import torch
from torch.func import jvp

from piezojet.model import ResponsePotential
from piezojet.tensor_ops import piezo_voigt_to_cartesian


def test_sketch_monte_carlo_identity():
    torch.manual_seed(0)
    matrix = torch.randn(3, 6)
    a, b = torch.randn(200_000, 3), torch.randn(200_000, 6)
    estimate = (torch.einsum("bi,ij,bj->b", a, matrix, b).square()).mean()
    assert torch.isclose(estimate, matrix.square().sum(), rtol=0.03)


def test_mixed_derivative_orders_agree():
    piezo = piezo_voigt_to_cartesian(torch.randn(1, 3, 6))
    response = ResponsePotential()
    field0, eta0 = torch.zeros(1, 3), torch.zeros(1, 6)
    a, b = torch.randn_like(field0), torch.randn_like(eta0)
    _, first = jvp(lambda field: jvp(lambda eta: response(piezo, field, eta), (eta0,), (b,))[1], (field0,), (a,))
    _, second = jvp(lambda eta: jvp(lambda field: response(piezo, field, eta), (field0,), (a,))[1], (eta0,), (b,))
    assert torch.allclose(first, second, atol=1e-5)
