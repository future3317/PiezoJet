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


def test_direct_and_jvp_sketch_scalar_and_parameter_gradient_agree():
    """The direct tensor contraction and mixed-Hessian JVP are the same map."""
    torch.manual_seed(4)
    piezo_voigt = torch.randn(2, 3, 6, requires_grad=True)
    target = torch.randn_like(piezo_voigt)
    field = torch.randn(2, 3)
    strain = torch.randn(2, 6)
    piezo_cart = piezo_voigt_to_cartesian(piezo_voigt)
    response = ResponsePotential()
    field0, strain0 = torch.zeros_like(field), torch.zeros_like(strain)
    _, mixed = jvp(
        lambda f: jvp(
            lambda eta: response(piezo_cart, f, eta),
            (strain0,),
            (strain,),
        )[1],
        (field0,),
        (field,),
    )
    direct = torch.einsum("bi,bij,bj->b", field, piezo_voigt, strain)
    jvp_value = -mixed
    assert torch.allclose(direct, jvp_value, atol=1e-5, rtol=1e-5)

    direct_target = torch.einsum("bi,bij,bj->b", field, target, strain)
    direct_loss = (direct - direct_target).square().mean()
    jvp_loss = (jvp_value - direct_target).square().mean()
    direct_grad = torch.autograd.grad(direct_loss, piezo_voigt, retain_graph=True)[0]
    jvp_grad = torch.autograd.grad(jvp_loss, piezo_voigt)[0]
    relative = torch.linalg.vector_norm(direct_grad - jvp_grad) / (torch.linalg.vector_norm(direct_grad) + 1e-12)
    assert float(relative) < 1e-4


def test_rademacher_sketch_monte_carlo_identity():
    torch.manual_seed(5)
    matrix = torch.randn(3, 6)
    a = torch.where(torch.rand(200_000, 3) < 0.5, -torch.ones(1), torch.ones(1))
    b = torch.where(torch.rand(200_000, 6) < 0.5, -torch.ones(1), torch.ones(1))
    estimate = (torch.einsum("bi,ij,bj->b", a, matrix, b).square()).mean()
    assert torch.isclose(estimate, matrix.square().sum(), rtol=0.03)
