import pytest
import torch

from piezojet.protocol_ablation import (
    factor_protected_gradient_projection,
    gradient_conflict_metrics,
)


def test_gradient_conflict_reports_aligned_shared_gradients_without_mutating_grad():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    lambda_loss = (parameter - 1.0).square()
    piezo_loss = 3.0 * (parameter - 1.0).square()
    result = gradient_conflict_metrics(lambda_loss, piezo_loss, [parameter])
    assert result["shared_encoder_trainable_parameters"] == 1
    assert result["lambda_gradient_norm"] == pytest.approx(2.0)
    assert result["piezo_gradient_norm"] == pytest.approx(6.0)
    assert result["gradient_cosine"] == pytest.approx(1.0)
    assert parameter.grad is None


def test_gradient_conflict_distinguishes_frozen_stack_from_zero_gradient():
    parameter = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
    result = gradient_conflict_metrics(parameter.detach(), parameter.detach(), [parameter])
    assert result["shared_encoder_trainable_parameters"] == 0
    assert result["lambda_gradient_norm"] is None
    assert result["piezo_gradient_norm"] is None
    assert result["gradient_cosine"] is None


def test_factor_protected_projection_removes_only_conflicting_response_component():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    factor_loss = (parameter - 1.0).square()
    response_loss = (parameter + 1.0).square()
    result = factor_protected_gradient_projection(
        factor_loss, response_loss, [parameter], [parameter]
    )
    # At zero, g_factor=-2 and g_response=+2.  The response gradient is
    # entirely conflicting, so the one-sided projection leaves g_factor.
    assert result["response_gradient_conflict_projected"] is True
    assert result["factor_response_gradient_cosine_before_projection"] == pytest.approx(-1.0)
    assert result["removed_response_gradient_fraction"] == pytest.approx(1.0)
    assert parameter.grad.item() == pytest.approx(-2.0)


def test_factor_protected_projection_preserves_compatible_response_gradient():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    factor_loss = (parameter - 1.0).square()
    response_loss = (parameter - 2.0).square()
    result = factor_protected_gradient_projection(
        factor_loss, response_loss, [parameter], [parameter]
    )
    assert result["response_gradient_conflict_projected"] is False
    assert result["removed_response_gradient_fraction"] == pytest.approx(0.0)
    assert parameter.grad.item() == pytest.approx(-6.0)


def test_factor_protected_projection_can_match_projected_response_norm_without_a_weight():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    factor_loss = (parameter - 1.0).square()
    response_loss = (parameter - 2.0).square()
    result = factor_protected_gradient_projection(
        factor_loss, response_loss, [parameter], [parameter], norm_match=True
    )
    # g_factor=-2 and g_response=-4.  Norm matching scales the latter by 1/2
    # before summation, so each task contributes equal protected-stack norm.
    assert result["response_gradient_scale_after_projection"] == pytest.approx(0.5)
    assert parameter.grad.item() == pytest.approx(-4.0)


def test_factor_protected_projection_leaves_response_only_parameters_unconstrained():
    factor_parameter = torch.nn.Parameter(torch.tensor(0.0))
    response_parameter = torch.nn.Parameter(torch.tensor(0.0))
    factor_loss = (factor_parameter - 1.0).square()
    response_loss = (factor_parameter + 1.0).square() + (response_parameter - 3.0).square()
    factor_protected_gradient_projection(
        factor_loss, response_loss, [factor_parameter], [factor_parameter, response_parameter]
    )
    assert factor_parameter.grad.item() == pytest.approx(-2.0)
    assert response_parameter.grad.item() == pytest.approx(-6.0)
