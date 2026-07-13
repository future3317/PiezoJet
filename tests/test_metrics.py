import torch

from piezojet.metrics import response_tensor_skill, stabilized_relative_residual, stratified_metrics, tensor_metrics
from piezojet.tensor_ops import rotate_piezo
from piezojet.train import full_loss, response_bin_weights


def test_tensor_metrics_use_stabilized_relative_error_and_strata():
    target = torch.zeros(6, 3, 3, 3)
    target[:, 0, 0, 0] = torch.arange(6, dtype=torch.float32)
    target[:, 0, 1, 2] = target[:, 0, 2, 1] = 0.25
    prediction = target.clone()
    prediction[:, 0, 0, 0] += 0.1
    prediction[:, 0, 1, 2] += 0.1
    prediction[:, 0, 2, 1] += 0.1
    centrosymmetric = torch.tensor([True, False, True, False, True, False])
    result = stratified_metrics(prediction, target, torch.linalg.vector_norm(target.reshape(6, -1), dim=-1), centrosymmetric, 0.05)
    assert torch.isfinite(torch.tensor(tensor_metrics(prediction, target, 0.05)["relative_error_tau"]))
    assert "non_centrosymmetric_summary" in result
    assert "q05_25" in result
    assert result["centro_fp"]["max"] > 0


def test_stabilized_relative_residual_does_not_amplify_near_zero_tensor_roundoff():
    expected = torch.zeros(2, 3, 3, 3)
    actual = expected.clone()
    actual[0, 0, 0, 0] = 1e-9
    absolute, relative = stabilized_relative_residual(actual, expected, floor=1e-2)
    assert torch.allclose(absolute, torch.tensor([1e-9, 0.0]))
    assert torch.allclose(relative, torch.tensor([1e-7, 0.0]))


def test_response_tensor_skill_is_calibrated_against_zero_and_rotation_invariant():
    target = torch.randn(4, 3, 3, 3)
    target = 0.5 * (target + target.transpose(-1, -2))
    zero_metrics = response_tensor_skill(torch.zeros_like(target), target)
    perfect_metrics = response_tensor_skill(target, target)
    assert abs(float(zero_metrics["tensor_response_skill_vs_zero"])) < 1e-6
    assert abs(float(perfect_metrics["tensor_response_skill_vs_zero"]) - 1.0) < 1e-6

    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = response_tensor_skill(rotate_piezo(target, rotation), rotate_piezo(target, rotation))
    assert abs(float(rotated["signal_weighted_relative_frobenius_error"])) < 1e-6


def test_balanced_robust_loss_retains_zero_negatives_and_is_rotation_invariant():
    target = torch.zeros(6, 3, 3, 3)
    target[5, 0, 0, 0] = 2.0
    prediction = target.clone()
    prediction[0, 0, 0, 0] = 0.1
    weights = response_bin_weights(target)
    assert weights[0] > 0
    assert weights[4] > weights[0]
    value = full_loss(prediction, target, torch.tensor(1.0), weights)
    assert value > 0
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = full_loss(rotate_piezo(prediction, rotation), rotate_piezo(target, rotation), torch.tensor(1.0), weights)
    assert torch.allclose(value, rotated, atol=1e-6)
