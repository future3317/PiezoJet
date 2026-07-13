import torch

from piezojet.metrics import stabilized_relative_residual, stratified_metrics, tensor_metrics


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
