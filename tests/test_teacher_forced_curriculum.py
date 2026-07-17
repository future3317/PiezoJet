import pytest
import torch

from piezojet.train import (
    _displacement_core_parameters,
    _joint_optimizer,
    _optimizer_for_resume,
    _paired_parameter_gradient_metrics,
    displacement_consistency_weight_for_epoch,
)


def test_first_order_consistency_schedule_warms_up_and_ramps() -> None:
    assert displacement_consistency_weight_for_epoch(0.1, 1, 3, 2) == 0.0
    assert displacement_consistency_weight_for_epoch(0.1, 3, 3, 2) == 0.0
    assert displacement_consistency_weight_for_epoch(0.1, 4, 3, 2) == pytest.approx(0.05)
    assert displacement_consistency_weight_for_epoch(0.1, 5, 3, 2) == pytest.approx(0.1)
    assert displacement_consistency_weight_for_epoch(0.1, 20, 3, 2) == pytest.approx(0.1)


def test_zero_schedule_parameters_apply_the_declared_weight_immediately() -> None:
    assert displacement_consistency_weight_for_epoch(0.1, 1, 0, 0) == pytest.approx(0.1)


@pytest.mark.parametrize("base,warmup,ramp", [(-0.1, 0, 0), (0.1, -1, 0), (0.1, 0, -1)])
def test_consistency_schedule_rejects_invalid_inputs(base: float, warmup: int, ramp: int) -> None:
    with pytest.raises(ValueError):
        displacement_consistency_weight_for_epoch(base, 1, warmup, ramp)


def test_paired_u_gradient_metrics_detect_opposition() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    first = parameter.sum()
    second = -parameter.sum()
    result = _paired_parameter_gradient_metrics(first, second, (parameter,))
    assert result["direct_u_gradient_norm"] == pytest.approx(2.0**0.5)
    assert result["true_born_ionic_gradient_norm"] == pytest.approx(2.0**0.5)
    assert result["gradient_cosine"] == pytest.approx(-1.0)


class _TinyResponseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.displacement_encoder = torch.nn.Linear(2, 2)
        self.displacement_local_polar = torch.nn.Linear(2, 2)
        self.displacement_global_context = torch.nn.Linear(2, 2)
        self.displacement_response_head = torch.nn.Linear(2, 2)
        self.displacement_auxiliary_head = torch.nn.Linear(2, 2)
        self.factor_head = torch.nn.Linear(2, 2)


def test_joint_optimizer_restores_teacher_u_moments_before_adding_groups() -> None:
    model = _TinyResponseModel()
    teacher = torch.optim.AdamW(
        _displacement_core_parameters(model), lr=1e-3, weight_decay=1e-6
    )
    loss = sum(parameter.square().sum() for parameter in _displacement_core_parameters(model))
    loss.backward()
    teacher.step()
    optimizer, restored = _joint_optimizer(
        model,
        {
            "learning_rate": 1e-3,
            "joint_displacement_learning_rate": 5e-4,
            "displacement_pretrain_learning_rate": 5e-4,
            "weight_decay": 1e-6,
            "preserve_displacement_optimizer_state": True,
        },
        displacement_optimizer_state=teacher.state_dict(),
    )
    assert restored
    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert optimizer.state
    resumed = _optimizer_for_resume(
        model,
        {"learning_rate": 1e-3, "weight_decay": 1e-6},
        optimizer.state_dict(),
    )
    assert len(resumed.param_groups) == 3
    assert resumed.state
