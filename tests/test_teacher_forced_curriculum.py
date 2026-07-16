import pytest

from piezojet.train import normal_equation_weight_for_epoch


def test_normal_equation_schedule_delays_zero_basin_constraint() -> None:
    assert normal_equation_weight_for_epoch(0.1, 1, 3, 2) == 0.0
    assert normal_equation_weight_for_epoch(0.1, 3, 3, 2) == 0.0
    assert normal_equation_weight_for_epoch(0.1, 4, 3, 2) == pytest.approx(0.05)
    assert normal_equation_weight_for_epoch(0.1, 5, 3, 2) == pytest.approx(0.1)
    assert normal_equation_weight_for_epoch(0.1, 20, 3, 2) == pytest.approx(0.1)


def test_zero_schedule_parameters_reproduce_immediate_historical_weight() -> None:
    assert normal_equation_weight_for_epoch(0.1, 1, 0, 0) == pytest.approx(0.1)


@pytest.mark.parametrize("base,warmup,ramp", [(-0.1, 0, 0), (0.1, -1, 0), (0.1, 0, -1)])
def test_normal_equation_schedule_rejects_invalid_inputs(base: float, warmup: int, ramp: int) -> None:
    with pytest.raises(ValueError):
        normal_equation_weight_for_epoch(base, 1, warmup, ramp)
