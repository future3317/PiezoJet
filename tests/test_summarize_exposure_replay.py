import pytest

from piezojet.summarize_exposure_replay import (
    _physical_statistics,
    hierarchical_seed_material_interval,
    paired_macro_difference_interval,
)


def _row(material_id: str, scale: float) -> dict[str, object]:
    target = [0.0] * 27
    target[0] = 1.0
    prediction = [scale * value for value in target]
    return {
        "material_id": material_id,
        "total_prediction": prediction,
        "total_target": target,
        "ionic_prediction": prediction,
        "ionic_target": target,
    }


def test_hierarchical_bootstrap_reports_seed_mean_without_selecting_a_point():
    rows = {
        7: [_row("a", 1.0), _row("b", 1.0)],
        42: [_row("a", 0.0), _row("b", 0.0)],
    }
    interval = hierarchical_seed_material_interval(
        rows,
        lambda values: _physical_statistics(values)[
            "ionic_mae_skill_vs_zero_macro"
        ],
        resamples=200,
        seed=3,
    )
    assert interval["point_estimate_seed_mean"] == pytest.approx(0.5)
    assert interval["lower_95"] <= 0.5 <= interval["upper_95"]


def test_paired_macro_interval_is_zero_for_identical_predictions():
    rows = {
        7: [_row("a", 0.5), _row("b", 1.0)],
        42: [_row("a", 0.0), _row("b", 0.5)],
    }
    interval = paired_macro_difference_interval(
        rows, rows, resamples=100, seed=4
    )
    assert interval["point_estimate_seed_mean"] == pytest.approx(0.0)
    assert interval["lower_95"] == pytest.approx(0.0)
    assert interval["upper_95"] == pytest.approx(0.0)
