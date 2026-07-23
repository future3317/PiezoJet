import torch

from piezojet.electronic_scale_shape_diagnostic import (
    apply_irrep_scales,
    fit_irrep_scales,
    fit_scale,
    oracle_shape_prediction,
    run_diagnostic,
)
from piezojet.tensor_ops import piezo_from_irreps


def test_scalar_and_irrep_scales_recover_known_mapping():
    target_coordinates = torch.randn(6, 18)
    prediction_coordinates = target_coordinates.clone()
    prediction_coordinates[..., :3] *= 0.5
    target = piezo_from_irreps(target_coordinates)
    prediction = piezo_from_irreps(prediction_coordinates)
    scalar = fit_scale(prediction, target)
    assert 1.0 < float(scalar) < 2.0
    scales = fit_irrep_scales(prediction, target)
    restored = apply_irrep_scales(prediction, scales)
    assert torch.linalg.vector_norm(restored - target) < torch.linalg.vector_norm(prediction - target)


def test_oracle_shape_has_unit_amplitude_and_run_is_leakage_safe():
    target = piezo_from_irreps(torch.randn(8, 18))
    prediction = target * torch.linspace(0.2, 1.4, 8).reshape(-1, 1, 1, 1)
    oracle = oracle_shape_prediction(prediction[4:], target[4:])
    assert torch.allclose(
        torch.linalg.vector_norm(oracle.reshape(4, -1), dim=-1),
        torch.linalg.vector_norm(target[4:].reshape(4, -1), dim=-1).to(torch.float64),
        atol=1e-6,
    )
    result = run_diagnostic(prediction, target, calibration_count=4)
    assert result["calibration_materials"] == 4
    assert result["audit_materials"] == 4
    assert result["audit_oracle_per_material_norm"]["active_cosine"] > 0.99
