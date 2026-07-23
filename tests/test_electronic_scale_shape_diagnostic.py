import torch

from piezojet.electronic_scale_shape_diagnostic import (
    apply_l1_mixer,
    apply_irrep_scales,
    fit_l1_mixer,
    fit_per_material_l1_mixers,
    fit_irrep_scales,
    fit_scale,
    oracle_shape_prediction,
    run_diagnostic,
)
from piezojet.tensor_ops import piezo_from_irreps, piezo_to_irreps


def test_l1_mixer_recovers_shared_multiplicity_map():
    torch.manual_seed(5)
    prediction = piezo_from_irreps(torch.randn(12, 18))
    target_coords = piezo_to_irreps(prediction)
    true_map = torch.tensor([[1.4, -0.2], [0.35, 0.8]], dtype=torch.float64)
    copies = torch.stack((target_coords[..., 0:3], target_coords[..., 3:6]), dim=-2)
    mapped = torch.einsum("ab,...bc->...ac", true_map.to(copies), copies)
    target_coords = target_coords.clone()
    target_coords[..., 0:3] = mapped[..., 0, :]
    target_coords[..., 3:6] = mapped[..., 1, :]
    target = piezo_from_irreps(target_coords)
    fitted = fit_l1_mixer(prediction, target)["unconstrained"]
    assert torch.allclose(fitted, true_map, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        apply_l1_mixer(prediction, fitted), target.to(torch.float64), atol=1e-6, rtol=1e-6
    )


def test_l1_mixer_is_batch_permutation_equivariant_and_rotation_independent():
    torch.manual_seed(9)
    prediction = piezo_from_irreps(torch.randn(7, 18))
    mixer = torch.tensor([[0.8, 0.1], [-0.3, 1.2]], dtype=torch.float64)
    mixed = apply_l1_mixer(prediction, mixer)
    permutation = torch.tensor([5, 1, 6, 0, 3, 2, 4])
    assert torch.allclose(
        apply_l1_mixer(prediction[permutation], mixer), mixed[permutation], atol=1e-12
    )


def test_per_material_mixer_diagnostic_recovers_each_map():
    torch.manual_seed(11)
    prediction = piezo_from_irreps(torch.randn(4, 18))
    coordinates = piezo_to_irreps(prediction)
    maps = torch.tensor(
        [[[1.0, 0.1], [0.2, 0.9]], [[0.8, -0.2], [0.3, 1.1]],
         [[1.2, 0.0], [-0.1, 0.7]], [[0.9, 0.4], [0.0, 1.3]]],
        dtype=torch.float64,
    )
    copies = torch.stack((coordinates[..., 0:3], coordinates[..., 3:6]), dim=-2)
    mapped = torch.einsum("nab,nbc->nac", maps.to(copies), copies)
    target_coordinates = coordinates.clone()
    target_coordinates[..., 0:3] = mapped[..., 0, :]
    target_coordinates[..., 3:6] = mapped[..., 1, :]
    fitted = fit_per_material_l1_mixers(prediction, piezo_from_irreps(target_coordinates))
    assert torch.allclose(fitted, maps, atol=1e-6, rtol=1e-6)
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
