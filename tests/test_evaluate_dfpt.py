import pytest
import torch

from piezojet.evaluate_dfpt import (
    FactorAccumulator,
    clean_force_constant_target,
    ionic_piezo_from_factors,
    optical_eigensystem,
    pair_metrics,
    replace_printed_internal_strain,
    selected_internal_strain,
    soft_mode_metrics,
)
from piezojet.model import AtomCoordinateResponsePotential


def test_pair_metrics_reports_zero_baseline_skill_in_physical_units():
    target = torch.tensor([1.0, -1.0, 2.0])
    perfect = pair_metrics(target, target, floor=0.1)
    zero = pair_metrics(torch.zeros_like(target), target, floor=0.1)
    assert perfect["component_mae"] == 0.0
    assert perfect["mae_skill_vs_zero"] == 1.0
    assert zero["mae_skill_vs_zero"] == pytest.approx(0.0)


def test_factor_accumulator_separates_micro_and_material_macro_metrics():
    accumulator = FactorAccumulator("born_charge")
    accumulator.add(torch.tensor([0.0]), torch.tensor([1.0]))
    accumulator.add(torch.zeros(9), torch.full((9,), 3.0))
    summary = accumulator.summary()
    assert summary["materials"] == 2
    assert summary["micro_component_mae"] == pytest.approx(2.8)
    assert summary["macro_material_component_mae"] == pytest.approx(2.0)
    assert summary["macro_material_mae_skill_vs_zero"] == pytest.approx(0.0)


def test_clean_force_constant_target_has_symmetry_and_three_translations():
    atoms = 3
    raw = torch.randn(atoms, atoms, 3, 3)
    cleaned = clean_force_constant_target(raw.reshape(-1), atoms)
    matrix = cleaned.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
    assert torch.allclose(matrix, matrix.T, atol=1e-6)
    for axis in range(3):
        translation = torch.zeros(3 * atoms)
        translation[axis::3] = 1.0
        assert torch.allclose(matrix @ translation, torch.zeros_like(translation), atol=1e-5)


def test_internal_strain_selects_only_printed_blocks():
    prediction = torch.arange(4 * 3 * 3 * 3, dtype=torch.float32).reshape(4, 3, 3, 3)
    ions = torch.tensor([2, 0])
    directions = torch.tensor([1, 2])
    target = torch.randn(2, 3, 3)
    selected, cleaned_target = selected_internal_strain(
        prediction, target.reshape(-1), ions, directions
    )
    assert torch.equal(selected, torch.stack((prediction[2, 1], prediction[0, 2])))
    assert torch.allclose(cleaned_target, cleaned_target.transpose(-1, -2))


def _two_atom_blocks(eigenvalues: torch.Tensor) -> torch.Tensor:
    relative = torch.zeros(3, 6, dtype=eigenvalues.dtype)
    for axis in range(3):
        relative[axis, axis] = 2.0 ** -0.5
        relative[axis, axis + 3] = -(2.0 ** -0.5)
    matrix = torch.einsum("a,ai,aj->ij", eigenvalues, relative, relative)
    return matrix.reshape(2, 3, 2, 3).permute(0, 2, 1, 3)


def test_optical_eigensystem_removes_translations_and_soft_metrics_are_exact():
    values = torch.tensor([-2.0, 1.0, 3.0])
    blocks = _two_atom_blocks(values)
    optical, _ = optical_eigensystem(blocks)
    assert torch.allclose(optical, values.to(torch.float64))
    born = torch.randn(2, 3, 3)
    metrics = soft_mode_metrics(blocks, blocks, born, born, mode_count=3)
    assert metrics["lowest_optical_eigenvalue_mae"] == pytest.approx(0.0)
    assert metrics["soft_mode_sign_accuracy"] == pytest.approx(1.0)
    assert metrics["soft_mode_subspace_overlap"] == pytest.approx(1.0)
    assert metrics["mode_effective_charge_norm_mae"] == pytest.approx(0.0)


def test_printed_lambda_replacement_changes_only_observed_blocks():
    prediction = torch.zeros(3, 3, 3, 3)
    target = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    ions, directions = torch.tensor([0, 2]), torch.tensor([1, 0])
    replaced = replace_printed_internal_strain(prediction, target.reshape(-1), ions, directions)
    assert torch.count_nonzero(replaced[1]) == 0
    assert torch.allclose(replaced[0, 1], 0.5 * (target[0] + target[0].T))
    assert torch.allclose(replaced[2, 0], 0.5 * (target[1] + target[1].T))


def test_oracle_ionic_response_uses_declared_exact_operator():
    response = AtomCoordinateResponsePotential(optical_stability_cutoff=1e-5)
    blocks = _two_atom_blocks(torch.tensor([2.0, 3.0, 4.0]))
    born = torch.randn(2, 3, 3)
    born = born - born.mean(dim=0, keepdim=True)
    internal = torch.randn(2, 3, 3, 3)
    internal = 0.5 * (internal + internal.transpose(-1, -2))
    internal = internal - internal.mean(dim=0, keepdim=True)
    exact = ionic_piezo_from_factors(response, born, blocks, internal, 10.0, "exact")
    auto = ionic_piezo_from_factors(response, born, blocks, internal, 10.0, "auto")
    assert torch.allclose(exact, auto, atol=1e-6, rtol=1e-6)
