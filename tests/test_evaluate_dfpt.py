import pytest
import torch

from piezojet.evaluate_dfpt import (
    FactorAccumulator,
    _read_material_ids,
    _oracle_operator_policy,
    clean_force_constant_target,
    electronic_basis_oracle,
    electronic_irrep_metrics,
    ionic_aggregate_metrics,
    ionic_piezo_from_factors,
    low_rank_displacement_oracle,
    optical_eigensystem,
    pair_metrics,
    replace_printed_internal_strain,
    response_active_alignment_metrics,
    response_weighted_log_stiffness_bias,
    response_decomposition_metrics,
    selected_internal_strain,
    soft_mode_metrics,
    spectrum_regularization_regions,
)
from piezojet.model import AtomCoordinateResponsePotential
from piezojet.tensor_ops import (
    PIEZO_IRREP_SLICES,
    electronic_irrep_decomposition,
    piezo_from_irreps,
    piezo_to_irreps,
    piezo_voigt_to_cartesian,
)


def test_pair_metrics_reports_zero_baseline_skill_in_physical_units():
    target = torch.tensor([1.0, -1.0, 2.0])
    perfect = pair_metrics(target, target, floor=0.1)
    zero = pair_metrics(torch.zeros_like(target), target, floor=0.1)
    assert perfect["component_mae"] == 0.0
    assert perfect["mae_skill_vs_zero"] == 1.0
    assert zero["mae_skill_vs_zero"] == pytest.approx(0.0)


def test_electronic_irrep_decomposition_is_complete_and_keeps_two_l1_copies():
    torch.manual_seed(12)
    coordinates = torch.randn(18, dtype=torch.float64)
    tensor = piezo_from_irreps(coordinates)
    blocks = electronic_irrep_decomposition(tensor)
    assert tuple(blocks) == tuple(PIEZO_IRREP_SLICES)
    assert [value.numel() for value in blocks.values()] == [3, 3, 5, 7]
    assert torch.allclose(torch.cat(list(blocks.values())), coordinates, atol=1e-10)
    metrics = electronic_irrep_metrics(tensor, tensor)
    assert all(row["frobenius_error"] == pytest.approx(0.0) for row in metrics.values())


def test_electronic_basis_oracle_reports_full_and_missing_l3_spans():
    basis = piezo_from_irreps(torch.eye(18, dtype=torch.float64))
    target_coordinates = torch.arange(1, 19, dtype=torch.float64)
    target = piezo_from_irreps(target_coordinates)
    full = electronic_basis_oracle(basis, target)
    assert full["rank_in_18d_irrep_space"] == 18
    assert full["minimum_stabilized_relative_residual"] < 1e-12
    assert full["theoretical_maximum_cosine"] == pytest.approx(1.0)
    missing_l3 = electronic_basis_oracle(basis[:11], target)
    assert missing_l3["rank_in_18d_irrep_space"] == 11
    assert missing_l3["per_irrep"]["l3"]["relative_residual"] > 0.99
    fitted_maximum = missing_l3["theoretical_maximum_cosine"]
    expected = torch.linalg.vector_norm(target_coordinates[:11]) / torch.linalg.vector_norm(
        piezo_to_irreps(target)
    )
    assert fitted_maximum == pytest.approx(float(expected))


def test_material_id_reader_accepts_windows_utf8_bom(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text('["JVASP-1"]', encoding="utf-8-sig")
    assert _read_material_ids(path) == ["JVASP-1"]


def test_factor_accumulator_separates_micro_and_material_macro_metrics():
    accumulator = FactorAccumulator("born_charge")
    accumulator.add(torch.tensor([0.0]), torch.tensor([1.0]))
    accumulator.add(torch.zeros(9), torch.full((9,), 3.0))
    summary = accumulator.summary()
    assert summary["materials"] == 2
    assert summary["micro_component_mae"] == pytest.approx(2.8)
    assert summary["macro_material_component_mae"] == pytest.approx(2.0)
    assert summary["macro_material_mae_skill_vs_zero"] == pytest.approx(0.0)


def test_ionic_aggregation_labels_material_macro_and_component_micro_separately():
    # The high-amplitude perfect material dominates a component-micro cosine,
    # while a material macro score gives the weak, failed material one vote.
    targets = [torch.tensor([[1.0]]), torch.tensor([[100.0]])]
    predictions = [torch.zeros(1, 1), torch.tensor([[100.0]])]
    metrics = ionic_aggregate_metrics(predictions, targets, floor=0.05)
    assert metrics["ionic_cosine_macro_material"] == pytest.approx(0.5)
    assert metrics["ionic_cosine_micro_components"] > 0.99
    assert metrics["directional_cosine"] == metrics["ionic_cosine_micro_components"]
    assert metrics["ionic_amplitude_ratio_macro"] == pytest.approx(0.5)
    assert metrics["ionic_active_norm_threshold_c_per_m2"] == pytest.approx(0.05)


def test_ionic_active_panel_uses_registered_physical_norm_floor():
    targets = [torch.tensor([[0.04]]), torch.tensor([[1.0]])]
    predictions = [torch.tensor([[-2.0]]), torch.tensor([[1.0]])]
    metrics = ionic_aggregate_metrics(predictions, targets, floor=0.05)
    assert metrics["ionic_materials"] == 2
    assert metrics["ionic_active_materials"] == 1
    assert metrics["ionic_cosine_active_only"] == pytest.approx(1.0)
    assert metrics["ionic_mae_skill_vs_zero_macro"] == pytest.approx(
        1.0 - (2.04 / 2.0) / ((0.04 + 1.0) / 2.0)
    )


def test_cartesian_piezo_active_threshold_counts_eighteen_independent_components():
    target = piezo_voigt_to_cartesian(torch.ones(3, 6))
    metrics = ionic_aggregate_metrics([target], [target], floor=0.05)
    assert metrics["ionic_active_norm_threshold_c_per_m2"] == pytest.approx(0.05 * 18.0 ** 0.5)


def test_response_decomposition_exposes_branch_magnitude_and_cancellation():
    metrics = response_decomposition_metrics(
        electronic_predictions=[torch.tensor([[3.0]])],
        ionic_predictions=[torch.tensor([[-2.0]])],
        total_targets=[torch.tensor([[1.0]])],
        ionic_targets=[torch.tensor([[0.0]])],
        floor=0.05,
    )
    assert metrics["predicted_electronic_norm_over_true_total_macro"] == pytest.approx(3.0)
    assert metrics["predicted_ionic_norm_over_true_total_macro"] == pytest.approx(2.0)
    assert metrics["predicted_total_norm_over_true_total_macro"] == pytest.approx(1.0)
    assert metrics["predicted_cancellation_ratio_macro"] == pytest.approx(0.2)


def test_oracle_operator_policies_are_explicitly_labelled():
    assert _oracle_operator_policy("true_z_true_phi_pred_lambda_regularized", "regularized") == "regularized"
    assert (
        _oracle_operator_policy("true_z_true_phi_pred_lambda_exact_true_stable", "regularized")
        == "exact_true_dfpt_stable_diagnostic"
    )
    assert (
        _oracle_operator_policy("direct_pred_z_pred_u_regularized", "regularized")
        == "direct_displacement:no_inverse_in_forward"
    )
    assert (
        _oracle_operator_policy(
            "factorized_pred_z_pred_phi_pred_lambda_regularized", "regularized"
        )
        == "model_configured:regularized"
    )


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


def test_spectrum_regularization_regions_partition_modes_without_a_zero_crossing_branch():
    regions = spectrum_regularization_regions(
        torch.tensor([-0.5, 0.0, 0.9, 1.0, 2.0, 3.0, 10.0]), delta=1.0
    )
    assert regions["mode_count"] == 7
    assert regions["below_delta_count"] == 3
    assert regions["delta_to_3delta_count"] == 2
    assert regions["above_3delta_count"] == 2
    assert sum(
        float(regions[name])
        for name in (
            "below_delta_fraction",
            "delta_to_3delta_fraction",
            "above_3delta_fraction",
        )
    ) == pytest.approx(1.0)


def test_rank_six_displacement_oracle_is_exact_but_does_not_claim_six_phonon_modes():
    torch.manual_seed(0)
    displacement = torch.randn(12, 6, dtype=torch.float64)
    born = torch.randn(4, 3, 3, dtype=torch.float64)
    oracle = low_rank_displacement_oracle(displacement, born)
    assert oracle["algebraic_rank_upper_bound"] == 6
    assert oracle["numerical_rank"] == 6
    assert oracle["per_rank"]["6"]["displacement_relative_frobenius_error"] < 1e-12
    assert oracle["per_rank"]["6"]["response_relative_frobenius_error"] < 1e-12
    assert oracle["per_rank"]["1"]["displacement_relative_frobenius_error"] > 0.0
    assert "does not imply" in oracle["interpretation"]


def test_response_active_alignment_uses_projectors_not_mode_pairing():
    torch.manual_seed(4)
    target = torch.randn(12, 6, dtype=torch.float64)
    # An invertible mixing changes the six columns but preserves their span.
    mixing = torch.eye(6, dtype=torch.float64)
    mixing[0, 1] = 0.4
    prediction = target @ mixing
    born = torch.randn(4, 3, 3, dtype=torch.float64)
    metrics = response_active_alignment_metrics(prediction, target, born)
    assert metrics["displacement_subspace_projector_overlap"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert metrics["displacement_subspace_minimum_principal_cosine"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert metrics["target_displacement_rank"] == 6
    assert metrics["predicted_displacement_rank"] == 6


def test_response_active_alignment_cross_covariance_detects_sign_failure():
    torch.manual_seed(5)
    target = torch.randn(9, 6, dtype=torch.float64)
    born = torch.randn(3, 3, 3, dtype=torch.float64)
    metrics = response_active_alignment_metrics(-target, target, born)
    # The coordinate subspace is identical, while the physical cross-covariance
    # has the wrong sign.  The two diagnostics must not be conflated.
    assert metrics["displacement_subspace_projector_overlap"] == pytest.approx(1.0)
    assert metrics["true_charge_cross_covariance_directional_cosine"] == pytest.approx(-1.0)
    assert metrics["true_charge_cross_covariance_amplitude_ratio"] == pytest.approx(1.0)


def test_response_weighted_log_stiffness_bias_uses_true_mode_rayleigh_quotients():
    values = torch.tensor([0.5, 1.5, 4.0], dtype=torch.float64)
    target = _two_atom_blocks(values)
    prediction = _two_atom_blocks(2.0 * values)
    born = torch.randn(2, 3, 3, dtype=torch.float64)
    born = born - born.mean(dim=0, keepdim=True)
    coupling = torch.randn(6, 6, dtype=torch.float64)
    metrics = response_weighted_log_stiffness_bias(
        prediction, target, born, coupling, delta=1e-3
    )
    assert metrics["mean_log_abs_stiffness_bias"] == pytest.approx(
        torch.log(torch.tensor(2.0, dtype=torch.float64)).item(), abs=1e-10
    )
    assert metrics["response_weighted_mean_log_abs_stiffness_bias"] == pytest.approx(
        torch.log(torch.tensor(2.0, dtype=torch.float64)).item(), abs=1e-10
    )


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
    regularized = ionic_piezo_from_factors(
        response, born, blocks, internal, 10.0, "regularized"
    )
    assert torch.isfinite(exact).all()
    assert torch.isfinite(regularized).all()
