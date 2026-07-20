import math

import pytest
import torch
from types import SimpleNamespace

from piezojet.electronic_capacity import (
    born_capacity_metrics,
    born_material_balanced_loss,
    capacity_checkpoint_provenance,
    electronic_capacity_metrics,
    irrep_balanced_capacity_loss,
    response_jet_probe_loss,
    validate_capacity_checkpoint_provenance,
)
from piezojet.model import ElectromechanicalJetPrediction
from piezojet.tensor_ops import piezo_from_irreps


def test_irrep_balanced_capacity_loss_gives_each_block_one_vote():
    target_irreps = torch.ones(1, 18)
    target = piezo_from_irreps(target_irreps)
    perfect = irrep_balanced_capacity_loss(target, target)
    assert perfect == pytest.approx(0.0)
    prediction_irreps = target_irreps.clone()
    prediction_irreps[:, 11:] = 0.0
    prediction = piezo_from_irreps(prediction_irreps)
    # Exactly one of four irreducible blocks has unit normalized error.
    assert irrep_balanced_capacity_loss(prediction, target) == pytest.approx(0.25)


def test_electronic_capacity_metrics_report_active_relative_error_and_cosine():
    target = piezo_from_irreps(torch.ones(2, 18))
    metrics = electronic_capacity_metrics(target, target)
    assert metrics["active_materials"] == 2
    assert metrics["mean_active_relative_frobenius_error"] == pytest.approx(0.0)
    assert metrics["mean_active_cosine"] == pytest.approx(1.0)
    assert metrics["mean_stabilized_amplitude_ratio"] == pytest.approx(1.0)


def test_born_loss_and_metrics_are_material_balanced_and_auditable():
    target = torch.arange(36, dtype=torch.float32).reshape(4, 3, 3) / 10
    batch_index = torch.tensor([0, 0, 1, 1])
    assert born_material_balanced_loss(target, target, batch_index) == pytest.approx(0.0)
    metrics = born_capacity_metrics(target, target, batch_index)
    assert metrics["materials"] == 2
    assert metrics["mean_relative_frobenius_error"] == pytest.approx(0.0)
    assert metrics["mean_stabilized_relative_frobenius_error"] == pytest.approx(0.0)
    assert metrics["mean_cosine"] == pytest.approx(1.0)


def test_born_metrics_stabilize_exact_zero_material_without_hiding_raw_audit():
    prediction = torch.full((2, 3, 3), 1e-7)
    target = torch.zeros_like(prediction)
    batch_index = torch.tensor([0, 0])

    metrics = born_capacity_metrics(prediction, target, batch_index)

    assert metrics["exact_zero_target_materials"] == 1
    assert metrics["mean_exact_zero_prediction_norm_e"] == pytest.approx(
        float(torch.linalg.vector_norm(prediction))
    )
    assert math.isfinite(metrics["mean_stabilized_relative_frobenius_error"])
    assert metrics["mean_stabilized_relative_frobenius_error"] < 1e-5
    assert metrics["mean_relative_frobenius_error"] > 1e20


def test_response_jet_probe_loss_is_zero_for_identical_jacobians():
    torch.manual_seed(17)
    batch = SimpleNamespace(
        pos=torch.zeros(3, 3),
        num_nodes=3,
        batch=torch.tensor([0, 0, 0]),
        cell=torch.eye(3).unsqueeze(0) * 4,
    )
    born = torch.randn(3, 3, 3)
    born = born - born.mean(dim=0)
    electronic = piezo_from_irreps(torch.randn(1, 18))
    target = ElectromechanicalJetPrediction(
        born, electronic, torch.zeros(1, 3, 3)
    )
    assert response_jet_probe_loss(target, target, batch, probes=4) == pytest.approx(0.0)
    shifted = ElectromechanicalJetPrediction(
        born * 0.5, electronic * 0.5, torch.zeros(1, 3, 3)
    )
    assert response_jet_probe_loss(shifted, target, batch, probes=32) > 0.01


def test_material_weighted_microbatch_gradient_equals_full_cohort_mean():
    torch.manual_seed(23)
    target = piezo_from_irreps(torch.randn(8, 18))
    full_coordinates = torch.randn(8, 18, requires_grad=True)
    full_loss = irrep_balanced_capacity_loss(
        piezo_from_irreps(full_coordinates), target
    )
    full_loss.backward()

    micro_coordinates = full_coordinates.detach().clone().requires_grad_(True)
    micro_loss = sum(
        0.5 * irrep_balanced_capacity_loss(
            piezo_from_irreps(micro_coordinates[start : start + 4]),
            target[start : start + 4],
        )
        for start in (0, 4)
    )
    micro_loss.backward()
    assert float(micro_loss.detach()) == pytest.approx(
        float(full_loss.detach()), abs=5e-7
    )
    assert torch.allclose(
        micro_coordinates.grad, full_coordinates.grad, atol=2e-7, rtol=2e-7
    )


def test_capacity_resume_rejects_a_different_same_size_material_cohort():
    args = SimpleNamespace(
        architecture="nonlinear_cartesian",
        seed=42,
        learning_rate=1e-3,
        bec_weight=1.0,
        jet_weight=0.0,
        jet_probes=3,
        train_batch_size=4,
    )
    config = {"jarvis_dfpt_dir": "dfpt"}
    saved = capacity_checkpoint_provenance(["a", "b"], args, config)
    current = capacity_checkpoint_provenance(["a", "c"], args, config)
    with pytest.raises(ValueError, match="same-ID cohort"):
        validate_capacity_checkpoint_provenance(
            {"capacity_checkpoint_provenance": saved}, current
        )
