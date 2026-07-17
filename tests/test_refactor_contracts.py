"""Regression tests for public contracts introduced by maintenance refactors."""

import inspect
from pathlib import Path

import torch

from piezojet import train
from piezojet.model import (
    AtomCoordinateResponsePotential,
    CartesianLocalEnvironmentEncoder,
    model_from_config,
)
from piezojet.projectors import translation_projector
from piezojet.train import first_order_spectral_residual_diagnostics


def test_model_from_config_defaults_to_the_energy_integrable_factor_architecture():
    model = model_from_config({
        "embedding_dim": 8,
        "cutoff": 5.0,
        "num_blocks": 1,
        "radial_basis": 4,
        "radial_hidden": 11,
    })
    assert model.factor_architecture == "independent_quadratic_response"
    assert hasattr(model, "macroscopic_response_density")
    assert not hasattr(model, "potential")


def test_cartesian_encoder_exposes_scalar_dimension_without_mlp_introspection():
    encoder = CartesianLocalEnvironmentEncoder(radial_hidden=13)
    assert encoder.scalar_dim == 13


def test_translation_projector_is_idempotent_and_removes_all_translations():
    reference = torch.empty(1, dtype=torch.float64)
    projector, translations = translation_projector(3, reference)
    assert torch.allclose(projector @ projector, projector)
    assert torch.allclose(projector @ translations, torch.zeros_like(translations))


def _two_atom_force_blocks(eigenvalues: torch.Tensor) -> torch.Tensor:
    relative = torch.zeros(3, 6, dtype=eigenvalues.dtype)
    for axis in range(3):
        relative[axis, axis] = 2.0 ** -0.5
        relative[axis, axis + 3] = -(2.0 ** -0.5)
    matrix = torch.einsum("a,ai,aj->ij", eigenvalues, relative, relative)
    return matrix.reshape(2, 3, 2, 3).permute(0, 2, 1, 3)


def test_first_order_diagnostic_bins_residual_by_true_stiffness():
    response = AtomCoordinateResponsePotential(optical_regularization=1e-3)
    true_blocks = _two_atom_force_blocks(
        torch.tensor([5e-4, 2e-3, 5e-3], dtype=torch.float64)
    )
    predicted_blocks = torch.zeros_like(true_blocks)
    displacement = torch.zeros(2, 3, 3, 3, dtype=torch.float64)
    displacement[0, 0, 0, 0] = 2.0 ** -0.5
    displacement[1, 0, 0, 0] = -(2.0 ** -0.5)
    internal = torch.zeros_like(displacement)
    metrics = first_order_spectral_residual_diagnostics(
        displacement,
        torch.zeros_like(displacement),
        predicted_blocks.reshape(-1),
        internal,
        true_blocks.reshape(-1),
        torch.tensor([True]),
        torch.tensor([True]),
        torch.tensor([0, 2]),
        response,
    )
    assert metrics["below_delta_mode_count"] == 1
    assert metrics["delta_to_3delta_mode_count"] == 1
    assert metrics["3delta_to_10delta_mode_count"] == 1
    assert metrics["above_10delta_mode_count"] == 0
    assert metrics["below_delta_residual_fraction"] == 1.0


def test_main_trainer_exposes_no_historical_memorization_or_weight_bundle_flags():
    source = inspect.getsource(train.main)
    for obsolete_flag in (
        "--m2-1",
        "--overfit-32",
        "--operator-learning-capacity",
    ):
        assert obsolete_flag not in source


def test_main_trainer_excludes_rejected_operator_auxiliary_surface() -> None:
    """Removed capacity-only operator losses cannot silently return to production."""
    source = (Path(__file__).parents[1] / "src" / "piezojet" / "train.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "operator_losses",
        "low_mode_action_loss_weight",
        "low_mode_leak_loss_weight",
        "phi_probe_loss_weight",
        "lambda_probe_loss_weight",
        "born_probe_loss_weight",
        "born_oracle_loss_weight",
        "phi_oracle_normal_loss_weight",
    )
    assert not any(token in source for token in forbidden)
