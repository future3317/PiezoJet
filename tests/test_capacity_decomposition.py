import torch

from piezojet.capacity_decomposition import (
    _factor_metrics,
    _symmetry_metrics,
    _transform_born,
    _transform_phi,
    _transform_rank3_atom_tensor,
)
from piezojet.strain_completion import CartesianSymmetryOperation


def _identity_operation(atoms: int) -> CartesianSymmetryOperation:
    return CartesianSymmetryOperation(
        rotation=torch.eye(3, dtype=torch.float64),
        permutation=torch.arange(atoms),
        mapping_error_angstrom=0.0,
    )


def test_identity_reynolds_projection_has_unit_cosine_ceiling() -> None:
    operation = _identity_operation(2)
    for value, transform in (
        (torch.randn(2, 3, 3), _transform_born),
        (torch.randn(2, 2, 3, 3), _transform_phi),
        (torch.randn(2, 3, 3, 3), _transform_rank3_atom_tensor),
    ):
        metrics = _symmetry_metrics(value, value, [operation], transform)
        assert metrics["target_reynolds_relative_residual"] < 1e-12
        assert metrics["prediction_reynolds_relative_residual"] < 1e-12
        assert metrics["theoretical_maximum_cosine_from_target_symmetry"] == 1.0


def test_factor_metrics_preserves_zero_target_information() -> None:
    metrics = _factor_metrics(torch.ones(3), torch.zeros(3), component_floor=0.1)
    assert not metrics["active_target"]
    assert metrics["target_frobenius_norm"] == 0.0
    assert metrics["prediction_frobenius_norm"] > 0.0
    assert metrics["relative_frobenius_error"] > 1e20
