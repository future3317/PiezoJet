import torch

from piezojet.evaluate_dfpt import ionic_piezo_from_factors
from piezojet.identifiability import (
    linear_map_metrics,
    macro_response_matrix,
    observable_null_projectors,
)
from piezojet.model import AtomCoordinateResponsePotential
from piezojet.strain_completion import vector_to_internal_tensor
from piezojet.tensor_ops import cartesian_to_piezo_voigt


def test_macro_response_matrix_matches_direct_factor_contraction():
    dtype = torch.float64
    response = AtomCoordinateResponsePotential(
        optical_solve_policy="regularized", optical_regularization=1e-3
    )
    spring = torch.diag(torch.tensor([2.0, 3.0, 5.0], dtype=dtype))
    matrix = torch.cat(
        (torch.cat((spring, -spring), dim=1), torch.cat((-spring, spring), dim=1)),
        dim=0,
    )
    force_constants = response._blocks_from_matrix(matrix, 2)
    born = torch.tensor(
        [
            [[1.0, 0.2, 0.0], [0.1, -0.4, 0.3], [0.0, 0.2, 0.7]],
            [[-1.0, -0.2, 0.0], [-0.1, 0.4, -0.3], [0.0, -0.2, -0.7]],
        ],
        dtype=dtype,
    )
    generator = torch.Generator().manual_seed(17)
    basis = torch.randn(36, 4, generator=generator, dtype=dtype)
    coefficients = torch.randn(4, generator=generator, dtype=dtype)
    operator = macro_response_matrix(
        basis=basis,
        born_charges=born,
        force_constants=force_constants,
        volume=20.0,
    )
    internal = vector_to_internal_tensor(basis @ coefficients, atoms=2)
    direct = cartesian_to_piezo_voigt(
        ionic_piezo_from_factors(
            response, born, force_constants, internal, 20.0,
            solve_policy="regularized", regularization=1e-3,
        )
    ).reshape(-1)
    assert torch.allclose(operator @ coefficients, direct, atol=1e-10, rtol=1e-10)


def test_observable_and_null_projectors_are_exact_orthogonal_projectors():
    matrix = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    observable, null, rank = observable_null_projectors(matrix)
    identity = torch.eye(3, dtype=torch.float64)
    assert rank == 2
    assert torch.allclose(observable @ observable, observable, atol=1e-12)
    assert torch.allclose(null @ null, null, atol=1e-12)
    assert torch.allclose(observable @ null, torch.zeros_like(null), atol=1e-12)
    assert torch.allclose(observable + null, identity, atol=1e-12)
    assert torch.allclose(matrix @ null, torch.zeros_like(matrix), atol=1e-12)


def test_rank_metrics_distinguish_observable_nullity_and_full_column_rank():
    underdetermined = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    metrics = linear_map_metrics(underdetermined)
    assert metrics["rank"] == 2
    assert metrics["nullity"] == 1
    assert metrics["full_column_rank"] is False
    assert metrics["condition_number_full"] is None
    empty = linear_map_metrics(torch.empty(5, 0))
    assert empty["full_column_rank"] is True
    assert empty["condition_number_full"] == 1.0
