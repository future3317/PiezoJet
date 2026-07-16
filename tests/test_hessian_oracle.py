import torch

from piezojet.hessian_oracle import (
    bond_laplacian_from_stiffness,
    project_force_constants,
    symmetric_edge_stiffness,
)


def test_bond_oracle_exactly_recovers_a_bond_laplacian() -> None:
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    stiffness = symmetric_edge_stiffness(torch.tensor([[2.0, 3.0, 4.0, 0.2, -0.1, 0.3], [1.0, 2.0, 3.0, -0.4, 0.5, 0.1]]))
    target = bond_laplacian_from_stiffness(2, edges, stiffness)

    result = project_force_constants(target, edges)

    assert result["relative_frobenius_error"] < 1e-6
    assert result["explained_frobenius_fraction"] > 0.999999
    assert result["translation_residual_max_abs"] < 1e-6


def test_bond_oracle_exposes_nonsymmetric_offdiagonal_block_limitation() -> None:
    # A globally symmetric acoustic 9x9 force matrix may still have
    # Phi_01 != Phi_01^T.
    # A sum of symmetric bond K matrices cannot reproduce this component.
    block = torch.tensor([[0.0, 1.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.zeros(9, 9)
    target[:3, 3:6] = block
    target[:3, 6:] = -block
    target[3:6, :3] = block.transpose(0, 1)
    target[3:6, 6:] = -block.transpose(0, 1)
    target[6:, :3] = -block.transpose(0, 1)
    target[6:, 3:6] = -block
    target[6:, 6:] = block + block.transpose(0, 1)
    edges = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)

    result = project_force_constants(target, edges)

    assert result["offdiagonal_block_skew_fraction"] > 0.1
    assert result["relative_frobenius_error"] > 0.1
