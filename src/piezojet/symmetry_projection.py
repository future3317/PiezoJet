"""Point-group Reynolds projection for rank-3 polar piezoelectric tensors."""

from __future__ import annotations

import torch
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .tensor_ops import rotate_piezo


def get_cartesian_point_group_operations(
    record, symprec: float = 1e-5,
) -> torch.Tensor:
    atoms = record["atoms"]
    structure = Structure(
        atoms["lattice_mat"],
        atoms["elements"],
        atoms["coords"],
        coords_are_cartesian=False,
    )
    operations = SpacegroupAnalyzer(
        structure, symprec=symprec
    ).get_point_group_operations(cartesian=True)
    if not operations:
        raise ValueError("No point-group operations returned by pymatgen")
    return torch.stack([
        torch.tensor(operation.rotation_matrix, dtype=torch.float32)
        for operation in operations
    ])


def project_piezo_to_point_group(
    tensor: torch.Tensor, rotations: torch.Tensor,
) -> torch.Tensor:
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape [G,3,3]")
    return rotate_piezo(tensor.unsqueeze(0), rotations).mean(dim=0)


def point_group_residual(
    tensor: torch.Tensor, rotations: torch.Tensor, eps: float = 1e-12,
) -> torch.Tensor:
    projected = project_piezo_to_point_group(tensor, rotations)
    return torch.linalg.vector_norm(tensor - projected) / (
        torch.linalg.vector_norm(tensor) + eps
    )
