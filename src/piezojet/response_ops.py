"""Convention-frozen dielectric and elastic tensor conversions for M5A."""

from __future__ import annotations

import torch
from e3nn.io import CartesianTensor


PAIR_ORDER = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
ELASTIC_TENSOR = CartesianTensor("ijkl=ijlk=jikl=klij")
DIELECTRIC_TENSOR = CartesianTensor("ij=ji")


def _basis(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    basis = torch.zeros(6, 3, 3, dtype=dtype, device=device)
    for index, (i, j) in enumerate(PAIR_ORDER):
        basis[index, i, j] = 1
        basis[index, j, i] = 1
        if i == j:
            basis[index, i, j] = 1
    return basis


def _dual_basis(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    basis = _basis(dtype, device)
    basis[3:] *= 0.5
    return basis


def elastic_voigt_to_cartesian(stiffness: torch.Tensor) -> torch.Tensor:
    """Convert engineering-shear 6x6 stiffness to C_ijkl."""
    if stiffness.shape[-2:] != (6, 6):
        raise ValueError(f"Expected [...,6,6], got {tuple(stiffness.shape)}")
    basis = _basis(stiffness.dtype, stiffness.device)
    return torch.einsum("...IJ,Iij,Jkl->...ijkl", stiffness, basis, basis)


def elastic_cartesian_to_voigt(stiffness: torch.Tensor) -> torch.Tensor:
    if stiffness.shape[-4:] != (3, 3, 3, 3):
        raise ValueError(f"Expected [...,3,3,3,3], got {tuple(stiffness.shape)}")
    dual = _dual_basis(stiffness.dtype, stiffness.device)
    return torch.einsum("Iij,...ijkl,Jkl->...IJ", dual, stiffness, dual)


def elastic_to_irreps(stiffness: torch.Tensor) -> torch.Tensor:
    if stiffness.shape[-4:] != (3, 3, 3, 3):
        raise ValueError(f"Expected [...,3,3,3,3], got {tuple(stiffness.shape)}")
    return ELASTIC_TENSOR.from_cartesian(stiffness)


def elastic_from_irreps(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != ELASTIC_TENSOR.dim:
        raise ValueError(f"Expected final dimension {ELASTIC_TENSOR.dim}, got {values.shape[-1]}")
    return ELASTIC_TENSOR.to_cartesian(values)


def dielectric_to_irreps(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [...,3,3], got {tuple(tensor.shape)}")
    return DIELECTRIC_TENSOR.from_cartesian(tensor)


def dielectric_from_irreps(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != DIELECTRIC_TENSOR.dim:
        raise ValueError(f"Expected final dimension {DIELECTRIC_TENSOR.dim}, got {values.shape[-1]}")
    return DIELECTRIC_TENSOR.to_cartesian(values)


def static_dielectric(electronic: torch.Tensor, ionic: torch.Tensor) -> torch.Tensor:
    if electronic.shape[-2:] != (3, 3) or ionic.shape != electronic.shape:
        raise ValueError("Electronic and ionic dielectric tensors must have matching [...,3,3] shapes")
    return electronic + ionic


def susceptibility_from_relative_permittivity(relative_permittivity: torch.Tensor) -> torch.Tensor:
    if relative_permittivity.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [...,3,3], got {tuple(relative_permittivity.shape)}")
    return relative_permittivity - torch.eye(3, dtype=relative_permittivity.dtype, device=relative_permittivity.device)
