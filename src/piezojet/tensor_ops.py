"""The sole definitions of PiezoJet Voigt, Cartesian, and rotation conventions."""

from __future__ import annotations

import torch
from e3nn.io import CartesianTensor


# GMTNet labels are [xx, yy, zz, xy, yz, xz].  PiezoJet uses the canonical
# [xx, yy, zz, yz, xz, xy] order everywhere outside data ingestion.
SOURCE_TO_CANONICAL = (0, 1, 2, 4, 5, 3)
CANONICAL_TO_SOURCE = (0, 1, 2, 5, 3, 4)
PIEZO_TYPE = CartesianTensor("ijk=ikj")
if PIEZO_TYPE.dim != 18:
    raise RuntimeError(f"Expected 18 piezoelectric irreps, got {PIEZO_TYPE.dim}")


def source_voigt_to_canonical(piezo: torch.Tensor) -> torch.Tensor:
    """Convert GMTNet's documented [xx, yy, zz, xy, yz, xz] columns."""
    if piezo.shape[-2:] != (3, 6):
        raise ValueError(f"Expected [..., 3, 6], got {tuple(piezo.shape)}")
    return piezo[..., list(SOURCE_TO_CANONICAL)]


def canonical_voigt_to_source(piezo: torch.Tensor) -> torch.Tensor:
    if piezo.shape[-2:] != (3, 6):
        raise ValueError(f"Expected [..., 3, 6], got {tuple(piezo.shape)}")
    return piezo[..., list(CANONICAL_TO_SOURCE)]


def voigt_to_symmetric_matrix(eta6: torch.Tensor, convention: str = "engineering") -> torch.Tensor:
    """Map canonical strain to a symmetric matrix.

    `engineering` means shear entries are gamma_ij=2*eta_ij.
    """
    if convention != "engineering":
        raise ValueError(f"Unsupported Voigt convention: {convention}")
    if eta6.shape[-1] != 6:
        raise ValueError(f"Expected [..., 6], got {tuple(eta6.shape)}")
    out = eta6.new_zeros((*eta6.shape[:-1], 3, 3))
    out[..., 0, 0], out[..., 1, 1], out[..., 2, 2] = eta6.unbind(-1)[:3]
    out[..., 1, 2] = out[..., 2, 1] = eta6[..., 3] / 2
    out[..., 0, 2] = out[..., 2, 0] = eta6[..., 4] / 2
    out[..., 0, 1] = out[..., 1, 0] = eta6[..., 5] / 2
    return out


def symmetric_matrix_to_voigt(eta: torch.Tensor, convention: str = "engineering") -> torch.Tensor:
    if convention != "engineering":
        raise ValueError(f"Unsupported Voigt convention: {convention}")
    if eta.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [..., 3, 3], got {tuple(eta.shape)}")
    if not torch.allclose(eta, eta.transpose(-1, -2), atol=1e-7, rtol=1e-7):
        raise ValueError("Strain matrix must be symmetric")
    return torch.stack(
        (eta[..., 0, 0], eta[..., 1, 1], eta[..., 2, 2], 2 * eta[..., 1, 2], 2 * eta[..., 0, 2], 2 * eta[..., 0, 1]),
        dim=-1,
    )


def piezo_voigt_to_cartesian(piezo: torch.Tensor) -> torch.Tensor:
    """Expand canonical engineering-strain coefficients to e_ijk=e_ikj.

    The source loader differentiates with respect to one off-diagonal entry of
    a symmetric strain matrix.  With gamma_ij=2*eta_ij, this derivative is
    exactly e_ij (not e_ij/2); the factor of two instead appears when eta6 is
    expanded into both eta_ij and eta_ji in :func:`voigt_to_symmetric_matrix`.
    """
    if piezo.shape[-2:] != (3, 6):
        raise ValueError(f"Expected [..., 3, 6], got {tuple(piezo.shape)}")
    out = piezo.new_zeros((*piezo.shape[:-2], 3, 3, 3))
    out[..., :, 0, 0], out[..., :, 1, 1], out[..., :, 2, 2] = (piezo[..., :, i] for i in range(3))
    out[..., :, 1, 2] = out[..., :, 2, 1] = piezo[..., :, 3]
    out[..., :, 0, 2] = out[..., :, 2, 0] = piezo[..., :, 4]
    out[..., :, 0, 1] = out[..., :, 1, 0] = piezo[..., :, 5]
    return out


def cartesian_to_piezo_voigt(piezo: torch.Tensor) -> torch.Tensor:
    if piezo.shape[-3:] != (3, 3, 3):
        raise ValueError(f"Expected [..., 3, 3, 3], got {tuple(piezo.shape)}")
    if not torch.allclose(piezo, piezo.transpose(-1, -2), atol=1e-6, rtol=1e-6):
        raise ValueError("Piezoelectric tensor must satisfy e_ijk=e_ikj")
    return torch.stack(
        (piezo[..., :, 0, 0], piezo[..., :, 1, 1], piezo[..., :, 2, 2], piezo[..., :, 1, 2], piezo[..., :, 0, 2], piezo[..., :, 0, 1]),
        dim=-1,
    )


def piezo_to_irreps(piezo: torch.Tensor) -> torch.Tensor:
    return PIEZO_TYPE.from_cartesian(piezo)


def piezo_from_irreps(irreps: torch.Tensor) -> torch.Tensor:
    if irreps.shape[-1] != PIEZO_TYPE.dim:
        raise ValueError(f"Expected final irrep dimension {PIEZO_TYPE.dim}, got {irreps.shape[-1]}")
    return PIEZO_TYPE.to_cartesian(irreps)


def rotate_piezo(piezo: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ia,...jb,...kc,...abc->...ijk", rotation, rotation, rotation, piezo)


def rotate_strain(strain: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return rotation @ strain @ rotation.transpose(-1, -2)


def piezo_scale(piezo_voigt: torch.Tensor) -> torch.Tensor:
    """One global positive normalization scale, fitted only on training data."""
    if piezo_voigt.numel() == 0:
        raise ValueError("Cannot calculate a scale from an empty tensor")
    return torch.sqrt(piezo_voigt.square().mean()).clamp_min(torch.finfo(piezo_voigt.dtype).eps)
