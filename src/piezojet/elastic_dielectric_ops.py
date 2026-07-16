"""Convention-frozen dielectric and elastic tensor conversions for M5A.

This module contains tensor representation conversions only.  It deliberately
does not implement the atom-coordinate optical response operator, which lives
in :class:`piezojet.model.AtomCoordinateResponsePotential`.
"""

from __future__ import annotations

import torch
from e3nn.io import CartesianTensor


PAIR_ORDER = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
KBAR_PER_GPA = 10.0
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


def elastic_kbar_to_gpa(stiffness_kbar: torch.Tensor) -> torch.Tensor:
    """The sole source-unit conversion: JARVIS kbar -> internal/model GPa."""
    if stiffness_kbar.shape[-2:] != (6, 6):
        raise ValueError(f"Expected [...,6,6], got {tuple(stiffness_kbar.shape)}")
    return stiffness_kbar / KBAR_PER_GPA


def elastic_gpa_to_kbar(stiffness_gpa: torch.Tensor) -> torch.Tensor:
    if stiffness_gpa.shape[-2:] != (6, 6):
        raise ValueError(f"Expected [...,6,6], got {tuple(stiffness_gpa.shape)}")
    return stiffness_gpa * KBAR_PER_GPA


def voigt_strain_to_cartesian(strain: torch.Tensor) -> torch.Tensor:
    """Expand canonical engineering strain `[xx,yy,zz,yz,xz,xy]`."""
    if strain.shape[-1] != 6:
        raise ValueError(f"Expected [...,6], got {tuple(strain.shape)}")
    output = strain.new_zeros(*strain.shape[:-1], 3, 3)
    output[..., 0, 0], output[..., 1, 1], output[..., 2, 2] = strain.unbind(-1)[:3]
    output[..., 1, 2] = output[..., 2, 1] = strain[..., 3] / 2
    output[..., 0, 2] = output[..., 2, 0] = strain[..., 4] / 2
    output[..., 0, 1] = output[..., 1, 0] = strain[..., 5] / 2
    return output


def cartesian_strain_to_voigt(strain: torch.Tensor) -> torch.Tensor:
    if strain.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [...,3,3], got {tuple(strain.shape)}")
    if not torch.allclose(strain, strain.transpose(-1, -2), atol=1e-7, rtol=1e-7):
        raise ValueError("Strain must be symmetric")
    return torch.stack((
        strain[..., 0, 0], strain[..., 1, 1], strain[..., 2, 2],
        2 * strain[..., 1, 2], 2 * strain[..., 0, 2], 2 * strain[..., 0, 1],
    ), dim=-1)


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


def compliance_voigt_to_cartesian(compliance: torch.Tensor) -> torch.Tensor:
    """Convert engineering-Voigt compliance to Cartesian ``S_ijkl``.

    Compliance uses the *dual* basis on both pairs, unlike stiffness, because
    stress Voigt shear entries are not doubled while strain Voigt entries are.
    """
    if compliance.shape[-2:] != (6, 6):
        raise ValueError(f"Expected [...,6,6], got {tuple(compliance.shape)}")
    dual = _dual_basis(compliance.dtype, compliance.device)
    return torch.einsum("...IJ,Iij,Jkl->...ijkl", compliance, dual, dual)


def compliance_cartesian_to_voigt(compliance: torch.Tensor) -> torch.Tensor:
    if compliance.shape[-4:] != (3, 3, 3, 3):
        raise ValueError(f"Expected [...,3,3,3,3], got {tuple(compliance.shape)}")
    basis = _basis(compliance.dtype, compliance.device)
    return torch.einsum("Iij,...ijkl,Jkl->...IJ", basis, compliance, basis)


def rotate_elastic(stiffness: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply a Cartesian orthogonal change of frame to ``C_ijkl``."""
    if stiffness.shape[-4:] != (3, 3, 3, 3) or rotation.shape[-2:] != (3, 3):
        raise ValueError("Expected C[...,3,3,3,3] and R[...,3,3]")
    return torch.einsum("...ia,...jb,...kc,...ld,...abcd->...ijkl", rotation, rotation, rotation, rotation, stiffness)


def elastic_energy_voigt(stiffness_gpa: torch.Tensor, strain_voigt: torch.Tensor) -> torch.Tensor:
    if stiffness_gpa.shape[-2:] != (6, 6) or strain_voigt.shape[-1] != 6:
        raise ValueError("Expected C[...,6,6] and engineering strain [...,6]")
    return 0.5 * torch.einsum("...I,...IJ,...J->...", strain_voigt, stiffness_gpa, strain_voigt)


def elastic_energy_cartesian(stiffness: torch.Tensor, strain: torch.Tensor) -> torch.Tensor:
    if stiffness.shape[-4:] != (3, 3, 3, 3) or strain.shape[-2:] != (3, 3):
        raise ValueError("Expected C[...,3,3,3,3] and strain [...,3,3]")
    return 0.5 * torch.einsum("...ij,...ijkl,...kl->...", strain, stiffness, strain)


def voigt_compliance_from_stiffness(stiffness_gpa: torch.Tensor) -> torch.Tensor:
    if stiffness_gpa.shape[-2:] != (6, 6):
        raise ValueError(f"Expected [...,6,6], got {tuple(stiffness_gpa.shape)}")
    return torch.linalg.inv(stiffness_gpa)


def voigt_bulk_shear_moduli(stiffness_gpa: torch.Tensor) -> dict[str, torch.Tensor]:
    """Voigt/Reuss/Hill bulk and shear moduli in GPa."""
    if stiffness_gpa.shape[-2:] != (6, 6):
        raise ValueError(f"Expected [...,6,6], got {tuple(stiffness_gpa.shape)}")
    c = stiffness_gpa
    bulk_voigt = (c[..., 0, 0] + c[..., 1, 1] + c[..., 2, 2] + 2 * (c[..., 0, 1] + c[..., 0, 2] + c[..., 1, 2])) / 9
    shear_voigt = (c[..., 0, 0] + c[..., 1, 1] + c[..., 2, 2] - c[..., 0, 1] - c[..., 0, 2] - c[..., 1, 2] + 3 * (c[..., 3, 3] + c[..., 4, 4] + c[..., 5, 5])) / 15
    s = voigt_compliance_from_stiffness(c)
    bulk_reuss = 1 / (s[..., 0, 0] + s[..., 1, 1] + s[..., 2, 2] + 2 * (s[..., 0, 1] + s[..., 0, 2] + s[..., 1, 2]))
    shear_reuss = 15 / (
        4 * (s[..., 0, 0] + s[..., 1, 1] + s[..., 2, 2] - s[..., 0, 1] - s[..., 0, 2] - s[..., 1, 2])
        + 3 * (s[..., 3, 3] + s[..., 4, 4] + s[..., 5, 5])
    )
    return {
        "bulk_voigt_gpa": bulk_voigt,
        "shear_voigt_gpa": shear_voigt,
        "bulk_reuss_gpa": bulk_reuss,
        "shear_reuss_gpa": shear_reuss,
        "bulk_hill_gpa": 0.5 * (bulk_voigt + bulk_reuss),
        "shear_hill_gpa": 0.5 * (shear_voigt + shear_reuss),
    }


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
