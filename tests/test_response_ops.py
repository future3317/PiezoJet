import torch

from piezojet.response_ops import (
    DIELECTRIC_TENSOR,
    ELASTIC_TENSOR,
    dielectric_from_irreps,
    dielectric_to_irreps,
    elastic_cartesian_to_voigt,
    elastic_from_irreps,
    elastic_to_irreps,
    elastic_voigt_to_cartesian,
    static_dielectric,
    susceptibility_from_relative_permittivity,
)


def _symmetric_stiffness() -> torch.Tensor:
    matrix = torch.randn(6, 6)
    return matrix @ matrix.T + torch.eye(6)


def test_elastic_voigt_cartesian_roundtrip_and_symmetry():
    voigt = _symmetric_stiffness()
    cartesian = elastic_voigt_to_cartesian(voigt)
    restored = elastic_cartesian_to_voigt(cartesian)
    assert torch.allclose(restored, voigt, atol=1e-6)
    assert torch.allclose(cartesian, cartesian.transpose(0, 1).transpose(2, 3), atol=1e-6)
    assert torch.allclose(cartesian, cartesian.permute(2, 3, 0, 1), atol=1e-6)


def test_elastic_pure_shear_energy_equivalence():
    voigt = _symmetric_stiffness()
    cartesian = elastic_voigt_to_cartesian(voigt)
    eta = torch.randn(6)
    strain = torch.zeros(3, 3)
    strain[0, 0], strain[1, 1], strain[2, 2] = eta[:3]
    strain[1, 2] = strain[2, 1] = eta[3] / 2
    strain[0, 2] = strain[2, 0] = eta[4] / 2
    strain[0, 1] = strain[1, 0] = eta[5] / 2
    voigt_energy = 0.5 * eta @ voigt @ eta
    cartesian_energy = 0.5 * torch.einsum("ij,ijkl,kl->", strain, cartesian, strain)
    assert torch.allclose(voigt_energy, cartesian_energy, atol=1e-5)


def test_elastic_and_dielectric_irrep_roundtrips():
    elastic = elastic_voigt_to_cartesian(_symmetric_stiffness())
    assert torch.allclose(elastic_from_irreps(elastic_to_irreps(elastic)), elastic, atol=1e-6)
    dielectric = torch.randn(3, 3)
    dielectric = 0.5 * (dielectric + dielectric.T)
    assert torch.allclose(dielectric_from_irreps(dielectric_to_irreps(dielectric)), dielectric, atol=1e-6)
    assert ELASTIC_TENSOR.dim == 21
    assert DIELECTRIC_TENSOR.dim == 6


def test_static_dielectric_and_susceptibility_definitions():
    electronic = torch.eye(3)
    ionic = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    static = static_dielectric(electronic, ionic)
    assert torch.allclose(static, electronic + ionic)
    assert torch.allclose(susceptibility_from_relative_permittivity(static), ionic)


def test_linear_response_background_returns_relative_permittivity():
    from piezojet.model import LinearResponseBackground

    background = LinearResponseBackground(context_dim=4)
    context = torch.zeros(2, 4)
    _, epsilon_relative = background(context)
    eigenvalues = torch.linalg.eigvalsh(epsilon_relative)
    assert torch.all(eigenvalues > 1.0)
