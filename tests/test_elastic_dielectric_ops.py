import torch

from piezojet.elastic_dielectric_ops import (
    cartesian_strain_to_voigt,
    compliance_cartesian_to_voigt,
    compliance_voigt_to_cartesian,
    DIELECTRIC_TENSOR,
    ELASTIC_TENSOR,
    dielectric_from_irreps,
    dielectric_to_irreps,
    elastic_cartesian_to_voigt,
    elastic_energy_cartesian,
    elastic_energy_voigt,
    elastic_from_irreps,
    elastic_gpa_to_kbar,
    elastic_kbar_to_gpa,
    elastic_to_irreps,
    elastic_voigt_to_cartesian,
    rotate_elastic,
    static_dielectric,
    susceptibility_from_relative_permittivity,
    voigt_bulk_shear_moduli,
    voigt_compliance_from_stiffness,
    voigt_strain_to_cartesian,
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


def test_engineering_strain_round_trip_and_energy_for_any_symmetric_strain():
    stiffness = _symmetric_stiffness()
    eta = torch.randn(6)
    strain = voigt_strain_to_cartesian(eta)
    assert torch.allclose(cartesian_strain_to_voigt(strain), eta)
    cartesian = elastic_voigt_to_cartesian(stiffness)
    assert torch.allclose(
        elastic_energy_voigt(stiffness, eta), elastic_energy_cartesian(cartesian, strain), atol=1e-5
    )


def test_elastic_rotation_covariance_and_energy_invariance():
    stiffness = elastic_voigt_to_cartesian(_symmetric_stiffness())
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    strain = voigt_strain_to_cartesian(torch.randn(6))
    rotated_stiffness = rotate_elastic(stiffness, rotation)
    rotated_strain = rotation @ strain @ rotation.T
    assert torch.allclose(
        elastic_energy_cartesian(stiffness, strain),
        elastic_energy_cartesian(rotated_stiffness, rotated_strain), atol=1e-5,
    )


def test_stiffness_compliance_shear_scaling_and_unit_conversion():
    stiffness = _symmetric_stiffness() * 100
    compliance = voigt_compliance_from_stiffness(stiffness)
    assert torch.allclose(stiffness @ compliance, torch.eye(6), atol=1e-5)
    c_cartesian = elastic_voigt_to_cartesian(stiffness)
    s_cartesian = compliance_voigt_to_cartesian(compliance)
    symmetric_identity = 0.5 * (
        torch.einsum("ik,jl->ijkl", torch.eye(3), torch.eye(3))
        + torch.einsum("il,jk->ijkl", torch.eye(3), torch.eye(3))
    )
    assert torch.allclose(torch.einsum("ijmn,mnkl->ijkl", c_cartesian, s_cartesian), symmetric_identity, atol=1e-5)
    assert torch.allclose(compliance_cartesian_to_voigt(s_cartesian), compliance, atol=1e-6)
    assert torch.allclose(elastic_gpa_to_kbar(elastic_kbar_to_gpa(stiffness)), stiffness)


def test_isotropic_moduli_are_consistent_in_gpa():
    lam, mu = 50.0, 30.0
    stiffness = torch.zeros(6, 6)
    stiffness[:3, :3] = lam
    stiffness[0, 0] = stiffness[1, 1] = stiffness[2, 2] = lam + 2 * mu
    stiffness[3, 3] = stiffness[4, 4] = stiffness[5, 5] = mu
    moduli = voigt_bulk_shear_moduli(stiffness)
    expected_bulk = lam + 2 * mu / 3
    for name in ("bulk_voigt_gpa", "bulk_reuss_gpa", "bulk_hill_gpa"):
        assert torch.allclose(moduli[name], torch.tensor(expected_bulk))
    for name in ("shear_voigt_gpa", "shear_reuss_gpa", "shear_hill_gpa"):
        assert torch.allclose(moduli[name], torch.tensor(mu))


def test_availability_masked_elastic_loss_uses_cartesian_frobenius_norm():
    from piezojet.train import elastic_auxiliary_loss

    target = _symmetric_stiffness().unsqueeze(0)
    prediction = target + 0.1 * _symmetric_stiffness().unsqueeze(0)
    assert elastic_auxiliary_loss(prediction, target, torch.tensor([True])) > 0
    assert elastic_auxiliary_loss(prediction, target, torch.tensor([False])) == 0


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
