import torch

from piezojet.dfpt_conventions import force_constant_convention_metrics
from piezojet.evaluate_dfpt import ionic_piezo_from_factors
from piezojet.model import AtomCoordinateResponsePotential
from piezojet.tensor_ops import (
    cartesian_to_piezo_voigt,
    piezo_voigt_to_cartesian,
    voigt_to_symmetric_matrix,
)


def test_force_constant_audit_identifies_mass_unweighted_parser_relation():
    masses = torch.tensor([2.0, 8.0])
    raw = torch.randn(2, 2, 3, 3)
    force = -raw * torch.sqrt(masses[:, None, None, None] * masses[None, :, None, None])
    metrics = force_constant_convention_metrics(
        {"masses": masses, "dynamical_matrix": raw, "force_constants": force}
    )
    assert metrics["mass_unweighting_relative_error"] < 1e-12


def test_all_three_engineering_shear_columns_produce_one_cartesian_piezo_coefficient():
    response = AtomCoordinateResponsePotential()
    field = torch.tensor([[1.0, 0.0, 0.0]])
    for column in (3, 4, 5):
        piezo_voigt = torch.zeros(1, 3, 6)
        piezo_voigt[0, 0, column] = 3.25
        eta = torch.zeros(1, 6)
        eta[0, column] = 1.0
        # This is the tensor-convention identity under test, not a model
        # energy: an arbitrary assembled piezo tensor does not establish a
        # shared microscopic response potential.
        energy = -torch.einsum(
            "bi,bijk,bjk->b",
            field,
            piezo_voigt_to_cartesian(piezo_voigt),
            voigt_to_symmetric_matrix(eta),
        ) / response.PIEZO_C_PER_M2
        assert torch.allclose(
            energy,
            torch.tensor([-3.25 / response.PIEZO_C_PER_M2]),
        )


def _two_atom_optical_blocks() -> torch.Tensor:
    """Stable optical force constants with exactly three translations."""
    relative = torch.zeros(3, 6)
    for axis in range(3):
        relative[axis, axis] = 2.0 ** -0.5
        relative[axis, axis + 3] = -(2.0 ** -0.5)
    matrix = torch.einsum("a,ai,aj->ij", torch.tensor([2.0, 3.0, 4.0]), relative, relative)
    return matrix.reshape(2, 3, 2, 3).permute(0, 2, 1, 3)


def test_pure_shear_force_finite_difference_and_ionic_response_golden_for_every_column():
    """Golden convention test independent of an OUTCAR parser.

    For each canonical engineering shear (yz, xz, xy), a scalar quadratic
    energy produces the force derivative printed by OUTCAR, and equilibrium
    finite differences of polarization reproduce ``Z^T Phi^-1 Lambda``.  A
    column permutation or an erroneous factor of two fails at least one of
    the three cases.
    """
    response = AtomCoordinateResponsePotential(optical_solve_policy="exact")
    blocks = _two_atom_optical_blocks()
    phi = blocks.permute(0, 2, 1, 3).reshape(6, 6)
    # Neutrality ensures BEC does not couple the acoustic nullspace.
    born = torch.tensor(
        [
            [[1.0, 0.2, -0.1], [0.1, 0.8, 0.0], [0.0, -0.2, 0.6]],
            [[-1.0, -0.2, 0.1], [-0.1, -0.8, 0.0], [0.0, 0.2, -0.6]],
        ]
    )
    volume = 11.0
    step = 1e-4
    for column, displacement_axis in zip((3, 4, 5), (0, 1, 2)):
        internal = torch.zeros(2, 3, 3, 3)
        # Lambda is acoustic-null and maps one selected atom-coordinate to
        # exactly one engineering shear column.
        pair = ((1, 2), (0, 2), (0, 1))[column - 3]
        internal[0, displacement_axis, pair[0], pair[1]] = 1.7
        internal[0, displacement_axis, pair[1], pair[0]] = 1.7
        internal[1, displacement_axis, pair[0], pair[1]] = -1.7
        internal[1, displacement_axis, pair[1], pair[0]] = -1.7
        coupling = response._coupling_voigt(internal).reshape(6, 6)
        assert coupling[:, column].abs().sum() > 0
        assert coupling[:, [index for index in range(6) if index != column]].abs().sum() == 0

        # E = u^T Phi u / 2 - u^T Lambda eta implies F=-Phi u+Lambda eta.
        # Central differences of forces obtained from the scalar energy
        # recover the same OUTCAR force derivative.
        eta_plus = torch.zeros(6)
        eta_minus = torch.zeros(6)
        eta_plus[column], eta_minus[column] = step, -step
        def force_from_energy(eta: torch.Tensor) -> torch.Tensor:
            displacement = torch.zeros(6, requires_grad=True)
            energy = 0.5 * displacement @ phi @ displacement - displacement @ coupling @ eta
            return -torch.autograd.grad(energy, displacement)[0]

        finite_force_derivative = (force_from_energy(eta_plus) - force_from_energy(eta_minus)) / (2 * step)
        assert torch.allclose(finite_force_derivative, coupling[:, column], atol=1e-8)

        # Solve only in the optical subspace, then finite-difference the BEC
        # polarization.  This is the ionic piezoelectric definition.
        u_plus = response.apply_optical_operator(blocks, coupling @ eta_plus[:, None], "exact").squeeze(-1)
        u_minus = response.apply_optical_operator(blocks, coupling @ eta_minus[:, None], "exact").squeeze(-1)
        polarization_derivative = response.PIEZO_C_PER_M2 * born.reshape(-1, 3).T @ ((u_plus - u_minus) / (2 * step)) / volume
        analytic = cartesian_to_piezo_voigt(
            ionic_piezo_from_factors(response, born, blocks, internal, volume, "exact")
        )
        assert torch.allclose(polarization_derivative, analytic[:, column], atol=1e-6, rtol=1e-6)
