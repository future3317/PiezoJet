import torch

from piezojet.tensor_ops import (
    PIEZO_TYPE,
    cartesian_to_piezo_voigt,
    piezo_from_irreps,
    piezo_to_irreps,
    piezo_voigt_to_cartesian,
    symmetric_matrix_to_voigt,
    voigt_to_symmetric_matrix,
)
from piezojet.model import ResponsePotential


def test_voigt_round_trip():
    eta6 = torch.randn(7, 6)
    assert torch.allclose(symmetric_matrix_to_voigt(voigt_to_symmetric_matrix(eta6)), eta6, atol=1e-7)


def test_piezo_cartesian_irrep_round_trip():
    source = torch.randn(5, 3, 6)
    cartesian = piezo_voigt_to_cartesian(source)
    restored = piezo_from_irreps(piezo_to_irreps(cartesian))
    assert PIEZO_TYPE.dim == 18
    assert torch.allclose(restored, cartesian, atol=1e-5)
    assert torch.allclose(cartesian_to_piezo_voigt(restored), source, atol=1e-5)


def test_engineering_shear_matches_single_symmetric_component_derivative():
    """e_i,xy multiplies gamma_xy, while e_i,xy=e_ixy=e_iyx in Cartesian form."""
    piezo_voigt = torch.zeros(1, 3, 6)
    piezo_voigt[..., 0, 5] = 2.75  # xy engineering-shear column
    piezo_cart = piezo_voigt_to_cartesian(piezo_voigt)
    assert piezo_cart[0, 0, 0, 1] == piezo_voigt[0, 0, 5]
    assert piezo_cart[0, 0, 1, 0] == piezo_voigt[0, 0, 5]
    field = torch.tensor([[1.0, 0.0, 0.0]])
    eta6 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(ResponsePotential()(piezo_cart, field, eta6), torch.tensor([-2.75]))
