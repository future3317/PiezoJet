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
