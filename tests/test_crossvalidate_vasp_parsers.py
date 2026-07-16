import pytest
import torch

from piezojet.crossvalidate_vasp_parsers import relative_error


def test_parser_crossvalidation_relative_error_is_shape_safe_and_scale_safe():
    reference = torch.tensor([[3.0, 4.0]])
    assert relative_error(reference, reference) == pytest.approx(0.0)
    assert relative_error(torch.zeros_like(reference), reference) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shapes differ"):
        relative_error(torch.zeros(3), reference)
