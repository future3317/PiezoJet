import torch

from piezojet.symmetry_projection import (
    point_group_residual,
    project_piezo_to_point_group,
)


def test_projector_is_idempotent_and_inversion_zeroes_polar_rank3():
    tensor = torch.randn(3, 3, 3)
    rotations = torch.stack((torch.eye(3), -torch.eye(3)))
    projected = project_piezo_to_point_group(tensor, rotations)
    assert torch.allclose(
        project_piezo_to_point_group(projected, rotations),
        projected,
        atol=1e-6,
    )
    assert torch.linalg.vector_norm(projected) < 1e-6
    assert point_group_residual(tensor, rotations) <= 1.0
