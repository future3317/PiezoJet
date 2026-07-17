import pytest
import torch

from piezojet.train import _epoch, restrict_splits_to_material_ids, soft_optical_eigenvalue_loss


def test_global_material_id_restriction_preserves_disjoint_split():
    splits = {
        "train": ["a", "b", "c"],
        "val": ["d", "e"],
        "test": ["f"],
    }
    restricted = restrict_splits_to_material_ids(splits, ["b", "d", "f"], "global")
    assert restricted == {"train": ["b"], "val": ["d"], "test": ["f"]}
    assert not set(restricted["train"]) & set(restricted["val"])


def test_same_material_id_mode_is_explicit_smoke_only():
    restricted = restrict_splits_to_material_ids(
        {"train": ["a"], "val": ["b"], "test": []}, ["a", "b"], "same"
    )
    assert restricted["train"] == restricted["val"] == ["a", "b"]


def test_global_material_id_restriction_requires_validation_coverage():
    with pytest.raises(ValueError, match="non-empty train and validation"):
        restrict_splits_to_material_ids(
            {"train": ["a"], "val": ["b"], "test": []}, ["a"], "global"
        )


def test_soft_optical_loss_is_zero_for_exact_variable_size_hessian():
    atoms = 2
    relative = torch.zeros(3, 6)
    for axis in range(3):
        relative[axis, axis] = 2.0 ** -0.5
        relative[axis, axis + 3] = -(2.0 ** -0.5)
    matrix = torch.einsum("a,ai,aj->ij", torch.tensor([-2.0, 1.0, 3.0]), relative, relative)
    blocks = matrix.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3).reshape(-1)
    loss = soft_optical_eigenvalue_loss(
        blocks, blocks, torch.tensor([0, atoms]), torch.tensor([True])
    )
    assert loss == pytest.approx(0.0)


def test_branch_sum_can_never_be_reenabled_as_an_optimization_loss():
    with pytest.raises(ValueError, match="closure diagnostic only"):
        _epoch(
            None,
            None,
            None,
            torch.tensor(1.0),
            torch.ones(5),
            torch.device("cpu"),
            branch_sum_weight=0.1,
        )
