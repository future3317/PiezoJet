import pytest

from piezojet.pretrain_bec_e3nn import _epoch_indices, _multi_epoch_batches
from piezojet.pretraining_protocol import provenance, validate_inductive_checkpoint


def test_inductive_pretraining_checkpoint_accepts_train_subset_only(tmp_path):
    split = tmp_path / "splits.json"
    split.write_text("{}", encoding="utf-8")
    entry = provenance(["A", "B"], split, "train")
    assert validate_inductive_checkpoint(
        {"pretraining_provenance": entry}, ["A", "B", "C"], ["D", "E"]
    ) == entry


def test_inductive_pretraining_checkpoint_rejects_held_out_id(tmp_path):
    split = tmp_path / "splits.json"
    split.write_text("{}", encoding="utf-8")
    entry = provenance(["A", "D"], split, "train")
    with pytest.raises(ValueError, match="non-train IDs|held-out IDs"):
        validate_inductive_checkpoint(
            {"pretraining_provenance": entry}, ["A", "B", "C"], ["D", "E"]
        )


def test_inductive_pretraining_checkpoint_requires_provenance():
    with pytest.raises(ValueError, match="no inductive provenance"):
        validate_inductive_checkpoint({}, ["A"], ["B"])


def test_multi_epoch_batches_keep_deterministic_epoch_boundaries():
    batches = _multi_epoch_batches(5, 42, 1, 2, 2)
    assert batches == [
        _epoch_indices(5, 42, 1)[0:2],
        _epoch_indices(5, 42, 1)[2:4],
        _epoch_indices(5, 42, 1)[4:5],
        _epoch_indices(5, 42, 2)[0:2],
        _epoch_indices(5, 42, 2)[2:4],
        _epoch_indices(5, 42, 2)[4:5],
    ]
