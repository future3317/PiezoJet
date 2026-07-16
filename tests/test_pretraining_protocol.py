import pytest

from piezojet.pretraining_protocol import provenance, validate_inductive_checkpoint


def test_inductive_pretraining_checkpoint_accepts_train_subset_only(tmp_path):
    entry = provenance(["A", "B"], tmp_path / "splits.json", "train")
    assert validate_inductive_checkpoint(
        {"pretraining_provenance": entry}, ["A", "B", "C"], ["D", "E"]
    ) == entry


def test_inductive_pretraining_checkpoint_rejects_held_out_id(tmp_path):
    entry = provenance(["A", "D"], tmp_path / "splits.json", "train")
    with pytest.raises(ValueError, match="non-train IDs|held-out IDs"):
        validate_inductive_checkpoint(
            {"pretraining_provenance": entry}, ["A", "B", "C"], ["D", "E"]
        )


def test_inductive_pretraining_checkpoint_requires_provenance():
    with pytest.raises(ValueError, match="no inductive provenance"):
        validate_inductive_checkpoint({}, ["A"], ["B"])
