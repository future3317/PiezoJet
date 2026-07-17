import json

import pytest

from piezojet.checkpoint_provenance import (
    build_checkpoint_provenance,
    validate_checkpoint_provenance,
)


def _write_split(path, train=("a", "b"), val=("c",), test=("d",)):
    splits = {"train": list(train), "val": list(val), "test": list(test)}
    path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return splits


def _config(tmp_path):
    manifest = tmp_path / "canonical.json"
    manifest.write_text('{"schema":1}', encoding="utf-8")
    return {
        "seed": 42,
        "data_commit": "a" * 40,
        "canonical_data_manifest_path": str(manifest),
    }


def test_checkpoint_provenance_accepts_exact_split_and_data_identity(tmp_path):
    source = tmp_path / "split.json"
    splits = _write_split(source)
    expected = build_checkpoint_provenance(
        splits, source, _config(tmp_path), split_kind="explicit_frozen"
    )
    assert validate_checkpoint_provenance(
        {"checkpoint_provenance": expected}, expected
    ) == expected


def test_checkpoint_provenance_rejects_same_size_different_materials(tmp_path):
    source = tmp_path / "split.json"
    config = _config(tmp_path)
    first = _write_split(source)
    saved = build_checkpoint_provenance(
        first, source, config, split_kind="explicit_frozen"
    )
    second = _write_split(source, train=("a", "x"))
    current = build_checkpoint_provenance(
        second, source, config, split_kind="explicit_frozen"
    )
    with pytest.raises(ValueError, match="split_id_sha256|all_ids_sha256"):
        validate_checkpoint_provenance({"checkpoint_provenance": saved}, current)


def test_checkpoint_provenance_rejects_missing_legacy_metadata(tmp_path):
    source = tmp_path / "split.json"
    expected = build_checkpoint_provenance(
        _write_split(source), source, _config(tmp_path), split_kind="explicit_frozen"
    )
    with pytest.raises(ValueError, match="no strict split/data provenance"):
        validate_checkpoint_provenance({}, expected)


def test_only_explicit_same_id_diagnostic_may_overlap_splits(tmp_path):
    source = tmp_path / "ids.json"
    source.write_text('["a"]', encoding="utf-8")
    splits = {"train": ["a"], "val": ["a"], "test": []}
    with pytest.raises(ValueError, match="shared across splits"):
        build_checkpoint_provenance(
            splits, source, _config(tmp_path), split_kind="explicit_frozen"
        )
    diagnostic = build_checkpoint_provenance(
        splits, source, _config(tmp_path), split_kind="material_ids_same"
    )
    assert diagnostic["noninductive_same_id"] is True
    inductive = {
        **diagnostic,
        "split_kind": "explicit_frozen",
        "noninductive_same_id": False,
    }
    with pytest.raises(ValueError, match="split_kind|noninductive_same_id"):
        validate_checkpoint_provenance(
            {"checkpoint_provenance": diagnostic}, inductive
        )
