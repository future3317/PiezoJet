"""Strict dataset/split identity for trainable and evaluable checkpoints."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_PROVENANCE_SCHEMA = 1


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Provenance source does not exist: {source}")
    return _sha256_bytes(source.read_bytes())


def _normalized_splits(
    splits: Mapping[str, list[str]],
    *,
    allow_cross_split_overlap: bool = False,
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        values = splits.get(name)
        if not isinstance(values, list):
            raise ValueError(f"Checkpoint provenance requires a {name} ID list")
        normalized[name] = sorted(str(value) for value in values)
        if len(normalized[name]) != len(set(normalized[name])):
            raise ValueError(f"Checkpoint provenance found duplicate {name} IDs")
    combined = normalized["train"] + normalized["val"] + normalized["test"]
    if not allow_cross_split_overlap and len(combined) != len(set(combined)):
        raise ValueError("Checkpoint provenance found IDs shared across splits")
    return normalized


def build_checkpoint_provenance(
    splits: Mapping[str, list[str]],
    split_source: str | Path,
    config: Mapping[str, Any],
    *,
    split_kind: str,
) -> dict[str, Any]:
    """Fingerprint the exact assignment and the files that define data roles."""
    noninductive_same_id = split_kind == "material_ids_same"
    normalized = _normalized_splits(
        splits, allow_cross_split_overlap=noninductive_same_id
    )
    source = Path(split_source).resolve()
    canonical_value = config.get("canonical_data_manifest_path")
    canonical = Path(str(canonical_value)).resolve() if canonical_value else None
    split_hashes = {
        name: _sha256_bytes("\n".join(values).encode("utf-8"))
        for name, values in normalized.items()
    }
    return {
        "schema": CHECKPOINT_PROVENANCE_SCHEMA,
        "split_kind": str(split_kind),
        "noninductive_same_id": noninductive_same_id,
        "split_source": str(source),
        "split_source_sha256": file_sha256(source),
        "split_counts": {name: len(values) for name, values in normalized.items()},
        "split_id_sha256": split_hashes,
        "all_ids_sha256": _sha256_bytes(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "canonical_data_manifest": str(canonical) if canonical is not None else None,
        "canonical_data_manifest_sha256": (
            file_sha256(canonical) if canonical is not None else None
        ),
        "data_commit": config.get("data_commit"),
        "fold_identity": config.get("fold_identity"),
        "seed": int(config["seed"]),
    }


def validate_checkpoint_provenance(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject checkpoints whose cohort/data identity differs from this run."""
    actual = payload.get("checkpoint_provenance")
    if not isinstance(actual, dict):
        raise ValueError(
            "Checkpoint has no strict split/data provenance; it cannot initialize "
            "or evaluate the maintained production path"
        )
    if actual.get("schema") != CHECKPOINT_PROVENANCE_SCHEMA:
        raise ValueError("Checkpoint provenance schema is unsupported")
    compared = (
        "split_kind",
        "noninductive_same_id",
        "split_source_sha256",
        "split_counts",
        "split_id_sha256",
        "all_ids_sha256",
        "canonical_data_manifest_sha256",
        "data_commit",
        "fold_identity",
        "seed",
    )
    mismatches = [name for name in compared if actual.get(name) != expected.get(name)]
    if mismatches:
        raise ValueError(
            "Checkpoint provenance does not match the active cohort: "
            + ", ".join(mismatches)
        )
    return actual
