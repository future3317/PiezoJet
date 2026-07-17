"""Inductive structural-pretraining provenance checks.

Structural labels are not used during masked-species/denoising pretraining,
but exact validation/test structures still constitute transductive information
in a formula-OOD benchmark.  Production checkpoints therefore record their
material IDs and are accepted only when those IDs are a subset of the current
training panel.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from .checkpoint_provenance import file_sha256


PRETRAINING_PROTOCOL_SCHEMA = 2


def _normalized_ids(ids: Iterable[str]) -> list[str]:
    values = sorted({str(value) for value in ids})
    if not values:
        raise ValueError("Inductive pretraining requires at least one material ID")
    return values


def _id_hash(ids: Iterable[str]) -> str:
    return sha256("\n".join(_normalized_ids(ids)).encode("utf-8")).hexdigest()


def provenance(
    material_ids: Iterable[str],
    split_file: Path,
    split_name: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return persisted, reviewable provenance for one pretraining panel."""
    ids = _normalized_ids(material_ids)
    if split_name != "train":
        raise ValueError("Production structural pretraining must use only the train split")
    source = split_file.resolve()
    canonical_value = None if config is None else config.get("canonical_data_manifest_path")
    canonical = Path(str(canonical_value)).resolve() if canonical_value else None
    return {
        "schema": PRETRAINING_PROTOCOL_SCHEMA,
        "mode": "inductive_train_only",
        "split_file": str(source),
        "split_file_sha256": file_sha256(source),
        "split_name": split_name,
        "material_ids": ids,
        "material_id_sha256": _id_hash(ids),
        "canonical_data_manifest_sha256": (
            file_sha256(canonical) if canonical is not None else None
        ),
        "data_commit": None if config is None else config.get("data_commit"),
    }


def validate_inductive_checkpoint(
    payload: dict[str, Any],
    train_ids: Iterable[str],
    held_out_ids: Iterable[str],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject missing, transductive, or panel-mismatched pretraining state."""
    entry = payload.get("pretraining_provenance")
    if not isinstance(entry, dict):
        raise ValueError(
            "Pretraining checkpoint has no inductive provenance. Regenerate it with "
            "piezojet.pretrain --splits-file using the current train panel."
        )
    if entry.get("schema") != PRETRAINING_PROTOCOL_SCHEMA or entry.get("mode") != "inductive_train_only":
        raise ValueError("Pretraining checkpoint is not an inductive train-only checkpoint")
    if entry.get("split_name") != "train":
        raise ValueError("Pretraining checkpoint was not trained on the train split")
    ids = _normalized_ids(entry.get("material_ids", []))
    train, held_out = set(_normalized_ids(train_ids)), set(str(value) for value in held_out_ids)
    if set(ids) & held_out:
        leaked = sorted(set(ids) & held_out)
        raise ValueError(f"Pretraining checkpoint contains held-out IDs: {leaked[:5]}")
    if not set(ids).issubset(train):
        leaked = sorted(set(ids) - train)
        raise ValueError(f"Pretraining checkpoint contains non-train IDs: {leaked[:5]}")
    if entry.get("material_id_sha256") != _id_hash(ids):
        raise ValueError("Pretraining checkpoint material-ID hash does not match its stored IDs")
    split_file = Path(str(entry.get("split_file", "")))
    if not split_file.is_file() or entry.get("split_file_sha256") != file_sha256(split_file):
        raise ValueError("Pretraining checkpoint split-file hash is missing or stale")
    if config is not None:
        canonical_value = config.get("canonical_data_manifest_path")
        canonical_hash = (
            file_sha256(Path(str(canonical_value)).resolve())
            if canonical_value else None
        )
        if entry.get("canonical_data_manifest_sha256") != canonical_hash:
            raise ValueError("Pretraining checkpoint canonical-data manifest differs")
        if entry.get("data_commit") != config.get("data_commit"):
            raise ValueError("Pretraining checkpoint GMTNet data commit differs")
    return entry
