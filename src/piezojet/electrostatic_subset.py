"""Fixed, auditable response-supervision subsets for electrostatic adjudication."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Iterable


SUBSET_SCHEMA = 1


def _normalized(values: Iterable[str]) -> list[str]:
    return [str(value) for value in values]


def material_id_sha256(values: Iterable[str]) -> str:
    ids = sorted(set(_normalized(values)))
    return sha256("\n".join(ids).encode("utf-8")).hexdigest()


def load_response_subset(
    path: Path,
    *,
    fold: int,
    allowed_ids: Iterable[str],
) -> tuple[list[str], dict[str, object]]:
    """Load one preregistered subset and reject duplicates or fold leakage."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema") != SUBSET_SCHEMA:
        raise ValueError("Unsupported electrostatic response-subset schema")
    if int(payload.get("fold", -1)) != fold:
        raise ValueError("Response-subset fold does not match the requested fold")
    ids = _normalized(payload.get("material_ids", []))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Response subset must contain unique material IDs")
    allowed = set(_normalized(allowed_ids))
    outside = sorted(set(ids) - allowed)
    if outside:
        raise ValueError(f"Response subset contains non-fold-train IDs: {outside[:5]}")
    if payload.get("material_id_sha256") != material_id_sha256(ids):
        raise ValueError("Response-subset material-ID hash is stale")
    declared = int(payload.get("materials", -1))
    if declared != len(ids):
        raise ValueError("Response-subset declared material count is stale")
    return ids, payload
