"""Lightweight GMTNet source loading without model-stack imports."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


PIEZO_FILE = "jarvis_diele_piezo.pkl"
PIEZO_FIELD = "piezoelectric_C_m2"


def load_gmtnet_records(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root) / "data" / PIEZO_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Expected official GMTNet piezoelectric file: {path}")
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty list in {path}")
    required = {"JARVIS_ID", "atoms", PIEZO_FIELD}
    missing = required.difference(records[0])
    if missing:
        raise ValueError(f"Missing required GMTNet fields: {sorted(missing)}")
    valid: list[dict[str, Any]] = []
    for record in records:
        value = record.get(PIEZO_FIELD)
        tensor = np.asarray(value) if value is not None else None
        if tensor is None or tensor.shape != (3, 6):
            continue
        if not np.isfinite(tensor).all() or np.abs(tensor).max() >= 100:
            continue
        valid.append(record)
    if not valid:
        raise ValueError("No finite 3x6 piezoelectric records found")
    return valid
