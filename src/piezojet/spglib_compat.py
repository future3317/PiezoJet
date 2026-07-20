"""Narrow compatibility wrapper for spglib's invalid-cell behavior."""

from __future__ import annotations

from typing import Any

import spglib

try:
    from spglib.error import SpglibError
except ImportError:  # spglib < 2.7 returns None for the same invalid cells.
    class SpglibError(Exception):
        """Compatibility sentinel used only when old spglib lacks the type."""


def symmetry_dataset_or_none(cell: Any, *, symprec: float) -> Any | None:
    """Return ``None`` for cells that spglib explicitly rejects as invalid."""

    try:
        return spglib.get_symmetry_dataset(cell, symprec=symprec)
    except SpglibError:
        return None
