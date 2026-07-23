"""Small runtime helpers that avoid redundant checkpoint serialization."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def atomic_link_or_copy(source: str | Path, destination: str | Path) -> None:
    """Make ``destination`` refer to ``source`` without serializing twice.

    A hard link is instantaneous on the experiment filesystem.  The copy
    fallback keeps the helper usable on filesystems that do not support links;
    replacing a temporary path makes readers see either the old or new file.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    temporary = destination_path.with_name(f".{destination_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source_path, temporary)
    except OSError:
        shutil.copyfile(source_path, temporary)
    temporary.replace(destination_path)
