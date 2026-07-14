"""Merge schema-validated JARVIS DFPT caches without altering either source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .jarvis_dfpt import DFPT_CACHE_SCHEMA, JarvisDFPTCache


def _same_payload(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Require duplicate JARVIS IDs to carry the same parsed source labels."""
    if left.get("schema") != DFPT_CACHE_SCHEMA or right.get("schema") != DFPT_CACHE_SCHEMA:
        return False
    if left.get("jid") != right.get("jid") or left.get("source_archive") != right.get("source_archive"):
        return False
    tensor_keys = (
        "born_charges", "dynamical_eigenvalues", "dynamical_eigenvectors", "masses",
        "force_constants", "dynamical_matrix", "ionic_piezo_source", "total_piezo_source",
        "internal_strain_tensors", "internal_strain_ions", "internal_strain_directions",
    )
    return all(
        isinstance(left.get(key), torch.Tensor)
        and isinstance(right.get(key), torch.Tensor)
        and torch.equal(left[key], right[key])
        for key in tensor_keys
    )


def merge_caches(source_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Copy non-conflicting cache entries into a fresh cache and persist provenance."""
    if output_dir in source_dirs:
        raise ValueError("Output directory must differ from every source directory")
    merged: dict[str, dict[str, Any]] = {}
    for directory in source_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        cache = JarvisDFPTCache(directory)
        for path in sorted(directory.glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            jid = str(payload.get("jid", ""))
            if payload.get("schema") != DFPT_CACHE_SCHEMA or not jid or cache.path(jid).name != path.name:
                raise ValueError(f"Invalid JARVIS DFPT cache payload: {path}")
            previous = merged.get(jid)
            if previous is not None and not _same_payload(previous, payload):
                raise ValueError(f"Conflicting JARVIS DFPT payloads for {jid}")
            merged[jid] = payload
    if not merged:
        raise ValueError("No cache payloads found")
    destination = JarvisDFPTCache(output_dir)
    for payload in merged.values():
        destination.save(payload)
    manifest = {
        "schema": DFPT_CACHE_SCHEMA,
        "sources": [str(path) for path in source_dirs],
        "cached": len(merged),
        "material_ids": sorted(merged),
        "policy": "fresh union of schema-validated source caches; duplicate JARVIS IDs must agree exactly",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_caches(args.source_dirs, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
