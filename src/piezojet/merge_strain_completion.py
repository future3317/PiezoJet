"""Merge independently audited strict-Lambda caches without mutating a source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .strain_completion import COMPLETION_SCHEMA


def _load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != COMPLETION_SCHEMA
        or payload.get("jid") != path.stem
        or not bool(payload.get("audit", {}).get("accepted", False))
    ):
        raise ValueError(f"Invalid strict-completion payload: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir in args.source_dirs:
        raise ValueError("Output directory must be distinct from every source directory")
    merged: dict[str, dict] = {}
    for directory in args.source_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Completion source directory does not exist: {directory}")
        for path in sorted(directory.glob("*.pt")):
            payload = _load(path)
            jid = str(payload["jid"])
            previous = merged.get(jid)
            if previous is not None and not torch.allclose(
                previous["internal_strain_full"], payload["internal_strain_full"], atol=1e-6, rtol=0.0
            ):
                raise ValueError(f"Conflicting strict completions for {jid}")
            merged[jid] = payload
    if not merged:
        raise ValueError("No accepted strict-completion payloads found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for jid, payload in merged.items():
        temporary = args.output_dir / f"{jid}.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(args.output_dir / f"{jid}.pt")
    manifest = {
        "schema": COMPLETION_SCHEMA,
        "sources": [str(path) for path in args.source_dirs],
        "accepted": len(merged),
        "policy": "merged only schema-validated, accepted strict completions; duplicate tensors must agree exactly to tolerance",
        "material_ids": sorted(merged),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
