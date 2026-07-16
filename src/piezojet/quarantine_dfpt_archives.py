"""Preserve public DFPT archives that cannot become GMTNet-aligned labels.

The raw bytes are useful provenance even when XML is malformed or the DFPT
reference structure differs from the GMTNet record.  This command creates no
tensor label and is never read by :class:`PiezoDataset`.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from .jarvis_dfpt import JarvisDFPTCache, JarvisRawFileIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-manifest", type=Path, required=True)
    parser.add_argument("--index-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    source = json.loads(args.failed_manifest.read_text(encoding="utf-8"))
    failed = source.get("failed")
    if not isinstance(failed, list) or not failed:
        raise ValueError("Failed manifest contains no quarantined archives")
    index = JarvisRawFileIndex(args.index_cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in failed:
        jid, error = str(item["jid"]), str(item["error"])
        archive = index.find(jid)
        blob = JarvisDFPTCache._download(archive.url, args.timeout)
        path = args.output_dir / archive.name
        if path.is_file():
            existing = path.read_bytes()
            if sha256(existing).digest() != sha256(blob).digest():
                raise ValueError(f"Existing quarantine archive hash differs: {path}")
        else:
            temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
            temporary.write_bytes(blob)
            temporary.replace(path)
        rows.append({
            "jid": jid,
            "archive_name": archive.name,
            "archive_sha256": sha256(blob).hexdigest(),
            "archive_bytes": len(blob),
            "parsed_tensor_label_available": False,
            "quarantine_reason": error,
        })
        print(f"quarantined {jid}: {len(blob)} bytes")
    manifest = {
        "schema": 1,
        "source": "JARVIS public raw_files / DFPT archives",
        "policy": (
            "Raw bytes retained for provenance only. Structure-mismatched or malformed-XML "
            "archives are never exposed as GMTNet-aligned tensor labels."
        ),
        "source_failed_manifest": str(args.failed_manifest),
        "archives": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
