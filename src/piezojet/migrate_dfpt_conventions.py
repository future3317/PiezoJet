"""Migrate immutable JARVIS DFPT cache copies to explicit VASP conventions.

The schema-2 cache preserved VASP BEC and internal-strain arrays verbatim.
Schema 4 fixes the source-to-internal boundary without redownloading raw
archives: it retains those source arrays and writes a separately named cache
with ``Z_internal = Z_source.T`` and the printed OUTCAR force derivative
``Lambda_internal = Lambda_source``.

This is an auditable migration, not a numerical relabelling.  Existing caches
and strict-completion directories remain untouched; callers must regenerate
completion labels against the output directory before training with it.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from .jarvis_dfpt import (
    DFPT_CACHE_SCHEMA,
    source_born_to_internal,
    source_internal_strain_to_internal,
    tensor_sha256,
)


SOURCE_SCHEMA = 2


def payload_digest(payload: dict[str, Any]) -> str:
    """Stable digest over the source tensors that determine this migration."""
    digest = sha256()
    for name in ("jid", "born_charges", "internal_strain_tensors", "force_constants"):
        value = payload[name]
        if isinstance(value, str):
            digest.update(value.encode("utf-8"))
        else:
            tensor = torch.as_tensor(value).detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _legacy_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    """Record precisely what a schema-2 migration can and cannot recover."""
    checksums: dict[str, str] = {}
    for name, value in sorted(payload.items()):
        if isinstance(value, torch.Tensor):
            checksums[name] = tensor_sha256(value)
        elif isinstance(value, dict) and name == "epsilon":
            for key, tensor in sorted(value.items()):
                checksums[f"epsilon.{key}"] = tensor_sha256(tensor)
    return {
        "schema": 1,
        "status": "legacy_migration_without_raw_archive",
        "source_archive": {
            "name": payload.get("source_archive"),
            "archive_sha256": None,
            "vasprun_xml_sha256": None,
            "outcar_sha256": None,
        },
        "parser": {
            "jarvis_tools_version": None,
            "jarvis_tools_commit": None,
            "parser_schema": "unknown: source cache predates parser provenance",
        },
        "force_constant_conversion": {
            "raw_fc_mass": None,
            "converted_fc_mass": None,
            "expected_relation": "not recoverable from schema-2 migration alone",
        },
        "tensor_sha256": checksums,
        "warning": (
            "This cache is reproducible only from the prior payload digest. "
            "Re-fetch the public archive into a fresh schema-4 cache before "
            "using raw-file provenance as scientific evidence."
        ),
    }


def migrate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a schema-4 internal-convention copy of one schema-2 payload."""
    if payload.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"Expected schema-{SOURCE_SCHEMA} source cache, got {payload.get('schema')}")
    required = {"jid", "born_charges", "internal_strain_tensors", "force_constants"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Source payload is missing {missing}")
    result = dict(payload)
    born_source = torch.as_tensor(payload["born_charges"]).clone()
    strain_source = torch.as_tensor(payload["internal_strain_tensors"]).clone()
    result.update(
        {
            "schema": DFPT_CACHE_SCHEMA,
            "born_charges_source": born_source,
            "born_charges": source_born_to_internal(born_source),
            "internal_strain_tensors_source": strain_source,
            "internal_strain_tensors": source_internal_strain_to_internal(strain_source),
            "conventions": {
                "migration": "schema-2 raw VASP arrays -> schema-3 PiezoJet internal arrays",
                "born_charges_source": "VASP Z[i,j]=dP_i/du_j",
                "born_charges_internal": "Z[j,i], coordinate/force row and polarization/field column",
                "internal_strain_source": "VASP OUTCAR printed dF/deta=-d^2E/(du deta)",
                "internal_strain_internal": "Lambda=dF/deta (identity transform)",
            },
            "source_schema": SOURCE_SCHEMA,
            "source_payload_sha256": payload_digest(payload),
            "provenance": _legacy_provenance(payload),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source_paths = sorted(args.source_dir.glob("*.pt"))
    if not source_paths:
        raise FileNotFoundError(f"No cache payloads found under {args.source_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    migrated: list[dict[str, str]] = []
    for source_path in source_paths:
        output_path = args.output_dir / source_path.name
        if output_path.exists() and not args.overwrite:
            payload = torch.load(output_path, map_location="cpu", weights_only=False)
            if payload.get("schema") != DFPT_CACHE_SCHEMA:
                raise ValueError(f"Existing output has incompatible schema: {output_path}")
            migrated.append({"jid": str(payload["jid"]), "path": output_path.name, "status": "existing"})
            continue
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        migrated_payload = migrate_payload(payload)
        torch.save(migrated_payload, output_path)
        migrated.append({"jid": str(migrated_payload["jid"]), "path": output_path.name, "status": "migrated"})
    manifest = {
        "schema": DFPT_CACHE_SCHEMA,
        "source_schema": SOURCE_SCHEMA,
        "source_directory": str(args.source_dir),
        "transform": {
            "born_charges": "transpose final two axes",
            "internal_strain_tensors": "identity (OUTCAR prints dF/deta)",
            "force_constants": "unchanged",
        },
        "cached": len(migrated),
        "materials": migrated,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "cached": len(migrated)}, indent=2))


if __name__ == "__main__":
    main()
