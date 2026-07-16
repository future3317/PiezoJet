"""Build a source-tagged Materials Project auxiliary table for JARVIS records.

The mapping is deliberately conservative: it accepts only an explicit
``reference == 'mp-<id>'`` in JARVIS ``dft_3d`` metadata.  Formula matching is
then an audit gate, not a substitute for an identifier.  The resulting table
is external-source auxiliary data only; neither splits nor JARVIS-only test
labels are modified here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jarvis.db.figshare import data as figshare_data
from mp_api.client import MPRester
from pymatgen.core import Composition

from .data import load_gmtnet_records


SUMMARY_FIELDS = (
    "material_id", "formula_pretty", "is_stable", "energy_above_hull",
    "band_gap", "formation_energy_per_atom", "total_magnetization",
    "ordering", "theoretical",
)


def _read_api_key(env_file: Path) -> str:
    """Read only the named key locally; never serialize or print credentials."""
    value = os.environ.get("MP_API_KEY")
    if value:
        return value
    if not env_file.is_file():
        raise FileNotFoundError(f"MP API key file is absent: {env_file}")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("MP_API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise ValueError("MP_API_KEY is absent or empty")


def _same_formula(left: str, right: str) -> bool:
    try:
        return Composition(left).reduced_composition == Composition(right).reduced_composition
    except Exception:
        return False


def build_auxiliary_table(
    *,
    data_root: Path,
    dft3d_cache_dir: Path,
    env_file: Path,
    output_dir: Path,
    batch_size: int = 500,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    key = _read_api_key(env_file)
    gmtnet = {str(record["JARVIS_ID"]): record for record in load_gmtnet_records(data_root)}
    jarvis_rows = figshare_data("dft_3d", store_dir=str(dft3d_cache_dir))
    reference = {str(row["jid"]): str(row.get("reference", "")).strip() for row in jarvis_rows}
    mapping = {
        jid: mp_id for jid, mp_id in reference.items()
        if jid in gmtnet and re.fullmatch(r"mp-\d+", mp_id)
    }
    requested = sorted(set(mapping.values()))
    documents: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    with MPRester(api_key=key) as mpr:
        for start in range(0, len(requested), batch_size):
            material_ids = requested[start : start + batch_size]
            try:
                response = mpr.materials.summary.search(material_ids=material_ids, fields=list(SUMMARY_FIELDS))
            except Exception as error:
                failures.extend({"material_id": mp_id, "error": str(error)} for mp_id in material_ids)
                continue
            documents.update({str(document.material_id): document for document in response})
    rows: list[dict[str, Any]] = []
    formula_mismatch: list[dict[str, str]] = []
    missing_summary: list[str] = []
    for jid, mp_id in sorted(mapping.items()):
        document = documents.get(mp_id)
        if document is None:
            missing_summary.append(mp_id)
            continue
        dumped = document.model_dump()
        formula = str(dumped["formula_pretty"])
        # GMTNet has an authoritative structure for this exact JARVIS ID.  An
        # explicit MP reference plus this formula gate is retained per row.
        gmtnet_formula = Composition(dict(Counter(gmtnet[jid]["atoms"]["elements"]))).reduced_formula
        if not _same_formula(gmtnet_formula, formula):
            formula_mismatch.append({"jid": jid, "mp_id": mp_id, "jarvis_formula": gmtnet_formula, "mp_formula": formula})
            continue
        rows.append({
            "jarvis_id": jid,
            "material_id": mp_id,
            "mapping_source": "JARVIS dft_3d reference exact mp-id",
            "formula_match": True,
            "source": "Materials Project summary",
            **{field: dumped.get(field) for field in SUMMARY_FIELDS if field != "material_id"},
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    table = output_dir / "materials_project_summary.jsonl"
    table.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Materials Project materials.summary",
        "fields": list(SUMMARY_FIELDS),
        "credential_policy": "MP_API_KEY read locally from environment or declared .env; never serialized",
        "mapping_policy": "explicit JARVIS dft_3d reference equal to mp-<id>, followed by reduced-formula audit",
        "gmt_net_records": len(gmtnet),
        "explicit_mp_mappings": len(mapping),
        "unique_mp_requests": len(requested),
        "matched_rows": len(rows),
        "missing_summary_material_ids": sorted(set(missing_summary)),
        "formula_mismatches": formula_mismatch,
        "request_failures": failures,
        "training_policy": "External-source auxiliary only. It must be availability-masked and source-tagged; JARVIS-only frozen validation/test reporting remains unchanged.",
        "table": str(table),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dft3d-cache-dir",
        type=Path,
        required=True,
        help="Dedicated jarvis-tools dft_3d cache; never reuse the raw_files cache directory.",
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(build_auxiliary_table(
        data_root=args.data_root, dft3d_cache_dir=args.dft3d_cache_dir,
        env_file=args.env_file, output_dir=args.output_dir, batch_size=args.batch_size,
    ), indent=2, default=str))


if __name__ == "__main__":
    main()
