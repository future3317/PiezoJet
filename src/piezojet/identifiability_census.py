"""Run a label-safe identifiability census over public JARVIS DFPT records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .data import load_gmtnet_records
from .evaluate_dfpt import optical_eigensystem
from .identifiability import build_identification_system, identification_certificate
from .jarvis_dfpt import JarvisDFPTCache
from .strain_completion import _structure


SCHEMA = 1


def _ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    values = payload.get("material_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("Population file must be a list or contain material_ids")
    return [str(value) for value in values]


def _atom_bin(atoms: int) -> str:
    if atoms <= 2:
        return "1-2"
    if atoms <= 4:
        return "3-4"
    if atoms <= 8:
        return "5-8"
    if atoms <= 16:
        return "9-16"
    return "17+"


def _metadata(record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
    optical, _ = optical_eigensystem(payload["force_constants"])
    minimum = float(optical.min()) if optical.numel() else float("inf")
    stability = "stable" if minimum > 1e-4 else "soft_positive" if minimum > 0 else "unstable"
    atoms = len(record["atoms"]["elements"])
    return {
        "atoms": atoms,
        "atom_count_bin": _atom_bin(atoms),
        "space_group_number": int(analyzer.get_space_group_number()),
        "space_group_symbol": str(analyzer.get_space_group_symbol()),
        "crystal_system": str(analyzer.get_crystal_system()),
        "minimum_optical_eigenvalue_eV_per_A2": minimum,
        "stability": stability,
    }


def _load_strict_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["jid"])
        for row in payload.get("rows", [])
        if bool(row.get("accepted", False))
    }


def _group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    output = []
    for value, members in sorted(groups.items()):
        full_joint = sum(bool(row["joint_full_identifiable"]) for row in members)
        new_joint = sum(
            bool(row["joint_full_identifiable"])
            and not bool(row["printed_full_identifiable"])
            for row in members
        )
        conditions = [
            float(row["condition_joint_scaled"])
            for row in members
            if row["condition_joint_scaled"] is not None
        ]
        output.append({
            key: value,
            "materials": len(members),
            "macro_full_identifiable": sum(bool(row["macro_full_identifiable"]) for row in members),
            "printed_full_identifiable": sum(bool(row["printed_full_identifiable"]) for row in members),
            "joint_full_identifiable": full_joint,
            "joint_increment_over_printed": new_joint,
            "joint_full_fraction": full_joint / len(members),
            "condition_joint_scaled_median": median(conditions) if conditions else None,
        })
    return output


def _finalize(output_dir: Path, shard_count: int, requested: int, population: Path) -> None:
    rows, errors = [], []
    for index in range(shard_count):
        shard = output_dir / f"shard_{index}_of_{shard_count}.json"
        payload = json.loads(shard.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA or payload.get("shard") != {"index": index, "count": shard_count}:
            raise ValueError(f"Incompatible census shard: {shard}")
        rows.extend(payload["rows"])
        errors.extend(payload["errors"])
    if len({row["jid"] for row in rows}) != len(rows):
        raise ValueError("Duplicate material IDs across census shards")
    rows.sort(key=lambda row: row["jid"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if rows:
        with (output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    incremental = [
        row for row in rows
        if row["joint_full_identifiable"] and not row["printed_full_identifiable"]
    ]
    summary = {
        "schema": SCHEMA,
        "scope": (
            "label-safe development census; frozen validation/test response labels are not read"
        ),
        "population_file": str(population),
        "requested_materials": requested,
        "audited_materials": len(rows),
        "errors": len(errors),
        "strict_complete": sum(bool(row["strict_completion"]) for row in rows),
        "macro_full_identifiable": sum(bool(row["macro_full_identifiable"]) for row in rows),
        "printed_full_identifiable": sum(bool(row["printed_full_identifiable"]) for row in rows),
        "joint_full_identifiable": sum(bool(row["joint_full_identifiable"]) for row in rows),
        "joint_increment_over_printed": len(incremental),
        "joint_increment_material_ids": [row["jid"] for row in incremental],
        "stability_counts": dict(Counter(row["stability"] for row in rows)),
        "group_statistics": {
            "crystal_system": _group(rows, "crystal_system"),
            "atom_count_bin": _group(rows, "atom_count_bin"),
            "space_group_number": _group(rows, "space_group_number"),
            "stability": _group(rows, "stability"),
            "strict_completion": _group(rows, "strict_completion"),
            "printed_blocks": _group(rows, "printed_blocks"),
        },
        "error_rows": errors,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "requested_materials", "audited_materials", "errors",
        "macro_full_identifiable", "printed_full_identifiable",
        "joint_full_identifiable", "joint_increment_over_printed",
    )}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dfpt-dir", type=Path, required=True)
    parser.add_argument(
        "--population",
        type=Path,
        default=Path("data/processed/electrostatic_development_folds_v2.json"),
    )
    parser.add_argument("--strict-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--finalize-shards", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard specification")
    if args.progress_interval < 1:
        raise ValueError("--progress-interval must be positive")
    material_ids = _ids(args.population)
    if args.finalize_shards:
        _finalize(args.output_dir, args.shard_count, len(material_ids), args.population)
        return
    records = {str(record["JARVIS_ID"]): record for record in load_gmtnet_records(args.data_root)}
    strict_ids = _load_strict_ids(args.strict_manifest)
    cache = JarvisDFPTCache(args.dfpt_dir)
    selected = [
        jid for position, jid in enumerate(material_ids)
        if position % args.shard_count == args.shard_index
    ]
    rows, errors = [], []
    for position, jid in enumerate(selected, start=1):
        try:
            payload = cache.load(jid)
            if payload is None:
                raise FileNotFoundError("parsed DFPT payload unavailable")
            system = build_identification_system(records[jid], payload)
            row = {
                "jid": jid,
                **_metadata(records[jid], payload),
                "strict_completion": jid in strict_ids,
                **identification_certificate(system),
            }
            rows.append(row)
            if position == 1 or position % args.progress_interval == 0 or position == len(selected):
                print(
                    f"[{position}/{len(selected)}] {jid} d={row['identifiable_dimension']} "
                    f"r_macro={row['rank_macro']} r_printed={row['rank_printed']} "
                    f"r_joint={row['rank_joint']}",
                    flush=True,
                )
        except Exception as error:
            errors.append({"jid": jid, "error": str(error)})
            print(f"[{position}/{len(selected)}] {jid} ERROR {error}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard = {
        "schema": SCHEMA,
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "population": str(args.population),
        "frozen_validation_test_labels_read": False,
        "rows": rows,
        "errors": errors,
    }
    (args.output_dir / f"shard_{args.shard_index}_of_{args.shard_count}.json").write_text(
        json.dumps(shard, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
