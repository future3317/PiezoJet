"""Audit strict and observed-only coverage of a JARVIS DFPT cache.

This module deliberately separates two facts which are often conflated:
an OUTCAR may be a provenance-complete, finite *partial* atom-resolved DFPT
label while its printed internal-strain blocks do not uniquely identify the
full strain-force tensor.  The former can supervise only observed/factor
quantities; the latter is required for the strict Lambda completion benchmark.
No acceptance tolerance in :mod:`piezojet.strain_completion` is modified here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .data import _raw_cartesian_target, load_gmtnet_records, response_norm_bin
from .jarvis_dfpt import DFPT_CACHE_SCHEMA, JarvisDFPTCache
from .strain_completion import _structure
from .completion_forensics import _failure_reasons
from .tensor_ops import piezo_voigt_to_cartesian, source_voigt_to_canonical


def atom_count_bin(atoms: int) -> str:
    if atoms <= 2:
        return "1-2"
    if atoms <= 4:
        return "3-4"
    if atoms <= 8:
        return "5-8"
    if atoms <= 16:
        return "9-16"
    return "17+"


def _finite_shape(value: Any, shape: tuple[int, ...]) -> bool:
    tensor = torch.as_tensor(value)
    return tuple(tensor.shape) == shape and bool(torch.isfinite(tensor).all())


def high_quality_partial_audit(record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Check source-array integrity without completing or changing Lambda.

    ``partial_qualified`` means that the independently useful BEC/Phi/branch
    arrays are finite, dimensionally coherent and archive-provenanced.
    Observed Lambda has its own mask because VASP numeric overflow can make a
    printed block irrecoverable without invalidating the other DFPT factors.
    """
    atoms = len(record["atoms"]["elements"])
    modes = 3 * atoms
    errors: list[str] = []
    internal_errors: list[str] = []
    if payload.get("schema") != DFPT_CACHE_SCHEMA:
        errors.append("not_schema4")
    if payload.get("jid") != str(record["JARVIS_ID"]):
        errors.append("jid_mismatch")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("source_archive"), dict):
        errors.append("missing_archive_provenance")
    required = {
        "born_charges": (atoms, 3, 3),
        "dynamical_eigenvalues": (modes,),
        "dynamical_eigenvectors": (modes, atoms, 3),
        "masses": (atoms,),
        "force_constants": (atoms, atoms, 3, 3),
        "dynamical_matrix": (atoms, atoms, 3, 3),
        "ionic_piezo_source": (3, 6),
        "total_piezo_source": (3, 6),
    }
    for name, shape in required.items():
        if name not in payload or not _finite_shape(payload[name], shape):
            errors.append(f"invalid_{name}")
    blocks = payload.get("internal_strain_tensors")
    ions, directions = payload.get("internal_strain_ions"), payload.get("internal_strain_directions")
    observed_blocks = 0
    if blocks is None or ions is None or directions is None:
        internal_errors.append("missing_internal_strain_observations")
    else:
        block_tensor = torch.as_tensor(blocks)
        ions_tensor, directions_tensor = torch.as_tensor(ions), torch.as_tensor(directions)
        observed_blocks = int(block_tensor.shape[0]) if block_tensor.ndim >= 1 else 0
        valid_blocks = (
            block_tensor.ndim == 3 and block_tensor.shape[1:] == (3, 3)
            and observed_blocks > 0 and bool(torch.isfinite(block_tensor).all())
            and tuple(ions_tensor.shape) == (observed_blocks,)
            and tuple(directions_tensor.shape) == (observed_blocks,)
            and bool(((ions_tensor >= 0) & (ions_tensor < atoms)).all())
            and bool(((directions_tensor >= 0) & (directions_tensor < 3)).all())
        )
        if not valid_blocks:
            internal_errors.append("invalid_internal_strain_observations")
    parse_audit = payload.get("internal_strain_parse_audit")
    complete_parse = (
        bool(parse_audit.get("complete_observed_block_parse", False))
        if isinstance(parse_audit, dict)
        else not internal_errors
    )
    return {
        "partial_qualified": not errors,
        "partial_failures": errors,
        "observed_internal_strain_qualified": not internal_errors and observed_blocks > 0,
        "observed_internal_strain_failures": internal_errors,
        "complete_observed_block_parse": complete_parse,
        "malformed_internal_strain_blocks": (
            len(parse_audit.get("malformed_blocks", [])) if isinstance(parse_audit, dict) else 0
        ),
        "observed_internal_strain_blocks": observed_blocks,
        "printed_block_coverage": observed_blocks / max(3 * atoms, 1),
    }


def _grouped(rows: list[dict[str, Any]], field: str, predicate: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    report: dict[str, dict[str, float | int]] = {}
    for key, values in sorted(groups.items()):
        numerator = sum(bool(value.get(predicate, False)) for value in values)
        report[key] = {
            "denominator": len(values), "numerator": numerator,
            "rate": numerator / max(len(values), 1),
        }
    return report


def _grouped_failure_reasons(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    output: dict[str, dict[str, Any]] = {}
    for key, values in sorted(groups.items()):
        rejected = [value for value in values if not value["strict_complete"]]
        output[key] = {
            "population": len(values),
            "strict_rejected": len(rejected),
            "nonexclusive_failure_counts": dict(Counter(
                reason for value in rejected for reason in value.get("strict_failure_reasons", [])
            )),
        }
    return output


def _response_bin(norm: float) -> str:
    return str(response_norm_bin(torch.tensor([norm], dtype=torch.float64)))


def audit_cache(
    records: list[dict[str, Any]],
    cache: JarvisDFPTCache,
    strict_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, record in enumerate(records, start=1):
        jid = str(record["JARVIS_ID"])
        atoms = len(record["atoms"]["elements"])
        target = _raw_cartesian_target(record)
        row: dict[str, Any] = {
            "jid": jid,
            "atoms": atoms,
            "atom_count_bin": atom_count_bin(atoms),
            "gmtnet_total_frobenius_norm_c_per_m2": float(torch.linalg.vector_norm(target)),
            "gmtnet_response_bin": _response_bin(float(torch.linalg.vector_norm(target))),
            "raw_dfpt_available": False,
            "partial_qualified": False,
            "observed_internal_strain_qualified": False,
            "complete_observed_block_parse": False,
            "malformed_internal_strain_blocks": 0,
            "strict_complete": bool(strict_rows.get(jid, {}).get("accepted", False)),
            "strict_failure_reasons": (
                _failure_reasons(strict_rows[jid]) if jid in strict_rows and not bool(strict_rows[jid].get("accepted", False))
                else (["raw_dfpt_unavailable"] if jid not in strict_rows else [])
            ),
        }
        try:
            analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
            row["crystal_system"] = str(analyzer.get_crystal_system())
            row["space_group_number"] = int(analyzer.get_space_group_number())
        except Exception as error:
            row["crystal_system"] = "unresolved"
            row["space_group_number"] = None
            row["structure_classification_error"] = str(error)
        try:
            payload = cache.load(jid)
        except Exception as error:
            payload = None
            row["cache_load_error"] = str(error)
        if payload is not None:
            row["raw_dfpt_available"] = True
            row.update(high_quality_partial_audit(record, payload))
            if row["partial_qualified"]:
                ionic = piezo_voigt_to_cartesian(source_voigt_to_canonical(payload["ionic_piezo_source"]))
                row["outcar_ionic_frobenius_norm_c_per_m2"] = float(torch.linalg.vector_norm(ionic))
                row["outcar_ionic_response_bin"] = _response_bin(float(torch.linalg.vector_norm(ionic)))
        rows.append(row)
        print(f"[{position}/{len(records)}] {jid}: raw={row['raw_dfpt_available']} partial={row['partial_qualified']} strict={row['strict_complete']}")
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row["raw_dfpt_available"]]
    partial = [row for row in rows if row["partial_qualified"]]
    observed_lambda = [row for row in rows if row["observed_internal_strain_qualified"]]
    complete_parse = [row for row in rows if row.get("complete_observed_block_parse", False)]
    malformed_parse = [row for row in rows if int(row.get("malformed_internal_strain_blocks", 0)) > 0]
    strict = [row for row in rows if row["strict_complete"]]
    dimensions = ("crystal_system", "atom_count_bin", "gmtnet_response_bin")
    return {
        "population": len(rows),
        "raw_dfpt_available": len(available),
        "high_quality_partial": len(partial),
        "observed_internal_strain_partial": len(observed_lambda),
        "complete_observed_internal_strain_parse": len(complete_parse),
        "records_with_malformed_internal_strain_blocks": len(malformed_parse),
        "strict_complete": len(strict),
        "raw_dfpt_rate": len(available) / max(len(rows), 1),
        "partial_rate_given_population": len(partial) / max(len(rows), 1),
        "strict_rate_given_raw_available": len(strict) / max(len(available), 1),
        "selection_bias": {
            dimension: {
                "raw_dfpt": _grouped(rows, dimension, "raw_dfpt_available"),
                "high_quality_partial": _grouped(rows, dimension, "partial_qualified"),
                "observed_internal_strain_partial": _grouped(rows, dimension, "observed_internal_strain_qualified"),
                "complete_observed_internal_strain_parse": _grouped(rows, dimension, "complete_observed_block_parse"),
                "strict_complete": _grouped(rows, dimension, "strict_complete"),
            }
            for dimension in dimensions
        },
        "strict_failure_reasons_by": {
            dimension: _grouped_failure_reasons(rows, dimension)
            for dimension in dimensions
        },
        "partial_failure_counts": dict(Counter(
            reason for row in available for reason in row.get("partial_failures", [])
        )),
        "observed_internal_strain_failure_counts": dict(Counter(
            reason for row in available for reason in row.get("observed_internal_strain_failures", [])
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dfpt-dir", type=Path, required=True)
    parser.add_argument("--strict-completion-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    strict_rows: dict[str, dict[str, Any]] = {}
    if args.strict_completion_dir is not None:
        manifest_path = args.strict_completion_dir / "manifest.json"
        if manifest_path.is_file():
            strict_rows = {str(row["jid"]): row for row in json.loads(manifest_path.read_text(encoding="utf-8")).get("rows", [])}
    rows = audit_cache(load_gmtnet_records(args.data_root), JarvisDFPTCache(args.dfpt_dir), strict_rows)
    result = {
        "schema": 1,
        "policy": (
            "Strict completion is unchanged. High-quality partial means a schema-4, archive-provenanced, "
            "finite observed-label payload; it never asserts unprinted Lambda entries are known."
        ),
        "dfpt_dir": str(args.dfpt_dir),
        "strict_completion_dir": None if args.strict_completion_dir is None else str(args.strict_completion_dir),
        "summary": summary(rows), "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    qualified = [row["jid"] for row in rows if row["partial_qualified"]]
    (args.output_dir / "high_quality_partial_ids.json").write_text(json.dumps({"material_ids": qualified}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
