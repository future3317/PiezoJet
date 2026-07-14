"""Create a fixed, source-linked report for one strict-completion expansion."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .completion_forensics import _failure_reasons, _negative_optical_modes
from .data import _raw_cartesian_target, load_gmtnet_records, response_norm_bin
from .jarvis_dfpt import JarvisDFPTCache
from .strain_completion import _structure


def _rate(rows: list[dict]) -> dict[str, float | int]:
    accepted = sum(bool(row["accepted"]) for row in rows)
    return {"materials": len(rows), "accepted": accepted, "rate": accepted / max(len(rows), 1)}


def _grouped_rates(rows: list[dict], field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: _rate(value) for key, value in sorted(groups.items())}


def _polar(record: dict) -> bool:
    operations = SpacegroupAnalyzer(_structure(record), symprec=1e-5).get_symmetry_operations(cartesian=True)
    # The average of the vector representation is the projector onto the
    # polar invariant subspace; a nonzero rank admits a polar vector.
    projector = torch.tensor([operation.rotation_matrix for operation in operations], dtype=torch.float64).mean(dim=0)
    return bool(torch.linalg.matrix_rank(projector, tol=1e-7) > 0)


def _atom_count_bin(atoms: int) -> int:
    return 0 if atoms <= 2 else 1 if atoms <= 4 else 2 if atoms <= 8 else 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--dfpt-dir", type=Path, default=Path("data/processed/jarvis_dfpt_v1"))
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--completion-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    material_metadata = {str(item["jid"]): item for item in cohort["materials"]}
    records = {str(item["JARVIS_ID"]): item for item in load_gmtnet_records(args.data_root)}
    manifest = json.loads(args.completion_manifest.read_text(encoding="utf-8"))
    audits = {str(item["jid"]): item for item in manifest["rows"]}
    cache = JarvisDFPTCache(args.dfpt_dir)
    profiles = []
    for jid in cohort["material_ids"]:
        jid = str(jid)
        record, audit, payload = records[jid], audits.get(jid), cache.load(jid)
        if audit is None or payload is None:
            profiles.append({"jid": jid, "available": False, "accepted": False, "failure_reasons": "download_or_parse_failure"})
            continue
        analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
        target = _raw_cartesian_target(record)
        metadata = material_metadata[jid]
        profiles.append({
            "jid": jid,
            "available": True,
            "accepted": bool(audit["accepted"]),
            "failure_reasons": ";".join(_failure_reasons(audit)) if not audit["accepted"] else "",
            "formula": metadata["formula"],
            "atoms": len(record["atoms"]["elements"]),
            "atom_count_bin": _atom_count_bin(len(record["atoms"]["elements"])),
            "space_group_number": int(analyzer.get_space_group_number()),
            "space_group_symbol": str(analyzer.get_space_group_symbol()),
            "space_group_operations": int(audit["space_group_operations"]),
            "crystal_system": str(analyzer.get_crystal_system()),
            "polar": _polar(record),
            "response_norm": float(torch.linalg.vector_norm(target)),
            "response_bin": response_norm_bin(target),
            "negative_optical_mode": _negative_optical_modes(payload),
            "printed_block_coverage": float(payload["internal_strain_tensors"].shape[0] / (3 * len(record["atoms"]["elements"]))),
            "invariant_dimensions": int(audit["invariant_dimensions"]),
            "observed_rank": int(audit["observed_rank"]),
            "fit_relative_residual": float(audit["fit_relative_residual"]),
            "ionic_closure_relative": float(audit["ionic_closure_relative_error"]),
            "elements": metadata["elements"],
            "cohort_source": "completion_likelihood_v1",
        })
    available = [profile for profile in profiles if profile["available"]]
    accepted = [profile for profile in available if profile["accepted"]]
    summary = {
        "requested": len(profiles), "available": len(available), "unavailable": len(profiles) - len(available),
        "strict_completion": _rate(available),
        "acceptance_by_space_group": _grouped_rates(available, "space_group_number"),
        "acceptance_by_crystal_system": _grouped_rates(available, "crystal_system"),
        "acceptance_by_atom_count_bin": _grouped_rates(available, "atom_count_bin"),
        "acceptance_by_response_bin": _grouped_rates(available, "response_bin"),
        "acceptance_by_coverage": _grouped_rates(available, "printed_block_coverage"),
        "failure_reason_counts": dict(Counter(reason for row in available if not row["accepted"] for reason in row["failure_reasons"].split(";"))),
        "accepted_element_count": len({element for row in accepted for element in row["elements"]}),
        "accepted_response_bins": dict(Counter(str(row["response_bin"]) for row in accepted)),
        "accepted_negative_optical_modes": sum(bool(row["negative_optical_mode"]) for row in accepted),
        "accepted_polar": sum(bool(row["polar"]) for row in accepted),
    }
    report = {
        "schema": 1,
        "scope": "fixed descriptive report; no labels, thresholds, or splits were changed",
        "cohort": str(args.cohort), "completion_manifest": str(args.completion_manifest),
        "summary": summary, "completion_propensity_profiles": profiles,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "expansion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = list(profiles[0])
    with (args.output_dir / "completion_propensity_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(profiles)
    rate = summary["strict_completion"]
    lines = [
        "# Completion-likelihood expansion report", "",
        "This is a fixed audit of the downloaded cohort. Strict thresholds were not relaxed.", "",
        "## Core statistics", "",
        f"- Requested/download queue: {summary['requested']}",
        f"- Parsed and audited: {summary['available']}; unavailable after download/OUTCAR parsing: {summary['unavailable']}",
        f"- Strictly accepted: {rate['accepted']}/{rate['materials']} ({rate['rate']:.1%})",
        f"- Accepted element coverage: {summary['accepted_element_count']}",
        f"- Accepted response bins: {summary['accepted_response_bins']}",
        f"- Accepted polar / negative-optical-mode materials: {summary['accepted_polar']} / {summary['accepted_negative_optical_modes']}",
        f"- Non-exclusive failure reasons: {summary['failure_reason_counts']}", "",
        "The CSV contains one completion-propensity profile per requested material, including the strict gate result, structural class, printed coverage, invariant dimension, response stratum, polar class, soft-mode flag, and cohort source.",
    ]
    (args.output_dir / "expansion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "expansion_report.md")


if __name__ == "__main__":
    main()
