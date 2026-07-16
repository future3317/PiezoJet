"""Forensic audit of accepted and rejected strict Lambda completions.

This is diagnostic-only: no tolerance in this module changes the formal
completion gate.  It identifies whether rejected printed strain-force blocks
show a channel, space-group-operation, or structural-pattern signature before
larger JARVIS retrieval cohorts are requested.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .data import _raw_cartesian_target, load_gmtnet_records, response_norm_bin
from .jarvis_dfpt import JarvisDFPTCache
from .strain_completion import (
    VOIGT_PAIRS,
    _observed_system,
    _structure,
    cartesian_space_group_operations,
    complete_internal_strain,
    internal_tensor_to_vector,
    invariant_basis,
    transform_internal_tensor,
    vector_to_internal_tensor,
)


VOIGT_NAMES = ("xx", "yy", "zz", "yz", "xz", "xy")


def _decode_index(index: int) -> dict[str, int | str]:
    return {
        "atom": index // 18,
        "displacement": (index % 18) // 6,
        "strain_channel": VOIGT_NAMES[index % 6],
    }


def _operation_matrix(atoms: int, operation) -> torch.Tensor:
    """Full packed-vector representation of one Cartesian space-group action."""
    unit = torch.eye(18 * atoms, dtype=torch.float64)
    transformed = transform_internal_tensor(vector_to_internal_tensor(unit, atoms), operation)
    return internal_tensor_to_vector(transformed).transpose(0, 1)


def _failure_reasons(audit: dict) -> list[str]:
    if "error" in audit:
        return ["audit_error"]
    reasons = []
    if not bool(audit.get("source_internal_strain_block_parse_complete", True)):
        reasons.append("incomplete_source_block_parse")
    if not bool(audit.get("uniquely_determined", False)):
        reasons.append("not_uniquely_identifiable")
    if float(audit.get("maximum_mapping_error_angstrom", float("inf"))) > 2e-5:
        reasons.append("atom_mapping")
    if float(audit.get("fit_relative_residual", float("inf"))) > 5e-3:
        reasons.append("invariant_fit")
    redundant = audit.get("leave_one_block_out_relative_max")
    if redundant is not None and float(redundant) > 5e-2:
        reasons.append("redundant_block")
    if float(audit.get("ionic_closure_relative_error", float("inf"))) > 5e-2:
        reasons.append("ionic_closure")
    stable_exact = audit.get("stable_exact_ionic_closure_relative_error")
    if stable_exact is not None and float(stable_exact) > 5e-2:
        reasons.append("stable_exact_ionic_closure")
    return reasons or ["unknown_gate"]


def _fit_and_pairwise(record: dict, payload: dict, symprec: float) -> dict:
    atoms = len(record["atoms"]["elements"])
    operations = cartesian_space_group_operations(record, symprec=symprec)
    basis = invariant_basis(atoms, operations)
    indices, values, _ = _observed_system(
        payload["internal_strain_tensors"], payload["internal_strain_ions"],
        payload["internal_strain_directions"], atoms,
    )
    observed_basis = basis[indices]
    rank = int(torch.linalg.matrix_rank(observed_basis, tol=1e-7))
    coefficients = torch.linalg.lstsq(observed_basis, values).solution
    fitted = observed_basis @ coefficients
    residual = fitted - values
    channel = indices.remainder(6)
    channel_relative = {}
    for index, name in enumerate(VOIGT_NAMES):
        mask = channel == index
        channel_relative[name] = (
            float(torch.linalg.vector_norm(residual[mask]) / torch.linalg.vector_norm(values[mask]).clamp_min(1e-12))
            if bool(mask.any()) else float("nan")
        )
    pairwise_rows, full_values = [], values.new_zeros(18 * atoms)
    full_values[indices] = values
    observed = torch.zeros(18 * atoms, dtype=torch.bool)
    observed[indices] = True
    for operation_index, operation in enumerate(operations):
        matrix = _operation_matrix(atoms, operation)
        if torch.allclose(matrix, torch.eye(matrix.shape[0], dtype=matrix.dtype), atol=1e-10, rtol=0.0):
            continue
        scalar_residuals, scalar_targets, scalar_channels, scalar_outputs, scalar_sources = [], [], [], [], []
        for output in indices.tolist():
            source = matrix[output].abs() > 1e-10
            if bool(source.any()) and bool(observed[source].all()):
                scalar_residuals.append(torch.dot(matrix[output], full_values) - full_values[output])
                scalar_targets.append(full_values[output])
                scalar_channels.append(output % 6)
                scalar_outputs.append(output)
                scalar_sources.append(torch.nonzero(source, as_tuple=False).reshape(-1).tolist())
        if scalar_residuals:
            residual_tensor = torch.stack(scalar_residuals)
            target_tensor = torch.stack(scalar_targets)
            maximum = int(residual_tensor.abs().argmax())
            pairwise_rows.append({
                "operation_index": operation_index,
                "comparable_scalars": len(scalar_residuals),
                "relative_residual": float(torch.linalg.vector_norm(residual_tensor) / torch.linalg.vector_norm(target_tensor).clamp_min(1e-12)),
                "maximum_mapping_error_angstrom": float(operation.mapping_error_angstrom),
                "max_channel": VOIGT_NAMES[scalar_channels[maximum]],
                "max_output": _decode_index(scalar_outputs[maximum]),
                "max_source_components": [_decode_index(index) for index in scalar_sources[maximum]],
                "max_observed_value": float(target_tensor[maximum]),
                "max_transformed_value": float(target_tensor[maximum] + residual_tensor[maximum]),
                "max_residual": float(residual_tensor[maximum]),
            })
    pairwise_rows.sort(key=lambda item: item["relative_residual"], reverse=True)
    maximum_pairwise = pairwise_rows[0] if pairwise_rows else None
    return {
        "space_group_operations": len(operations),
        "invariant_dimensions": int(basis.shape[1]),
        "observed_rank": rank,
        "fit_relative_residual": float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(values).clamp_min(1e-12)),
        "fit_relative_by_channel": channel_relative,
        "pairwise_comparable_operations": len(pairwise_rows),
        "pairwise_maximum": maximum_pairwise,
        "pairwise_operations": pairwise_rows,
    }


def _symprec_sweep(record: dict, payload: dict, values: list[float]) -> dict[str, dict]:
    output = {}
    for symprec in values:
        try:
            result = _fit_and_pairwise(record, payload, symprec)
            output[f"{symprec:.0e}"] = {
                key: result[key] for key in (
                    "space_group_operations", "invariant_dimensions", "observed_rank", "fit_relative_residual",
                )
            }
        except Exception as error:
            output[f"{symprec:.0e}"] = {"error": str(error)}
    return output


def _space_group(record: dict) -> tuple[int | None, str | None]:
    try:
        analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
        return int(analyzer.get_space_group_number()), str(analyzer.get_space_group_symbol())
    except Exception:
        return None, None


def _negative_optical_modes(payload: dict) -> bool:
    blocks = payload["force_constants"]
    atoms = blocks.shape[0]
    matrix = blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms).to(torch.float64)
    values = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    optical = values[torch.argsort(values.abs())[3:]] if values.numel() > 3 else values.new_empty(0)
    return bool((optical < 0).any())


def _figures(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    accepted = np.array([bool(row["accepted"]) for row in rows])
    residual = np.array([max(float(row["fit_relative_residual"]), 1e-12) for row in rows])
    coverage = np.array([float(row["printed_block_coverage"]) for row in rows])
    group_order = np.array([float(row["space_group_operations"]) for row in rows])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for mask, label, color in ((~accepted, "rejected", "#b45309"), (accepted, "accepted", "#1d4ed8")):
        axes[0, 0].scatter(coverage[mask], residual[mask], label=label, color=color, alpha=0.8)
        axes[0, 1].scatter(group_order[mask], residual[mask], label=label, color=color, alpha=0.8)
    for axis, xlabel in ((axes[0, 0], "Printed block coverage"), (axes[0, 1], "Space-group operation count")):
        axis.set_yscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Invariant-fit relative residual")
        axis.legend(frameon=False)
    atom_bins = sorted({str(row["atom_count_bin"]) for row in rows})
    response_bins = sorted({str(row["response_bin"]) for row in rows})
    for axis, keys, column, title in (
        (axes[0, 2], atom_bins, "atom_count_bin", "Completion rate by atom-count bin"),
        (axes[1, 0], response_bins, "response_bin", "Completion rate by response bin"),
    ):
        rates = [np.mean([bool(row["accepted"]) for row in rows if str(row[column]) == key]) for key in keys]
        counts = [sum(str(row[column]) == key for row in rows) for key in keys]
        axis.bar(keys, rates, color="#1d4ed8")
        for index, (rate, count) in enumerate(zip(rates, counts)):
            axis.text(index, rate + 0.02, f"n={count}", ha="center", fontsize=8)
        axis.set_ylim(0, 1.05)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("Strict-completion rate")
    rejected = [row for row in rows if not row["accepted"] and "invariant_fit" in row["failure_reasons"]]
    worst = Counter(row["max_fit_channel"] for row in rejected)
    axes[1, 1].bar(VOIGT_NAMES, [worst[name] for name in VOIGT_NAMES], color="#b45309")
    axes[1, 1].set_title("Worst fit-residual channel among fit failures", fontsize=10)
    axes[1, 1].set_ylabel("Material count")
    normal = [row[f"fit_channel_{name}"] for row in rows for name in VOIGT_NAMES[:3] if np.isfinite(row[f"fit_channel_{name}"])]
    shear = [row[f"fit_channel_{name}"] for row in rows for name in VOIGT_NAMES[3:] if np.isfinite(row[f"fit_channel_{name}"])]
    axes[1, 2].boxplot([normal, shear], tick_labels=["normal", "shear"], showfliers=True)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_ylabel("Per-channel invariant-fit residual")
    axes[1, 2].set_title("Normal vs shear residual distribution", fontsize=10)
    fig.suptitle("Strict Lambda completion forensic audit (n=50)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "completion_forensics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dfpt-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--completion-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--symprec-grid", default="1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4")
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    ids = [str(value) for value in cohort["material_ids"]]
    manifest = json.loads(args.completion_manifest.read_text(encoding="utf-8"))
    audit_by_id = {str(row["jid"]): row for row in manifest["rows"]}
    records = {str(record["JARVIS_ID"]): record for record in load_gmtnet_records(args.data_root)}
    cache = JarvisDFPTCache(args.dfpt_dir)
    sweep_values = [float(value) for value in args.symprec_grid.split(",")]
    rows, detailed, unavailable = [], [], []
    for position, jid in enumerate(ids, start=1):
        record, payload, audit = records[jid], cache.load(jid), audit_by_id.get(jid)
        if payload is None or audit is None:
            unavailable.append({
                "jid": jid,
                "reason": "DFPT payload unavailable (download or OUTCAR parse failure)",
            })
            print(f"[{position}/{len(ids)}] {jid}: unavailable")
            continue
        analysis = _fit_and_pairwise(record, payload, symprec=1e-5)
        number, symbol = _space_group(record)
        channels = analysis["fit_relative_by_channel"]
        finite_channels = {name: value for name, value in channels.items() if np.isfinite(value)}
        maximum = max(finite_channels, key=finite_channels.get)
        row = {
            "jid": jid, "accepted": bool(audit.get("accepted", False)),
            "failure_reasons": ";".join(_failure_reasons(audit)),
            "space_group_number": number, "space_group_symbol": symbol,
            "space_group_operations": analysis["space_group_operations"],
            "atoms": len(record["atoms"]["elements"]),
            "atom_count_bin": 0 if len(record["atoms"]["elements"]) <= 2 else 1 if len(record["atoms"]["elements"]) <= 4 else 2 if len(record["atoms"]["elements"]) <= 8 else 3,
            "response_norm": float(torch.linalg.vector_norm(_raw_cartesian_target(record))),
            "response_bin": response_norm_bin(_raw_cartesian_target(record)),
            "negative_optical_mode": _negative_optical_modes(payload),
            "printed_block_coverage": payload["internal_strain_tensors"].shape[0] / (3 * len(record["atoms"]["elements"])),
            "invariant_dimensions": analysis["invariant_dimensions"],
            "observed_rank": analysis["observed_rank"],
            "fit_relative_residual": analysis["fit_relative_residual"],
            "max_fit_channel": maximum,
            "pairwise_comparable_operations": analysis["pairwise_comparable_operations"],
            "pairwise_maximum_operation": None if analysis["pairwise_maximum"] is None else analysis["pairwise_maximum"]["operation_index"],
            "pairwise_maximum_relative_residual": None if analysis["pairwise_maximum"] is None else analysis["pairwise_maximum"]["relative_residual"],
            "pairwise_maximum_channel": None if analysis["pairwise_maximum"] is None else analysis["pairwise_maximum"]["max_channel"],
            "maximum_mapping_error_angstrom": audit.get("maximum_mapping_error_angstrom"),
            "redundant_validation_relative": audit.get("leave_one_block_out_relative_max"),
            "ionic_closure_relative": audit.get("ionic_closure_relative_error"),
            "constrained_asr_residual": 0.0,
        }
        row.update({f"fit_channel_{name}": value for name, value in channels.items()})
        rows.append(row)
        detailed.append({"row": row, "pairwise_operations": analysis["pairwise_operations"], "symprec_sweep": _symprec_sweep(record, payload, sweep_values)})
        print(f"[{position}/{len(ids)}] {jid}: accepted={row['accepted']} fit={row['fit_relative_residual']:.3g}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": 1,
        "scope": "diagnostic only; formal strict-completion thresholds were not changed",
        "cohort": str(args.cohort), "completion_manifest": str(args.completion_manifest),
        "symprec_grid": sweep_values, "rows": rows, "details": detailed,
        "unavailable": unavailable,
        "summary": {
            "requested_materials": len(ids), "materials": len(rows), "accepted": sum(row["accepted"] for row in rows),
            "unavailable_materials": len(unavailable),
            "failure_reason_counts": dict(Counter(reason for row in rows if not row["accepted"] for reason in row["failure_reasons"].split(";"))),
            "worst_fit_channel_counts": dict(Counter(row["max_fit_channel"] for row in rows if "invariant_fit" in row["failure_reasons"])),
        },
    }
    fit_failures = [row for row in rows if "invariant_fit" in row["failure_reasons"]]
    normal = [row[f"fit_channel_{name}"] for row in fit_failures for name in VOIGT_NAMES[:3] if np.isfinite(row[f"fit_channel_{name}"])]
    shear = [row[f"fit_channel_{name}"] for row in fit_failures for name in VOIGT_NAMES[3:] if np.isfinite(row[f"fit_channel_{name}"])]
    p187 = [row for row in rows if row["space_group_number"] == 187]
    report["summary"].update({
        "fit_failure_count": len(fit_failures),
        "normal_channel_fit_residual_median": float(np.median(normal)) if normal else None,
        "shear_channel_fit_residual_median": float(np.median(shear)) if shear else None,
        "space_group_187_materials": len(p187),
        "space_group_187_accepted": sum(row["accepted"] for row in p187),
        "space_group_187_pairwise_sign_conflicts": sum((row["pairwise_maximum_relative_residual"] or 0.0) > 1.5 for row in p187),
    })
    (args.output_dir / "forensics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0])
    with (args.output_dir / "forensics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _figures(rows, args.output_dir / "figures")
    lines = [
        "# Strict Lambda completion forensics", "", "Formal completion thresholds were not changed.", "", "## Summary", "",
        f"- Requested: {len(ids)}", f"- Audited: {len(rows)}", f"- Unavailable (download/parse): {len(unavailable)}", f"- Accepted: {sum(row['accepted'] for row in rows)}",
        f"- Failure reasons (not mutually exclusive): {report['summary']['failure_reason_counts']}",
        f"- Worst fit channels: {report['summary']['worst_fit_channel_counts']}",
        f"- Fit-failure median residual, normal/shear: {report['summary']['normal_channel_fit_residual_median']:.6g} / {report['summary']['shear_channel_fit_residual_median']:.6g}",
        f"- Space group 187: {report['summary']['space_group_187_materials']} materials, {report['summary']['space_group_187_accepted']} accepted, {report['summary']['space_group_187_pairwise_sign_conflicts']} direct pairwise sign conflicts.",
        "", "The CSV gives one material per row; JSON retains pairwise-operation and symmetry-tolerance details.",
        "The group-187 pattern is an unresolved source/reference or group-specific convention discrepancy, not a basis for relaxing the gate.",
    ]
    (args.output_dir / "forensics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "forensics.md")


if __name__ == "__main__":
    main()
