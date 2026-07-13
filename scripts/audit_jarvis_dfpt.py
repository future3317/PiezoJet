"""Audit cached same-ID JARVIS DFPT tensors before model training.

The output is deliberately machine-readable: ``summary.json`` is the quality
gate, while ``materials.csv`` preserves per-material evidence for reports and
regression checks.  No API credentials are read by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from piezojet.data import load_gmtnet_records
from piezojet.jarvis_dfpt import JarvisDFPTCache


EPS = 1e-12
NEGATIVE_MODE_THRESHOLD = -1e-3  # eV / Angstrom^2
SOFT_MODE_THRESHOLD = 1e-3


def _matrix(blocks: torch.Tensor) -> torch.Tensor:
    atoms = blocks.shape[0]
    return blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms).to(torch.float64)


def _optical_basis(atoms: int, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    translation = reference.new_zeros(3 * atoms, 3)
    for axis in range(3):
        translation[axis::3, axis] = atoms ** -0.5
    projector = torch.eye(3 * atoms, dtype=reference.dtype) - translation @ translation.T
    values, vectors = torch.linalg.eigh(projector)
    return vectors[:, values > 0.5], projector


def _relative_norm(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value) / torch.linalg.vector_norm(reference).clamp_min(EPS))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q))


def audit_record(record: dict, payload: dict) -> dict[str, float | int | str | bool]:
    jid = str(record["JARVIS_ID"])
    born = payload["born_charges"].to(torch.float64)
    force_blocks = payload["force_constants"].to(torch.float64)
    dynamical_blocks = payload["dynamical_matrix"].to(torch.float64)
    masses = payload["masses"].to(torch.float64)
    internal = payload["internal_strain_tensors"].to(torch.float64)
    atoms = born.shape[0]
    modes = 3 * atoms
    force = _matrix(force_blocks)
    dynamical = _matrix(dynamical_blocks)
    force_symmetric = 0.5 * (force + force.T)
    optical_basis, projector = _optical_basis(atoms, force)
    cleaned_force = projector @ force_symmetric @ projector
    optical_eigenvalues = torch.linalg.eigvalsh(optical_basis.T @ cleaned_force @ optical_basis)
    dynamical_eigenvalues = torch.linalg.eigvalsh(0.5 * (dynamical + dynamical.T))
    cached_eigenvalues = payload["dynamical_eigenvalues"].to(torch.float64).sort().values
    mass_scale = torch.sqrt(masses[:, None] * masses[None, :])[:, :, None, None]
    reconstructed_force = -payload["dynamical_matrix"].to(torch.float64) * mass_scale
    source_total = payload["total_piezo_source"].to(torch.float64)
    record_total = torch.as_tensor(record["piezoelectric_C_m2"], dtype=torch.float64)
    finite_tensors = (
        born, force_blocks, dynamical_blocks, masses, internal,
        payload["ionic_piezo_source"], payload["total_piezo_source"],
    )
    internal_symmetric = 0.5 * (internal + internal.transpose(-1, -2))
    projected_born = born - born.mean(dim=0, keepdim=True)
    return {
        "jid": jid,
        "atoms": atoms,
        "gamma_modes": modes,
        "printed_internal_blocks": int(internal.shape[0]),
        "printed_internal_coverage": float(internal.shape[0] / modes),
        "all_finite": all(bool(torch.isfinite(tensor).all()) for tensor in finite_tensors),
        "force_block_symmetry_rel": _relative_norm(force - force.T, force),
        "force_asr_rel": _relative_norm(force.reshape(atoms, 3, atoms, 3).sum(dim=2), force),
        "born_asr_max_abs_e": float(born.sum(dim=0).abs().max()),
        "born_asr_rel": _relative_norm(born.sum(dim=0), born),
        "born_projection_max_abs_e": float((born - projected_born).abs().max()),
        "born_projection_rel": _relative_norm(born - projected_born, born),
        "internal_antisym_rel": _relative_norm(internal - internal.transpose(-1, -2), internal),
        "internal_antisym_max_abs_eV_A": float((internal - internal.transpose(-1, -2)).abs().max()),
        "internal_symmetric_norm": float(torch.linalg.vector_norm(internal_symmetric)),
        "negative_optical_modes": int((optical_eigenvalues < NEGATIVE_MODE_THRESHOLD).sum()),
        "soft_optical_modes": int((optical_eigenvalues.abs() <= SOFT_MODE_THRESHOLD).sum()),
        "min_optical_eigenvalue": float(optical_eigenvalues.min()),
        "max_optical_eigenvalue": float(optical_eigenvalues.max()),
        "dynamical_eigenvalue_max_abs_error": float((dynamical_eigenvalues - cached_eigenvalues).abs().max()),
        "mass_conversion_max_abs_error": float((force_blocks - reconstructed_force).abs().max()),
        "total_piezo_record_max_abs_error": float((source_total - record_total).abs().max()),
        "ionic_piezo_frobenius": float(torch.linalg.vector_norm(payload["ionic_piezo_source"].to(torch.float64))),
        "total_piezo_frobenius": float(torch.linalg.vector_norm(source_total)),
    }


def summarize(rows: list[dict], requested: int) -> dict:
    numeric = lambda name: [float(row[name]) for row in rows]
    critical_failures = [
        str(row["jid"]) for row in rows
        if not row["all_finite"]
        or float(row["force_block_symmetry_rel"]) > 1e-10
        or float(row["dynamical_eigenvalue_max_abs_error"]) > 1e-5
        or float(row["mass_conversion_max_abs_error"]) > 1e-5
        or float(row["total_piezo_record_max_abs_error"]) > 1e-6
    ]
    return {
        "quality_gate": "passed" if not critical_failures and len(rows) == requested else "failed",
        "requested_materials": requested,
        "cached_materials": len(rows),
        "cache_success_rate": len(rows) / requested if requested else 0.0,
        "critical_failure_jids": critical_failures,
        "thresholds": {
            "negative_optical_mode_eV_A2": NEGATIVE_MODE_THRESHOLD,
            "soft_optical_mode_abs_eV_A2": SOFT_MODE_THRESHOLD,
            "total_piezo_match_max_abs_C_m2": 1e-6,
            "dynamical_eigenvalue_max_abs": 1e-5,
            "mass_conversion_max_abs": 1e-5,
        },
        "atom_count": {
            "min": min(int(row["atoms"]) for row in rows),
            "median": _quantile(numeric("atoms"), 0.5),
            "max": max(int(row["atoms"]) for row in rows),
        },
        "negative_mode_materials": sum(int(row["negative_optical_modes"]) > 0 for row in rows),
        "negative_mode_material_rate": sum(int(row["negative_optical_modes"]) > 0 for row in rows) / len(rows),
        "negative_optical_modes": sum(int(row["negative_optical_modes"]) for row in rows),
        "soft_mode_materials": sum(int(row["soft_optical_modes"]) > 0 for row in rows),
        "force_asr_rel": {
            "median": _quantile(numeric("force_asr_rel"), 0.5),
            "max": max(numeric("force_asr_rel")),
        },
        "born_asr_max_abs_e": {
            "median": _quantile(numeric("born_asr_max_abs_e"), 0.5),
            "max": max(numeric("born_asr_max_abs_e")),
        },
        "born_asr_rel": {
            "median": _quantile(numeric("born_asr_rel"), 0.5),
            "max": max(numeric("born_asr_rel")),
        },
        "born_projection_max_abs_e": {
            "median": _quantile(numeric("born_projection_max_abs_e"), 0.5),
            "p95": _quantile(numeric("born_projection_max_abs_e"), 0.95),
            "max": max(numeric("born_projection_max_abs_e")),
        },
        "born_projection_rel": {
            "median": _quantile(numeric("born_projection_rel"), 0.5),
            "p95": _quantile(numeric("born_projection_rel"), 0.95),
            "max": max(numeric("born_projection_rel")),
        },
        "internal_antisym_rel": {
            "median": _quantile(numeric("internal_antisym_rel"), 0.5),
            "max": max(numeric("internal_antisym_rel")),
        },
        "printed_internal_coverage": {
            "median": _quantile(numeric("printed_internal_coverage"), 0.5),
            "min": min(numeric("printed_internal_coverage")),
            "max": max(numeric("printed_internal_coverage")),
        },
        "max_consistency_errors": {
            "force_block_symmetry_rel": max(numeric("force_block_symmetry_rel")),
            "dynamical_eigenvalue_abs": max(numeric("dynamical_eigenvalue_max_abs_error")),
            "mass_conversion_abs": max(numeric("mass_conversion_max_abs_error")),
            "total_piezo_record_abs_C_m2": max(numeric("total_piezo_record_max_abs_error")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--dfpt-dir", type=Path, default=Path("data/processed/jarvis_dfpt_v1"))
    parser.add_argument("--limit", type=int, required=True, help="Audit the first N source records, matching cache pilot selection.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dfpt_pilot_audit"))
    args = parser.parse_args()
    records = load_gmtnet_records(args.data_root)[: args.limit]
    cache = JarvisDFPTCache(args.dfpt_dir)
    rows = []
    for record in records:
        payload = cache.load(str(record["JARVIS_ID"]))
        if payload is not None:
            rows.append(audit_record(record, payload))
    if not rows:
        raise RuntimeError("No cached DFPT records were found for the requested pilot")
    summary = summarize(rows, len(records))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "material_ids.json").write_text(
        json.dumps([row["jid"] for row in rows], indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "materials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    if summary["quality_gate"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
