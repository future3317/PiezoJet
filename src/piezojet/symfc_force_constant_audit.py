"""Read-only symfc symmetry/ASR audit for cached Gamma force constants.

The JARVIS archive stores a Gamma-point force-constant block, not a complete
real-space supercell interaction model.  This utility therefore never feeds a
symfc projection back into cache, completion, training, or evaluation.  It
uses synthetic infinitesimal displacements generated from the cached Phi to
ask how far that finite block is from the force-constant subspace imposed by
the same primitive-cell space group and acoustic sum rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Element
from symfc import Symfc
from symfc.utils.utils import SymfcAtoms

from .data import load_gmtnet_records
from .jarvis_dfpt import JarvisDFPTCache


def _relative(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-30))


def symfc_project_force_constants(record: dict[str, Any], force_constants: torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
    """Project a finite Gamma block through symfc without mutating its source."""
    phi = force_constants.detach().cpu().to(torch.float64).numpy()
    atoms = phi.shape[0]
    if phi.shape != (atoms, atoms, 3, 3):
        raise ValueError("Force constants must have shape [atoms, atoms, 3, 3]")
    atom_data = record["atoms"]
    supercell = SymfcAtoms(
        numbers=[Element(symbol).Z for symbol in atom_data["elements"]],
        scaled_positions=np.asarray(atom_data["coords"], dtype=float),
        cell=np.asarray(atom_data["lattice_mat"], dtype=float),
    )
    amplitude = 1e-2
    displacements = np.eye(3 * atoms, dtype=float).reshape(3 * atoms, atoms, 3) * amplitude
    forces = -np.einsum("ijab,sjb->sia", phi, displacements)
    symfc = Symfc(supercell, displacements=displacements, forces=forces)
    symfc.run(orders=[2], is_compact_fc=False)
    projected = np.asarray(symfc.force_constants[2], dtype=float)
    audit = {
        "symfc_projection_relative_error": _relative(projected, phi),
        "source_acoustic_relative_residual": float(
            np.linalg.norm(phi.sum(axis=1)) / max(np.linalg.norm(phi), 1e-30)
        ),
        "symfc_acoustic_relative_residual": float(
            np.linalg.norm(projected.sum(axis=1)) / max(np.linalg.norm(projected), 1e-30)
        ),
        "symfc_force_constant_symmetry_relative_residual": float(
            np.linalg.norm(projected - projected.transpose(1, 0, 3, 2))
            / max(np.linalg.norm(projected), 1e-30)
        ),
    }
    return projected, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dfpt-dir", type=Path, required=True)
    parser.add_argument("--strict-completion-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/symfc_force_constant_audit_v1/report.json"))
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    records = {str(record["JARVIS_ID"]): record for record in load_gmtnet_records(args.data_root)}
    material_ids = []
    for path in sorted(args.strict_completion_dir.glob("JVASP-*.pt")):
        completion = torch.load(path, map_location="cpu", weights_only=False)
        if bool(completion.get("audit", {}).get("accepted", False)):
            material_ids.append(str(completion["jid"]))
    cache = JarvisDFPTCache(args.dfpt_dir)
    rows = []
    for jid in material_ids[: args.limit]:
        payload = cache.load(jid)
        if payload is None:
            rows.append({"jid": jid, "error": "cache payload unavailable"})
            continue
        try:
            _, row = symfc_project_force_constants(records[jid], payload["force_constants"])
            rows.append({"jid": jid, **row})
        except Exception as error:
            rows.append({"jid": jid, "error": str(error)})
    successful = [row for row in rows if "error" not in row]
    result: dict[str, Any] = {
        "schema": 1,
        "scope": "read-only symfc projection of finite JARVIS Gamma force-constant blocks",
        "limitation": (
            "The projected arrays are an external diagnostic only. A primitive-cell Gamma block is not a "
            "complete real-space force-constant model, so this audit must not replace cache Phi or alter labels."
        ),
        "requested": len(rows),
        "successful": len(successful),
        "rows": rows,
        "median": {
            key: float(np.median([row[key] for row in successful])) if successful else None
            for key in (
                "symfc_projection_relative_error",
                "source_acoustic_relative_residual",
                "symfc_acoustic_relative_residual",
                "symfc_force_constant_symmetry_relative_residual",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["successful"] != result["requested"]:
        raise RuntimeError("symfc audit did not complete for every selected material")


if __name__ == "__main__":
    main()
