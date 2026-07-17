"""Read-only P0 audit for JARVIS/VASP DFPT tensor conventions.

This module records the cache's mass-unweighting relation, its acoustic
residuals, and the ionic-closure consequences of four Born/Lambda variants.
It makes a parser-boundary decision auditable: the empirically supported BEC
transpose is fixed in schema 3, while a sign flip of OUTCAR's printed force
derivative is rejected by the same controlled response check.

Strict-completion closure is an internal consistency check, not an independent
VASP-convention proof: the same convention participates in strict acceptance.
The emitted JSON labels that limitation explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .data import load_gmtnet_records
from .evaluate_dfpt import ionic_piezo_from_factors
from .jarvis_dfpt import (
    JarvisDFPTCache,
    source_born_to_internal,
)
from .model import AtomCoordinateResponsePotential
from .projectors import translation_projector
from .tensor_ops import piezo_voigt_to_cartesian, source_voigt_to_canonical


def _matrix(blocks: torch.Tensor) -> torch.Tensor:
    atoms = blocks.shape[0]
    return blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)


def force_constant_convention_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Measure the relation implemented by jarvis-tools' ``phonon_data``.

    JARVIS currently exposes the raw VASP dynamical block in
    ``dynamical_matrix`` and the default mass-unweighted, sign-restored
    force-constant block in ``force_constants``.  This routine proves that the
    cached tensors obey that parser relation and reports uniform-translation
    residuals for both arrays.  It does not substitute for a phonopy or finite
    displacement comparison.
    """
    force = payload["force_constants"].to(torch.float64)
    raw = payload["dynamical_matrix"].to(torch.float64)
    masses = payload["masses"].to(torch.float64)
    scale = -torch.sqrt(masses[:, None, None, None] * masses[None, :, None, None])
    expected = raw * scale
    force_matrix, raw_matrix = _matrix(force), _matrix(raw)
    _, translation = translation_projector(force.shape[0], force)
    return {
        "mass_unweighting_relative_error": float(
            torch.linalg.vector_norm(force - expected)
            / torch.linalg.vector_norm(expected).clamp_min(1e-30)
        ),
        "force_uniform_translation_relative_residual": float(
            torch.linalg.vector_norm(force_matrix @ translation)
            / torch.linalg.vector_norm(force_matrix).clamp_min(1e-30)
        ),
        "raw_uniform_translation_relative_residual": float(
            torch.linalg.vector_norm(raw_matrix @ translation)
            / torch.linalg.vector_norm(raw_matrix).clamp_min(1e-30)
        ),
    }


def ionic_closure_variants(
    record: dict[str, Any],
    payload: dict[str, Any],
    internal_strain: torch.Tensor,
    response: AtomCoordinateResponsePotential,
) -> dict[str, float]:
    """Compare source-consistent axis/sign alternatives on one full Lambda."""
    volume = torch.linalg.det(
        torch.as_tensor(record["atoms"]["lattice_mat"], dtype=torch.float32)
    ).abs()
    target = piezo_voigt_to_cartesian(source_voigt_to_canonical(payload["ionic_piezo_source"]))
    # Schema 3 records both tensors in their source convention.  Keeping all
    # four combinations in the report makes the parser-boundary decision
    # falsifiable without ever patching a trained model in place.  Schema 2 is
    # accepted only for a read-only historical comparison.
    source_born = payload.get("born_charges_source", payload["born_charges"])
    # The completed tensor is the OUTCAR printed force derivative, which has
    # the same sign as internal Lambda.  The cache's raw source blocks are
    # intentionally partial and cannot be used as a 3N-by-6 right-hand side.
    source_lambda_full = internal_strain
    variants = {
        "source_born_source_lambda": (source_born, source_lambda_full),
        "internal_born_source_lambda": (source_born_to_internal(source_born), source_lambda_full),
        "source_born_negated_lambda": (
            source_born,
            -source_lambda_full,
        ),
        "internal_born_negated_lambda": (
            source_born_to_internal(source_born),
            -source_lambda_full,
        ),
    }
    values: dict[str, float] = {}
    for name, (born, coupling) in variants.items():
        prediction = ionic_piezo_from_factors(
            response, born, payload["force_constants"], coupling, volume, "regularized"
        )
        values[name] = float(
            torch.linalg.vector_norm(prediction - target)
            / torch.linalg.vector_norm(target).clamp_min(0.05)
        )
    return values


def _median(rows: list[float]) -> float | None:
    return None if not rows else float(torch.tensor(rows, dtype=torch.float64).median())


def audit(
    records: list[dict[str, Any]],
    cache: JarvisDFPTCache,
    completion_dir: Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the P0 cache audit without modifying caches, splits, or labels."""
    by_id = {str(record["JARVIS_ID"]): record for record in records}
    response = AtomCoordinateResponsePotential(
        optical_solve_policy="regularized", optical_regularization=1e-3
    )
    force_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    for path in sorted(completion_dir.glob("JVASP-*.pt")):
        if limit is not None and len(force_rows) >= limit:
            break
        completion = torch.load(path, map_location="cpu", weights_only=False)
        jid = str(completion.get("jid", ""))
        if jid not in by_id or not bool(completion.get("audit", {}).get("accepted", False)):
            continue
        payload = cache.load(jid)
        if payload is None:
            continue
        force_rows.append({"jid": jid, **force_constant_convention_metrics(payload)})
        closure_rows.append(
            {
                "jid": jid,
                **ionic_closure_variants(
                    by_id[jid], payload, completion["internal_strain_full"], response
                ),
            }
        )
    if not force_rows:
        raise ValueError("No accepted completions with matching DFPT cache entries were available")
    force_keys = [key for key in force_rows[0] if key != "jid"]
    closure_keys = [key for key in closure_rows[0] if key != "jid"]
    return {
        "schema": 2,
        "scope": "read-only JARVIS cache convention audit",
        "independence_limit": (
            "Closure rows validate the explicit source-to-internal implementation but are not an "
            "independent finite-field proof of VASP axes/signs. Use phonopy/finite-displacement "
            "and a VASP/py4vasp parser comparison as an external cross-check."
        ),
        "force_constant_parser_relation": (
            "expected force_constants = -dynamical_matrix * sqrt(m_i*m_j), "
            "matching jarvis-tools phonon_data(fc_mass=True) relative to fc_mass=False."
        ),
        "strict_materials_audited": len(force_rows),
        "force_constant_medians": {
            key: _median([float(row[key]) for row in force_rows]) for key in force_keys
        },
        "ionic_closure_variant_medians": {
            key: _median([float(row[key]) for row in closure_rows]) for key in closure_keys
        },
        "force_constant_rows": force_rows,
        "ionic_closure_rows": closure_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dfpt-dir", type=Path, required=True)
    parser.add_argument("--strict-completion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/dfpt_convention_audit_v2/convention_audit.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    result = audit(
        load_gmtnet_records(args.data_root),
        JarvisDFPTCache(args.dfpt_dir),
        args.strict_completion_dir,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
