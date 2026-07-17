"""Audit the usable JARVIS electrostatic pool without reading frozen panels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .data import formula, load_gmtnet_records
from .jarvis_dfpt import JarvisDFPTCache
from .project_config import load_project_config
from .tensor_ops import piezo_voigt_to_cartesian, source_voigt_to_canonical


def _finite_shape(value: object, shape: tuple[int, ...]) -> bool:
    tensor = torch.as_tensor(value)
    return tensor.shape == shape and bool(torch.isfinite(tensor).all())


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        str(probability): float(torch.quantile(tensor, probability))
        for probability in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    }


def audit(config: dict[str, object]) -> dict[str, object]:
    records = load_gmtnet_records(config["data_root"])
    cache = JarvisDFPTCache(config["jarvis_dfpt_dir"])
    availability = Counter()
    atom_counts = Counter()
    formula_counts = Counter()
    cancellation: list[float] = []
    rows: list[dict[str, object]] = []
    for record in records:
        jid = str(record["JARVIS_ID"])
        payload = cache.load(jid)
        atoms = len(record["atoms"]["elements"])
        atom_counts[str(atoms)] += 1
        formula_counts[formula(record)] += 1
        row: dict[str, object] = {"material_id": jid, "atoms": atoms}
        if payload is None:
            availability["missing_payload"] += 1
            rows.append(row)
            continue
        availability["payload"] += 1
        born_ok = _finite_shape(payload.get("born_charges", []), (atoms, 3, 3))
        total_ok = _finite_shape(payload.get("total_piezo_source", []), (3, 6))
        ionic_ok = _finite_shape(payload.get("ionic_piezo_source", []), (3, 6))
        epsilon_value = payload.get("epsilon", {}).get("epsilon", [])
        epsilon_ok = _finite_shape(epsilon_value, (3, 3))
        force_ok = _finite_shape(payload.get("force_constants", []), (atoms, atoms, 3, 3))
        for name, ok in {
            "born": born_ok,
            "total_piezo": total_ok,
            "ionic_piezo": ionic_ok,
            "electronic_dielectric": epsilon_ok,
            "force_constants": force_ok,
        }.items():
            if ok:
                availability[name] += 1
        row.update({
            "born": born_ok,
            "total_piezo": total_ok,
            "ionic_piezo": ionic_ok,
            "electronic_dielectric": epsilon_ok,
            "force_constants": force_ok,
        })
        if total_ok and ionic_ok:
            total = piezo_voigt_to_cartesian(
                source_voigt_to_canonical(torch.as_tensor(payload["total_piezo_source"]))
            )
            ionic = piezo_voigt_to_cartesian(
                source_voigt_to_canonical(torch.as_tensor(payload["ionic_piezo_source"]))
            )
            electronic = total - ionic
            numerator = torch.linalg.vector_norm(total) + torch.linalg.vector_norm(ionic)
            denominator = torch.linalg.vector_norm(electronic).clamp_min(1e-8)
            value = float(numerator / denominator)
            cancellation.append(value)
            row["electronic_cancellation_index"] = value
        rows.append(row)
    complete_electrostatic = sum(
        1 for row in rows
        if all(bool(row.get(key, False)) for key in (
            "born", "total_piezo", "ionic_piezo", "electronic_dielectric"
        ))
    )
    return {
        "schema": 1,
        "purpose": "source-label availability and electronic cancellation audit",
        "frozen_validation_test_labels_read": False,
        "records": len(records),
        "availability": dict(availability),
        "complete_electrostatic_records": complete_electrostatic,
        "atom_count_histogram": dict(sorted(atom_counts.items(), key=lambda item: int(item[0]))),
        "unique_formulas": len(formula_counts),
        "electronic_cancellation_index": {
            "definition": "(||e_total||_F + ||e_ion||_F) / (||e_el||_F + 1e-8)",
            "materials": len(cancellation),
            "quantiles": _quantiles(cancellation),
            "strata": {
                ">=2": sum(value >= 2.0 for value in cancellation),
                ">=5": sum(value >= 5.0 for value in cancellation),
                ">=10": sum(value >= 10.0 for value in cancellation),
            },
        },
        "per_material": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(load_project_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "records": result["records"],
        "complete_electrostatic_records": result["complete_electrostatic_records"],
    }))


if __name__ == "__main__":
    main()
