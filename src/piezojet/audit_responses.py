"""Audit GMTNet piezo, dielectric, and elastic response sources before M5."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_stats(values: list[Any]) -> dict[str, Any]:
    tensors = [torch.as_tensor(value, dtype=torch.float64) for value in values if value is not None]
    flattened = torch.cat([tensor.reshape(-1) for tensor in tensors]) if tensors else torch.empty(0, dtype=torch.float64)
    return {
        "populated": len(tensors),
        "nan_inf": int((~torch.isfinite(flattened)).sum()),
        "min": float(flattened.min()) if flattened.numel() else None,
        "max": float(flattened.max()) if flattened.numel() else None,
    }


def field_audit(records: list[dict[str, Any]], field: str, shape: list[int], unit: str, convention: str, transform: str) -> dict[str, Any]:
    values = [record.get(field) for record in records]
    shape_set = {tuple(torch.as_tensor(value).shape) for value in values if value is not None}
    shapes = [list(shape) for shape in sorted(shape_set)]
    return {
        "field": field,
        "shape_observed": shapes,
        "shape_expected": shape,
        "shape_match": shapes == [shape],
        "unit": unit,
        "voigt_or_tensor_convention": convention,
        "loader_transform": transform,
        "statistics": finite_stats(values),
        "material_ids": sum(value is not None for value in values),
        "missing_fraction": sum(value is None for value in values) / max(len(values), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root / "data"
    piezo_path, elastic_path = root / "jarvis_diele_piezo.pkl", root / "jarvis_elastic.pkl"
    with piezo_path.open("rb") as handle:
        piezo = pickle.load(handle)
    with elastic_path.open("rb") as handle:
        elastic = pickle.load(handle)
    piezo_ids, elastic_ids = {record["JARVIS_ID"] for record in piezo}, {record["JARVIS_ID"] for record in elastic}
    intersections = {
        "piezo_count": len(piezo_ids), "elastic_count": len(elastic_ids), "piezo_elastic_intersection": len(piezo_ids & elastic_ids),
        "piezo_only": len(piezo_ids - elastic_ids), "elastic_only": len(elastic_ids - piezo_ids),
    }
    fields = {
        "piezoelectric_C_m2": field_audit(piezo, "piezoelectric_C_m2", [3, 6], "C/m^2", "source [xx,yy,zz,xy,yz,xz]; converted internally", "GMTNet abs(max)<100 screen"),
        "piezoelectric_e_Angst": field_audit(piezo, "piezoelectric_e_Angst", [3, 6], "not independently declared in loader; source field name contains Angstrom", "same source 3x6 ordering", "none"),
        "dielectric": field_audit(piezo, "dielectric", [3, 3], "raw dielectric field; unit requires source confirmation before M5", "symmetric Cartesian 3x3 observed", "none"),
        "dielectric_ionic": field_audit(piezo, "dielectric_ionic", [3, 3], "raw dielectric field; unit requires source confirmation before M5", "symmetric Cartesian 3x3 observed", "none"),
        "elastic_total_kbar": field_audit(elastic, "elastic_total_kbar", [6, 6], "kbar source; GMTNet loader divides by 10 for GPa", "6x6 elastic Voigt source; exact shear convention requires audit", "divide by 10 in GMTNet loader"),
        "elastic_sym_kbar": field_audit(elastic, "elastic_sym_kbar", [6, 6], "kbar source; GMTNet loader convention", "6x6 elastic Voigt source; exact shear convention requires audit", "none"),
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    response_fields = {"files": {"piezo": {"path": str(piezo_path), "sha256": sha256_file(piezo_path), "records": len(piezo)}, "elastic": {"path": str(elastic_path), "sha256": sha256_file(elastic_path), "records": len(elastic)}}, "fields": fields}
    (output / "response_fields.json").write_text(json.dumps(response_fields, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "response_intersections.json").write_text(json.dumps(intersections, indent=2) + "\n", encoding="utf-8")
    units = """# Response units and convention audit

- `piezoelectric_C_m2`: C/m^2, confirmed by the field name and GMTNet loader usage.
- `piezoelectric_e_Angst`: field is present but its unit/conversion is not used by the MVP.
- `dielectric` and `dielectric_ionic`: 3x3 symmetric-looking fields; the GMTNet loader does not declare a unit conversion, so M5 coefficient training is blocked pending authoritative unit confirmation.
- `elastic_total_kbar` and `elastic_sym_kbar`: 6x6 source fields in kbar; the official loader divides `elastic_total_kbar` by 10, corresponding to GPa. Exact shear/Voigt convention still requires a dedicated round-trip check.
"""
    (output / "response_units.md").write_text(units, encoding="utf-8")
    torch.save({"piezo_examples": piezo[:2], "elastic_examples": elastic[:2]}, output / "response_examples.pt")
    print(f"response audit written to {output}")


if __name__ == "__main__":
    main()
