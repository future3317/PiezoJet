"""Reproducible repository, data, split, and tensor-convention audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import spglib
import torch
from pymatgen.core import Element

from .data import (
    PIEZO_FIELD,
    PIEZO_FILE,
    SPLIT_SCHEMA,
    create_or_load_splits,
    formula,
    load_gmtnet_records,
)
from .project_config import load_project_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def structure_hash(record: dict[str, Any]) -> str:
    atoms = record["atoms"]
    normalized = {
        "elements": sorted(atoms["elements"]),
        "lattice": [[round(float(value), 5) for value in row] for row in atoms["lattice_mat"]],
        "coords": sorted([[round(float(value) % 1.0, 5) for value in row] for row in atoms["coords"]]),
    }
    return hashlib.sha256(json.dumps(normalized, separators=(",", ":")).encode("utf-8")).hexdigest()


def chemical_system(record: dict[str, Any]) -> str:
    return "-".join(sorted(set(record["atoms"]["elements"])))


def _centrosymmetric(record: dict[str, Any]) -> bool:
    atoms = record["atoms"]
    cell = (
        atoms["lattice_mat"],
        atoms["coords"],
        [Element(symbol).Z for symbol in atoms["elements"]],
    )
    symmetry = spglib.get_symmetry(cell, symprec=1e-5)
    return bool((symmetry["rotations"] == -torch.eye(3, dtype=torch.int32).numpy()).all(axis=(1, 2)).any())


def build_audit(config_path: Path, output: Path) -> None:
    config = load_project_config(config_path)
    data_root = Path(config["data_root"])
    raw_path = data_root / "data" / PIEZO_FILE
    with raw_path.open("rb") as handle:
        raw_records = pickle.load(handle)
    records = load_gmtnet_records(data_root)
    splits = create_or_load_splits(records, config["processed_dir"], int(config["seed"]))
    by_id = {str(record["JARVIS_ID"]): record for record in records}
    split_hash = Path(config["processed_dir"]) / f"splits_formula_stratified_v{SPLIT_SCHEMA}.json"
    valid_tensors = [torch.tensor(record[PIEZO_FIELD], dtype=torch.float64) for record in records]
    populated = [tensor for tensor in valid_tensors if tensor.numel()]
    centers = sum(_centrosymmetric(record) for record in records)
    ids = [str(record["JARVIS_ID"]) for record in records]
    formulas = {material_id: formula(by_id[material_id]) for material_id in ids}
    systems = {material_id: chemical_system(by_id[material_id]) for material_id in ids}
    hashes = {material_id: structure_hash(by_id[material_id]) for material_id in ids}

    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{left}_{right}_material_id"] = sorted(set(splits[left]) & set(splits[right]))
        overlap[f"{left}_{right}_formula"] = sorted({formulas[item] for item in splits[left]} & {formulas[item] for item in splits[right]})
        overlap[f"{left}_{right}_chemical_system"] = sorted({systems[item] for item in splits[left]} & {systems[item] for item in splits[right]})
        overlap[f"{left}_{right}_structure_hash"] = sorted({hashes[item] for item in splits[left]} & {hashes[item] for item in splits[right]})

    output.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_geometric": package_version("torch-geometric"),
        "e3nn": package_version("e3nn"),
        "pymatgen": package_version("pymatgen"),
        "spglib": package_version("spglib"),
        "numpy": package_version("numpy"),
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    repository = {
        "git_commit": git_value("rev-parse", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain")),
        "remote": git_value("remote", "get-url", "origin"),
    }
    data_manifest = {
        "path": str(raw_path),
        "sha256": sha256_file(raw_path),
        "raw_records": len(raw_records),
        "valid_records": len(records),
        "filtered_records": len(raw_records) - len(records),
        "filter_rule": "finite 3x6 piezoelectric_C_m2 and abs(max)<100, matching GMTNet loader",
        "tensor_field": PIEZO_FIELD,
        "shape": [3, 6],
        "unit": "C/m^2",
        "source_voigt": ["xx", "yy", "zz", "xy", "yz", "xz"],
        "internal_voigt": ["xx", "yy", "zz", "yz", "xz", "xy"],
        "engineering_shear": "eta=[exx,eyy,ezz,2eyz,2exz,2exy]; source e_iJ equals Cartesian e_ij",
        "centrosymmetric_records": centers,
        "non_centrosymmetric_records": len(records) - centers,
        "duplicate_material_ids": len(ids) - len(set(ids)),
        "nan_inf_count": sum(not torch.isfinite(tensor).all() for tensor in valid_tensors),
        "tensor_rms": float(torch.sqrt(torch.cat(populated).square().mean())),
    }
    split_manifest = {
        "seed": int(config["seed"]),
        "method": "material-id sorted input with deterministic random.Random(seed) shuffle; 80/10/10",
        "counts": {name: len(ids) for name, ids in splits.items()},
        "ids": splits,
        "split_sha256": sha256_file(split_hash),
        "overlap": overlap,
    }
    tensor_convention = {
        "piezo_cartesian": "e_ijk=e_ikj",
        "cartesian_tensor": "CartesianTensor('ijk=ikj')",
        "irreps_dimension": 18,
        "strain_convention": "engineering shear",
        "voigt_to_matrix": "off-diagonal eta_ij=eta6/2",
        "piezo_to_cartesian": "off-diagonal e_ij=e_iJ, duplicated symmetrically",
        "rotation": "rho3(R)e = R_ia R_jb R_kc e_abc",
    }
    for name, payload in (("environment.json", environment), ("repository.json", repository), ("data_manifest.json", data_manifest), ("split_manifest.json", split_manifest), ("tensor_convention.json", tensor_convention)):
        (output / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_md = f"""# PiezoJet reproducibility audit

## Git commit

`{repository['git_commit']}`; dirty={repository['dirty']}

## Data manifest

- `{raw_path}`
- SHA256: `{data_manifest['sha256']}`
- raw/valid/filtered: {len(raw_records)}/{len(records)}/{len(raw_records)-len(records)}
- centrosymmetric/non-centrosymmetric: {centers}/{len(records)-centers}
- NaN/Inf after loader validation: {data_manifest['nan_inf_count']}

## Split manifest

- seed: {config['seed']}
- counts: {split_manifest['counts']}
- split SHA256: `{split_manifest['split_sha256']}`
- material-ID, formula, chemical-system and structure-hash overlaps are recorded in `split_manifest.json`.

## Tensor convention

Source Voigt is `[xx, yy, zz, xy, yz, xz]`; internal Voigt is `[xx, yy, zz, yz, xz, xy]`. Engineering shear is used and the factor two is applied only when expanding strain into a symmetric Cartesian matrix.

## Interpretation boundary

This audit verifies provenance, schema and split integrity. It does not establish model generalization or physical accuracy.
"""
    (output / "audit.md").write_text(audit_md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_audit(args.config, args.output)
    print(f"audit written to {args.output}")


if __name__ == "__main__":
    main()
