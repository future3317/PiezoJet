#!/usr/bin/env python
"""Inspect, without guessing, the only allowed GMTNet tensor source."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import spglib
import torch
from pymatgen.core import Element

from piezojet.data import PIEZO_FIELD, PIEZO_FILE, load_gmtnet_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    records = load_gmtnet_records(args.root)
    raw_path = args.root / "data" / PIEZO_FILE
    raw = __import__("pickle").load(raw_path.open("rb"))
    tensors = torch.stack([torch.tensor(item[PIEZO_FIELD], dtype=torch.float64) for item in records])
    atom_counts = [len(item["atoms"]["elements"]) for item in records]
    invalid = sum(not torch.isfinite(torch.tensor(item[PIEZO_FIELD], dtype=torch.float64)).all() for item in raw if item.get(PIEZO_FIELD) is not None)
    centrosymmetric = 0
    centrosymmetric_false_positive = 0
    for record, tensor in zip(records, tensors):
        atoms = record["atoms"]
        cell = (
            np.asarray(atoms["lattice_mat"], dtype=float),
            np.asarray(atoms["coords"], dtype=float),
            [Element(symbol).Z for symbol in atoms["elements"]],
        )
        symmetry = spglib.get_symmetry(cell, symprec=1e-5)
        if np.any(np.all(symmetry["rotations"] == -np.eye(3, dtype=int), axis=(1, 2))):
            centrosymmetric += 1
            centrosymmetric_false_positive += int(float(tensor.abs().max()) >= 1e-4)
    print(f"file: {raw_path}")
    commit_path = args.root / "SOURCE_COMMIT.txt"
    print(f"source commit: {commit_path.read_text(encoding='utf-8').strip() if commit_path.is_file() else 'MISSING (training is blocked)'}")
    print(f"raw samples: {len(raw)}; valid finite 3x6 samples: {len(records)}")
    print(f"first record fields: {sorted(raw[0].keys())}")
    print("structure: atoms OrderedDict with lattice_mat [3,3], fractional coords [N,3], elements, cartesian=False")
    print(f"tensor: {PIEZO_FIELD}, shape {tuple(tensors.shape[1:])}, unit C/m^2")
    print("source Voigt order: [xx, yy, zz, xy, yz, xz] (GMTNet_piezo/transformer.py index [0,4,8,1,5,6])")
    print("PiezoJet canonical order: [xx, yy, zz, yz, xz, xy]; source is converted at ingestion")
    print("shear: engineering strain; eta=[exx,eyy,ezz,2eyz,2exz,2exy], fixed in piezojet.tensor_ops")
    print("existing train/val/test split: no")
    print(f"excluded by GMTNet loader screen (empty/malformed/non-finite/abs(max)>=100): {len(raw) - len(records)} (non-finite among populated tensors: {invalid})")
    print(f"atoms per sample: min {min(atom_counts)}, max {max(atom_counts)}")
    print(f"centrosymmetric samples (spglib inversion operation): {centrosymmetric}; tensors above 1e-4: {centrosymmetric_false_positive}")
    if not math.isfinite(float(tensors.abs().max())):
        raise RuntimeError("Non-finite tensor survived validation")


if __name__ == "__main__":
    main()
