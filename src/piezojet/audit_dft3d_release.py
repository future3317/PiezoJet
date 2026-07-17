"""Audit the official JARVIS dft_3d release against PiezoJet structures."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from .data import load_gmtnet_records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fractional(atoms: dict[str, object]) -> np.ndarray:
    coordinates = np.asarray(atoms["coords"], dtype=np.float64)
    if bool(atoms.get("cartesian", False)):
        cell = np.asarray(atoms["lattice_mat"], dtype=np.float64)
        return coordinates @ np.linalg.inv(cell)
    return coordinates


def audit_release(zip_path: Path, gmtnet_root: Path) -> dict[str, object]:
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        if members != ["jdft_3d-12-12-2022.json"]:
            raise ValueError(f"Unexpected dft_3d ZIP members: {members}")
        crc_failure = archive.testzip()
        if crc_failure is not None:
            raise ValueError(f"ZIP CRC failure in {crc_failure}")
        records = json.load(archive.open(members[0]))
    if not isinstance(records, list) or not records:
        raise ValueError("Official dft_3d JSON must be a non-empty record list")
    official = {str(record["jid"]): record for record in records}
    if len(official) != len(records):
        raise ValueError("Official dft_3d release contains duplicate JIDs")
    gmtnet = {
        str(record["JARVIS_ID"]): record
        for record in load_gmtnet_records(gmtnet_root)
    }
    common = sorted(set(official) & set(gmtnet))
    direct = 0
    same_cell_and_order = 0
    maximum_cell_error = 0.0
    maximum_fractional_error = 0.0
    for jid in common:
        left = gmtnet[jid]["atoms"]
        right = official[jid]["atoms"]
        same_elements = list(left["elements"]) == list(right["elements"])
        left_cell = np.asarray(left["lattice_mat"], dtype=np.float64)
        right_cell = np.asarray(right["lattice_mat"], dtype=np.float64)
        cell_error = (
            float(np.max(np.abs(left_cell - right_cell)))
            if left_cell.shape == right_cell.shape else float("inf")
        )
        maximum_cell_error = max(maximum_cell_error, cell_error)
        if same_elements and cell_error <= 1e-5:
            same_cell_and_order += 1
            delta = _fractional(left) - _fractional(right)
            fractional_error = float(np.max(np.abs(delta - np.round(delta))))
            maximum_fractional_error = max(maximum_fractional_error, fractional_error)
            if fractional_error <= 2e-5:
                direct += 1
    return {
        "schema": 1,
        "source": "official jarvis-tools dft_3d 12-12-2022 release",
        "source_url": "https://ndownloader.figshare.com/files/38521619",
        "zip_path": str(zip_path.resolve()),
        "zip_sha256": _sha256(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_crc_ok": True,
        "official_records": len(records),
        "official_unique_jids": len(official),
        "gmtnet_records": len(gmtnet),
        "jid_intersection": len(common),
        "gmtnet_ids_absent_from_2022_release": sorted(set(gmtnet) - set(official)),
        "official_ids_absent_from_gmtnet_piezo": len(set(official) - set(gmtnet)),
        "same_cell_and_atom_order_at_1e_5": same_cell_and_order,
        "direct_periodic_coordinate_matches_at_2e_5": direct,
        "maximum_cell_error_on_common_ids": maximum_cell_error,
        "maximum_fractional_error_on_same_cell_and_order_ids": maximum_fractional_error,
        "interpretation": (
            "The official 2022 dft_3d table is an auxiliary metadata/structure release. "
            "It does not replace the GMTNet-pinned structures for the JARVIS piezo benchmark: "
            "some historical JIDs are absent and some common JIDs use a different relaxed cell."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--gmtnet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_release(args.zip, args.gmtnet_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
