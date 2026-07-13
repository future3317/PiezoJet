"""Public JARVIS DFPT raw-file retrieval and convention-checked caching.

The GMTNet piezoelectric labels are indexed by ``JVASP-*`` identifiers.  The
same identifiers occur in JARVIS's public ``raw_files`` Figshare index, where
the ``DFPT`` archive contains the VASP ``vasprun.xml`` needed for Born charges,
Gamma-point dynamical eigenpairs, force constants, and dielectric tensors.

The API summary tables deliberately do not expose these large arrays.  This
module keeps the raw-file route explicit, cacheable, and separate from model
training: run ``python -m piezojet.jarvis_dfpt`` before enabling DFPT losses.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
import torch
from jarvis.db.figshare import data as figshare_data
from jarvis.io.vasp.outputs import Outcar, Vasprun


DFPT_CACHE_SCHEMA = 2
STRUCTURE_TOLERANCE = 2e-5


def _cache_name(jid: str) -> str:
    return sha256(jid.encode("utf-8")).hexdigest()[:16] + ".pt"


def _record_jid(name: str) -> str:
    return Path(name).name.split("_")[0].removesuffix(".zip")


@dataclass(frozen=True)
class DFPTArchive:
    jid: str
    name: str
    url: str


class JarvisRawFileIndex:
    """Lookup public DFPT archive URLs without querying the private REST API."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._archives: dict[str, DFPTArchive] | None = None

    def archives(self) -> dict[str, DFPTArchive]:
        if self._archives is None:
            raw_index = figshare_data("raw_files", store_dir=str(self.cache_dir))
            records = raw_index.get("DFPT", []) if isinstance(raw_index, dict) else []
            archives: dict[str, DFPTArchive] = {}
            for record in records:
                if not isinstance(record, dict):
                    continue
                name, url = str(record.get("name", "")), record.get("download_url")
                if not name or not isinstance(url, str) or not url:
                    continue
                jid = _record_jid(name)
                if jid in archives:
                    raise ValueError(f"Duplicate public JARVIS DFPT archive for {jid}")
                archives[jid] = DFPTArchive(jid=jid, name=name, url=url)
            if not archives:
                raise RuntimeError("JARVIS raw_files index contained no DFPT archives")
            self._archives = archives
        return self._archives

    def find(self, jid: str) -> DFPTArchive:
        try:
            return self.archives()[jid]
        except KeyError as error:
            raise LookupError(f"No public JARVIS DFPT archive found for {jid}") from error


class JarvisDFPTCache:
    """Atomic cache of parsed, same-structure JARVIS DFPT labels."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, jid: str) -> Path:
        return self.directory / _cache_name(jid)

    def load(self, jid: str) -> dict[str, Any] | None:
        path = self.path(jid)
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != DFPT_CACHE_SCHEMA or payload.get("jid") != jid:
            raise ValueError(f"Incompatible JARVIS DFPT cache payload: {path}")
        return payload

    def save(self, payload: dict[str, Any]) -> Path:
        jid = str(payload["jid"])
        if payload.get("schema") != DFPT_CACHE_SCHEMA:
            raise ValueError("Cannot save a DFPT payload with an unknown schema")
        path = self.path(jid)
        temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    @staticmethod
    def _validate_structure(record: dict[str, Any], structure: Any) -> None:
        atoms = record["atoms"]
        elements = [str(element) for element in structure.elements]
        if elements != list(atoms["elements"]):
            raise ValueError(f"{record['JARVIS_ID']}: DFPT archive atom ordering/species differs from GMTNet")
        lattice = np.asarray(structure.lattice_mat, dtype=float)
        target_lattice = np.asarray(atoms["lattice_mat"], dtype=float)
        if lattice.shape != (3, 3) or not np.allclose(lattice, target_lattice, atol=STRUCTURE_TOLERANCE, rtol=0.0):
            raise ValueError(f"{record['JARVIS_ID']}: DFPT archive lattice differs from GMTNet")
        frac = np.asarray(structure.frac_coords, dtype=float)
        target_frac = np.asarray(atoms["coords"], dtype=float)
        displacement = frac - target_frac
        displacement -= np.round(displacement)
        if frac.shape != target_frac.shape or not np.allclose(displacement, 0.0, atol=STRUCTURE_TOLERANCE, rtol=0.0):
            raise ValueError(f"{record['JARVIS_ID']}: DFPT archive fractional coordinates differ from GMTNet")

    @staticmethod
    def _download(url: str, timeout: float) -> bytes:
        request = Request(url, headers={"User-Agent": "PiezoJet/0.1 JARVIS-DFPT-cache"})
        with urlopen(request, timeout=timeout) as response:
            return response.read()

    @classmethod
    def _outcar_dfpt_labels(cls, contents: bytes, temporary: Path, jid: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse VASP's ionic/total piezo tensors and raw internal-strain blocks.

        VASP prints internal-strain blocks only for symmetry-inequivalent ionic
        perturbations.  We retain their ion and Cartesian-direction indices
        rather than applying an unvalidated symmetry expansion here.
        """
        text = contents.decode("utf-8", errors="replace")
        outcar_path = temporary / "OUTCAR"
        outcar_path.write_text(text, encoding="utf-8")
        ionic, total = Outcar(str(outcar_path)).piezoelectric_tensor
        ionic, total = np.asarray(ionic, dtype=np.float32), np.asarray(total, dtype=np.float32)
        if ionic.shape != (3, 6) or total.shape != (3, 6):
            raise ValueError(f"{jid}: OUTCAR did not contain one 3x6 ionic and total piezoelectric tensor")
        blocks, ions, directions = [], [], []
        pattern = re.compile(
            r"INTERNAL STRAIN TENSOR\s+FOR ION\s+(\d+)\s+DIRECTION\s+(\d+).*?\n\s*-+\n((?:.*\n){3})"
        )
        for match in pattern.finditer(text):
            rows = [re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", row) for row in match.group(3).strip().splitlines()]
            if len(rows) != 3 or any(len(row) != 3 for row in rows):
                raise ValueError(f"{jid}: malformed OUTCAR internal-strain block")
            blocks.append([[float(value) for value in row] for row in rows])
            ions.append(int(match.group(1)) - 1)
            directions.append(int(match.group(2)) - 1)
        if not blocks:
            raise ValueError(f"{jid}: OUTCAR contained no internal-strain tensor blocks")
        return (
            torch.as_tensor(ionic), torch.as_tensor(total), torch.as_tensor(np.asarray(blocks, dtype=np.float32)),
            torch.as_tensor(ions, dtype=torch.long), torch.as_tensor(directions, dtype=torch.long),
        )

    @classmethod
    def _parse_archive(cls, record: dict[str, Any], archive: DFPTArchive, timeout: float) -> dict[str, Any]:
        blob = cls._download(archive.url, timeout)
        with zipfile.ZipFile(io.BytesIO(blob)) as zipped:
            xml_members = [name for name in zipped.namelist() if Path(name).name.lower() == "vasprun.xml"]
            if len(xml_members) != 1:
                raise ValueError(f"{record['JARVIS_ID']}: expected exactly one vasprun.xml in {archive.name}")
            with tempfile.TemporaryDirectory() as temporary:
                temporary_path = Path(temporary)
                xml_path = temporary_path / "vasprun.xml"
                xml_path.write_bytes(zipped.read(xml_members[0]))
                vrun = Vasprun(str(xml_path))
                cls._validate_structure(record, vrun.all_structures[-1])
                dfpt = vrun.dfpt_data
                raw_dynamical_matrix = vrun.phonon_data(fc_mass=False)["force_constants"]
                outcar_members = [name for name in zipped.namelist() if Path(name).name == "OUTCAR"]
                if len(outcar_members) != 1:
                    raise ValueError(f"{record['JARVIS_ID']}: expected exactly one OUTCAR in {archive.name}")
                ionic_piezo, total_piezo, internal_strain, internal_strain_ions, internal_strain_directions = cls._outcar_dfpt_labels(
                    zipped.read(outcar_members[0]), temporary_path, str(record["JARVIS_ID"])
                )
        required = ("born_charges", "phonon_eigenvalues", "phonon_eigenvectors", "masses", "force_constants", "epsilon")
        missing = [name for name in required if name not in dfpt]
        if missing:
            raise ValueError(f"{record['JARVIS_ID']}: DFPT parser omitted {missing}")
        born = torch.as_tensor(np.asarray(dfpt["born_charges"]), dtype=torch.float32)
        eigenvalues = torch.as_tensor(np.asarray(dfpt["phonon_eigenvalues"]), dtype=torch.float32)
        eigenvectors = torch.as_tensor(np.asarray(dfpt["phonon_eigenvectors"]), dtype=torch.float32)
        masses = torch.as_tensor(np.asarray(dfpt["masses"]), dtype=torch.float32)
        force_constants = torch.as_tensor(np.asarray(dfpt["force_constants"]), dtype=torch.float32)
        dynamical_matrix = torch.as_tensor(np.asarray(raw_dynamical_matrix), dtype=torch.float32)
        atoms, modes = born.shape[0], eigenvalues.numel()
        if born.shape != (atoms, 3, 3) or modes != 3 * atoms or eigenvectors.shape != (modes, atoms, 3):
            raise ValueError(f"{record['JARVIS_ID']}: invalid Gamma-mode/BEC tensor shapes")
        if masses.shape != (atoms,) or force_constants.shape != (atoms, atoms, 3, 3):
            raise ValueError(f"{record['JARVIS_ID']}: invalid mass/force-constant tensor shapes")
        if dynamical_matrix.shape != force_constants.shape:
            raise ValueError(f"{record['JARVIS_ID']}: invalid dynamical-matrix tensor shape")
        if int(internal_strain_ions.min()) < 0 or int(internal_strain_ions.max()) >= atoms or int(internal_strain_directions.min()) < 0 or int(internal_strain_directions.max()) >= 3:
            raise ValueError(f"{record['JARVIS_ID']}: OUTCAR internal-strain index out of range")
        epsilon = {name: torch.as_tensor(np.asarray(value), dtype=torch.float32) for name, value in dfpt["epsilon"].items()}
        return {
            "schema": DFPT_CACHE_SCHEMA,
            "jid": str(record["JARVIS_ID"]),
            "source_archive": archive.name,
            "born_charges": born,
            # These are the VASP dynamical-matrix eigenvalues, not silently
            # relabelled THz frequencies.  OUTCAR may contain repeated blocks.
            "dynamical_eigenvalues": eigenvalues,
            "dynamical_eigenvectors": eigenvectors,
            "masses": masses,
            "force_constants": force_constants,
            "dynamical_matrix": dynamical_matrix,
            "ionic_piezo_source": ionic_piezo,
            "total_piezo_source": total_piezo,
            "internal_strain_tensors": internal_strain,
            "internal_strain_ions": internal_strain_ions,
            "internal_strain_directions": internal_strain_directions,
            "epsilon": epsilon,
        }

    def ensure(self, record: dict[str, Any], index: JarvisRawFileIndex, timeout: float = 180.0, overwrite: bool = False) -> dict[str, Any]:
        jid = str(record["JARVIS_ID"])
        cached = None if overwrite else self.load(jid)
        if cached is not None:
            return cached
        payload = self._parse_archive(record, index.find(jid), timeout)
        self.save(payload)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache public same-source JARVIS DFPT labels for PiezoJet.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/jarvis_dfpt_v1"))
    parser.add_argument("--index-cache-dir", type=Path, default=Path.home() / ".cache" / "piezojet" / "jarvis_raw_files")
    parser.add_argument("--limit", type=int, help="Bounded number of records; omit to cache all records.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    from .data import load_gmtnet_records

    records = load_gmtnet_records(args.data_root)
    if args.limit is not None:
        records = records[: args.limit]
    cache, index = JarvisDFPTCache(args.output_dir), JarvisRawFileIndex(args.index_cache_dir)
    succeeded, failed = [], []
    for position, record in enumerate(records, start=1):
        jid = str(record["JARVIS_ID"])
        try:
            payload = cache.ensure(record, index, timeout=args.timeout, overwrite=args.overwrite)
            succeeded.append(jid)
            print(f"[{position}/{len(records)}] cached {jid}: {payload['dynamical_eigenvalues'].numel()} Gamma modes")
        except Exception as error:  # retain failures for an auditable partial run
            failed.append({"jid": jid, "error": str(error)})
            print(f"[{position}/{len(records)}] failed {jid}: {error}")
    manifest = {
        "schema": DFPT_CACHE_SCHEMA,
        "requested": len(records),
        "cached": len(succeeded),
        "failed": failed,
        "source": "JARVIS public raw_files / DFPT archives",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"DFPT cache completed with {len(failed)} failures; see {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
