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
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from inspect import getsourcefile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
import torch
from jarvis.db.figshare import data as figshare_data
from jarvis.io.vasp.outputs import Outcar, Vasprun


# Schema 4 stores tensors in PiezoJet's *internal* coordinate convention and
# records the exact archive/parser/conversion provenance needed to reproduce a
# cache entry. Older schemas require the explicit offline migration command;
# the maintained data path never silently accepts provenance-incomplete data.
# The two VASP-source tensors that have a non-trivial convention transform
# retain their raw counterparts as ``*_source`` so the transformation is
# auditable and a cache does not silently lose information.
DFPT_CACHE_SCHEMA = 4
STRUCTURE_TOLERANCE = 2e-5


def tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    """Hash tensor content together with dtype and shape, never object reprs."""
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _jarvis_parser_identity() -> dict[str, str | None]:
    """Return reproducible parser identity without inventing a git revision.

    Wheels normally do not preserve the upstream jarvis-tools git commit.  A
    caller that installed from a checkout may set ``JARVIS_TOOLS_COMMIT``;
    otherwise the field is explicitly ``None`` and the installed parser source
    hash remains available for byte-level provenance.
    """
    source_file = getsourcefile(Vasprun)
    source_hash = None
    if source_file is not None:
        try:
            source_hash = sha256(Path(source_file).read_bytes()).hexdigest()
        except OSError:
            pass
    try:
        version = metadata.version("jarvis-tools")
    except metadata.PackageNotFoundError:
        version = None
    return {
        "jarvis_tools_version": version,
        "jarvis_tools_commit": os.environ.get("JARVIS_TOOLS_COMMIT") or None,
        "vasprun_parser_source_sha256": source_hash,
        "parser_schema": "jarvis.io.vasp.outputs.Vasprun.dfpt_data; "
        "Vasprun.phonon_data(fc_mass=False); Outcar.piezoelectric_tensor",
    }


def build_dfpt_provenance(
    *,
    archive: DFPTArchive,
    archive_bytes: bytes,
    vasprun_bytes: bytes,
    outcar_bytes: bytes,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Construct auditable raw/source/internal checksum metadata for schema 4."""
    tensor_names = (
        "born_charges_source",
        "born_charges",
        "dynamical_eigenvalues",
        "dynamical_eigenvectors",
        "masses",
        "dynamical_matrix",
        "force_constants",
        "ionic_piezo_source",
        "total_piezo_source",
        "internal_strain_tensors_source",
        "internal_strain_tensors",
        "internal_strain_rounding_halfwidth_source",
        "internal_strain_rounding_halfwidth",
        "internal_strain_ions",
        "internal_strain_directions",
    )
    checksums = {name: tensor_sha256(payload[name]) for name in tensor_names}
    checksums.update(
        {
            f"epsilon.{name}": tensor_sha256(value)
            for name, value in sorted(payload["epsilon"].items())
        }
    )
    return {
        "schema": 1,
        "source_archive": {
            "name": archive.name,
            "url": archive.url,
            "archive_sha256": sha256(archive_bytes).hexdigest(),
            "vasprun_xml_sha256": sha256(vasprun_bytes).hexdigest(),
            "outcar_sha256": sha256(outcar_bytes).hexdigest(),
        },
        "parser": _jarvis_parser_identity(),
        "force_constant_conversion": {
            "raw_call": "Vasprun.phonon_data(fc_mass=False)['force_constants']",
            "raw_fc_mass": False,
            "converted_source": "Vasprun.dfpt_data['force_constants']",
            "converted_fc_mass": True,
            "expected_relation": "force_constants = -dynamical_matrix * sqrt(m_i*m_j)",
            "unit": "eV/Angstrom^2",
            "sign_conversion": "negative mass-unweighting performed by jarvis-tools",
        },
        "tensor_sha256": checksums,
        "units": {
            "born_charges": "electron charge",
            "force_constants": "eV/Angstrom^2",
            "internal_strain": "eV/Angstrom per unit engineering strain",
            "piezoelectric": "C/m^2",
        },
        "coordinate_conversions": {
            "born_charges": "source Z[i,j]=dP_i/du_j -> internal Z[j,i]",
            "internal_strain": "source printed dF/deta -> internal Lambda=dF/deta (identity sign)",
            "internal_strain_voigt": "canonical engineering order (xx, yy, zz, yz, xz, xy)",
        },
    }


def source_born_to_internal(born_charges: torch.Tensor) -> torch.Tensor:
    """Convert VASP's ``Z[i,j] = dP_i / du_j`` to PiezoJet's ``Z[j,i]``.

    PiezoJet contracts a flattened atom-coordinate row with an electric-field
    column, ``Z_internal.T @ u``.  VASP reports the polarization/field index
    first, so every per-ion 3-by-3 source tensor must be transposed exactly
    once at ingestion.  Keeping the conversion here avoids a model-dependent
    transpose in training or evaluation.
    """
    if born_charges.ndim != 3 or born_charges.shape[-2:] != (3, 3):
        raise ValueError("VASP Born charges must have shape [atoms, 3, 3]")
    return born_charges.transpose(-1, -2).contiguous()


def source_internal_strain_to_internal(internal_strain: torch.Tensor) -> torch.Tensor:
    """Keep VASP OUTCAR's printed force derivative as PiezoJet ``Lambda``.

    Formally ``Xi=d^2E/(du deta)`` and PiezoJet writes
    ``Lambda=-Xi``.  The *printed OUTCAR internal-strain block*, however, is
    the force derivative ``dF/deta=-Xi``.  Its sign already equals the
    internal convention.  This is checked against the source OUTCAR ionic
    branch over every strict completion; a further negation reverses the
    response.  Voigt ordering is handled separately and once by tensor helpers.
    """
    if internal_strain.ndim != 3 or internal_strain.shape[-2:] != (3, 3):
        raise ValueError("VASP internal-strain blocks must have shape [blocks, 3, 3]")
    return internal_strain.contiguous()


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
        schema = payload.get("schema")
        if schema != DFPT_CACHE_SCHEMA or payload.get("jid") != jid:
            raise ValueError(f"Incompatible JARVIS DFPT cache payload: {path}")
        if not isinstance(payload.get("provenance"), dict):
            raise ValueError(f"Schema-{DFPT_CACHE_SCHEMA} payload lacks provenance: {path}")
        return payload

    def save(self, payload: dict[str, Any]) -> Path:
        jid = str(payload["jid"])
        if payload.get("schema") != DFPT_CACHE_SCHEMA:
            raise ValueError("Cannot save a DFPT payload with an unknown schema")
        if not isinstance(payload.get("provenance"), dict):
            raise ValueError("Schema-4 DFPT cache payloads require explicit provenance")
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
    def _outcar_dfpt_labels(cls, contents: bytes, temporary: Path, jid: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
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
        blocks, halfwidths, ions, directions = [], [], [], []
        malformed: list[dict[str, Any]] = []
        pattern = re.compile(
            r"INTERNAL STRAIN TENSOR\s+FOR ION\s+(\d+)\s+DIRECTION\s+(\d+).*?\n\s*-+\n((?:.*\n){3})"
        )
        for match in pattern.finditer(text):
            rows = [re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", row) for row in match.group(3).strip().splitlines()]
            if len(rows) != 3 or any(len(row) != 3 for row in rows):
                # VASP prints ********* on numeric overflow.  That value is
                # not recoverable and must never be guessed, but it must not
                # discard the independently valid BEC/Phi/epsilon labels in
                # the same archive.  Retain the block identity and mark it
                # unavailable for observed-Lambda or strict completion.
                malformed.append({
                    "ion": int(match.group(1)) - 1,
                    "direction": int(match.group(2)) - 1,
                    "reason": "numeric_overflow" if "*" in match.group(3) else "unparseable_numeric_layout",
                    "row_numeric_counts": [len(row) for row in rows],
                })
                continue
            blocks.append([[float(value) for value in row] for row in rows])
            # The OUTCAR text is the label source.  Preserve its displayed
            # rounding interval so completion sensitivity can be bootstrapped
            # from a declared source uncertainty rather than an arbitrary loss
            # scale.  For a mantissa with d decimal digits and exponent e the
            # nearest-printing halfwidth is 0.5 * 10**(e-d).
            widths: list[list[float]] = []
            for row in rows:
                row_widths = []
                for value in row:
                    mantissa, exponent = re.split(r"[Ee]", value) if re.search(r"[Ee]", value) else (value, "0")
                    decimals = len(mantissa.split(".", maxsplit=1)[1]) if "." in mantissa else 0
                    row_widths.append(0.5 * 10.0 ** (int(exponent) - decimals))
                widths.append(row_widths)
            halfwidths.append(widths)
            ions.append(int(match.group(1)) - 1)
            directions.append(int(match.group(2)) - 1)
        block_tensor = (
            torch.as_tensor(np.asarray(blocks, dtype=np.float32))
            if blocks else torch.empty(0, 3, 3, dtype=torch.float32)
        )
        halfwidth_tensor = (
            torch.as_tensor(np.asarray(halfwidths, dtype=np.float32))
            if halfwidths else torch.empty(0, 3, 3, dtype=torch.float32)
        )
        return (
            torch.as_tensor(ionic), torch.as_tensor(total), block_tensor,
            torch.as_tensor(ions, dtype=torch.long), torch.as_tensor(directions, dtype=torch.long),
            halfwidth_tensor,
            {
                "matched_blocks": len(blocks) + len(malformed),
                "valid_blocks": len(blocks),
                "malformed_blocks": malformed,
                "complete_observed_block_parse": len(malformed) == 0 and len(blocks) > 0,
            },
        )

    @classmethod
    def _parse_archive(cls, record: dict[str, Any], archive: DFPTArchive, timeout: float) -> dict[str, Any]:
        blob = cls._download(archive.url, timeout)
        with zipfile.ZipFile(io.BytesIO(blob)) as zipped:
            xml_members = [name for name in zipped.namelist() if Path(name).name.lower() == "vasprun.xml"]
            if len(xml_members) != 1:
                raise ValueError(f"{record['JARVIS_ID']}: expected exactly one vasprun.xml in {archive.name}")
            xml_bytes = zipped.read(xml_members[0])
            outcar_members = [name for name in zipped.namelist() if Path(name).name == "OUTCAR"]
            if len(outcar_members) != 1:
                raise ValueError(f"{record['JARVIS_ID']}: expected exactly one OUTCAR in {archive.name}")
            outcar_bytes = zipped.read(outcar_members[0])
            with tempfile.TemporaryDirectory() as temporary:
                temporary_path = Path(temporary)
                xml_path = temporary_path / "vasprun.xml"
                xml_path.write_bytes(xml_bytes)
                vrun = Vasprun(str(xml_path))
                cls._validate_structure(record, vrun.all_structures[-1])
                dfpt = vrun.dfpt_data
                raw_dynamical_matrix = vrun.phonon_data(fc_mass=False)["force_constants"]
                (
                    ionic_piezo, total_piezo, internal_strain,
                    internal_strain_ions, internal_strain_directions,
                    internal_strain_halfwidth, internal_strain_parse_audit,
                ) = cls._outcar_dfpt_labels(
                    outcar_bytes, temporary_path, str(record["JARVIS_ID"])
                )
        required = ("born_charges", "phonon_eigenvalues", "phonon_eigenvectors", "masses", "force_constants", "epsilon")
        missing = [name for name in required if name not in dfpt]
        if missing:
            raise ValueError(f"{record['JARVIS_ID']}: DFPT parser omitted {missing}")
        born_source = torch.as_tensor(np.asarray(dfpt["born_charges"]), dtype=torch.float32)
        eigenvalues = torch.as_tensor(np.asarray(dfpt["phonon_eigenvalues"]), dtype=torch.float32)
        eigenvectors = torch.as_tensor(np.asarray(dfpt["phonon_eigenvectors"]), dtype=torch.float32)
        masses = torch.as_tensor(np.asarray(dfpt["masses"]), dtype=torch.float32)
        force_constants = torch.as_tensor(np.asarray(dfpt["force_constants"]), dtype=torch.float32)
        dynamical_matrix = torch.as_tensor(np.asarray(raw_dynamical_matrix), dtype=torch.float32)
        atoms, modes = born_source.shape[0], eigenvalues.numel()
        if born_source.shape != (atoms, 3, 3) or modes != 3 * atoms or eigenvectors.shape != (modes, atoms, 3):
            raise ValueError(f"{record['JARVIS_ID']}: invalid Gamma-mode/BEC tensor shapes")
        if masses.shape != (atoms,) or force_constants.shape != (atoms, atoms, 3, 3):
            raise ValueError(f"{record['JARVIS_ID']}: invalid mass/force-constant tensor shapes")
        if dynamical_matrix.shape != force_constants.shape:
            raise ValueError(f"{record['JARVIS_ID']}: invalid dynamical-matrix tensor shape")
        if internal_strain_ions.numel() and (
            int(internal_strain_ions.min()) < 0
            or int(internal_strain_ions.max()) >= atoms
            or int(internal_strain_directions.min()) < 0
            or int(internal_strain_directions.max()) >= 3
        ):
            raise ValueError(f"{record['JARVIS_ID']}: OUTCAR internal-strain index out of range")
        epsilon = {name: torch.as_tensor(np.asarray(value), dtype=torch.float32) for name, value in dfpt["epsilon"].items()}
        payload = {
            "schema": DFPT_CACHE_SCHEMA,
            "jid": str(record["JARVIS_ID"]),
            "source_archive": archive.name,
            # Retain the unmodified VASP orientation and make the one public
            # source-to-internal conversion at cache creation, never in a
            # training loss or response contraction.
            "born_charges_source": born_source,
            "born_charges": source_born_to_internal(born_source),
            # These are the VASP dynamical-matrix eigenvalues, not silently
            # relabelled THz frequencies.  OUTCAR may contain repeated blocks.
            "dynamical_eigenvalues": eigenvalues,
            "dynamical_eigenvectors": eigenvectors,
            "masses": masses,
            "force_constants": force_constants,
            "dynamical_matrix": dynamical_matrix,
            "ionic_piezo_source": ionic_piezo,
            "total_piezo_source": total_piezo,
            "internal_strain_tensors_source": internal_strain,
            "internal_strain_tensors": source_internal_strain_to_internal(internal_strain),
            "internal_strain_rounding_halfwidth_source": internal_strain_halfwidth,
            "internal_strain_rounding_halfwidth": source_internal_strain_to_internal(internal_strain_halfwidth),
            "internal_strain_ions": internal_strain_ions,
            "internal_strain_directions": internal_strain_directions,
            "internal_strain_parse_audit": internal_strain_parse_audit,
            "epsilon": epsilon,
            "conventions": {
                "force_constants": "VASP dynmat hessian converted by jarvis-tools: -raw*sqrt(m_i*m_j), eV/Angstrom^2",
                "born_charges_source": "VASP Z[i,j]=dP_i/du_j; source axes retained",
                "born_charges_internal": "PiezoJet Z[j,i], coordinate/force row and polarization/field column",
                "internal_strain_source": "VASP OUTCAR printed dF/deta=-d^2E/(du deta)",
                "internal_strain_internal": "PiezoJet Lambda=dF/deta; no additional sign change",
            },
        }
        payload["provenance"] = build_dfpt_provenance(
            archive=archive,
            archive_bytes=blob,
            vasprun_bytes=xml_bytes,
            outcar_bytes=outcar_bytes,
            payload=payload,
        )
        return payload

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
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-cache-dir", type=Path, default=Path.home() / ".cache" / "piezojet" / "jarvis_raw_files")
    parser.add_argument("--limit", type=int, help="Bounded number of records; omit to cache all records.")
    parser.add_argument(
        "--material-ids-file", type=Path,
        help="JSON list or newline-delimited JARVIS IDs for an explicit, auditable retrieval cohort.",
    )
    parser.add_argument(
        "--accepted-completion-dir",
        type=Path,
        help="Select only schema-validated accepted IDs from an existing strict-completion directory; useful for provenance-only cache regeneration.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--shard-count", type=int, default=1,
        help=(
            "Deterministically partition the selected records into this many "
            "zero-based shards.  Use only disjoint shards when they share an "
            "atomic payload cache."
        ),
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="Zero-based shard index; valid only together with --shard-count.",
    )
    parser.add_argument(
        "--no-manifest", action="store_true",
        help=(
            "Do not write manifest.json.  Required for parallel acquisition "
            "workers so that a single final owner writes the cache manifest."
        ),
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < --shard-count")
    selectors = sum(value is not None for value in (args.limit, args.material_ids_file, args.accepted_completion_dir))
    if selectors > 1:
        raise ValueError("--limit, --material-ids-file, and --accepted-completion-dir are mutually exclusive")
    from .gmtnet_io import load_gmtnet_records

    records = load_gmtnet_records(args.data_root)
    if args.accepted_completion_dir is not None:
        if not args.accepted_completion_dir.is_dir():
            raise FileNotFoundError(f"Strict-completion directory does not exist: {args.accepted_completion_dir}")
        selected = []
        for path in sorted(args.accepted_completion_dir.glob("JVASP-*.pt")):
            completion = torch.load(path, map_location="cpu", weights_only=False)
            if completion.get("schema") != 2 or not bool(completion.get("audit", {}).get("accepted", False)):
                continue
            selected.append(str(completion.get("jid", "")))
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("Strict-completion directory did not contain unique accepted schema-2 labels")
        by_id = {str(record["JARVIS_ID"]): record for record in records}
        unknown = sorted(set(selected) - set(by_id))
        if unknown:
            raise ValueError(f"Strict-completion directory contains unknown IDs: {unknown[:5]}")
        records = [by_id[jid] for jid in selected]
    elif args.material_ids_file is not None:
        if not args.material_ids_file.is_file():
            raise FileNotFoundError(f"Material-ID file does not exist: {args.material_ids_file}")
        text = args.material_ids_file.read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parsed = parsed.get("material_ids")
            if not isinstance(parsed, list):
                raise ValueError("JSON material-ID input must be a list or contain a material_ids list")
            selected = [str(value) for value in parsed]
        except json.JSONDecodeError:
            selected = [line.strip() for line in text.splitlines() if line.strip()]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("Material-ID file must contain a non-empty list of unique IDs")
        by_id = {str(record["JARVIS_ID"]): record for record in records}
        unknown = sorted(set(selected) - set(by_id))
        if unknown:
            raise ValueError(f"Material-ID file contains unknown IDs: {unknown[:5]}")
        records = [by_id[jid] for jid in selected]
    elif args.limit is not None:
        records = records[: args.limit]
    if args.shard_count > 1:
        records = [
            record for position, record in enumerate(records)
            if position % args.shard_count == args.shard_index
        ]
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
        "material_ids": succeeded,
        "failed": failed,
        "source": "JARVIS public raw_files / DFPT archives",
        "provenance": (
            "Each schema-4 .pt records archive/XML/OUTCAR SHA256, jarvis-tools version and optional "
            "commit, parser source SHA256, explicit fc_mass calls, units/conversions, and source/internal tensor SHA256."
        ),
    }
    if not args.no_manifest:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"DFPT cache completed with {len(failed)} failures; see {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
