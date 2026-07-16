"""Independent pymatgen cross-check of cached JARVIS VASP DFPT parsing.

This module deliberately re-downloads the public JARVIS archive and parses
``vasprun.xml``/``OUTCAR`` through pymatgen rather than reusing jarvis-tools.
It checks the raw Gamma dynamical matrix, source-oriented BECs, and the total
OUTCAR piezo tensor.  Pymatgen exposes the total printed piezo block but not
JARVIS's ionic/total split, so the latter remains a separate source-parser
check.  This is a parser cross-validation, not a phonopy finite-displacement
calculation and not evidence about non-analytic Gamma boundary conditions.
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.io.vasp.outputs import Outcar as PymatgenOutcar
from pymatgen.io.vasp.outputs import Vasprun as PymatgenVasprun

try:  # Keep the pymatgen-only audit importable in minimal environments.
    from phonopy.interface.vasp import get_born_vasprunxml, get_force_constants_OUTCAR
except ImportError:  # pragma: no cover - depends on an optional installation.
    get_born_vasprunxml = None
    get_force_constants_OUTCAR = None

from .data import load_gmtnet_records
from .jarvis_dfpt import JarvisDFPTCache, JarvisRawFileIndex


def relative_error(observed: np.ndarray | torch.Tensor, reference: np.ndarray | torch.Tensor) -> float:
    """Scale-safe Frobenius relative error for parser array comparisons."""
    left = torch.as_tensor(np.asarray(observed), dtype=torch.float64)
    right = torch.as_tensor(np.asarray(reference), dtype=torch.float64)
    if left.shape != right.shape:
        raise ValueError(f"Parser shapes differ: {tuple(left.shape)} versus {tuple(right.shape)}")
    return float(torch.linalg.vector_norm(left - right) / torch.linalg.vector_norm(right).clamp_min(1e-30))


def _archive_members(blob: bytes, jid: str) -> tuple[bytes, bytes]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zipped:
        xml_members = [name for name in zipped.namelist() if Path(name).name.lower() == "vasprun.xml"]
        outcar_members = [name for name in zipped.namelist() if Path(name).name == "OUTCAR"]
        if len(xml_members) != 1 or len(outcar_members) != 1:
            raise ValueError(f"{jid}: archive must contain exactly one vasprun.xml and one OUTCAR")
        return zipped.read(xml_members[0]), zipped.read(outcar_members[0])


def crossvalidate_one(
    *,
    jid: str,
    cache: JarvisDFPTCache,
    index: JarvisRawFileIndex,
    timeout: float,
    raw_matrix_tolerance: float,
    born_tolerance: float,
    total_piezo_tolerance: float,
    phonopy_outcar_force_tolerance: float = 1e-2,
    phonopy_born_tolerance: float = 5e-5,
    phonopy_dielectric_tolerance: float = 5e-5,
    require_phonopy: bool = False,
) -> dict[str, Any]:
    """Parse one public archive independently and compare it with the cache."""
    payload = cache.load(jid)
    if payload is None:
        raise FileNotFoundError(f"No cached DFPT payload for {jid}")
    archive = index.find(jid)
    blob = cache._download(archive.url, timeout)
    xml, outcar = _archive_members(blob, jid)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        xml_path, outcar_path = directory / "vasprun.xml", directory / "OUTCAR"
        xml_path.write_bytes(xml)
        outcar_path.write_bytes(outcar)
        # Disable DOS/eigen parsing: the independent checks below need only
        # the DFPT Hessian, so this keeps the 10-archive audit lightweight.
        pymatgen_vasprun = PymatgenVasprun(
            xml_path,
            parse_dos=False,
            parse_eigen=False,
            parse_potcar_file=False,
        )
        pymatgen_outcar = PymatgenOutcar(outcar_path)
        pymatgen_outcar.read_piezo_tensor()
        phonopy_metrics: dict[str, Any] | None = None
        if get_born_vasprunxml is not None and get_force_constants_OUTCAR is not None:
            phonopy_born, phonopy_epsilon, phonopy_independent_atoms = get_born_vasprunxml(
                xml_path, is_symmetry=False
            )
            phonopy_force_constants = get_force_constants_OUTCAR(outcar_path)
            epsilon = payload["epsilon"].get("epsilon")
            if epsilon is None:
                raise ValueError(f"{jid}: cache lacks static electronic epsilon for phonopy comparison")
            phonopy_metrics = {
                "born_source_relative_error": relative_error(
                    phonopy_born, payload.get("born_charges_source", payload["born_charges"])
                ),
                # OUTCAR second derivatives are printed with finite precision,
                # so this is deliberately compared to physical Phi, not raw D.
                "outcar_force_constants_relative_error": relative_error(
                    phonopy_force_constants, payload["force_constants"]
                ),
                "epsilon_relative_error": relative_error(phonopy_epsilon, epsilon),
                "independent_atom_count": int(np.asarray(phonopy_independent_atoms).size),
            }
    if not hasattr(pymatgen_outcar, "born"):
        raise ValueError(f"{jid}: pymatgen OUTCAR parser did not expose Born charges")
    total_piezo = pymatgen_outcar.data.get("piezo_tensor")
    if total_piezo is None:
        raise ValueError(f"{jid}: pymatgen OUTCAR parser did not expose the total piezo tensor")
    raw_relative = relative_error(pymatgen_vasprun.force_constants, payload["dynamical_matrix"])
    born_relative = relative_error(pymatgen_outcar.born, payload.get("born_charges_source", payload["born_charges"]))
    total_relative = relative_error(total_piezo, payload["total_piezo_source"])
    cached_provenance = payload.get("provenance", {})
    cached_archive_sha = cached_provenance.get("source_archive", {}).get("archive_sha256")
    archive_sha = sha256(blob).hexdigest()
    checks = {
        "raw_dynamical_matrix": raw_relative <= raw_matrix_tolerance,
        "born_charges_source": born_relative <= born_tolerance,
        "total_outcar_piezo": total_relative <= total_piezo_tolerance,
        "archive_digest": cached_archive_sha in {None, archive_sha},
    }
    if phonopy_metrics is not None:
        checks.update(
            {
                "phonopy_born_source": phonopy_metrics["born_source_relative_error"] <= phonopy_born_tolerance,
                "phonopy_outcar_force_constants": phonopy_metrics["outcar_force_constants_relative_error"] <= phonopy_outcar_force_tolerance,
                "phonopy_epsilon": phonopy_metrics["epsilon_relative_error"] <= phonopy_dielectric_tolerance,
            }
        )
    elif require_phonopy:
        checks["phonopy_available"] = False
    return {
        "jid": jid,
        "archive_sha256": archive_sha,
        "cached_archive_sha256": cached_archive_sha,
        "cache_schema": payload.get("schema"),
        "pymatgen_parser": "pymatgen.io.vasp.outputs.Vasprun/Outcar",
        "raw_dynamical_matrix_relative_error": raw_relative,
        "born_charges_source_relative_error": born_relative,
        "total_outcar_piezo_relative_error": total_relative,
        "phonopy": phonopy_metrics if phonopy_metrics is not None else {"status": "unavailable"},
        "checks": checks,
        "accepted": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dfpt-dir", type=Path, required=True)
    parser.add_argument("--strict-completion-dir", type=Path, required=True)
    parser.add_argument("--index-cache-dir", type=Path, default=Path.home() / ".cache" / "piezojet" / "jarvis_raw_files")
    parser.add_argument("--material-ids-file", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--raw-matrix-tolerance", type=float, default=5e-5)
    parser.add_argument("--born-tolerance", type=float, default=5e-5)
    parser.add_argument("--total-piezo-tolerance", type=float, default=5e-5)
    parser.add_argument("--phonopy-outcar-force-tolerance", type=float, default=1e-2)
    parser.add_argument("--phonopy-born-tolerance", type=float, default=5e-5)
    parser.add_argument("--phonopy-dielectric-tolerance", type=float, default=5e-5)
    parser.add_argument("--require-phonopy", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/pymatgen_parser_crosscheck_v1/report.json"))
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if min(
        args.raw_matrix_tolerance,
        args.born_tolerance,
        args.total_piezo_tolerance,
        args.phonopy_outcar_force_tolerance,
        args.phonopy_born_tolerance,
        args.phonopy_dielectric_tolerance,
    ) <= 0:
        raise ValueError("All parser tolerances must be positive")
    if args.material_ids_file is not None:
        parsed = json.loads(args.material_ids_file.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            parsed = parsed.get("material_ids")
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("--material-ids-file must contain a non-empty JSON material_ids list")
        material_ids = [str(value) for value in parsed]
    else:
        material_ids = []
        for path in sorted(args.strict_completion_dir.glob("JVASP-*.pt")):
            completion = torch.load(path, map_location="cpu", weights_only=False)
            if bool(completion.get("audit", {}).get("accepted", False)):
                material_ids.append(str(completion["jid"]))
        material_ids = material_ids[: args.limit]
    if not material_ids:
        raise ValueError("No material IDs selected for parser cross-validation")
    known_ids = {str(record["JARVIS_ID"]) for record in load_gmtnet_records(args.data_root)}
    unknown = sorted(set(material_ids) - known_ids)
    if unknown:
        raise ValueError(f"Unknown JARVIS IDs: {unknown[:5]}")
    cache, index = JarvisDFPTCache(args.dfpt_dir), JarvisRawFileIndex(args.index_cache_dir)
    rows = []
    for position, jid in enumerate(material_ids, start=1):
        try:
            row = crossvalidate_one(
                jid=jid,
                cache=cache,
                index=index,
                timeout=args.timeout,
                raw_matrix_tolerance=args.raw_matrix_tolerance,
                born_tolerance=args.born_tolerance,
                total_piezo_tolerance=args.total_piezo_tolerance,
                phonopy_outcar_force_tolerance=args.phonopy_outcar_force_tolerance,
                phonopy_born_tolerance=args.phonopy_born_tolerance,
                phonopy_dielectric_tolerance=args.phonopy_dielectric_tolerance,
                require_phonopy=args.require_phonopy,
            )
        except Exception as error:
            row = {"jid": jid, "accepted": False, "error": str(error)}
        rows.append(row)
        print(f"[{position}/{len(material_ids)}] {jid}: accepted={row['accepted']}")
    result = {
        "schema": 1,
        "scope": "independent pymatgen/phonopy parser cross-validation of public JARVIS archives",
        "limitations": [
            "Phonopy parses the same VASP XML/OUTCAR text; this is not a finite-displacement calculation.",
            "This audit cannot establish Gamma non-analytic/LO-TO boundary semantics.",
            "Pymatgen exposes the total OUTCAR piezo block, not the ionic/total decomposition used by jarvis-tools.",
        ],
        "selected_materials": material_ids,
        "tolerances": {
            "raw_dynamical_matrix_relative": args.raw_matrix_tolerance,
            "born_charges_source_relative": args.born_tolerance,
            "total_outcar_piezo_relative": args.total_piezo_tolerance,
            "phonopy_outcar_force_constants_relative": args.phonopy_outcar_force_tolerance,
            "phonopy_born_source_relative": args.phonopy_born_tolerance,
            "phonopy_epsilon_relative": args.phonopy_dielectric_tolerance,
        },
        "rows": rows,
        "accepted": sum(bool(row.get("accepted", False)) for row in rows),
        "requested": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["accepted"] != result["requested"]:
        raise RuntimeError(f"Parser cross-validation failed for {result['requested'] - result['accepted']} archives")


if __name__ == "__main__":
    main()
