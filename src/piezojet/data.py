"""GMTNet piezoelectric ingestion, persistent splitting, and periodic graphs."""

from __future__ import annotations

import json
import pickle
from uuid import uuid4
import random
from hashlib import sha256
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import torch
from pymatgen.core import Element
from torch_geometric.data import Data, Dataset

from .jarvis_dfpt import JarvisDFPTCache
from .projector import get_cartesian_point_group_operations, project_piezo_to_point_group
from .tensor_ops import cartesian_to_piezo_voigt, piezo_voigt_to_cartesian, source_voigt_to_canonical


PIEZO_FILE = "jarvis_diele_piezo.pkl"
PIEZO_FIELD = "piezoelectric_C_m2"
GRAPH_CACHE_SCHEMA = 3
SPLIT_SCHEMA = 2
SYMMETRY_TARGET_CACHE_SCHEMA = 2
RESPONSE_NORM_BOUNDS = (0.0, 0.05, 0.5, 1.0)
MAX_POINT_GROUP_OPERATIONS = 48


def load_gmtnet_records(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root) / "data" / PIEZO_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Expected official GMTNet piezoelectric file: {path}")
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty list in {path}")
    required = {"JARVIS_ID", "atoms", PIEZO_FIELD}
    missing = required.difference(records[0])
    if missing:
        raise ValueError(f"Missing required GMTNet fields: {sorted(missing)}")
    valid: list[dict[str, Any]] = []
    for record in records:
        value = record.get(PIEZO_FIELD)
        tensor = torch.as_tensor(value, dtype=torch.float32) if value is not None else None
        if tensor is None or tensor.shape != (3, 6):
            continue
        # Match GMTNet_piezo/data.py's documented screening exactly.
        if not torch.isfinite(tensor).all() or tensor.abs().max() >= 100:
            continue
        valid.append(record)
    if not valid:
        raise ValueError("No finite 3x6 piezoelectric records found")
    return valid


def formula(record: dict[str, Any]) -> str:
    counts = Counter(record["atoms"]["elements"])
    return "".join(f"{element}{counts[element] if counts[element] != 1 else ''}" for element in sorted(counts))


def response_norm_bin(target: torch.Tensor) -> int:
    """Return a rotation-invariant response stratum for one Cartesian tensor."""
    norm = float(torch.linalg.vector_norm(target))
    if norm == 0.0:
        return 0
    if norm < RESPONSE_NORM_BOUNDS[1]:
        return 1
    if norm < RESPONSE_NORM_BOUNDS[2]:
        return 2
    if norm < RESPONSE_NORM_BOUNDS[3]:
        return 3
    return 4


def _raw_cartesian_target(record: dict[str, Any]) -> torch.Tensor:
    return piezo_voigt_to_cartesian(source_voigt_to_canonical(torch.tensor(record[PIEZO_FIELD], dtype=torch.float32)))


def _formula_group_stratified_splits(records: list[dict[str, Any]], seed: int) -> dict[str, list[str]]:
    """Assign whole formulas while matching zero/weak/high response counts.

    This is a deterministic greedy group-stratification procedure.  It avoids
    composition/formula leakage across splits while matching five invariant
    target-norm bins as closely as the indivisible formula groups permit.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(formula(record), []).append(record)
    names, fractions = ("train", "val", "test"), (0.8, 0.1, 0.1)
    total = torch.zeros(5, dtype=torch.float64)
    grouped: list[tuple[str, list[dict[str, Any]], torch.Tensor]] = []
    for key, members in groups.items():
        counts = torch.zeros(5, dtype=torch.float64)
        for record in members:
            counts[response_norm_bin(_raw_cartesian_target(record))] += 1
        total += counts
        grouped.append((key, members, counts))
    desired = {name: total * fraction for name, fraction in zip(names, fractions)}
    assigned = {name: torch.zeros(5, dtype=torch.float64) for name in names}
    split_ids = {name: [] for name in names}
    rng = random.Random(seed)
    rng.shuffle(grouped)
    # Rare, high-response groups are placed first, then groups with more
    # records.  This prevents the easy zero-heavy groups consuming capacity.
    grouped.sort(key=lambda item: (int((item[2][1:] > 0).sum()), float(item[2][1:].sum()), len(item[1])), reverse=True)
    for _, members, counts in grouped:
        best_name, best_score = None, None
        for name in names:
            proposal = assigned[name] + counts
            # Bin matching is meaningful only after split sizes are controlled.
            # Formula groups are small relative to the corpus, so a dominant
            # size term keeps the 80/10/10 contract while the bin term breaks
            # ties toward response-stratified allocation.
            bin_score = ((proposal - desired[name]).square() / desired[name].clamp_min(1.0)).sum()
            # Allocate the next indivisible formula group to the least-filled
            # split in relative terms.  The response term resolves close ties
            # without allowing the small validation/test partitions to grow.
            size_score = 1000.0 * proposal.sum() / desired[name].sum().clamp_min(1.0)
            score = float(size_score + 0.01 * bin_score)
            if best_score is None or score < best_score:
                best_name, best_score = name, score
        assert best_name is not None
        assigned[best_name] += counts
        split_ids[best_name].extend(str(record["JARVIS_ID"]) for record in members)
    return split_ids


def create_or_load_splits(records: list[dict[str, Any]], processed_dir: str | Path, seed: int = 42) -> dict[str, list[str]]:
    path = Path(processed_dir) / f"splits_formula_stratified_v{SPLIT_SCHEMA}.json"
    ids = sorted(str(record["JARVIS_ID"]) for record in records)
    if path.is_file():
        splits = json.loads(path.read_text(encoding="utf-8"))
        restored = sorted(splits.get("train", []) + splits.get("val", []) + splits.get("test", []))
        if restored != ids:
            raise ValueError(f"Existing split {path} does not match the GMTNet records")
        return splits
    splits = _formula_group_stratified_splits(records, seed)
    restored = sorted(splits["train"] + splits["val"] + splits["test"])
    if restored != ids:
        raise RuntimeError("Formula-stratified split did not preserve the record population")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    return splits


def _periodic_edges(frac: torch.Tensor, cell: torch.Tensor, cutoff: float, max_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
    if frac.ndim != 2 or frac.shape[-1] != 3 or cell.shape != (3, 3):
        raise ValueError("Expected fractional coordinates [N,3] and lattice [3,3]")
    shortest = torch.linalg.vector_norm(cell, dim=-1).min().item()
    if shortest <= 0:
        raise ValueError("Lattice vectors must be nonzero")
    span = int(cutoff / shortest) + 1
    shifts = torch.tensor(list(product(range(-span, span + 1), repeat=3)), dtype=frac.dtype, device=frac.device)
    shift_cartesian = shifts @ cell
    shift_count, atoms = shifts.shape[0], frac.shape[0]
    candidates_per_target = atoms * shift_count
    selected_per_target = min(max_neighbors, candidates_per_target)
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    edge_shifts: list[torch.Tensor] = []
    # Construct all source/image candidates for a small block of targets at
    # once.  This replaces the former Python target--source--image triple loop
    # with vectorized distance evaluation and top-k selection, while bounding
    # peak memory for large unit cells.
    for start in range(0, atoms, 32):
        stop = min(start + 32, atoms)
        delta_fractional = (
            frac.unsqueeze(0).unsqueeze(2)
            - frac[start:stop].unsqueeze(1).unsqueeze(2)
            + shifts.unsqueeze(0).unsqueeze(0)
        )
        distances = torch.linalg.vector_norm(delta_fractional @ cell, dim=-1)
        valid = (distances > 1e-7) & (distances <= cutoff)
        scores = distances.masked_fill(~valid, float("inf")).reshape(stop - start, -1)
        # Stable sorting reproduces the former Python ``sorted`` tie order
        # (source-major, then periodic-image-major) for high-symmetry shells.
        flat_indices = torch.argsort(scores, dim=-1, stable=True)[..., :selected_per_target]
        nearest = scores.gather(-1, flat_indices)
        keep = torch.isfinite(nearest)
        local_targets = torch.arange(start, stop, dtype=torch.long, device=frac.device).unsqueeze(-1).expand_as(flat_indices)
        source_indices = torch.div(flat_indices, shift_count, rounding_mode="floor")
        shift_indices = flat_indices.remainder(shift_count)
        sources.append(source_indices[keep])
        targets.append(local_targets[keep])
        edge_shifts.append(shift_cartesian[shift_indices[keep]])
    if not sources or not any(source.numel() for source in sources):
        raise ValueError("No periodic neighbors found; increase cutoff")
    return torch.stack((torch.cat(sources), torch.cat(targets))), torch.cat(edge_shifts)


def record_to_graph(record: dict[str, Any], cutoff: float, max_neighbors: int) -> Data:
    atoms = record["atoms"]
    if atoms.get("cartesian") is not False:
        raise ValueError("GMTNet atoms must use fractional coordinates (cartesian=False)")
    frac = torch.tensor(atoms["coords"], dtype=torch.float32)
    cell = torch.tensor(atoms["lattice_mat"], dtype=torch.float32)
    z = torch.tensor([Element(symbol).Z for symbol in atoms["elements"]], dtype=torch.long)
    edge_index, edge_shift = _periodic_edges(frac, cell, cutoff, max_neighbors)
    target_voigt = cartesian_to_piezo_voigt(_raw_cartesian_target(record))
    dielectric_value = record.get("dielectric")
    dielectric = torch.as_tensor(
        [] if dielectric_value is None else dielectric_value,
        dtype=torch.float32,
    )
    has_dielectric = dielectric.shape == (3, 3) and bool(torch.isfinite(dielectric).all())
    return Data(
        z=z,
        # Fractional coordinates are retained for the reciprocal-space global
        # context; Cartesian positions remain the input to local e3nn messages.
        frac=frac,
        pos=frac @ cell,
        cell=cell.unsqueeze(0),
        edge_index=edge_index,
        edge_shift=edge_shift,
        y=piezo_voigt_to_cartesian(target_voigt).unsqueeze(0),
        y_voigt=target_voigt.unsqueeze(0),
        y_dielectric=(dielectric if has_dielectric else torch.zeros(3, 3, dtype=torch.float32)).unsqueeze(0),
        dielectric_mask=torch.tensor(has_dielectric, dtype=torch.bool),
        material_id=str(record["JARVIS_ID"]),
        num_nodes=z.numel(),
    )


class SymmetryTargetCache:
    """Persistent Reynolds-projected piezoelectric labels.

    The source tensors are already largely symmetry compliant, but projection
    removes the small residual inconsistency before it enters supervision.
    Target projection is computed once per structure and cached independently
    of the geometry graph cache.
    """

    def __init__(self, processed_dir: str | Path, symprec: float = 1e-5):
        self.directory = Path(processed_dir) / f"piezo_symmetry_targets_v{SYMMETRY_TARGET_CACHE_SCHEMA}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.symprec = symprec

    def path(self, record: dict[str, Any]) -> Path:
        material_id = str(record["JARVIS_ID"])
        return self.directory / f"{sha256(material_id.encode('utf-8')).hexdigest()[:16]}.pt"

    def payload(self, record: dict[str, Any]) -> dict[str, torch.Tensor | float | int]:
        path = self.path(record)
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema") == SYMMETRY_TARGET_CACHE_SCHEMA and "rotations" in payload:
                return payload
        raw = _raw_cartesian_target(record)
        rotations = get_cartesian_point_group_operations(record, self.symprec).to(dtype=raw.dtype)
        if rotations.shape[0] > MAX_POINT_GROUP_OPERATIONS:
            raise ValueError(f"Point group has {rotations.shape[0]} operations; maximum is {MAX_POINT_GROUP_OPERATIONS}")
        projected = project_piezo_to_point_group(raw, rotations)
        residual = torch.linalg.vector_norm(raw - projected) / torch.linalg.vector_norm(raw).clamp_min(1e-12)
        temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        payload = {"schema": SYMMETRY_TARGET_CACHE_SCHEMA, "target": projected, "rotations": rotations, "residual": float(residual)}
        torch.save(payload, temporary)
        temporary.replace(path)
        return payload

    def get(self, record: dict[str, Any]) -> torch.Tensor:
        return self.payload(record)["target"]

    def rotations(self, record: dict[str, Any]) -> torch.Tensor:
        return self.payload(record)["rotations"]


def pad_point_group_operations(rotations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a crystal point group to a batchable fixed-size Reynolds projector."""
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape [G,3,3]")
    if rotations.shape[0] > MAX_POINT_GROUP_OPERATIONS:
        raise ValueError(f"Point group has {rotations.shape[0]} operations; maximum is {MAX_POINT_GROUP_OPERATIONS}")
    padded = torch.eye(3, dtype=rotations.dtype).expand(MAX_POINT_GROUP_OPERATIONS, 3, 3).clone()
    padded[: rotations.shape[0]] = rotations
    mask = torch.zeros(MAX_POINT_GROUP_OPERATIONS, dtype=torch.bool)
    mask[: rotations.shape[0]] = True
    return padded, mask


def is_polar_point_group(rotations: torch.Tensor, tolerance: float = 1e-5) -> bool:
    """Whether a point group admits a nonzero invariant polar vector.

    The Reynolds average of a vector is the projector onto the point group's
    polar invariant subspace.  This separates polar point groups from the
    nonpolar-but-piezoelectric classes without relying on a hand-written
    international-symbol lookup.
    """
    projector = rotations.mean(dim=0)
    return bool(torch.linalg.matrix_norm(projector) > tolerance)


def precompute_symmetry_targets(records: list[dict[str, Any]], processed_dir: str | Path) -> Path:
    """Materialize symmetry-projected targets before a long training job."""
    cache = SymmetryTargetCache(processed_dir)
    for record in records:
        cache.get(record)
    return cache.directory


def graph_cache_key(records: list[dict[str, Any]], cutoff: float, max_neighbors: int) -> str:
    """Hash graph-defining geometry and construction parameters, not labels."""
    digest = sha256()
    digest.update(f"schema={GRAPH_CACHE_SCHEMA};cutoff={cutoff:.8g};max_neighbors={max_neighbors}".encode("utf-8"))
    for record in records:
        atoms = record["atoms"]
        payload = {
            "id": str(record["JARVIS_ID"]),
            "elements": atoms["elements"],
            "coords": atoms["coords"],
            "lattice_mat": atoms["lattice_mat"],
            "cartesian": atoms.get("cartesian"),
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()[:20]


class PersistentGraphCache:
    """Atomic, versioned CPU graph cache reusable across Python processes."""

    def __init__(self, processed_dir: str | Path, records: list[dict[str, Any]], cutoff: float, max_neighbors: int, cache_key: str | None = None):
        self.key = cache_key or graph_cache_key(records, cutoff, max_neighbors)
        self.directory = Path(processed_dir) / "pbc_graph_cache" / self.key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cutoff, self.max_neighbors = cutoff, max_neighbors
        if not (self.directory / "manifest.json").exists():
            self.write_manifest(records)

    def write_manifest(self, records: list[dict[str, Any]]) -> None:
        (self.directory / "manifest.json").write_text(json.dumps({
            "schema": GRAPH_CACHE_SCHEMA,
            "cache_key": self.key,
            "cutoff": self.cutoff,
            "max_neighbors": self.max_neighbors,
            "graph_count": len(records),
            "material_ids": [str(record["JARVIS_ID"]) for record in records],
        }, indent=2) + "\n", encoding="utf-8")

    def path(self, record: dict[str, Any]) -> Path:
        material_id = str(record["JARVIS_ID"])
        return self.directory / f"{sha256(material_id.encode('utf-8')).hexdigest()[:16]}.pt"

    def load(self, record: dict[str, Any]) -> Data | None:
        path = self.path(record)
        return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None

    def has(self, record: dict[str, Any]) -> bool:
        return self.path(record).exists()

    def save(self, record: dict[str, Any], graph: Data) -> None:
        path = self.path(record)
        # Concurrent training/precomputation processes may request the same
        # missing graph.  A unique sibling keeps each atomic replacement
        # independent on Windows rather than racing on one fixed ``.tmp``.
        temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        torch.save(graph, temporary)
        temporary.replace(path)


def precompute_pbc_graphs(records: list[dict[str, Any]], processed_dir: str | Path, cutoff: float, max_neighbors: int) -> Path:
    """Materialize the persistent graph cache and return its versioned directory."""
    cache = PersistentGraphCache(processed_dir, records, cutoff, max_neighbors)
    cache.write_manifest(records)
    for record in records:
        if not cache.has(record):
            cache.save(record, record_to_graph(record, cutoff, max_neighbors))
    return cache.directory


class PiezoDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], ids: list[str], cutoff: float, max_neighbors: int, processed_dir: str | Path | None = None, cache_key: str | None = None, project_targets: bool = True, dfpt_dir: str | Path | None = None, strain_completion_dir: str | Path | None = None):
        super().__init__()
        wanted = set(ids)
        self.records = [record for record in records if str(record["JARVIS_ID"]) in wanted]
        if len(self.records) != len(ids):
            raise ValueError("Split contains material IDs absent from loaded GMTNet records")
        self.cutoff, self.max_neighbors = cutoff, max_neighbors
        self._graph_cache: dict[int, Data] = {}
        self._disk_cache = PersistentGraphCache(processed_dir, self.records, cutoff, max_neighbors, cache_key=cache_key) if processed_dir is not None else None
        self._target_cache = SymmetryTargetCache(processed_dir) if processed_dir is not None and project_targets else None
        self._dfpt_cache = JarvisDFPTCache(dfpt_dir) if dfpt_dir is not None else None
        self._strain_completion_dir = (
            Path(strain_completion_dir) if strain_completion_dir is not None else None
        )

    def target(self, record: dict[str, Any]) -> torch.Tensor:
        return _raw_cartesian_target(record) if self._target_cache is None else self._target_cache.get(record)

    def len(self) -> int:
        return len(self.records)

    def get(self, index: int) -> Data:
        if index not in self._graph_cache:
            record = self.records[index]
            graph = self._disk_cache.load(record) if self._disk_cache is not None else None
            if graph is None:
                graph = record_to_graph(record, self.cutoff, self.max_neighbors)
                if self._disk_cache is not None:
                    self._disk_cache.save(record, graph)
            if self._target_cache is None:
                target = _raw_cartesian_target(record)
                rotations = get_cartesian_point_group_operations(record).to(dtype=target.dtype)
            else:
                payload = self._target_cache.payload(record)
                target, rotations = payload["target"], payload["rotations"]
            graph.y = target.unsqueeze(0)
            graph.y_voigt = cartesian_to_piezo_voigt(target).unsqueeze(0)
            dielectric_value = record.get("dielectric")
            dielectric = torch.as_tensor(
                [] if dielectric_value is None else dielectric_value,
                dtype=target.dtype,
            )
            has_dielectric = dielectric.shape == (3, 3) and bool(torch.isfinite(dielectric).all())
            graph.y_dielectric = (dielectric if has_dielectric else torch.zeros(3, 3, dtype=target.dtype)).unsqueeze(0)
            graph.dielectric_mask = torch.tensor(has_dielectric, dtype=torch.bool)
            # BEC is a node-aligned tensor and can therefore be batched safely.
            # The variable-length Gamma-mode arrays are retained in flattened
            # form plus their dimensions for downstream mode-specific audits.
            dfpt = self._dfpt_cache.load(str(record["JARVIS_ID"])) if self._dfpt_cache is not None else None
            has_dfpt = dfpt is not None
            completion_path = (
                self._strain_completion_dir / f"{record['JARVIS_ID']}.pt"
                if self._strain_completion_dir is not None else None
            )
            completion = (
                torch.load(completion_path, map_location="cpu", weights_only=False)
                if completion_path is not None and completion_path.is_file() else None
            )
            if completion is not None and (
                completion.get("schema") != 2
                or completion.get("jid") != str(record["JARVIS_ID"])
                or not bool(completion.get("audit", {}).get("accepted", False))
            ):
                raise ValueError(f"Invalid strain-completion payload: {completion_path}")
            raw_born = (
                dfpt["born_charges"]
                if has_dfpt
                else torch.zeros(graph.num_nodes, 3, 3, dtype=target.dtype)
            ).to(dtype=target.dtype)
            # The Born-charge acoustic sum rule is exact for a neutral
            # periodic cell.  Public DFPT archives contain a small typical
            # residual and a few substantial upstream outliers, so supervise
            # the unique nearest zero-sum target and retain the size of the
            # correction explicitly for audit/weighting decisions.
            born_mean = raw_born.mean(dim=0, keepdim=True) if has_dfpt else torch.zeros_like(raw_born[:1])
            graph.y_born = raw_born - born_mean
            born_norm = torch.linalg.vector_norm(raw_born).clamp_min(torch.finfo(target.dtype).eps)
            born_sum = raw_born.sum(dim=0)
            graph.born_raw_asr_max_abs_e = born_sum.abs().max().reshape(1)
            graph.born_raw_asr_rel = (torch.linalg.vector_norm(born_sum) / born_norm).reshape(1)
            graph.born_projection_rel = (
                torch.linalg.vector_norm(raw_born - graph.y_born) / born_norm
            ).reshape(1)
            graph.born_mask = torch.full((graph.num_nodes,), has_dfpt, dtype=torch.bool)
            if has_dfpt:
                modes = int(dfpt["dynamical_eigenvalues"].numel())
                graph.dfpt_dynamical_eigenvalues = dfpt["dynamical_eigenvalues"].to(dtype=target.dtype)
                graph.dfpt_dynamical_eigenvectors_flat = dfpt["dynamical_eigenvectors"].reshape(-1).to(dtype=target.dtype)
                graph.dfpt_force_constants_flat = dfpt["force_constants"].reshape(-1).to(dtype=target.dtype)
                graph.force_constant_mask = torch.tensor(True, dtype=torch.bool)
                ionic_source = dfpt["ionic_piezo_source"].to(dtype=target.dtype)
                graph.y_ionic_piezo = piezo_voigt_to_cartesian(source_voigt_to_canonical(ionic_source)).unsqueeze(0)
                graph.ionic_piezo_mask = torch.tensor(True, dtype=torch.bool)
                graph.dfpt_internal_strain_flat = dfpt["internal_strain_tensors"].reshape(-1).to(dtype=target.dtype)
                graph.dfpt_internal_strain_ions = dfpt["internal_strain_ions"]
                graph.dfpt_internal_strain_directions = dfpt["internal_strain_directions"]
                graph.dfpt_internal_strain_count = torch.tensor(
                    [dfpt["internal_strain_tensors"].shape[0]], dtype=torch.long
                )
            else:
                modes = 0
                graph.dfpt_dynamical_eigenvalues = torch.empty(0, dtype=target.dtype)
                graph.dfpt_dynamical_eigenvectors_flat = torch.empty(0, dtype=target.dtype)
                graph.dfpt_force_constants_flat = torch.empty(0, dtype=target.dtype)
                graph.force_constant_mask = torch.tensor(False, dtype=torch.bool)
                graph.y_ionic_piezo = torch.zeros(1, 3, 3, 3, dtype=target.dtype)
                graph.ionic_piezo_mask = torch.tensor(False, dtype=torch.bool)
                graph.dfpt_internal_strain_flat = torch.empty(0, dtype=target.dtype)
                graph.dfpt_internal_strain_ions = torch.empty(0, dtype=torch.long)
                graph.dfpt_internal_strain_directions = torch.empty(0, dtype=torch.long)
                graph.dfpt_internal_strain_count = torch.tensor([0], dtype=torch.long)
            if completion is not None:
                full_internal = completion["internal_strain_full"].to(dtype=target.dtype)
                if full_internal.shape != (graph.num_nodes, 3, 3, 3):
                    raise ValueError(f"Invalid completed strain-force shape: {completion_path}")
                graph.dfpt_internal_strain_full = full_internal
                graph.internal_strain_full_mask = torch.tensor(True, dtype=torch.bool)
            else:
                graph.dfpt_internal_strain_full = torch.zeros(
                    graph.num_nodes, 3, 3, 3, dtype=target.dtype
                )
                graph.internal_strain_full_mask = torch.tensor(False, dtype=torch.bool)
            graph.dfpt_mode_count = torch.tensor([modes], dtype=torch.long)
            graph.point_group_ops, graph.point_group_mask = pad_point_group_operations(rotations)
            graph.is_polar_point_group = torch.tensor(is_polar_point_group(rotations), dtype=torch.bool)
            self._graph_cache[index] = graph
        return self._graph_cache[index]
