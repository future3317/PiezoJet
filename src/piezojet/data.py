"""GMTNet piezoelectric ingestion, persistent splitting, and periodic graphs."""

from __future__ import annotations

import json
import math
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
from .gmtnet_io import (
    PIEZO_FIELD,
    PIEZO_FILE as _PIEZO_FILE,
    load_gmtnet_records as _load_gmtnet_records,
)
from .symmetry_projection import (
    get_cartesian_point_group_operations,
    project_piezo_to_point_group,
)
from .tensor_ops import cartesian_to_piezo_voigt, piezo_voigt_to_cartesian, source_voigt_to_canonical


GRAPH_CACHE_SCHEMA = 5
LAMBDA_LABEL_TYPES = (
    "full_dfpt",
    "full_finite_strain",
    "strict_completion",
    "joint_identifiable",
    "partial_blocks",
    "macro_only",
)
FULL_LAMBDA_LABEL_TYPES = frozenset(LAMBDA_LABEL_TYPES[:4])
LAMBDA_LABEL_TYPE_CODES = {name: index for index, name in enumerate(LAMBDA_LABEL_TYPES)}


def completion_label_metadata(completion: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Validate a persisted full-Lambda label and expose its certificate."""
    schema = completion.get("schema")
    if schema == 2:
        label_type = "strict_completion"
        certificate = completion.get("audit", {})
    elif schema == 3:
        label_type = completion.get("lambda_label_type")
        certificate = completion.get("identifiability", {})
    else:
        raise ValueError(f"Unsupported strain-completion schema: {schema}")
    if label_type not in FULL_LAMBDA_LABEL_TYPES:
        raise ValueError(
            "A completion payload may contain internal_strain_full only for an "
            "atom-resolved or certified full-Lambda label"
        )
    if not bool(completion.get("audit", {}).get("accepted", False)):
        raise ValueError("Full-Lambda completion payload is not accepted")
    return str(label_type), certificate


SPLIT_SCHEMA = 3
SYMMETRY_TARGET_CACHE_SCHEMA = 2
RESPONSE_NORM_BOUNDS = (0.0, 0.05, 0.5, 1.0)
MAX_POINT_GROUP_OPERATIONS = 48
# Stable public re-exports retained for the data-ingestion surface.
PIEZO_FILE = _PIEZO_FILE
load_gmtnet_records = _load_gmtnet_records


def deterministic_subset(values: list[str], limit: int, seed: int) -> list[str]:
    """Return a seeded subset while preserving the source-list order."""
    if limit <= 0 or limit >= len(values):
        return list(values)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(values), generator=generator)[:limit].tolist()
    return [values[index] for index in sorted(order)]


def formula(record: dict[str, Any]) -> str:
    """Canonical reduced composition, independent of unit-cell multiplicity."""
    counts = Counter(record["atoms"]["elements"])
    divisor = math.gcd(*counts.values())
    return "".join(
        f"{element}{counts[element] // divisor if counts[element] // divisor != 1 else ''}"
        for element in sorted(counts)
    )


def response_norm_bin(target: torch.Tensor) -> int:
    """Return a rotation-invariant response stratum for one Cartesian tensor.

    For ``e_ijk=e_ikj``, the Cartesian Frobenius metric includes both symmetric
    shear entries. This is not double counting: it is exactly the Euclidean
    norm of the orthonormal 18-dimensional irrep coordinates. An unweighted
    six-column engineering-Voigt norm would not be rotation invariant.
    """
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
    """Construct periodic edges without splitting a degenerate distance shell.

    ``max_neighbors`` is a target budget, not a hard symmetry-breaking cap.
    When the final retained neighbor belongs to an equal-distance shell, every
    member of that shell is retained.  This keeps cubic and other
    high-symmetry edge multisets closed under atom permutations and point-group
    operations while still bounding generic, nondegenerate environments.
    """
    if frac.ndim != 2 or frac.shape[-1] != 3 or cell.shape != (3, 3):
        raise ValueError("Expected fractional coordinates [N,3] and lattice [3,3]")
    if not torch.isfinite(cell).all() or abs(float(torch.linalg.det(cell))) <= 1e-12:
        raise ValueError("Lattice must be finite and nonsingular")
    # For r=(n+delta)L with each delta component in (-1,1),
    # |n_i| <= cutoff*||L^{-1}_{:,i}||+1.  This reciprocal-basis bound is
    # complete for arbitrarily skew cells; the shortest direct lattice-vector
    # norm is not, because large integer combinations can cancel.
    inverse = torch.linalg.inv(cell)
    spans = [
        max(1, int(math.ceil(cutoff * float(torch.linalg.vector_norm(inverse[:, axis])) + 1.0)))
        for axis in range(3)
    ]
    shifts = torch.tensor(
        list(product(*(range(-span, span + 1) for span in spans))),
        dtype=frac.dtype,
        device=frac.device,
    )
    shift_cartesian = shifts @ cell
    shift_count, atoms = shifts.shape[0], frac.shape[0]
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")
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
        sorted_indices = torch.argsort(scores, dim=-1, stable=True)
        for local_index, target_index in enumerate(range(start, stop)):
            ordered = sorted_indices[local_index]
            ordered_distances = scores[local_index, ordered]
            finite = torch.isfinite(ordered_distances)
            ordered = ordered[finite]
            ordered_distances = ordered_distances[finite]
            if ordered.numel() == 0:
                continue
            if ordered.numel() > max_neighbors:
                boundary = ordered_distances[max_neighbors - 1]
                # Float32 coordinate conversion introduces roundoff at about
                # 1e-7 relative scale.  The tolerance is used only to retain
                # an already-degenerate boundary shell, never to exceed the
                # physical radial cutoff.
                shell_tolerance = torch.maximum(
                    boundary.abs() * 1e-6,
                    boundary.new_tensor(1e-7),
                )
                ordered = ordered[ordered_distances <= boundary + shell_tolerance]
            source_indices = torch.div(ordered, shift_count, rounding_mode="floor")
            shift_indices = ordered.remainder(shift_count)
            sources.append(source_indices)
            targets.append(torch.full_like(source_indices, target_index))
            edge_shifts.append(shift_cartesian[shift_indices])
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
    def __init__(self, records: list[dict[str, Any]], ids: list[str], cutoff: float, max_neighbors: int, processed_dir: str | Path | None = None, cache_key: str | None = None, project_targets: bool = True, dfpt_dir: str | Path | None = None, strain_completion_dir: str | Path | None = None, elastic_targets_path: str | Path | None = None, dfpt_total_consistency_relative_tolerance: float = 0.05, dfpt_total_consistency_absolute_tolerance: float = 0.05, dfpt_profile: str = "full"):
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
        if dfpt_profile not in {"full", "electrostatic"}:
            raise ValueError("dfpt_profile must be 'full' or 'electrostatic'")
        # Electrostatic-generator experiments require only BEC and electronic
        # response labels.  Retaining each material's force constants and
        # dynamical eigenvectors in the in-memory graph cache can consume many
        # GiB without contributing to their objective.
        self._dfpt_profile = dfpt_profile
        self._strain_completion_dir = (
            Path(strain_completion_dir) if strain_completion_dir is not None else None
        )
        self._elastic_targets: dict[str, torch.Tensor] = {}
        if elastic_targets_path is not None:
            payload = torch.load(elastic_targets_path, map_location="cpu", weights_only=False)
            if payload.get("schema") != 1 or payload.get("unit") != "GPa" or not isinstance(payload.get("targets"), dict):
                raise ValueError("Elastic auxiliary payload must be schema-1 GPa targets")
            self._elastic_targets = {str(jid): torch.as_tensor(value) for jid, value in payload["targets"].items()}
        if dfpt_total_consistency_relative_tolerance < 0 or dfpt_total_consistency_absolute_tolerance < 0:
            raise ValueError("DFPT/GMTNet total-piezo consistency tolerances must be non-negative")
        self._dfpt_total_consistency_relative_tolerance = float(dfpt_total_consistency_relative_tolerance)
        self._dfpt_total_consistency_absolute_tolerance = float(dfpt_total_consistency_absolute_tolerance)

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
            elastic_target = self._elastic_targets.get(str(record["JARVIS_ID"]))
            has_elastic = elastic_target is not None and elastic_target.shape == (6, 6) and bool(torch.isfinite(elastic_target).all())
            graph.y_elastic_gpa = (
                elastic_target.to(dtype=target.dtype) if has_elastic else torch.zeros(6, 6, dtype=target.dtype)
            ).unsqueeze(0)
            graph.elastic_mask = torch.tensor(has_elastic, dtype=torch.bool)
            # BEC is a node-aligned tensor and can therefore be batched safely.
            # The variable-length Gamma-mode arrays are retained in flattened
            # form plus their dimensions for downstream mode-specific audits.
            dfpt = self._dfpt_cache.load(str(record["JARVIS_ID"])) if self._dfpt_cache is not None else None
            has_dfpt = dfpt is not None
            completion_path = (
                self._strain_completion_dir / f"{record['JARVIS_ID']}.pt"
                if self._dfpt_profile == "full" and self._strain_completion_dir is not None else None
            )
            completion = (
                torch.load(completion_path, map_location="cpu", weights_only=False)
                if completion_path is not None and completion_path.is_file() else None
            )
            completion_metadata = None
            if completion is not None:
                if completion.get("jid") != str(record["JARVIS_ID"]):
                    raise ValueError(f"Invalid strain-completion payload: {completion_path}")
                try:
                    completion_metadata = completion_label_metadata(completion)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid strain-completion payload: {completion_path}: {error}"
                    ) from error
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
            if self._dfpt_profile == "electrostatic":
                if not has_dfpt:
                    raise ValueError(
                        "The electrostatic profile requires a DFPT payload for "
                        f"{record['JARVIS_ID']}"
                    )
                ionic_source = dfpt["ionic_piezo_source"].to(dtype=target.dtype)
                ionic_cartesian = piezo_voigt_to_cartesian(
                    source_voigt_to_canonical(ionic_source)
                )
                graph.y_ionic_piezo = project_piezo_to_point_group(
                    ionic_cartesian, rotations
                ).unsqueeze(0)
                total_source = dfpt["total_piezo_source"].to(dtype=target.dtype)
                total_cartesian = piezo_voigt_to_cartesian(
                    source_voigt_to_canonical(total_source)
                )
                graph.y_dfpt_total_piezo = project_piezo_to_point_group(
                    total_cartesian, rotations
                ).unsqueeze(0)
                graph.y_electronic_piezo = (
                    graph.y_dfpt_total_piezo - graph.y_ionic_piezo
                )
                electronic_dielectric = torch.as_tensor(
                    dfpt.get("epsilon", {}).get("epsilon", []),
                    dtype=target.dtype,
                )
                has_electronic_dielectric = (
                    electronic_dielectric.shape == (3, 3)
                    and bool(torch.isfinite(electronic_dielectric).all())
                )
                if has_electronic_dielectric:
                    electronic_dielectric = 0.5 * (
                        electronic_dielectric + electronic_dielectric.transpose(-1, -2)
                    )
                    electronic_dielectric = torch.einsum(
                        "rij,jk,rlk->il",
                        rotations,
                        electronic_dielectric,
                        rotations,
                    ) / rotations.shape[0]
                else:
                    electronic_dielectric = torch.zeros(3, 3, dtype=target.dtype)
                graph.y_dfpt_electronic_dielectric = electronic_dielectric.unsqueeze(0)
                graph.dfpt_electronic_dielectric_mask = torch.tensor(
                    has_electronic_dielectric, dtype=torch.bool
                )
                graph.point_group_ops, graph.point_group_mask = pad_point_group_operations(rotations)
                graph.is_polar_point_group = torch.tensor(
                    is_polar_point_group(rotations), dtype=torch.bool
                )
                self._graph_cache[index] = graph
                return graph
            if has_dfpt:
                modes = int(dfpt["dynamical_eigenvalues"].numel())
                graph.dfpt_dynamical_eigenvalues = dfpt["dynamical_eigenvalues"].to(dtype=target.dtype)
                graph.dfpt_dynamical_eigenvectors_flat = dfpt["dynamical_eigenvectors"].reshape(-1).to(dtype=target.dtype)
                graph.dfpt_force_constants_flat = dfpt["force_constants"].reshape(-1).to(dtype=target.dtype)
                graph.force_constant_mask = torch.tensor(True, dtype=torch.bool)
                ionic_source = dfpt["ionic_piezo_source"].to(dtype=target.dtype)
                ionic_cartesian = piezo_voigt_to_cartesian(
                    source_voigt_to_canonical(ionic_source)
                )
                graph.y_ionic_piezo = project_piezo_to_point_group(
                    ionic_cartesian, rotations
                ).unsqueeze(0)
                graph.ionic_piezo_mask = torch.tensor(True, dtype=torch.bool)
                # These three branch labels all come from the same JARVIS
                # OUTCAR.  They are intentionally distinct from the GMTNet
                # corpus target: the latter remains the all-material total
                # response objective, while branch supervision never mixes
                # sources or invents an electronic label for a missing archive.
                total_source = dfpt["total_piezo_source"].to(dtype=target.dtype)
                total_cartesian = piezo_voigt_to_cartesian(
                    source_voigt_to_canonical(total_source)
                )
                graph.y_dfpt_total_piezo = project_piezo_to_point_group(
                    total_cartesian, rotations
                ).unsqueeze(0)
                graph.y_electronic_piezo = graph.y_dfpt_total_piezo - graph.y_ionic_piezo
                # Every branch is now in the same Reynolds-projected target
                # space as the GMTNet objective. Linearity guarantees
                # P_G(e_el)+P_G(e_ion)=P_G(e_total) to floating-point roundoff.
                # A material for which the two projected training totals still
                # disagree beyond both physical tolerances cannot safely
                # receive simultaneous macro ionic/electronic/branch losses.
                # Keep its GMTNet total target, retain the source arrays for
                # forensic reporting, and mask only the conflicting DFPT
                # response supervision until its structure/response symmetry
                # convention is resolved.
                total_difference = torch.linalg.vector_norm(
                    target - graph.y_dfpt_total_piezo.squeeze(0)
                )
                total_scale = torch.linalg.vector_norm(
                    graph.y_dfpt_total_piezo.squeeze(0)
                ).clamp_min(0.05 * (18.0 ** 0.5))
                graph.dfpt_total_consistency_abs_c_per_m2 = total_difference.reshape(1)
                graph.dfpt_total_consistency_rel = (total_difference / total_scale).reshape(1)
                consistent_total = not bool(
                    (total_difference > self._dfpt_total_consistency_absolute_tolerance)
                    & (total_difference / total_scale > self._dfpt_total_consistency_relative_tolerance)
                )
                graph.dfpt_total_consistency_mask = torch.tensor(consistent_total, dtype=torch.bool)
                graph.ionic_piezo_mask = torch.tensor(consistent_total, dtype=torch.bool)
                graph.dfpt_branch_mask = torch.tensor(consistent_total, dtype=torch.bool)
                graph.dfpt_internal_strain_flat = dfpt["internal_strain_tensors"].reshape(-1).to(dtype=target.dtype)
                graph.dfpt_internal_strain_ions = dfpt["internal_strain_ions"]
                graph.dfpt_internal_strain_directions = dfpt["internal_strain_directions"]
                graph.dfpt_internal_strain_count = torch.tensor(
                    [dfpt["internal_strain_tensors"].shape[0]], dtype=torch.long
                )
                epsilon_ion = torch.as_tensor(
                    dfpt.get("epsilon", {}).get("epsilon_ion", []), dtype=target.dtype
                )
                has_ionic_dielectric = (
                    epsilon_ion.shape == (3, 3)
                    and bool(torch.isfinite(epsilon_ion).all())
                )
                graph.y_dfpt_ionic_dielectric = (
                    0.5 * (epsilon_ion + epsilon_ion.transpose(-1, -2))
                    if has_ionic_dielectric else torch.zeros(3, 3, dtype=target.dtype)
                ).unsqueeze(0)
                graph.dfpt_ionic_dielectric_mask = torch.tensor(
                    has_ionic_dielectric, dtype=torch.bool
                )
                electronic_dielectric = torch.as_tensor(
                    dfpt.get("epsilon", {}).get("epsilon", []),
                    dtype=target.dtype,
                )
                has_electronic_dielectric = (
                    electronic_dielectric.shape == (3, 3)
                    and bool(torch.isfinite(electronic_dielectric).all())
                )
                if has_electronic_dielectric:
                    electronic_dielectric = 0.5 * (
                        electronic_dielectric + electronic_dielectric.transpose(-1, -2)
                    )
                    electronic_dielectric = torch.einsum(
                        "rij,jk,rlk->il",
                        rotations,
                        electronic_dielectric,
                        rotations,
                    ) / rotations.shape[0]
                else:
                    electronic_dielectric = torch.zeros(
                        3, 3, dtype=target.dtype
                    )
                graph.y_dfpt_electronic_dielectric = electronic_dielectric.unsqueeze(0)
                graph.dfpt_electronic_dielectric_mask = torch.tensor(
                    has_electronic_dielectric, dtype=torch.bool
                )
            else:
                modes = 0
                graph.dfpt_dynamical_eigenvalues = torch.empty(0, dtype=target.dtype)
                graph.dfpt_dynamical_eigenvectors_flat = torch.empty(0, dtype=target.dtype)
                graph.dfpt_force_constants_flat = torch.empty(0, dtype=target.dtype)
                graph.force_constant_mask = torch.tensor(False, dtype=torch.bool)
                graph.y_ionic_piezo = torch.zeros(1, 3, 3, 3, dtype=target.dtype)
                graph.ionic_piezo_mask = torch.tensor(False, dtype=torch.bool)
                graph.y_dfpt_total_piezo = torch.zeros(1, 3, 3, 3, dtype=target.dtype)
                graph.y_electronic_piezo = torch.zeros(1, 3, 3, 3, dtype=target.dtype)
                graph.dfpt_branch_mask = torch.tensor(False, dtype=torch.bool)
                graph.dfpt_total_consistency_abs_c_per_m2 = torch.zeros(1, dtype=target.dtype)
                graph.dfpt_total_consistency_rel = torch.zeros(1, dtype=target.dtype)
                graph.dfpt_total_consistency_mask = torch.tensor(False, dtype=torch.bool)
                graph.dfpt_internal_strain_flat = torch.empty(0, dtype=target.dtype)
                graph.dfpt_internal_strain_ions = torch.empty(0, dtype=torch.long)
                graph.dfpt_internal_strain_directions = torch.empty(0, dtype=torch.long)
                graph.dfpt_internal_strain_count = torch.tensor([0], dtype=torch.long)
                graph.y_dfpt_ionic_dielectric = torch.zeros(1, 3, 3, dtype=target.dtype)
                graph.dfpt_ionic_dielectric_mask = torch.tensor(False, dtype=torch.bool)
                graph.y_dfpt_electronic_dielectric = torch.zeros(
                    1, 3, 3, dtype=target.dtype
                )
                graph.dfpt_electronic_dielectric_mask = torch.tensor(
                    False, dtype=torch.bool
                )
            if completion is not None:
                label_type, certificate = completion_metadata
                full_internal = completion["internal_strain_full"].to(dtype=target.dtype)
                if full_internal.shape != (graph.num_nodes, 3, 3, 3):
                    raise ValueError(f"Invalid completed strain-force shape: {completion_path}")
                graph.dfpt_internal_strain_full = full_internal
                graph.internal_strain_full_mask = torch.tensor(True, dtype=torch.bool)
                graph.lambda_label_type_code = torch.tensor(
                    LAMBDA_LABEL_TYPE_CODES[label_type], dtype=torch.long
                )
                graph.lambda_identifiable_dimension = torch.tensor(
                    int(certificate.get("identifiable_dimension", certificate.get("invariant_dimensions", 0))),
                    dtype=torch.long,
                )
                graph.lambda_null_dimension = torch.tensor(
                    int(certificate.get("null_dimension_joint", 0)), dtype=torch.long
                )
            else:
                graph.dfpt_internal_strain_full = torch.zeros(
                    graph.num_nodes, 3, 3, 3, dtype=target.dtype
                )
                graph.internal_strain_full_mask = torch.tensor(False, dtype=torch.bool)
                partial = has_dfpt and int(graph.dfpt_internal_strain_count.item()) > 0
                label_type = "partial_blocks" if partial else "macro_only"
                graph.lambda_label_type_code = torch.tensor(
                    LAMBDA_LABEL_TYPE_CODES[label_type], dtype=torch.long
                )
                graph.lambda_identifiable_dimension = torch.tensor(0, dtype=torch.long)
                graph.lambda_null_dimension = torch.tensor(0, dtype=torch.long)
            graph.dfpt_mode_count = torch.tensor([modes], dtype=torch.long)
            graph.point_group_ops, graph.point_group_mask = pad_point_group_operations(rotations)
            graph.is_polar_point_group = torch.tensor(is_polar_point_group(rotations), dtype=torch.bool)
            self._graph_cache[index] = graph
        return self._graph_cache[index]
