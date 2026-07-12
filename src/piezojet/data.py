"""GMTNet piezoelectric ingestion, persistent splitting, and periodic graphs."""

from __future__ import annotations

import json
import pickle
import random
from hashlib import sha256
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import torch
from pymatgen.core import Element
from torch_geometric.data import Data, Dataset

from .tensor_ops import piezo_voigt_to_cartesian, source_voigt_to_canonical


PIEZO_FILE = "jarvis_diele_piezo.pkl"
PIEZO_FIELD = "piezoelectric_C_m2"
GRAPH_CACHE_SCHEMA = 1


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


def create_or_load_splits(records: list[dict[str, Any]], processed_dir: str | Path, seed: int = 42) -> dict[str, list[str]]:
    path = Path(processed_dir) / "splits.json"
    ids = sorted(str(record["JARVIS_ID"]) for record in records)
    if path.is_file():
        splits = json.loads(path.read_text(encoding="utf-8"))
        restored = sorted(splits.get("train", []) + splits.get("val", []) + splits.get("test", []))
        if restored != ids:
            raise ValueError(f"Existing split {path} does not match the GMTNet records")
        return splits
    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end, val_end = int(0.8 * count), int(0.9 * count)
    splits = {"train": shuffled[:train_end], "val": shuffled[train_end:val_end], "test": shuffled[val_end:]}
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
    shifts = list(product(range(-span, span + 1), repeat=3))
    sources: list[int] = []
    targets: list[int] = []
    edge_shifts: list[torch.Tensor] = []
    for target in range(frac.shape[0]):
        neighbors: list[tuple[float, int, torch.Tensor]] = []
        for source in range(frac.shape[0]):
            for shift_tuple in shifts:
                shift = torch.tensor(shift_tuple, dtype=frac.dtype)
                delta = (frac[source] - frac[target] + shift) @ cell
                distance = torch.linalg.vector_norm(delta).item()
                if distance > 1e-7 and distance <= cutoff:
                    neighbors.append((distance, source, shift @ cell))
        for _, source, shift_cart in sorted(neighbors, key=lambda item: item[0])[:max_neighbors]:
            sources.append(source)
            targets.append(target)
            edge_shifts.append(shift_cart)
    if not sources:
        raise ValueError("No periodic neighbors found; increase cutoff")
    return torch.tensor([sources, targets], dtype=torch.long), torch.stack(edge_shifts)


def record_to_graph(record: dict[str, Any], cutoff: float, max_neighbors: int) -> Data:
    atoms = record["atoms"]
    if atoms.get("cartesian") is not False:
        raise ValueError("GMTNet atoms must use fractional coordinates (cartesian=False)")
    frac = torch.tensor(atoms["coords"], dtype=torch.float32)
    cell = torch.tensor(atoms["lattice_mat"], dtype=torch.float32)
    z = torch.tensor([Element(symbol).Z for symbol in atoms["elements"]], dtype=torch.long)
    edge_index, edge_shift = _periodic_edges(frac, cell, cutoff, max_neighbors)
    source = torch.tensor(record[PIEZO_FIELD], dtype=torch.float32)
    target_voigt = source_voigt_to_canonical(source)
    return Data(
        z=z,
        pos=frac @ cell,
        cell=cell.unsqueeze(0),
        edge_index=edge_index,
        edge_shift=edge_shift,
        y=piezo_voigt_to_cartesian(target_voigt).unsqueeze(0),
        y_voigt=target_voigt.unsqueeze(0),
        material_id=str(record["JARVIS_ID"]),
        num_nodes=z.numel(),
    )


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
        temporary = path.with_suffix(".tmp")
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
    def __init__(self, records: list[dict[str, Any]], ids: list[str], cutoff: float, max_neighbors: int, processed_dir: str | Path | None = None, cache_key: str | None = None):
        super().__init__()
        wanted = set(ids)
        self.records = [record for record in records if str(record["JARVIS_ID"]) in wanted]
        if len(self.records) != len(ids):
            raise ValueError("Split contains material IDs absent from loaded GMTNet records")
        self.cutoff, self.max_neighbors = cutoff, max_neighbors
        self._graph_cache: dict[int, Data] = {}
        self._disk_cache = PersistentGraphCache(processed_dir, self.records, cutoff, max_neighbors, cache_key=cache_key) if processed_dir is not None else None

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
            self._graph_cache[index] = graph
        return self._graph_cache[index]
