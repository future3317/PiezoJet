"""GMTNet piezoelectric ingestion, persistent splitting, and periodic graphs."""

from __future__ import annotations

import json
import pickle
import random
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


class PiezoDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], ids: list[str], cutoff: float, max_neighbors: int):
        super().__init__()
        wanted = set(ids)
        self.records = [record for record in records if str(record["JARVIS_ID"]) in wanted]
        if len(self.records) != len(ids):
            raise ValueError("Split contains material IDs absent from loaded GMTNet records")
        self.cutoff, self.max_neighbors = cutoff, max_neighbors
        self._graph_cache: dict[int, Data] = {}

    def len(self) -> int:
        return len(self.records)

    def get(self, index: int) -> Data:
        if index not in self._graph_cache:
            self._graph_cache[index] = record_to_graph(self.records[index], self.cutoff, self.max_neighbors)
        return self._graph_cache[index]
