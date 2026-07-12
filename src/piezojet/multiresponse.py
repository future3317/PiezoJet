"""Shared-encoder multiresponse heads and masked heterogeneous loss."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from e3nn import o3
from torch import nn
from torch_geometric.data import Dataset
from torch_geometric.utils import scatter

from .data import PersistentGraphCache, record_to_graph
from .model import PeriodicCrystalEncoder
from .response_ops import DIELECTRIC_TENSOR, ELASTIC_TENSOR, elastic_voigt_to_cartesian
from .tensor_ops import PIEZO_TYPE, piezo_from_irreps, piezo_voigt_to_cartesian, source_voigt_to_canonical


def _as_tensor(value: Any) -> torch.Tensor | None:
    return None if value is None else torch.as_tensor(value, dtype=torch.float32)


def load_multresponse_records(data_root: str | Path) -> list[dict[str, Any]]:
    root = Path(data_root) / "data"
    with (root / "jarvis_diele_piezo.pkl").open("rb") as handle:
        piezo_records = pickle.load(handle)
    with (root / "jarvis_elastic.pkl").open("rb") as handle:
        elastic_records = pickle.load(handle)
    elastic_by_id = {str(record["JARVIS_ID"]): record for record in elastic_records}
    records = []
    for record in piezo_records:
        material_id = str(record["JARVIS_ID"])
        elastic = elastic_by_id.get(material_id)
        if elastic is None:
            continue
        piezo = _as_tensor(record.get("piezoelectric_C_m2"))
        dielectric = _as_tensor(record.get("dielectric"))
        dielectric_ionic = _as_tensor(record.get("dielectric_ionic"))
        elastic_voigt = _as_tensor(elastic.get("elastic_total_kbar"))
        if piezo is None or elastic_voigt is None or piezo.shape != (3, 6) or elastic_voigt.shape != (6, 6):
            continue
        if (dielectric is not None and dielectric.shape != (3, 3)) or (dielectric_ionic is not None and dielectric_ionic.shape != (3, 3)):
            continue
        present = [value for value in (piezo, elastic_voigt, dielectric, dielectric_ionic) if value is not None]
        if not all(torch.isfinite(value).all() for value in present):
            continue
        merged = dict(record)
        merged["elastic_total_kbar"] = elastic_voigt
        merged["dielectric"] = dielectric
        merged["dielectric_ionic"] = dielectric_ionic
        records.append(merged)
    if not records:
        raise ValueError("No complete piezo/elastic/dielectric records found")
    return records


class MultiResponseDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], ids: list[str], cutoff: float, max_neighbors: int, processed_dir: str | Path | None = None, cache_key: str | None = None):
        super().__init__()
        wanted = set(ids)
        self.records = [record for record in records if str(record["JARVIS_ID"]) in wanted]
        if len(self.records) != len(ids):
            raise ValueError("Multiresponse split contains absent material IDs")
        self.cutoff, self.max_neighbors = cutoff, max_neighbors
        self._graph_cache: dict[int, Any] = {}
        self._disk_cache = PersistentGraphCache(processed_dir, self.records, cutoff, max_neighbors, cache_key=cache_key) if processed_dir is not None else None

    def len(self) -> int:
        return len(self.records)

    def get(self, index: int):
        record = self.records[index]
        if index not in self._graph_cache:
            graph = self._disk_cache.load(record) if self._disk_cache is not None else None
            if graph is None:
                graph = record_to_graph(record, self.cutoff, self.max_neighbors)
                if self._disk_cache is not None:
                    self._disk_cache.save(record, graph)
            self._graph_cache[index] = graph
        graph = self._graph_cache[index]
        graph.y_piezo = piezo_voigt_to_cartesian(source_voigt_to_canonical(_as_tensor(record["piezoelectric_C_m2"]))).unsqueeze(0)
        graph.y_elastic = elastic_voigt_to_cartesian(_as_tensor(record["elastic_total_kbar"]) * 0.1).unsqueeze(0)
        graph.y_dielectric_e = (_as_tensor(record["dielectric"]) if record["dielectric"] is not None else torch.zeros(3, 3)).unsqueeze(0)
        graph.y_dielectric_i = (_as_tensor(record["dielectric_ionic"]) if record["dielectric_ionic"] is not None else torch.zeros(3, 3)).unsqueeze(0)
        graph.mask_piezo = torch.tensor(True)
        graph.mask_elastic = torch.tensor(True)
        graph.mask_dielectric_e = torch.tensor(record["dielectric"] is not None)
        graph.mask_dielectric_i = torch.tensor(record["dielectric_ionic"] is not None)
        return graph


class TensorHead(nn.Module):
    def __init__(self, irreps_in: o3.Irreps, tensor_type: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in, tensor_type)
        self.tensor_type = tensor_type

    def forward(self, node_features: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        graph_features = scatter(node_features, batch_index, dim=0, dim_size=int(batch_index.max()) + 1, reduce="mean")
        return self.tensor_type.to_cartesian(self.linear(graph_features))


class SharedMultiResponseModel(nn.Module):
    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = PeriodicCrystalEncoder(**encoder_kwargs)
        self.piezo_head = TensorHead(self.encoder.hidden_irreps, PIEZO_TYPE)
        self.elastic_head = TensorHead(self.encoder.hidden_irreps, ELASTIC_TENSOR)
        self.dielectric_electronic_head = TensorHead(self.encoder.hidden_irreps, DIELECTRIC_TENSOR)
        self.dielectric_ionic_head = TensorHead(self.encoder.hidden_irreps, DIELECTRIC_TENSOR)

    def forward(self, batch) -> dict[str, torch.Tensor]:
        features = self.encoder(batch)
        return {
            "piezo": self.piezo_head(features, batch.batch),
            "elastic": self.elastic_head(features, batch.batch),
            "dielectric_electronic": self.dielectric_electronic_head(features, batch.batch),
            "dielectric_ionic": self.dielectric_ionic_head(features, batch.batch),
        }


TASK_TARGETS = {
    "piezo": "y_piezo",
    "elastic": "y_elastic",
    "dielectric_electronic": "y_dielectric_e",
    "dielectric_ionic": "y_dielectric_i",
}
TASK_MASKS = {
    "piezo": "mask_piezo",
    "elastic": "mask_elastic",
    "dielectric_electronic": "mask_dielectric_e",
    "dielectric_ionic": "mask_dielectric_i",
}


def masked_task_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, int]:
    mask = mask.to(dtype=torch.bool)
    count = int(mask.sum())
    if count == 0:
        return prediction.sum() * 0.0, 0
    error = ((prediction[mask] - target[mask]) / scale).square().mean()
    return error, count


def multiresponse_loss(predictions: dict[str, torch.Tensor], batch, scales: dict[str, torch.Tensor], weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, float | int]]:
    weights = weights or {name: 1.0 for name in predictions}
    total = predictions["piezo"].sum() * 0.0
    details: dict[str, float | int] = {}
    for name, prediction in predictions.items():
        target = getattr(batch, TASK_TARGETS[name])
        mask = getattr(batch, TASK_MASKS[name])
        loss, count = masked_task_loss(prediction, target, mask, scales[name])
        total = total + weights[name] * loss
        details[f"{name}_loss"] = float(loss.detach())
        details[f"{name}_count"] = count
    return total, details
