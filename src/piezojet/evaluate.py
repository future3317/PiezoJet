"""Metrics, rotation checks, point-group residuals, and loss-cost comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.loader import DataLoader

from .data import PiezoDataset, create_or_load_splits, load_gmtnet_records, record_to_graph
from .model import PiezoJet
from .tensor_ops import cartesian_to_piezo_voigt, rotate_piezo, rotate_strain, symmetric_matrix_to_voigt
from .train import device_from_config, full_loss, sketch_loss


def _random_rotation(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    matrix = torch.randn(3, 3, dtype=dtype, device=device)
    q, _ = torch.linalg.qr(matrix)
    return q


def _rotated_graph(graph, rotation: torch.Tensor):
    rotated = graph.clone()
    rotated.pos = graph.pos @ rotation.transpose(0, 1)
    rotated.edge_shift = graph.edge_shift @ rotation.transpose(0, 1)
    rotated.cell = graph.cell @ rotation.transpose(0, 1)
    return rotated


def _point_group_rotations(record) -> list[torch.Tensor]:
    atoms = record["atoms"]
    structure = Structure(atoms["lattice_mat"], atoms["elements"], atoms["coords"], coords_are_cartesian=False)
    operations = SpacegroupAnalyzer(structure, symprec=1e-5).get_point_group_operations(cartesian=True)
    return [torch.tensor(op.rotation_matrix, dtype=torch.float32) for op in operations]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-equivalence-samples", type=int, default=32)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    device = device_from_config(cfg["device"])
    records = load_gmtnet_records(cfg["data_root"])
    splits = create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
    dataset = PiezoDataset(records, splits[args.split], cfg["cutoff"], cfg["max_neighbors"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    model = PiezoJet(
        embedding_dim=cfg["embedding_dim"], cutoff=cfg["cutoff"], lmax=cfg["lmax"], num_blocks=cfg["num_blocks"],
        radial_basis=cfg["radial_basis"], radial_hidden=cfg["radial_hidden"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu())
            targets.append(batch.y.cpu())
    prediction, target = torch.cat(predictions), torch.cat(targets)
    diff = prediction - target
    sample_error = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=-1)
    target_norm = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=-1).clamp_min(1e-12)
    metrics: dict[str, float | int | None] = {
        "cartesian_component_mae": float(diff.abs().mean()),
        "sample_frobenius_mae": float(sample_error.mean()),
        "normalized_frobenius_error": float((sample_error / target_norm).mean()),
        "max_component_mae": float((prediction.abs().amax(dim=(1, 2, 3)) - target.abs().amax(dim=(1, 2, 3))).abs().mean()),
    }
    selected = dataset.records[: args.max_equivalence_samples]
    rotation_residuals, group_residuals, centro_norms = [], [], []
    with torch.no_grad():
        for record in selected:
            graph = record_to_graph(record, cfg["cutoff"], cfg["max_neighbors"])
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
            graph = graph.to(device)
            rotation = _random_rotation(torch.float32, device)
            rotated_prediction = model(_rotated_graph(graph, rotation))
            base_prediction = model(graph)
            expected = rotate_piezo(base_prediction, rotation)
            rotation_residuals.append(float(torch.linalg.vector_norm(rotated_prediction - expected) / torch.linalg.vector_norm(base_prediction).clamp_min(1e-12)))
            operations = [op.to(device) for op in _point_group_rotations(record)]
            per_op = [torch.linalg.vector_norm(base_prediction - rotate_piezo(base_prediction, op)) / torch.linalg.vector_norm(base_prediction).clamp_min(1e-12) for op in operations]
            group_residuals.append(float(torch.stack(per_op).mean()))
            if any(torch.allclose(op, -torch.eye(3, device=device), atol=1e-5) for op in operations):
                centro_norms.append(float(torch.linalg.vector_norm(base_prediction)))
    metrics["rotation_equivariance_residual"] = float(sum(rotation_residuals) / len(rotation_residuals))
    metrics["point_group_residual"] = float(sum(group_residuals) / len(group_residuals))
    metrics["centrosymmetric_false_positive_norm"] = float(sum(centro_norms) / len(centro_norms)) if centro_norms else None
    batch = next(iter(loader)).to(device)
    scale = torch.tensor(checkpoint["piezo_scale"], device=device)
    with torch.no_grad():
        start = time.perf_counter()
        full = full_loss(model(batch), batch.y, scale)
        metrics["full_loss_seconds"] = time.perf_counter() - start
        metrics["full_loss"] = float(full)
    start = time.perf_counter()
    with torch.no_grad():
        sketch = sketch_loss(model, batch, cartesian_to_piezo_voigt(batch.y))
    metrics["sketch_loss_seconds"] = time.perf_counter() - start
    metrics["sketch_loss"] = float(sketch)
    metrics["full_peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    metrics["sketch_peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    output = Path(cfg["output_dir"]) / f"evaluation_{args.split}.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
