"""Chunked, inference-mode prediction for fixed crystal structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .data import PiezoDataset, create_or_load_splits, graph_cache_key, load_gmtnet_records
from .model import model_from_config
from .train import device_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    device = device_from_config(cfg["device"])
    records = load_gmtnet_records(cfg["data_root"])
    splits = create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
    ids = splits[args.split][: args.max_samples]
    cache_key = graph_cache_key(records, cfg["cutoff"], cfg["max_neighbors"])
    dataset = PiezoDataset(records, ids, cfg["cutoff"], cfg["max_neighbors"], processed_dir=cfg["processed_dir"], cache_key=cache_key)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    model = model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks, count = [], 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            material_ids = list(batch.material_id)
            prediction = model(batch.to(device, non_blocking=device.type == "cuda")).cpu()
            filename = f"chunk_{index:05d}.pt"
            torch.save({"material_ids": material_ids, "piezo_cartesian": prediction}, args.output_dir / filename)
            chunks.append({"file": filename, "samples": len(material_ids)})
            count += len(material_ids)
    manifest = {"checkpoint": str(args.checkpoint), "split": args.split, "samples": count, "batch_size": cfg["batch_size"], "device": str(device), "graph_cache_key": cache_key, "chunks": chunks}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / "manifest.json")


if __name__ == "__main__":
    main()
