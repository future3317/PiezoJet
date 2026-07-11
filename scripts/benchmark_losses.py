#!/usr/bin/env python
"""Benchmark full, direct sketch, and JVP sketch on one reproducible batch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from piezojet.data import PiezoDataset, create_or_load_splits, load_gmtnet_records
from piezojet.model import PiezoJet
from piezojet.tensor_ops import cartesian_to_piezo_voigt, piezo_scale
from piezojet.train import direct_sketch_loss, full_loss, sketch_loss


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_gmtnet_records(cfg["data_root"])
    splits = create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
    dataset = PiezoDataset(records, splits["train"][:32], cfg["cutoff"], cfg["max_neighbors"])
    batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False))).to(device)
    model = PiezoJet(embedding_dim=cfg["embedding_dim"], cutoff=cfg["cutoff"], lmax=cfg["lmax"], num_blocks=cfg["num_blocks"], radial_basis=cfg["radial_basis"], radial_hidden=cfg["radial_hidden"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()
    scale = torch.tensor(checkpoint["piezo_scale"], device=device)
    target_voigt = cartesian_to_piezo_voigt(batch.y)
    results = {}
    for name in ("full", "direct_sketch", "jvp_sketch"):
        for _ in range(args.warmup):
            model.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = full_loss(prediction, batch.y, scale) if name == "full" else direct_sketch_loss(prediction, target_voigt) if name == "direct_sketch" else sketch_loss(model, batch, target_voigt, prediction)
            loss.backward()
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            model.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = full_loss(prediction, batch.y, scale) if name == "full" else direct_sketch_loss(prediction, target_voigt) if name == "direct_sketch" else sketch_loss(model, batch, target_voigt, prediction)
            loss.backward()
        sync(device)
        elapsed = time.perf_counter() - start
        results[name] = {"steps": args.steps, "step_seconds": elapsed / args.steps, "samples_per_second": batch.num_graphs * args.steps / elapsed, "peak_cuda_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "peak_cuda_reserved": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"device": str(device), "results": results, "warmup": args.warmup, "steps": args.steps}, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
