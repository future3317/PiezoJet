#!/usr/bin/env python
"""Short-cycle shared-encoder M5A multiresponse overfit/dry-run."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from piezojet.data import graph_cache_key, load_gmtnet_records
from piezojet.multiresponse import MultiResponseDataset, SharedMultiResponseModel, load_multresponse_records, multiresponse_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/m5/multitask_overfit"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if cfg.get("device") == "auto" and torch.cuda.is_available() else "cpu")
    records = load_multresponse_records(cfg["data_root"])
    piezo_records = load_gmtnet_records(cfg["data_root"])
    ids = sorted(str(record["JARVIS_ID"]) for record in records)[: args.limit]
    dataset = MultiResponseDataset(records, ids, cfg["cutoff"], cfg["max_neighbors"], processed_dir=cfg["processed_dir"], cache_key=graph_cache_key(piezo_records, cfg["cutoff"], cfg["max_neighbors"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=cfg["num_workers"])
    first = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False))).to(device)
    scales = {}
    for name, target_name in (("piezo", "y_piezo"), ("elastic", "y_elastic"), ("dielectric_electronic", "y_dielectric_e"), ("dielectric_ionic", "y_dielectric_i")):
        target = getattr(first, target_name)
        scales[name] = torch.sqrt(target.square().mean()).clamp_min(torch.finfo(target.dtype).eps)
    model = SharedMultiResponseModel(embedding_dim=cfg["embedding_dim"], cutoff=cfg["cutoff"], lmax=cfg["lmax"], num_blocks=cfg["num_blocks"], radial_basis=cfg["radial_basis"], radial_hidden=cfg["radial_hidden"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total, train_count = 0.0, 0
        train_details = {}
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)
            loss, details = multiresponse_loss(prediction, batch, scales)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite M5A loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                raise FloatingPointError("Non-finite M5A gradient")
            optimizer.step()
            train_total += float(loss.detach()) * batch.num_graphs
            train_count += batch.num_graphs
            train_details = details
        model.eval()
        with torch.no_grad():
            batch = first
            prediction = model(batch)
            val_loss, val_details = multiresponse_loss(prediction, batch, scales)
        row = {"epoch": epoch, "train_loss": train_total / max(train_count, 1), "val_loss": float(val_loss), **{f"train_{key}": value for key, value in train_details.items()}, **{f"val_{key}": value for key, value in val_details.items()}}
        rows.append(row)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": cfg, "scales": {name: float(value) for name, value in scales.items()}, "ids": ids}
        torch.save(checkpoint, args.output / "last.pt")
        if float(val_loss) < best:
            best = float(val_loss)
            torch.save(checkpoint, args.output / "best.pt")
        print(json.dumps(row))
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"epochs": args.epochs, "records_available": len(records), "cohort_size": len(dataset), "best_val_loss": best, "all_finite": True, "task_counts": {key: len(dataset) for key in scales}, "device": str(device), "interpretation_boundary": "M5A shared multiresponse short-cycle overfit/dry-run; not generalization."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text("\n".join(["# M5A short-cycle run", "", f"- Device: `{device}`", f"- Complete response records available: `{len(records)}`", f"- Cohort size: `{len(dataset)}`", f"- Epochs: `{args.epochs}`", f"- Best normalized validation loss: `{best:.6g}`", "- Shared encoder with piezo, elastic, electronic dielectric, and ionic dielectric heads.", "- Missing-label masks and per-task effective counts are recorded in `metrics.csv`.", "- This is an overfit/dry-run diagnostic, not a generalization result."]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
