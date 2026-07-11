"""Reproducible full-tensor and random mixed-Hessian-sketch training."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.func import jvp
from torch_geometric.loader import DataLoader

from .data import PiezoDataset, create_or_load_splits, load_gmtnet_records
from .model import PiezoJet
from .tensor_ops import cartesian_to_piezo_voigt, piezo_scale, source_voigt_to_canonical


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_config(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def full_loss(prediction: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.mean(((prediction - target) / scale).square())


def sketch_loss(model: PiezoJet, batch, target_voigt: torch.Tensor) -> torch.Tensor:
    """One Gaussian projection per graph, evaluated with nested forward-mode JVP."""
    graphs = target_voigt.shape[0]
    field0 = torch.zeros(graphs, 3, device=target_voigt.device, dtype=target_voigt.dtype)
    eta0 = torch.zeros(graphs, 6, device=target_voigt.device, dtype=target_voigt.dtype)
    a, b = torch.randn_like(field0), torch.randn_like(eta0)

    def eta_direction(field: torch.Tensor) -> torch.Tensor:
        _, tangent = jvp(lambda eta: model.potential(batch, field, eta), (eta0,), (b,))
        return tangent

    _, mixed = jvp(eta_direction, (field0,), (a,))
    target = torch.einsum("bi,bij,bj->b", a, target_voigt, b)
    return torch.mean((-mixed - target).square())


def _epoch(model, loader, optimizer, loss_name: str, scale: torch.Tensor, device: torch.device, full_weight: float) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total, count, elapsed = 0.0, 0, 0.0
    for batch in loader:
        batch = batch.to(device)
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            prediction = model(batch)
            full = full_loss(prediction, batch.y, scale)
            if loss_name == "full":
                loss = full
            else:
                sketch = sketch_loss(model, batch, cartesian_to_piezo_voigt(batch.y))
                loss = sketch if loss_name == "sketch" else sketch + full_weight * full
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        count += batch.num_graphs
        elapsed += time.perf_counter() - start
    return total / max(count, 1), elapsed


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _data_commit(data_root: str | Path) -> str:
    path = Path(data_root) / "SOURCE_COMMIT.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Reproducible training requires the exact GMTNet repository commit SHA; "
            "write it to SOURCE_COMMIT.txt before starting an experiment."
        )
    commit = path.read_text(encoding="utf-8").strip()
    if len(commit) != 40 or any(char not in "0123456789abcdefABCDEF" for char in commit):
        raise ValueError(f"{path} must contain one 40-character Git commit SHA")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--loss", choices=("full", "sketch", "hybrid"), default="full")
    parser.add_argument("--overfit-32", action="store_true")
    parser.add_argument("--epochs", type=int, help="Override config epochs for a bounded smoke run")
    parser.add_argument("--batch-size", type=int, help="Override config batch size")
    parser.add_argument("--learning-rate", type=float, help="Override config learning rate")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        cfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive")
        cfg["learning_rate"] = args.learning_rate
    seed_everything(int(cfg["seed"]))
    device = device_from_config(cfg["device"])
    data_commit = _data_commit(cfg["data_root"])
    records = load_gmtnet_records(cfg["data_root"])
    splits = create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
    if args.overfit_32:
        splits["train"] = splits["train"][:32]
        splits["val"] = splits["train"]
    train_set = PiezoDataset(records, splits["train"], cfg["cutoff"], cfg["max_neighbors"])
    val_set = PiezoDataset(records, splits["val"], cfg["cutoff"], cfg["max_neighbors"])
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    train_ids = set(splits["train"])
    first_target = torch.stack(
        [source_voigt_to_canonical(torch.tensor(record["piezoelectric_C_m2"], dtype=torch.float32))
         for record in records if str(record["JARVIS_ID"]) in train_ids]
    )
    scale = piezo_scale(first_target).to(device)
    processed = Path(cfg["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "stats.json").write_text(json.dumps({"piezo_scale": float(scale), "source": "train split only"}, indent=2) + "\n", encoding="utf-8")
    model = PiezoJet(
        embedding_dim=cfg["embedding_dim"], cutoff=cfg["cutoff"], lmax=cfg["lmax"], num_blocks=cfg["num_blocks"],
        radial_basis=cfg["radial_basis"], radial_hidden=cfg["radial_hidden"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cfg["loss"] = args.loss
    cfg["git_commit"] = _git_commit()
    cfg["data_commit"] = data_commit
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    best = float("inf")
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_value, train_seconds = _epoch(model, train_loader, optimizer, args.loss, scale, device, cfg["hybrid_full_weight"])
        val_value, val_seconds = _epoch(model, val_loader, None, args.loss, scale, device, cfg["hybrid_full_weight"])
        row = {"epoch": epoch, "train_loss": train_value, "val_loss": val_value, "train_seconds": train_seconds, "val_seconds": val_seconds}
        rows.append(row)
        checkpoint = {"model": model.state_dict(), "config": cfg, "piezo_scale": float(scale), "epoch": epoch}
        torch.save(checkpoint, output / "last.pt")
        if val_value < best:
            best = val_value
            torch.save(checkpoint, output / "best.pt")
        print(f"epoch={epoch} train={train_value:.6g} val={val_value:.6g}")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps({"best_val_loss": best, "loss": args.loss, "epochs": len(rows)}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
