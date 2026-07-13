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

from .data import PiezoDataset, create_or_load_splits, graph_cache_key, load_gmtnet_records
from .model import PiezoJet, model_from_config
from .tensor_ops import cartesian_to_piezo_voigt, piezo_scale, piezo_to_irreps, source_voigt_to_canonical


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


def sketch_loss(model: PiezoJet, batch, target_voigt: torch.Tensor, piezo_cart: torch.Tensor | None = None) -> torch.Tensor:
    """One Gaussian projection per graph, evaluated with nested forward-mode JVP."""
    graphs = target_voigt.shape[0]
    field0 = torch.zeros(graphs, 3, device=target_voigt.device, dtype=target_voigt.dtype)
    eta0 = torch.zeros(graphs, 6, device=target_voigt.device, dtype=target_voigt.dtype)
    a, b = torch.randn_like(field0), torch.randn_like(eta0)

    piezo_cart = model(batch) if piezo_cart is None else piezo_cart

    def eta_direction(field: torch.Tensor) -> torch.Tensor:
        _, tangent = jvp(lambda eta: model.response(piezo_cart, field, eta), (eta0,), (b,))
        return tangent

    _, mixed = jvp(eta_direction, (field0,), (a,))
    target = torch.einsum("bi,bij,bj->b", a, target_voigt, b)
    return torch.mean((-mixed - target).square())


def direct_sketch_loss(prediction: torch.Tensor, target_voigt: torch.Tensor, sketches: int = 1) -> torch.Tensor:
    values = []
    predicted = cartesian_to_piezo_voigt(prediction)
    for _ in range(sketches):
        field = torch.randn(target_voigt.shape[0], 3, device=target_voigt.device, dtype=target_voigt.dtype)
        strain = torch.randn_like(target_voigt[..., 0, :])
        values.append((torch.einsum("bi,bij,bj->b", field, predicted, strain) - torch.einsum("bi,bij,bj->b", field, target_voigt, strain)).square())
    return torch.stack(values).mean()


def _diagnostics(prediction: torch.Tensor, target: torch.Tensor, batch, model: PiezoJet, normalized_scale: torch.Tensor) -> list[dict[str, float | int]]:
    pred_voigt = cartesian_to_piezo_voigt(prediction)
    target_voigt = cartesian_to_piezo_voigt(target)
    diff_voigt = pred_voigt - target_voigt
    pred_irreps = piezo_to_irreps(prediction)
    target_irreps = piezo_to_irreps(target)
    irrep_diff = pred_irreps - target_irreps
    block_slices = (("1o", 0, 6), ("2o", 6, 11), ("3o", 11, 18))
    node_counts = (batch.ptr[1:] - batch.ptr[:-1]).tolist() if hasattr(batch, "ptr") else [batch.num_nodes]
    degree = torch.zeros(batch.num_nodes, device=prediction.device, dtype=torch.long)
    degree.scatter_add_(0, batch.edge_index[1], torch.ones_like(batch.edge_index[1]))
    isolated = int((degree == 0).sum())
    grad_norm = torch.sqrt(sum((parameter.grad.detach().square().sum() for parameter in model.parameters() if parameter.grad is not None), torch.zeros((), device=prediction.device)))
    rows = []
    for index in range(prediction.shape[0]):
        row: dict[str, float | int] = {
            "sample_index": index,
            "unnormalized_tensor_mse": float(diff_voigt[index].square().mean()),
            "frob_error": float(torch.linalg.vector_norm(diff_voigt[index]) / torch.sqrt(torch.tensor(18.0, device=prediction.device))),
            "normalized_frob_error": float(torch.linalg.vector_norm(diff_voigt[index]) / (normalized_scale * torch.sqrt(torch.tensor(18.0, device=prediction.device)))),
            "predicted_tensor_norm": float(torch.linalg.vector_norm(pred_voigt[index])),
            "target_tensor_norm": float(torch.linalg.vector_norm(target_voigt[index])),
            "gradient_norm": float(grad_norm),
            "atom_count": int(node_counts[index]),
            "isolated_nodes_in_batch": isolated,
        }
        for name, start, end in block_slices:
            row[f"irrep_{name}_error"] = float(torch.linalg.vector_norm(irrep_diff[index, start:end]))
        rows.append(row)
    return rows


def _epoch(model, loader, optimizer, loss_name: str, scale: torch.Tensor, device: torch.device, full_weight: float, collect_diagnostics: bool = False, sketch_implementation: str = "jvp", sketch_count: int = 1) -> tuple[float, float, list[dict[str, float | int]]]:
    training = optimizer is not None
    model.train(training)
    total, count, elapsed, diagnostics = 0.0, 0, 0.0, []
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            prediction = model(batch)
            full = full_loss(prediction, batch.y, scale)
            if loss_name == "full":
                loss = full
            else:
                target_voigt = cartesian_to_piezo_voigt(batch.y)
                sketch = direct_sketch_loss(prediction, target_voigt, sketch_count) if sketch_implementation == "direct" else sketch_loss(model, batch, target_voigt, prediction)
                loss = sketch if loss_name == "sketch" else sketch + full_weight * full
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite optimization loss encountered")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                    raise FloatingPointError("Non-finite parameter gradient encountered")
                optimizer.step()
        if collect_diagnostics:
            diagnostics.extend(_diagnostics(prediction.detach(), batch.y.detach(), batch, model, scale))
        total += float(loss.detach()) * batch.num_graphs
        count += batch.num_graphs
        elapsed += time.perf_counter() - start
    return total / max(count, 1), elapsed, diagnostics


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
    parser.add_argument("--weight-decay", type=float, help="Override config weight decay")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--m2-1", action="store_true", help="Strict 300-epoch 32-sample memorization experiment")
    parser.add_argument("--resume", type=Path, help="Resume a saved checkpoint at its next epoch")
    parser.add_argument("--sketch-implementation", choices=("direct", "jvp"), default="jvp")
    parser.add_argument("--sketch-count", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--seed", type=int, help="Override config seed for multi-seed experiments")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.m2_1:
        cfg["epochs"] = 300
        cfg["batch_size"] = 32
        cfg["weight_decay"] = 0.0
        cfg["output_dir"] = "outputs/m2_1"
        args.overfit_32 = True
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
    if args.weight_decay is not None:
        if args.weight_decay < 0:
            raise ValueError("--weight-decay must be non-negative")
        cfg["weight_decay"] = args.weight_decay
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    seed_everything(int(cfg["seed"]))
    device = device_from_config(cfg["device"])
    data_commit = _data_commit(cfg["data_root"])
    records = load_gmtnet_records(cfg["data_root"])
    splits = create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
    if args.overfit_32:
        splits["train"] = splits["train"][:32]
        splits["val"] = splits["train"]
    cache_key = graph_cache_key(records, cfg["cutoff"], cfg["max_neighbors"])
    train_set = PiezoDataset(records, splits["train"], cfg["cutoff"], cfg["max_neighbors"], processed_dir=cfg["processed_dir"], cache_key=cache_key)
    val_set = PiezoDataset(records, splits["val"], cfg["cutoff"], cfg["max_neighbors"], processed_dir=cfg["processed_dir"], cache_key=cache_key)
    loader_options = {"num_workers": cfg["num_workers"], "pin_memory": device.type == "cuda"}
    if cfg["num_workers"] > 0:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, **loader_options)
    train_ids = set(splits["train"])
    first_target = torch.stack(
        [source_voigt_to_canonical(torch.tensor(record["piezoelectric_C_m2"], dtype=torch.float32))
         for record in records if str(record["JARVIS_ID"]) in train_ids]
    )
    scale = piezo_scale(first_target).to(device)
    processed = Path(cfg["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "stats.json").write_text(json.dumps({"piezo_scale": float(scale), "source": "train split only"}, indent=2) + "\n", encoding="utf-8")
    model = model_from_config(cfg).to(device)
    pretraining_path = Path(cfg["pretrained_encoder"])
    if not pretraining_path.is_file():
        raise FileNotFoundError(
            f"PiezoJet fine-tuning requires the structural pretraining checkpoint {pretraining_path}. "
            "Run `python -m piezojet.pretrain --config config.yaml` or use `python scripts/run_pipeline.py --config config.yaml`."
        )
    pretrained = torch.load(pretraining_path, map_location=device, weights_only=False)
    if "encoder" not in pretrained:
        raise ValueError(f"Pretraining checkpoint {pretraining_path} has no encoder state")
    model.encoder.load_state_dict(pretrained["encoder"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cfg["loss"] = args.loss
    cfg["pretrained_encoder"] = str(pretraining_path)
    cfg["pretraining_epoch"] = pretrained.get("epoch")
    cfg["git_commit"] = _git_commit()
    cfg["data_commit"] = data_commit
    start_epoch = 1
    resumed_from = None
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        resumed_from = int(saved["epoch"])
        start_epoch = resumed_from + 1
        cfg["resumed_from_epoch"] = resumed_from
        cfg["resumed_from_commit"] = saved.get("config", {}).get("git_commit", "unknown")
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    best = float("inf")
    rows: list[dict[str, float | int]] = []
    all_sample_rows: list[dict[str, float | int]] = []
    diagnostic_rows: list[dict[str, float | int]] = []
    existing_metrics = output / "metrics.csv"
    if args.resume is not None and existing_metrics.is_file():
        with existing_metrics.open(newline="", encoding="utf-8") as handle:
            rows = [{key: (int(value) if key == "epoch" else float(value)) for key, value in row.items()} for row in csv.DictReader(handle)]
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        train_value, train_seconds, train_diagnostics = _epoch(model, train_loader, optimizer, args.loss, scale, device, cfg["hybrid_full_weight"], args.m2_1, args.sketch_implementation, args.sketch_count)
        val_value, val_seconds, val_diagnostics = _epoch(model, val_loader, None, args.loss, scale, device, cfg["hybrid_full_weight"], args.m2_1, args.sketch_implementation, args.sketch_count)
        row = {"epoch": epoch, "train_loss": train_value, "val_loss": val_value, "train_seconds": train_seconds, "val_seconds": val_seconds}
        rows.append(row)
        if args.m2_1:
            all_sample_rows.extend({"epoch": epoch, "phase": "train", **item} for item in train_diagnostics)
            all_sample_rows.extend({"epoch": epoch, "phase": "eval", **item} for item in val_diagnostics)
            diagnostic_rows.append({"epoch": epoch, "phase": "train", **{key: value for key, value in train_diagnostics[0].items() if key != "sample_index"}})
            diagnostic_rows.append({"epoch": epoch, "phase": "eval", **{key: value for key, value in val_diagnostics[0].items() if key != "sample_index"}})
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg, "piezo_scale": float(scale), "epoch": epoch}
        if not args.m2_1 or epoch % 10 == 0 or epoch == int(cfg["epochs"]):
            torch.save(checkpoint, output / "last.pt")
        if val_value < best:
            best = val_value
            torch.save(checkpoint, output / "best.pt")
        if not args.m2_1 or epoch == 1 or epoch % 10 == 0 or epoch == int(cfg["epochs"]):
            print(f"epoch={epoch} train={train_value:.6g} val={val_value:.6g}")
        if args.m2_1:
            with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"best_val_loss": best, "loss": args.loss, "epochs": int(cfg["epochs"]), "metrics_rows": len(rows), "optimization_loss": args.loss, "memorization_loss": args.loss if args.m2_1 else None, "all_finite": True}
    if resumed_from is not None:
        summary["resumed_from_epoch"] = resumed_from
        summary["resumed_from_commit"] = cfg.get("resumed_from_commit")
        summary["metrics_coverage"] = [int(rows[0]["epoch"]), int(rows[-1]["epoch"])] if rows else []
    if args.m2_1:
        with (output / "sample_errors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_sample_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_sample_rows)
        with (output / "diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=diagnostic_rows[0].keys())
            writer.writeheader()
            writer.writerows(diagnostic_rows)
        summary["diagnostics"] = ["sample_errors.csv", "diagnostics.csv"]
        summary["interpretation_boundary"] = "Same-cohort memorization only; not validation generalization."
        report = f"""# M2.1 strict memorization test

## Git commit

`{cfg['git_commit']}`

## Data manifest

GMTNet commit `{cfg['data_commit']}`; fixed first 32 material IDs from the existing seed-42 training split.

## Configuration

300 epochs, batch size 32, full tensor loss, weight decay 0, seed {cfg['seed']}, dropout disabled.

## What was implemented

Per-sample Cartesian/Voigt errors, irreps block errors, gradient norms, predicted/target tensor norms, graph atom statistics, and non-finite checks.

## Interpretation boundary

This is a strict same-cohort memorization test. It does not estimate random-split or chemical OOD generalization.
"""
        (output / "report.md").write_text(report, encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
