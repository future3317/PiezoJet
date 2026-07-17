"""Train a validation-selected matched direct PiezoJet baseline.

The model deliberately shares only the Cartesian local encoder and equivariant
piezo readout with PiezoJet.  It is the required control for deciding whether
atom-coordinate factorization improves the same data, split, optimization
budget, and tensor convention.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .baselines import direct_cartesian_baseline_from_config, e3nn_direct_baseline_from_config
from .data import PiezoDataset, graph_cache_key, load_gmtnet_records
from .metrics import response_tensor_skill
from .pretraining_protocol import validate_inductive_checkpoint
from .project_config import load_project_config
from .tensor_ops import piezo_scale
from .train import (
    _data_commit,
    _git_commit,
    device_from_config,
    full_loss,
    load_explicit_splits,
    response_bin_weights,
    seed_everything,
)


def _epoch(model, loader, optimizer, scale, bin_weights, device, *, accumulate_to_one_update: bool = False, max_train_updates: int | None = None) -> tuple[float, float, int]:
    training = optimizer is not None
    model.train(training)
    total, count, predictions, targets = 0.0, 0, [], []
    if training and accumulate_to_one_update:
        optimizer.zero_grad(set_to_none=True)
    updates = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        with torch.set_grad_enabled(training):
            prediction = model(batch)
            loss = full_loss(prediction, batch.y, scale, bin_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite direct-baseline loss")
            if training:
                if accumulate_to_one_update:
                    # Match the full-batch protocol when a steerable encoder
                    # needs microbatches to fit accelerator memory.
                    (loss * batch.num_graphs / len(loader.dataset)).backward()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                        raise FloatingPointError("Non-finite direct-baseline gradient")
                    optimizer.step()
                    updates += 1
        total += float(loss.detach()) * batch.num_graphs
        count += batch.num_graphs
        predictions.append(prediction.detach().cpu())
        targets.append(batch.y.detach().cpu())
        if training and max_train_updates is not None and updates >= max_train_updates:
            break
    if training and accumulate_to_one_update:
        if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
            raise FloatingPointError("Non-finite accumulated direct-baseline gradient")
        optimizer.step()
        updates += 1
    skill = response_tensor_skill(torch.cat(predictions), torch.cat(targets))["tensor_response_skill_vs_zero"]
    return total / max(count, 1), float(skill), updates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--family", choices=("cartesian", "e3nn"), default="cartesian")
    parser.add_argument("--pretrained-encoder", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--updates-per-epoch", type=int, help="Cap optimizer updates per epoch for fixed-update full-corpus controls")
    parser.add_argument("--accumulate-to-one-update", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.updates_per_epoch is not None and args.updates_per_epoch < 1:
        raise ValueError("--updates-per-epoch must be positive")
    cfg = load_project_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        cfg["batch_size"] = args.batch_size
    seed_everything(int(cfg["seed"]))
    device = device_from_config(str(cfg["device"]))
    runtime_device = (
        f"cuda ({torch.cuda.get_device_name(device)})"
        if device.type == "cuda" else device.type
    )
    cfg["runtime_device"] = runtime_device
    records = load_gmtnet_records(cfg["data_root"])
    splits = load_explicit_splits(args.splits_file, {str(record["JARVIS_ID"]) for record in records})
    cache_key = graph_cache_key(records, float(cfg["cutoff"]), int(cfg["max_neighbors"]))
    dataset_kwargs = dict(
        records=records, cutoff=float(cfg["cutoff"]), max_neighbors=int(cfg["max_neighbors"]),
        processed_dir=cfg["processed_dir"], cache_key=cache_key, project_targets=True,
    )
    train_set = PiezoDataset(ids=splits["train"], **dataset_kwargs)
    val_set = PiezoDataset(ids=splits["val"], **dataset_kwargs)
    options = {"num_workers": int(cfg["num_workers"]), "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, batch_size=int(cfg["batch_size"]), shuffle=True, **options)
    val_loader = DataLoader(val_set, batch_size=int(cfg["batch_size"]), shuffle=False, **options)
    scale = piezo_scale(torch.cat([train_set[index].y_voigt for index in range(len(train_set))])).to(device)
    bins = response_bin_weights(torch.stack([train_set[index].y.squeeze(0) for index in range(len(train_set))])).to(device)
    builders = {"cartesian": direct_cartesian_baseline_from_config, "e3nn": e3nn_direct_baseline_from_config}
    architectures = {"cartesian": "cartesian_local_environment_v1", "e3nn": "e3nn_periodic_v1"}
    model = builders[args.family](cfg).to(device)
    pretrained_path = args.pretrained_encoder or Path(str(cfg["pretrained_encoder"]))
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"Missing inductive pretraining checkpoint: {pretrained_path}")
    pretrained = torch.load(pretrained_path, map_location=device, weights_only=False)
    validate_inductive_checkpoint(pretrained, splits["train"], splits["val"] + splits["test"])
    if pretrained.get("architecture") != architectures[args.family]:
        raise ValueError(f"Pretraining checkpoint is not compatible with the {args.family} direct baseline")
    model.encoder.load_state_dict(pretrained["encoder"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, best, best_row = [], float("inf"), None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_skill, train_updates = _epoch(
            model, train_loader, optimizer, scale, bins, device,
            accumulate_to_one_update=args.accumulate_to_one_update,
            max_train_updates=args.updates_per_epoch,
        )
        val_loss, val_skill, _ = _epoch(model, val_loader, None, scale, bins, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_tensor_response_skill_vs_zero": train_skill,
            "val_loss": val_loss,
            "val_tensor_response_skill_vs_zero": val_skill,
            "train_optimizer_updates": train_updates,
            "macro_effective_passes": epoch,
            "macro_examples_seen": epoch * len(train_set),
        }
        history.append(row)
        checkpoint = {
            "model": model.state_dict(), "config": cfg, "epoch": epoch,
            "model_family": f"matched_direct_{args.family}_piezo", "validation": {"loss": val_loss, "tensor_response_skill_vs_zero": val_skill},
            "splits_file": str(args.splits_file), "pretraining_provenance": pretrained["pretraining_provenance"],
            "piezo_scale": float(scale),
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            best_row = dict(row)
            torch.save(checkpoint, args.output_dir / "loss_best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch} train={train_loss:.6g} val={val_loss:.6g}")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    if best_row is None:
        raise RuntimeError("Direct baseline completed without a selected validation checkpoint")
    (args.output_dir / "config.resolved.json").write_text(
        json.dumps(cfg, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(json.dumps({
        "schema": 4, "model_family": f"matched_direct_{args.family}_piezo", "epochs": args.epochs,
        "runtime_device": runtime_device,
        "checkpoint_selection": "minimum validation loss", "frozen_test_used_for_selection": False,
        "splits_file": str(args.splits_file), "data_commit": _data_commit(cfg["data_root"]), "code_commit": _git_commit(),
        "best_validation_loss": best,
        "selected_epoch": int(best_row["epoch"]),
        "selected_validation_tensor_response_skill_vs_zero": float(
            best_row["val_tensor_response_skill_vs_zero"]
        ),
        "piezo_scale_c_per_m2": float(scale),
        "batch_size": int(cfg["batch_size"]),
        "updates_per_epoch": int(history[0]["train_optimizer_updates"]),
        "optimizer_updates_total": int(sum(row["train_optimizer_updates"] for row in history)),
        "exposure_unit": "complete_macro_stream_passes",
        "macro_effective_passes": args.epochs,
        "macro_examples_seen": args.epochs * len(train_set),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
