"""Read-only response-active task-gradient audit for a Stage-A checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .build_electrostatic_development_folds import electrostatic_fold_train_ids
from .data import deterministic_subset, graph_cache_key, load_gmtnet_records
from .electrostatic_fold_adjudication import (
    ARCHITECTURES,
    _dataset,
    make_model,
    response_active_diagnostic_indices,
    task_gradient_geometry,
)
from .project_config import load_project_config
from .train import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--folds", type=Path,
        default=Path("data/processed/electrostatic_development_folds_v2.json"),
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)

    config = load_project_config(args.config)
    folds = json.loads(args.folds.read_text(encoding="utf-8-sig"))
    fold = next(
        (entry for entry in folds["folds"] if entry["fold"] == args.fold),
        None,
    )
    if fold is None:
        raise ValueError(f"Fold {args.fold} is absent from {args.folds}")
    train_ids = deterministic_subset(
        electrostatic_fold_train_ids(folds, args.fold),
        args.train_limit,
        args.seed + 1000,
    )
    seed_everything(args.seed)
    records = load_gmtnet_records(config["data_root"])
    cache_key = graph_cache_key(
        records, float(config["cutoff"]), int(config["max_neighbors"])
    )
    dataset = _dataset(config, records, train_ids, cache_key)
    indices, materials = response_active_diagnostic_indices(
        dataset, min(args.batch_size, len(dataset))
    )
    device = torch.device(args.device)
    batch = next(iter(DataLoader(
        Subset(dataset, indices), batch_size=len(indices), shuffle=False,
        num_workers=0,
    ))).to(device)
    model = make_model(args.architecture, config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != args.architecture:
        raise ValueError("Checkpoint architecture does not match the audit architecture")
    model.load_state_dict(checkpoint["model"], strict=True)
    geometry = task_gradient_geometry(model, batch, args.architecture)
    payload = {
        "schema": 1,
        "protocol": "read-only response-active norm-stratified task-gradient audit",
        "architecture": args.architecture,
        "fold": args.fold,
        "seed": args.seed,
        "train_limit": args.train_limit,
        "checkpoint": str(args.checkpoint.resolve()),
        "selected_update": checkpoint.get("selected_update"),
        "frozen_validation_test_labels_read": False,
        "selection": "fixed train prefix; response-active; norm-stratified ranks",
        "materials": materials,
        "losses": {
            "electronic": geometry["electronic_loss"],
            "born": geometry["born_loss"],
        },
        "gradient_geometry": geometry,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
