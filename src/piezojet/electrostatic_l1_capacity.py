"""Train-only capacity gate for the independent electronic ``l=1`` readout.

This executor is deliberately separate from formula-disjoint adjudication. It
uses only an explicit material-ID file, never evaluates development or frozen
panels, and reports capacity evidence rather than inductive performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .data import graph_cache_key, load_gmtnet_records
from .electrostatic_fold_adjudication import (
    _data_commit,
    _dataset,
    encoder_width_multiplier_for_architecture,
    load_structure_pretraining,
    make_model,
)
from .build_electrostatic_development_folds import electrostatic_fold_train_ids
from .electronic_capacity import electronic_capacity_metrics, irrep_balanced_capacity_loss
from .electrostatic_subset import load_response_subset
from .project_config import load_project_config
from .train import seed_everything


ARCHITECTURE = "a0_independent_l1_readout"
BASELINE_ARCHITECTURE = "a0_parameter_matched_irreps"


def _evaluate(tower, loader, device: torch.device) -> tuple[dict[str, object], torch.Tensor]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    tower.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            _, graph_features, context = tower.encode_response_features(batch)
            predictions.append(tower.decode_electronic_piezo(graph_features, context).cpu())
            targets.append(batch.y_electronic_piezo.cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return electronic_capacity_metrics(prediction, target), prediction


def run_capacity(args: argparse.Namespace) -> dict[str, object]:
    if args.architecture not in {ARCHITECTURE, BASELINE_ARCHITECTURE}:
        raise ValueError("Capacity executor accepts only the candidate or A0-PM baseline")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Capacity output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    config = load_project_config(args.config)
    config["data_commit"] = _data_commit(config["data_root"])
    config["fold_identity"] = f"electrostatic-development-fold-{args.fold}"
    folds = json.loads(args.folds.read_text(encoding="utf-8-sig"))
    fold = next(item for item in folds["folds"] if int(item["fold"]) == args.fold)
    train_ids = electrostatic_fold_train_ids(folds, args.fold)
    development_ids = [str(value) for value in fold["development"]]
    ids, subset_manifest = load_response_subset(
        args.material_ids_file,
        fold=args.fold,
        allowed_ids=train_ids,
    )
    if set(ids) & set(development_ids):
        raise ValueError("Capacity IDs overlap the development panel")
    records = load_gmtnet_records(config["data_root"])
    cache_key = graph_cache_key(
        records, float(config["cutoff"]), int(config["max_neighbors"])
    )
    dataset = _dataset(config, records, ids, cache_key, cache_graphs=True)
    dataset.warm_graph_cache()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = make_model(args.architecture, config).to(device)
    if args.pretrained_encoder is not None:
        load_structure_pretraining(
            model,
            args.architecture,
            args.pretrained_encoder,
            device,
            train_ids,
            development_ids,
            config,
        )
    tower = model.piezo_generator
    optimizer = torch.optim.AdamW(tower.parameters(), lr=args.learning_rate, weight_decay=0.0)
    history: list[dict[str, object]] = []
    best_score = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        tower.train()
        total = 0.0
        steps = 0
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            _, graph_features, context = tower.encode_response_features(batch)
            prediction = tower.decode_electronic_piezo(graph_features, context)
            loss = irrep_balanced_capacity_loss(prediction, batch.y_electronic_piezo)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite independent-l1 capacity loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        metrics, _ = _evaluate(tower, eval_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total / max(steps, 1),
            "electronic_metrics": metrics,
        }
        history.append(row)
        score = float(metrics["mean_active_relative_frobenius_error"])
        torch.save(
            {
                "schema": 1,
                "architecture": args.architecture,
                "epoch": epoch,
                "model": tower.state_dict(),
                "optimizer": optimizer.state_dict(),
                "capacity_provenance": {
                    "material_ids": sorted(ids),
                    "material_id_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
                    "fold": args.fold,
                    "seed": args.seed,
                    "frozen_validation_test_labels_read": False,
                },
            },
            args.output_dir / "last.pt",
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "schema": 1,
                    "architecture": args.architecture,
                    "epoch": epoch,
                    "model": tower.state_dict(),
                    "metrics": metrics,
                    "capacity_provenance": {
                        "material_ids": sorted(ids),
                        "material_id_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
                        "fold": args.fold,
                        "seed": args.seed,
                        "frozen_validation_test_labels_read": False,
                    },
                },
                args.output_dir / "best.pt",
            )
    summary = {
        "schema": 1,
        "protocol": "train-only same-ID electronic l=1 readout capacity",
        "architecture": args.architecture,
        "material_count": len(ids),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_active_relative_error": best_score,
        "frozen_validation_test_labels_read": False,
        "pretrained_encoder": str(args.pretrained_encoder) if args.pretrained_encoder else None,
        "subset_manifest": subset_manifest,
        "encoder_width_multiplier": encoder_width_multiplier_for_architecture(
            args.architecture, config
        ),
        "graph_cache_key": cache_key,
        "history": history,
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--material-ids-file", type=Path, required=True)
    parser.add_argument("--pretrained-encoder", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=(BASELINE_ARCHITECTURE, ARCHITECTURE),
        default=ARCHITECTURE,
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run_capacity(args), indent=2))


if __name__ == "__main__":
    main()
