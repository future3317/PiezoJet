"""Inductive self-supervised pretraining for the matched PBC e3nn baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from e3nn import o3
from torch import nn
from torch_geometric.loader import DataLoader

from .baselines import e3nn_direct_baseline_from_config
from .build_electrostatic_development_folds import electrostatic_fold_train_ids
from .data import (
    PiezoDataset,
    deterministic_subset,
    graph_cache_key,
    load_gmtnet_records,
)
from .pretrain import _corrupt_structure
from .pretraining_protocol import provenance
from .project_config import load_project_config
from .train import device_from_config, load_explicit_splits, seed_everything


class E3nnStructurePretrainingHead(nn.Module):
    """Steerable masked-species and vector-denoising readouts."""

    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.species = o3.Linear(irreps, o3.Irreps("119x0e"))
        self.displacement = o3.Linear(irreps, o3.Irreps("1x1o"))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.species(features), self.displacement(features)


def electrostatic_pretraining_ids(
    folds: dict[str, object],
    fold_index: int,
    known_ids: set[str],
    train_limit: int,
    seed: int,
) -> list[str]:
    """Resolve one formula-safe fold train panel without reading held-out labels."""
    fold = next(
        (entry for entry in folds["folds"] if entry["fold"] == fold_index),
        None,
    )
    if fold is None:
        raise ValueError(f"Fold {fold_index} is absent from electrostatic folds")
    ids = electrostatic_fold_train_ids(folds, fold_index)
    unknown = set(ids) - known_ids
    if unknown:
        raise ValueError(f"Unknown electrostatic train IDs: {sorted(unknown)[:5]}")
    return deterministic_subset(ids, train_limit, seed + 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--splits-file", type=Path)
    source.add_argument("--electrostatic-folds", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulate-to-one-update", action="store_true")
    args = parser.parse_args()
    cfg = load_project_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        cfg["batch_size"] = args.batch_size
    epochs = int(cfg["pretrain_epochs"] if args.epochs is None else args.epochs)
    if epochs < 1:
        raise ValueError("Pretraining epochs must be positive")
    seed_everything(int(cfg["seed"]))
    device = device_from_config(str(cfg["device"]))
    records = load_gmtnet_records(cfg["data_root"])
    known_ids = {str(record["JARVIS_ID"]) for record in records}
    if args.splits_file is not None:
        if args.fold is not None or args.train_limit:
            raise ValueError("--fold/--train-limit require --electrostatic-folds")
        splits = load_explicit_splits(args.splits_file, known_ids)
        ids = sorted(splits["train"])
        pretraining_provenance = provenance(ids, args.splits_file, "train")
    else:
        if args.fold is None:
            raise ValueError("--fold is required with --electrostatic-folds")
        folds = json.loads(args.electrostatic_folds.read_text(encoding="utf-8-sig"))
        ids = electrostatic_pretraining_ids(
            folds,
            args.fold,
            known_ids,
            args.train_limit,
            int(cfg["seed"]),
        )
        pretraining_provenance = provenance(
            ids, args.electrostatic_folds, "train"
        )
        pretraining_provenance.update({
            "development_fold": args.fold,
            "train_limit": args.train_limit,
            "selection_seed": int(cfg["seed"]) + 1000,
            "frozen_validation_test_labels_read": False,
        })
    cache_key = graph_cache_key(records, float(cfg["cutoff"]), int(cfg["max_neighbors"]))
    dataset = PiezoDataset(
        records, ids, float(cfg["cutoff"]), int(cfg["max_neighbors"]),
        processed_dir=cfg["processed_dir"], cache_key=cache_key, project_targets=False,
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    model = e3nn_direct_baseline_from_config(cfg).to(device)
    head = E3nnStructurePretrainingHead(model.encoder.hidden_irreps).to(device)
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(head.parameters()),
        lr=float(cfg["pretrain_learning_rate"]), weight_decay=float(cfg["weight_decay"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        total, graphs = 0.0, 0
        if args.accumulate_to_one_update:
            optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            corrupted, masked_z, mask, noise = _corrupt_structure(
                batch, float(cfg["mask_probability"]), float(cfg["coordinate_noise_std"])
            )
            features = model.encoder(corrupted, masked_z)
            species_logits, predicted_noise = head(features)
            loss = nn.functional.cross_entropy(species_logits[mask], batch.z[mask])
            loss = loss + float(cfg["denoising_weight"]) * nn.functional.mse_loss(predicted_noise, noise)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite e3nn pretraining loss")
            if args.accumulate_to_one_update:
                (loss * batch.num_graphs / len(dataset)).backward()
            else:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(head.parameters()), max_norm=10.0)
                optimizer.step()
            total += float(loss.detach()) * batch.num_graphs
            graphs += batch.num_graphs
        if args.accumulate_to_one_update:
            parameters = list(model.encoder.parameters()) + list(head.parameters())
            if not all(torch.isfinite(parameter.grad).all() for parameter in parameters if parameter.grad is not None):
                raise FloatingPointError("Non-finite accumulated e3nn pretraining gradient")
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            optimizer.step()
        value = total / max(graphs, 1)
        history.append({"epoch": epoch, "loss": value})
        payload = {
            "encoder": model.encoder.state_dict(), "config": cfg, "epoch": epoch, "loss": value,
            "architecture": "e3nn_periodic_v1",
            "objective": "masked_species_plus_translation_free_coordinate_denoising",
            "pretraining_provenance": pretraining_provenance,
        }
        torch.save(payload, args.output_dir / "last_encoder.pt")
        if value < best:
            best = value
            torch.save(payload, args.output_dir / "best_encoder.pt")
        print(f"pretrain_e3nn epoch={epoch} loss={value:.6g}")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
