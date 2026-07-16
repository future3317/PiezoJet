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
from .data import PiezoDataset, graph_cache_key, load_gmtnet_records
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
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
    splits = load_explicit_splits(args.splits_file, {str(record["JARVIS_ID"]) for record in records})
    ids = sorted(splits["train"])
    pretraining_provenance = provenance(ids, args.splits_file, "train")
    cache_key = graph_cache_key(records, float(cfg["cutoff"]), int(cfg["max_neighbors"]))
    dataset = PiezoDataset(
        records, ids, float(cfg["cutoff"]), int(cfg["max_neighbors"]),
        processed_dir=cfg["processed_dir"], cache_key=cache_key, project_targets=False,
    )
    loader = DataLoader(
        dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=int(cfg["num_workers"]), pin_memory=device.type == "cuda",
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
