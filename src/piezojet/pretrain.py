"""Self-supervised structural pretraining for the production PiezoJet encoder.

The targets are deliberately independent of piezoelectric labels: masked atomic
species and translation-free Cartesian coordinate denoising.  This lets the
encoder learn local polar motifs before the scarce response labels are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from .data import PiezoDataset, graph_cache_key, load_gmtnet_records
from .model import model_from_config
from .pretraining_protocol import provenance
from .project_config import load_project_config
from .train import _data_commit, device_from_config, load_explicit_splits, seed_everything


class StructurePretrainingHead(nn.Module):
    """Pretraining heads for invariant scalars and Cartesian polar modes."""

    def __init__(self, scalar_dim: int, channels: int):
        super().__init__()
        self.species = nn.Sequential(nn.Linear(scalar_dim, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, 119))
        self.displacement = nn.Sequential(nn.Linear(scalar_dim, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, channels))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.species(features.scalar), torch.einsum("nc,nci->ni", self.displacement(features.scalar), features.vector)


def _corrupt_structure(batch, mask_probability: float, noise_std: float):
    corrupted = batch.clone()
    mask = torch.rand(batch.num_nodes, device=batch.pos.device) < mask_probability
    if not bool(mask.any()):
        mask[torch.randint(batch.num_nodes, (1,), device=batch.pos.device)] = True
    masked_z = batch.z.clone()
    masked_z[mask] = 0
    noise = torch.randn_like(batch.pos) * noise_std
    # A global Cartesian translation is unobservable to an E(3)-equivariant
    # encoder, so remove it from every graph before constructing the target.
    mean_noise = scatter(noise, batch.batch, dim=0, dim_size=batch.num_graphs, reduce="mean")
    noise = noise - mean_noise[batch.batch]
    corrupted.pos = batch.pos + noise
    return corrupted, masked_z, mask, noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None, help="Exact optimizer-update budget; takes precedence over --epochs.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--splits-file", type=Path,
        help="Frozen train/val/test panel. Production pretraining uses its train IDs only.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Bounded smoke-run only; the production pipeline uses all structures.")
    args = parser.parse_args()
    cfg = load_project_config(args.config)
    cfg["data_commit"] = _data_commit(cfg["data_root"])
    epochs = int(cfg["pretrain_epochs"] if args.epochs is None else args.epochs)
    output = Path(cfg["pretraining_output_dir"] if args.output_dir is None else args.output_dir)
    if epochs < 1:
        raise ValueError("Pretraining epochs must be positive")
    if args.updates is not None and args.updates < 1:
        raise ValueError("--updates must be positive")
    seed_everything(int(cfg["seed"]))
    device = device_from_config(cfg["device"])
    records = load_gmtnet_records(cfg["data_root"])
    split_file = args.splits_file or (
        Path(cfg["pretrain_splits_file"]) if cfg.get("pretrain_splits_file") else None
    )
    if split_file is None or not split_file.is_file():
        raise ValueError(
            "Inductive production pretraining requires --splits-file (or pretrain_splits_file in config)."
        )
    splits = load_explicit_splits(split_file, {str(record["JARVIS_ID"]) for record in records})
    all_ids = sorted(splits["train"])
    pretraining_provenance = provenance(all_ids, split_file, "train", cfg)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        all_ids = all_ids[: args.max_samples]
        pretraining_provenance = provenance(all_ids, split_file, "train", cfg)
    cache_key = graph_cache_key(records, float(cfg["cutoff"]), int(cfg["max_neighbors"]))
    dataset = PiezoDataset(records, all_ids, float(cfg["cutoff"]), int(cfg["max_neighbors"]), processed_dir=cfg["processed_dir"], cache_key=cache_key, project_targets=False)
    loader_options = {"num_workers": int(cfg["num_workers"]), "pin_memory": device.type == "cuda"}
    if loader_options["num_workers"] > 0:
        loader_options["persistent_workers"] = True
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True, **loader_options)
    model = model_from_config(cfg).to(device)
    head = StructurePretrainingHead(model.encoder.scalar_dim, model.encoder.channels).to(device)
    optimizer = torch.optim.AdamW(list(model.encoder.parameters()) + list(head.parameters()), lr=float(cfg["pretrain_learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    updates_completed = 0
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            corrupted, masked_z, mask, noise = _corrupt_structure(batch, float(cfg["mask_probability"]), float(cfg["coordinate_noise_std"]))
            features = model.encode(corrupted, masked_z)
            species_logits, predicted_noise = head(features)
            species_loss = nn.functional.cross_entropy(species_logits[mask], batch.z[mask])
            denoise_loss = nn.functional.mse_loss(predicted_noise, noise)
            loss = species_loss + float(cfg["denoising_weight"]) * denoise_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(head.parameters()), max_norm=10.0)
            optimizer.step()
            updates_completed += 1
            total += float(loss.detach()) * batch.num_graphs
            count += batch.num_graphs
            if args.updates is not None and updates_completed >= args.updates:
                break
        value = total / max(count, 1)
        history.append({"epoch": epoch, "loss": value, "optimizer_updates": updates_completed})
        payload = {
            "encoder": model.encoder.state_dict(), "config": cfg, "epoch": epoch, "loss": value,
            "architecture": "cartesian_local_environment_v1",
            "objective": "masked_species_plus_translation_free_coordinate_denoising",
            "pretraining_provenance": pretraining_provenance,
        }
        torch.save(payload, output / "last_encoder.pt")
        if value < best:
            best = value
            torch.save(payload, output / "best_encoder.pt")
        print(f"pretrain epoch={epoch} loss={value:.6g}")
        if args.updates is not None and updates_completed >= args.updates:
            break
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
