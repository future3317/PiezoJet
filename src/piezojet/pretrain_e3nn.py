"""Inductive self-supervised pretraining for the matched PBC e3nn baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from e3nn import o3
from torch import nn
from torch_geometric.loader import DataLoader

from .baselines import e3nn_direct_baseline_from_config
from .build_electrostatic_development_folds import electrostatic_fold_train_ids
from .data import (
    GRAPH_CACHE_SCHEMA,
    PiezoDataset,
    deterministic_subset,
    formula,
    graph_cache_key,
    load_gmtnet_records,
)
from .pretrain import _corrupt_structure
from .pretraining_protocol import provenance
from .loader_runtime import loader_options
from .project_config import load_project_config
from .train import (
    _data_commit,
    _git_commit,
    device_from_config,
    load_explicit_splits,
    seed_everything,
)


class E3nnStructurePretrainingHead(nn.Module):
    """Steerable masked-species and vector-denoising readouts."""

    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.species = o3.Linear(irreps, o3.Irreps("119x0e"))
        self.displacement = o3.Linear(irreps, o3.Irreps("1x1o"))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.species(features), self.displacement(features)


def validate_resume_payload(
    saved: dict[str, object],
    expected_provenance: dict[str, object],
    expected_contract: dict[str, object] | None = None,
) -> None:
    """Require an exact-panel, optimizer-complete resumable checkpoint."""
    if saved.get("architecture") != "e3nn_periodic_v1":
        raise ValueError("Resume checkpoint is not an e3nn periodic pretraining state")
    if saved.get("pretraining_provenance") != expected_provenance:
        raise ValueError(
            "Resume checkpoint pretraining provenance differs from the exact current panel"
        )
    if not isinstance(saved.get("head"), dict) or not isinstance(saved.get("optimizer"), dict):
        raise ValueError("Legacy pretraining checkpoint lacks resumable head/optimizer state")
    if expected_contract is not None and saved.get("pretraining_contract") != expected_contract:
        raise ValueError("Resume checkpoint pretraining contract differs from this run")


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


def logical_pretraining_batch_sizes(
    materials: int, physical_batch_size: int, logical_batch_size: int
) -> list[int]:
    """Return complete material counts for one exposure epoch of AdamW updates."""
    if min(materials, physical_batch_size, logical_batch_size) < 1:
        raise ValueError("Material and batch sizes must be positive")
    if logical_batch_size < physical_batch_size:
        raise ValueError("Logical batch size cannot be smaller than physical batch size")
    if logical_batch_size != materials and logical_batch_size % physical_batch_size:
        raise ValueError("Logical batch size must be divisible by physical batch size")
    full, remainder = divmod(materials, logical_batch_size)
    return [logical_batch_size] * full + ([remainder] if remainder else [])


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
    parser.add_argument(
        "--encoder-width-multiplier",
        type=float,
        default=None,
        help="Explicit hidden-width contract for a parameter-matched response encoder",
    )
    parser.add_argument(
        "--code-commit",
        help="Pinned 40-character source commit recorded by the execution plan",
    )
    parser.add_argument(
        "--logical-batch-size",
        type=int,
        default=0,
        help="Graphs per AdamW update; must be a multiple of the physical batch size",
    )
    parser.add_argument("--accumulate-to-one-update", action="store_true")
    parser.add_argument(
        "--empty-cuda-cache-each-epoch",
        action="store_true",
        help="Release cached CUDA blocks after each epoch on memory-constrained GPUs",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="Resume an exact-panel e3nn pretraining checkpoint at its next epoch",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Background graph-loading workers; does not alter the logical batch",
    )
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"),
        default="highest", help="PyTorch float32 matmul precision policy",
    )
    args = parser.parse_args()
    cfg = load_project_config(args.config)
    cfg["data_commit"] = _data_commit(cfg["data_root"])
    code_commit = args.code_commit or _git_commit()
    if len(code_commit) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in code_commit
    ):
        raise ValueError("--code-commit must be one 40-character Git commit SHA")
    cfg["code_commit"] = code_commit.lower()
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        cfg["batch_size"] = args.batch_size
    if args.encoder_width_multiplier is not None:
        if args.encoder_width_multiplier <= 0.0 or not math.isfinite(
            args.encoder_width_multiplier
        ):
            raise ValueError("--encoder-width-multiplier must be finite and positive")
        cfg["electrostatic_encoder_width_multiplier"] = (
            args.encoder_width_multiplier
        )
    if args.logical_batch_size < 0:
        raise ValueError("--logical-batch-size cannot be negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.accumulate_to_one_update and args.logical_batch_size:
        raise ValueError(
            "--accumulate-to-one-update and --logical-batch-size are mutually exclusive"
        )
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
        pretraining_provenance = provenance(ids, args.splits_file, "train", cfg)
    else:
        if args.fold is None:
            raise ValueError("--fold is required with --electrostatic-folds")
        folds = json.loads(args.electrostatic_folds.read_text(encoding="utf-8-sig"))
        fold = next(
            (
                entry
                for entry in folds["folds"]
                if entry["fold"] == args.fold
            ),
            None,
        )
        if fold is None:
            raise ValueError(f"Fold {args.fold} is absent from electrostatic folds")
        ids = electrostatic_pretraining_ids(
            folds,
            args.fold,
            known_ids,
            args.train_limit,
            int(cfg["seed"]),
        )
        pretraining_provenance = provenance(
            ids, args.electrostatic_folds, "train", cfg
        )
        pretraining_provenance.update({
            "development_fold": args.fold,
            "train_limit": args.train_limit,
            "selection_seed": int(cfg["seed"]) + 1000,
            "frozen_validation_test_labels_read": False,
        })
        by_id = {str(record["JARVIS_ID"]): record for record in records}
        development_ids = [str(value) for value in fold["development"]]
        train_formulas = {formula(by_id[identifier]) for identifier in ids}
        development_formulas = {
            formula(by_id[identifier]) for identifier in development_ids
        }
        formula_overlap = sorted(train_formulas & development_formulas)
        if formula_overlap:
            raise ValueError(
                "Electrostatic structural pretraining overlaps development formulas: "
                f"{formula_overlap[:5]}"
            )
        pretraining_provenance.update({
            "structure_material_count": len(ids),
            "response_label_count": 0,
            "development_material_count": len(development_ids),
            "development_formula_overlap_count": 0,
            "development_formula_overlap": [],
            "graph_cache_schema": GRAPH_CACHE_SCHEMA,
            "code_commit": cfg["code_commit"],
        })
    cache_key = graph_cache_key(records, float(cfg["cutoff"]), int(cfg["max_neighbors"]))
    dataset = PiezoDataset(
        records, ids, float(cfg["cutoff"]), int(cfg["max_neighbors"]),
        processed_dir=cfg["processed_dir"], cache_key=cache_key, project_targets=False,
    )
    physical_batch_size = int(cfg["batch_size"])
    logical_batch_size = (
        len(dataset)
        if args.accumulate_to_one_update
        else (args.logical_batch_size or physical_batch_size)
    )
    logical_update_sizes = logical_pretraining_batch_sizes(
        len(dataset), physical_batch_size, logical_batch_size
    )
    loader = DataLoader(
        dataset, batch_size=physical_batch_size, shuffle=True,
        **loader_options(args.num_workers, cuda=device.type == "cuda"),
    )
    model = e3nn_direct_baseline_from_config(cfg).to(device)
    head = E3nnStructurePretrainingHead(model.encoder.hidden_irreps).to(device)
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(head.parameters()),
        lr=float(cfg["pretrain_learning_rate"]), weight_decay=float(cfg["weight_decay"]),
    )
    if args.resume is None:
        if args.output_dir.exists():
            raise FileExistsError(
                f"Fresh pretraining output directory already exists: {args.output_dir}"
            )
        args.output_dir.mkdir(parents=True, exist_ok=False)
    elif not args.output_dir.is_dir():
        raise FileNotFoundError(f"Resume output directory is absent: {args.output_dir}")
    pretraining_contract = {
        "structure_material_count": len(dataset),
        "response_label_count": 0,
        "physical_batch_size": physical_batch_size,
        "logical_batch_size": logical_batch_size,
        "optimizer_updates_per_exposure_epoch": len(logical_update_sizes),
        "optimizer": "AdamW",
        "electrostatic_encoder_width_multiplier": float(
            cfg.get("electrostatic_encoder_width_multiplier", 1.0)
        ),
        "objective": "masked_species_plus_translation_free_coordinate_denoising",
        "code_commit": cfg["code_commit"],
    }
    history, best, start_epoch = [], float("inf"), 1
    training_schedule: list[dict[str, object]] = []
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        validate_resume_payload(saved, pretraining_provenance, pretraining_contract)
        model.encoder.load_state_dict(saved["encoder"], strict=True)
        head.load_state_dict(saved["head"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history = list(saved.get("history", []))
        start_epoch = int(saved["epoch"]) + 1
        best = min((float(row["loss"]) for row in history), default=float(saved["loss"]))
        training_schedule = list(saved.get("training_schedule", []))
        if not training_schedule:
            training_schedule.append({
                "start_epoch": 1,
                "end_epoch": int(saved["epoch"]),
                "physical_batch_size": int(saved.get("config", {}).get("batch_size", cfg["batch_size"])),
                "logical_batch_size": int(
                    saved.get("pretraining_contract", {}).get(
                        "logical_batch_size", cfg["batch_size"]
                    )
                ),
                "empty_cuda_cache_each_epoch": False,
            })
        if start_epoch > epochs:
            raise ValueError("Resume checkpoint has already reached the requested epoch count")
    training_schedule.append({
        "start_epoch": start_epoch,
        "end_epoch": epochs,
        "physical_batch_size": int(cfg["batch_size"]),
        "logical_batch_size": logical_batch_size,
        "empty_cuda_cache_each_epoch": bool(args.empty_cuda_cache_each_epoch),
        "num_workers": int(args.num_workers),
        "matmul_precision": args.matmul_precision,
    })
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        head.train()
        total, graphs, optimizer_updates = 0.0, 0, 0
        logical_graphs = 0
        logical_target = min(logical_batch_size, len(dataset))
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
            if int(batch.num_graphs) > logical_target - logical_graphs:
                raise RuntimeError(
                    "A physical pretraining batch crossed a logical-batch boundary"
                )
            (loss * int(batch.num_graphs) / logical_target).backward()
            logical_graphs += int(batch.num_graphs)
            if logical_graphs == logical_target:
                parameters = list(model.encoder.parameters()) + list(head.parameters())
                if not all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in parameters
                    if parameter.grad is not None
                ):
                    raise FloatingPointError(
                        "Non-finite accumulated e3nn pretraining gradient"
                    )
                torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(head.parameters()), max_norm=10.0)
                optimizer.step()
                optimizer_updates += 1
                optimizer.zero_grad(set_to_none=True)
                logical_graphs = 0
                remaining = len(dataset) - (graphs + int(batch.num_graphs))
                logical_target = min(logical_batch_size, remaining)
            total += float(loss.detach()) * batch.num_graphs
            graphs += batch.num_graphs
        if logical_graphs:
            raise RuntimeError("Incomplete logical pretraining batch at epoch end")
        if optimizer_updates != len(logical_update_sizes):
            raise RuntimeError("Pretraining optimizer-update accounting drifted")
        value = total / max(graphs, 1)
        history.append({
            "epoch": epoch,
            "loss": value,
            "optimizer_updates": optimizer_updates,
            "structure_exposures": graphs,
        })
        if device.type == "cuda" and args.empty_cuda_cache_each_epoch:
            torch.cuda.empty_cache()
        payload = {
            "encoder": model.encoder.state_dict(), "config": cfg, "epoch": epoch, "loss": value,
            "head": head.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
            "training_schedule": training_schedule,
            "architecture": "e3nn_periodic_v1",
            "objective": "masked_species_plus_translation_free_coordinate_denoising",
            "pretraining_provenance": pretraining_provenance,
            "pretraining_contract": pretraining_contract,
            "code_commit": cfg["code_commit"],
        }
        torch.save(payload, args.output_dir / "last_encoder.pt")
        if value < best:
            best = value
            torch.save(payload, args.output_dir / "best_encoder.pt")
        print(f"pretrain_e3nn epoch={epoch} loss={value:.6g}")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
