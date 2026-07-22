"""Fold-train-only BEC response-aware initialization for parameter-matched A0.

This is supervised pretraining, not a new response model or a self-supervised
objective.  It uses only source BEC labels from the complete training side of
one electrostatic development fold, then exports exactly one A0 BEC tower.
The downstream runner verifies this contract before loading any parameters.
"""

from __future__ import annotations

import argparse
import json
import math
from hashlib import sha256
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .build_electrostatic_development_folds import electrostatic_fold_train_ids
from .data import (
    deterministic_subset,
    formula,
    graph_cache_key,
    load_gmtnet_records,
)
from .electronic_capacity import born_material_balanced_loss
from .electrostatic_fold_adjudication import (
    _dataset,
    encoder_width_multiplier_for_architecture,
    make_model,
)
from .loader_runtime import loader_options
from .pretrain_e3nn import logical_pretraining_batch_sizes
from .pretraining_protocol import (
    BEC_RESPONSE_PRETRAINING_ARCHITECTURE,
    BEC_RESPONSE_PRETRAINING_OBJECTIVE,
    provenance,
)
from .project_config import load_project_config
from .train import _data_commit, _git_commit, seed_everything


_ARCHITECTURE = "a0_parameter_matched_irreps"


def _valid_commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError("--code-commit must be one 40-character Git commit SHA")
    return value.lower()


def _epoch_indices(materials: int, seed: int, epoch: int) -> list[int]:
    """Deterministic epoch order makes a resumed run schedule-identical."""
    if materials < 1:
        raise ValueError("BEC response pretraining needs at least one material")
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(materials, generator=generator).tolist()


def _response_pretraining_ids(
    folds: dict[str, object], fold_index: int, known_ids: set[str], train_limit: int, seed: int
) -> tuple[list[str], list[str]]:
    fold = next((entry for entry in folds["folds"] if entry["fold"] == fold_index), None)
    if fold is None:
        raise ValueError(f"Fold {fold_index} is absent from electrostatic folds")
    full_train_ids = electrostatic_fold_train_ids(folds, fold_index)
    if set(full_train_ids) - known_ids:
        raise ValueError("Electrostatic fold contains unknown training IDs")
    development_ids = [str(value) for value in fold["development"]]
    ids = deterministic_subset(full_train_ids, train_limit, seed + 3000)
    if not ids:
        raise ValueError("BEC response-pretraining panel is empty")
    return ids, development_ids


def validate_resume_payload(
    saved: dict[str, object],
    expected_provenance: dict[str, object],
    expected_contract: dict[str, object],
) -> None:
    """Require an optimizer-complete, exact-panel BEC pretraining resume."""
    if saved.get("architecture") != BEC_RESPONSE_PRETRAINING_ARCHITECTURE:
        raise ValueError("Resume checkpoint is not BEC response-aware pretraining")
    if saved.get("pretraining_provenance") != expected_provenance:
        raise ValueError("Resume checkpoint BEC-pretraining provenance differs")
    if saved.get("response_pretraining_contract") != expected_contract:
        raise ValueError("Resume checkpoint BEC-pretraining contract differs")
    if not isinstance(saved.get("born_tower"), dict) or not isinstance(saved.get("optimizer"), dict):
        raise ValueError("BEC response-pretraining checkpoint lacks tower/optimizer state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--folds", type=Path,
        default=Path("data/processed/electrostatic_development_folds_v2.json"),
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logical-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--code-commit")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"), default="highest"
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.logical_batch_size) < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if args.logical_batch_size < args.batch_size or args.logical_batch_size % args.batch_size:
        raise ValueError("logical batch size must be a multiple of physical batch size")
    if args.train_limit < 0 or args.num_workers < 0:
        raise ValueError("train limit and num workers cannot be negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("weight decay must be finite and nonnegative")

    config = load_project_config(args.config)
    config["data_commit"] = _data_commit(config["data_root"])
    config["code_commit"] = _valid_commit(args.code_commit or _git_commit())
    config["seed"] = args.seed
    config["fold_identity"] = f"electrostatic-development-fold-{args.fold}"
    config["electrostatic_encoder_width_multiplier"] = encoder_width_multiplier_for_architecture(
        _ARCHITECTURE, config
    )
    torch.set_float32_matmul_precision(args.matmul_precision)
    seed_everything(args.seed)

    folds = json.loads(args.folds.read_text(encoding="utf-8-sig"))
    frozen_read = folds.get("frozen_validation_test_labels_read", False)
    role = str(folds.get("role", "")).lower()
    if frozen_read is not False or "frozen" not in role or "unread" not in role:
        raise ValueError("Electrostatic fold map is not explicitly frozen-panel safe")
    records = load_gmtnet_records(config["data_root"])
    known_ids = {str(record["JARVIS_ID"]) for record in records}
    ids, development_ids = _response_pretraining_ids(
        folds, args.fold, known_ids, args.train_limit, args.seed
    )
    by_id = {str(record["JARVIS_ID"]): record for record in records}
    train_formulas = {formula(by_id[identifier]) for identifier in ids}
    development_formulas = {formula(by_id[identifier]) for identifier in development_ids}
    overlap = sorted(train_formulas & development_formulas)
    if overlap:
        raise ValueError(
            "BEC response pretraining overlaps development formulas: " f"{overlap[:5]}"
        )
    pretraining_provenance = provenance(ids, args.folds, "train", config)
    pretraining_provenance.update({
        "development_fold": args.fold,
        "development_ids": sorted(development_ids),
        "development_material_id_sha256": sha256(
            "\n".join(sorted(set(development_ids))).encode("utf-8")
        ).hexdigest(),
        "development_formula_overlap_count": 0,
        "response_task": "born",
        "response_label_count": len(ids),
        "frozen_validation_test_labels_read": False,
    })
    cache_key = graph_cache_key(
        records, float(config["cutoff"]), int(config["max_neighbors"])
    )
    dataset = _dataset(config, records, ids, cache_key)
    if len(dataset) != len(ids):
        raise RuntimeError("BEC response-pretraining dataset lost material IDs")
    logical_sizes = logical_pretraining_batch_sizes(
        len(dataset), args.batch_size, args.logical_batch_size
    )
    contract = {
        "objective": BEC_RESPONSE_PRETRAINING_OBJECTIVE,
        "response_task": "born",
        "downstream_architecture": _ARCHITECTURE,
        "encoder_width_multiplier": float(config["electrostatic_encoder_width_multiplier"]),
        "response_material_count": len(dataset),
        "physical_batch_size": args.batch_size,
        "logical_batch_size": args.logical_batch_size,
        "optimizer_updates_per_exposure_epoch": len(logical_sizes),
        "optimizer": "AdamW",
        "code_commit": config["code_commit"],
    }
    device = torch.device(args.device)
    model = make_model(_ARCHITECTURE, config)
    tower = model.born_generator.to(device)
    del model
    optimizer = torch.optim.AdamW(
        tower.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.resume is None:
        if args.output_dir.exists():
            raise FileExistsError(f"BEC pretraining output directory exists: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        history: list[dict[str, object]] = []
        best = float("inf")
        start_epoch = 1
    else:
        if not args.output_dir.is_dir() or not args.resume.resolve().is_relative_to(args.output_dir.resolve()):
            raise ValueError("Resume checkpoint must be inside an existing output directory")
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        validate_resume_payload(saved, pretraining_provenance, contract)
        tower.load_state_dict(saved["born_tower"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history = list(saved.get("history", []))
        best = min((float(row["loss"]) for row in history), default=float(saved["loss"]))
        start_epoch = int(saved["epoch"]) + 1
        if start_epoch > args.epochs:
            raise ValueError("Resume checkpoint has already reached the requested epoch count")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_set = Subset(dataset, _epoch_indices(len(dataset), args.seed, epoch))
        loader = DataLoader(
            epoch_set,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_options(args.num_workers, cuda=device.type == "cuda"),
        )
        tower.train()
        optimizer.zero_grad(set_to_none=True)
        total, graphs, logical_graphs, updates = 0.0, 0, 0, 0
        logical_target = min(args.logical_batch_size, len(dataset))
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            if not torch.isfinite(batch.y_born).all():
                raise ValueError("BEC response-pretraining panel has a non-finite BEC label")
            node_features, _, context = tower.encode_response_features(batch)
            prediction = tower.decode_born(node_features, context, batch.batch)
            loss = born_material_balanced_loss(prediction, batch.y_born, batch.batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite BEC response-pretraining loss")
            graphs_in_batch = int(batch.num_graphs)
            if graphs_in_batch > logical_target - logical_graphs:
                raise RuntimeError("A BEC batch crossed a logical-batch boundary")
            (loss * graphs_in_batch / logical_target).backward()
            total += float(loss.detach()) * graphs_in_batch
            graphs += graphs_in_batch
            logical_graphs += graphs_in_batch
            if logical_graphs == logical_target:
                if not all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in tower.parameters() if parameter.grad is not None
                ):
                    raise FloatingPointError("Non-finite BEC response-pretraining gradient")
                torch.nn.utils.clip_grad_norm_(tower.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                logical_graphs = 0
                logical_target = min(args.logical_batch_size, len(dataset) - graphs)
        if logical_graphs or updates != len(logical_sizes):
            raise RuntimeError("BEC response-pretraining update accounting drifted")
        value = total / max(graphs, 1)
        row = {
            "epoch": epoch,
            "loss": value,
            "optimizer_updates": updates,
            "response_material_exposures": graphs,
        }
        history.append(row)
        payload = {
            "architecture": BEC_RESPONSE_PRETRAINING_ARCHITECTURE,
            "objective": BEC_RESPONSE_PRETRAINING_OBJECTIVE,
            "born_tower": tower.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "loss": value,
            "history": history,
            "pretraining_provenance": pretraining_provenance,
            "response_pretraining_contract": contract,
            "code_commit": config["code_commit"],
        }
        torch.save(payload, args.output_dir / "last_bec_tower.pt")
        if value < best:
            best = value
            torch.save(payload, args.output_dir / "best_bec_tower.pt")
        print(f"pretrain_bec_e3nn epoch={epoch} loss={value:.6g}")
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
