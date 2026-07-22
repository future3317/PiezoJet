"""Fold-train-only electronic-piezo response-aware initialization for A0-PM.

Only ``piezo_generator`` is optimized and exported.  BEC and dielectric
parameters are never instantiated as trainable targets in this executor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from hashlib import sha256
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .data import formula, graph_cache_key, load_gmtnet_records
from .electrostatic_subset import load_response_subset
from .electrostatic_fold_adjudication import (
    _dataset,
    encoder_width_multiplier_for_architecture,
    make_model,
)
from .electronic_capacity import irrep_balanced_capacity_loss
from .loader_runtime import loader_options
from .pretrain_bec_e3nn import (
    _multi_epoch_batches,
    _response_pretraining_ids,
    _valid_commit,
)
from .pretrain_e3nn import logical_pretraining_batch_sizes
from .pretraining_protocol import (
    ELECTRONIC_RESPONSE_PRETRAINING_ARCHITECTURE,
    ELECTRONIC_RESPONSE_PRETRAINING_OBJECTIVE,
    provenance,
)
from .project_config import load_project_config
from .train import _data_commit, _git_commit, seed_everything


_ARCHITECTURE = "a0_parameter_matched_irreps"


def validate_resume_payload(
    saved: dict[str, object],
    expected_provenance: dict[str, object],
    expected_contract: dict[str, object],
) -> None:
    if saved.get("architecture") != ELECTRONIC_RESPONSE_PRETRAINING_ARCHITECTURE:
        raise ValueError("Resume checkpoint is not electronic response pretraining")
    if saved.get("pretraining_provenance") != expected_provenance:
        raise ValueError("Resume checkpoint electronic provenance differs")
    if saved.get("response_pretraining_contract") != expected_contract:
        raise ValueError("Resume checkpoint electronic contract differs")
    if not isinstance(saved.get("piezo_tower"), dict) or not isinstance(
        saved.get("optimizer"), dict
    ):
        raise ValueError("Electronic resume checkpoint lacks tower/optimizer state")


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
    parser.add_argument(
        "--train-ids-file", type=Path,
        help="Preregistered fold-train response panel (mutually exclusive with --train-limit)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--code-commit")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per persistent worker without changing order",
    )
    parser.add_argument(
        "--graph-cache-key",
        help="Existing canonical graph-cache key; avoids recomputing a corpus hash",
    )
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
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive")
    if args.train_ids_file is not None and args.train_limit:
        raise ValueError("--train-ids-file and --train-limit are mutually exclusive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("weight decay must be finite and nonnegative")

    config = load_project_config(args.config)
    config["data_commit"] = _data_commit(config["data_root"])
    config["code_commit"] = _valid_commit(args.code_commit or _git_commit())
    config["seed"] = args.seed
    config["fold_identity"] = f"electrostatic-development-fold-{args.fold}"
    config["electrostatic_encoder_width_multiplier"] = (
        encoder_width_multiplier_for_architecture(_ARCHITECTURE, config)
    )
    torch.set_float32_matmul_precision(args.matmul_precision)
    seed_everything(args.seed)

    folds = json.loads(args.folds.read_text(encoding="utf-8-sig"))
    if (
        folds.get("frozen_validation_test_labels_read", False) is not False
        or "frozen" not in str(folds.get("role", "")).lower()
        or "unread" not in str(folds.get("role", "")).lower()
    ):
        raise ValueError("Electrostatic fold map is not explicitly frozen-panel safe")
    records = load_gmtnet_records(config["data_root"])
    known_ids = {str(record["JARVIS_ID"]) for record in records}
    fold = next((entry for entry in folds["folds"] if entry["fold"] == args.fold), None)
    if fold is None:
        raise ValueError(f"Fold {args.fold} is absent from electrostatic folds")
    full_fold_train_ids = _response_pretraining_ids(
        folds, args.fold, known_ids, 0, args.seed
    )[0]
    subset_manifest = None
    if args.train_ids_file is not None:
        ids, subset_manifest = load_response_subset(
            args.train_ids_file, fold=args.fold, allowed_ids=full_fold_train_ids
        )
    else:
        ids = _response_pretraining_ids(
            folds, args.fold, known_ids, args.train_limit, args.seed
        )[0]
    development_ids = [str(value) for value in fold["development"]]
    by_id = {str(record["JARVIS_ID"]): record for record in records}
    overlap = sorted(
        {formula(by_id[value]) for value in ids}
        & {formula(by_id[value]) for value in development_ids}
    )
    if overlap:
        raise ValueError(f"Electronic response pretraining overlaps development formulas: {overlap[:5]}")
    pretraining_provenance = provenance(ids, args.folds, "train", config)
    pretraining_provenance.update({
        "development_fold": args.fold,
        "development_ids": sorted(development_ids),
        "development_material_id_sha256": sha256(
            "\n".join(sorted(set(development_ids))).encode("utf-8")
        ).hexdigest(),
        "development_formula_overlap_count": 0,
        "response_task": "electronic",
        "response_label_count": len(ids),
        "frozen_validation_test_labels_read": False,
    })
    if subset_manifest is not None:
        pretraining_provenance.update({
            "response_subset_manifest": str(args.train_ids_file.resolve()),
            "response_subset_material_id_sha256": subset_manifest["material_id_sha256"],
        })
    if args.graph_cache_key is not None:
        cache_key = args.graph_cache_key
    else:
        # The response panel is the only graph population used by this
        # executor. Hashing the complete corpus needlessly serializes all
        # structures and can consume GiB before the first update.
        panel_records = [by_id[value] for value in ids]
        cache_key = graph_cache_key(
            panel_records, float(config["cutoff"]), int(config["max_neighbors"])
        )
    cache_manifest = (
        Path(config["processed_dir"]) / "pbc_graph_cache" / cache_key / "manifest.json"
    )
    if args.graph_cache_key is not None and not cache_manifest.is_file():
        raise FileNotFoundError(
            f"Requested graph cache key has no manifest: {cache_manifest}"
        )
    dataset = _dataset(config, records, ids, cache_key, cache_graphs=False)
    logical_sizes = logical_pretraining_batch_sizes(
        len(dataset), args.batch_size, args.logical_batch_size
    )
    contract = {
        "objective": ELECTRONIC_RESPONSE_PRETRAINING_OBJECTIVE,
        "response_task": "electronic",
        "downstream_architecture": _ARCHITECTURE,
        "encoder_width_multiplier": float(config["electrostatic_encoder_width_multiplier"]),
        "response_material_count": len(dataset),
        "physical_batch_size": args.batch_size,
        "logical_batch_size": args.logical_batch_size,
        "optimizer_updates_per_exposure_epoch": len(logical_sizes),
        "optimizer": "AdamW",
        "code_commit": config["code_commit"],
        "graph_cache_key": cache_key,
        "response_subset_manifest": (
            str(args.train_ids_file.resolve()) if args.train_ids_file is not None else None
        ),
        "response_subset_material_id_sha256": (
            subset_manifest["material_id_sha256"] if subset_manifest is not None else None
        ),
    }
    device = torch.device(args.device)
    model = make_model(_ARCHITECTURE, config)
    tower = model.piezo_generator.to(device)
    del model
    optimizer = torch.optim.AdamW(
        tower.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.resume is None:
        if args.output_dir.exists():
            raise FileExistsError(
                f"Electronic pretraining output directory exists: {args.output_dir}"
            )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        history: list[dict[str, object]] = []
        best = float("inf")
        start_epoch = 1
    else:
        if not args.output_dir.is_dir() or not args.resume.resolve().is_relative_to(
            args.output_dir.resolve()
        ):
            raise ValueError("Resume checkpoint must be inside an existing output directory")
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        validate_resume_payload(saved, pretraining_provenance, contract)
        tower.load_state_dict(saved["piezo_tower"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history = list(saved.get("history", []))
        best = min((float(row["loss"]) for row in history), default=float(saved["loss"]))
        start_epoch = int(saved["epoch"]) + 1
        if start_epoch > args.epochs:
            raise ValueError("Resume checkpoint already reached requested epochs")

    training_loader = DataLoader(
        dataset,
        batch_sampler=_multi_epoch_batches(
            len(dataset), args.seed, start_epoch, args.epochs, args.batch_size
        ),
        **loader_options(
            args.num_workers,
            cuda=device.type == "cuda",
            persistent=True,
            prefetch_factor=args.prefetch_factor,
        ),
    )
    iterator = iter(training_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        tower.train()
        optimizer.zero_grad(set_to_none=True)
        total, graphs, updates = 0.0, 0, 0
        for logical_target in logical_sizes:
            logical_graphs = 0
            while logical_graphs < logical_target:
                batch = next(iterator)
                batch = batch.to(device, non_blocking=device.type == "cuda")
                if not torch.isfinite(batch.y_electronic_piezo).all():
                    raise ValueError("Electronic response-pretraining panel has non-finite labels")
                _, graph_features, context = tower.encode_response_features(batch)
                prediction = tower.decode_electronic_piezo(graph_features, context)
                loss = irrep_balanced_capacity_loss(prediction, batch.y_electronic_piezo)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite electronic response-pretraining loss")
                graphs_in_batch = int(batch.num_graphs)
                if graphs_in_batch > logical_target - logical_graphs:
                    raise RuntimeError("Electronic batch crossed a logical-batch boundary")
                (loss * graphs_in_batch / logical_target).backward()
                total += float(loss.detach()) * graphs_in_batch
                graphs += graphs_in_batch
                logical_graphs += graphs_in_batch
            if not all(
                torch.isfinite(parameter.grad).all()
                for parameter in tower.parameters() if parameter.grad is not None
            ):
                raise FloatingPointError("Non-finite electronic pretraining gradient")
            torch.nn.utils.clip_grad_norm_(tower.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        if updates != len(logical_sizes):
            raise RuntimeError("Electronic response-pretraining update accounting drifted")
        value = total / max(graphs, 1)
        row = {
            "epoch": epoch,
            "loss": value,
            "optimizer_updates": updates,
            "response_material_exposures": graphs,
        }
        history.append(row)
        payload = {
            "architecture": ELECTRONIC_RESPONSE_PRETRAINING_ARCHITECTURE,
            "objective": ELECTRONIC_RESPONSE_PRETRAINING_OBJECTIVE,
            "piezo_tower": tower.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "loss": value,
            "history": history,
            "pretraining_provenance": pretraining_provenance,
            "response_pretraining_contract": contract,
            "code_commit": config["code_commit"],
        }
        last_path = args.output_dir / "last_electronic_tower.pt"
        torch.save(payload, last_path)
        if value < best:
            best = value
            best_path = args.output_dir / "best_electronic_tower.pt"
            temporary = best_path.with_name(f".{best_path.name}.tmp")
            temporary.unlink(missing_ok=True)
            try:
                os.link(last_path, temporary)
            except OSError:
                shutil.copyfile(last_path, temporary)
            temporary.replace(best_path)
        print(f"pretrain_electronic_e3nn epoch={epoch} loss={value:.6g}")
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
