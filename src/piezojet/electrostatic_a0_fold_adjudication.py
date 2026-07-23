"""Resource-bounded formula-disjoint A0 electrostatic adjudication.

A0 contains three statistically independent response generators.  Their
AdamW updates commute because the parameter sets are disjoint.  This executor
therefore advances them in matched update blocks while keeping only one tower
and its optimizer state on CUDA.  The schedule is mathematically identical to
three independent optimizers receiving the same material batches; it avoids
resident parameters, gradients, optimizer states, and allocator reservations
from all three towers competing for one 16 GiB device.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .build_electrostatic_development_folds import electrostatic_fold_train_ids
from .checkpoint_provenance import build_checkpoint_provenance, file_sha256
from .data import deterministic_subset, graph_cache_key, load_gmtnet_records
from .electronic_capacity import (
    born_capacity_metrics,
    born_material_balanced_loss,
    dielectric_capacity_metrics,
    dielectric_material_balanced_loss,
    electronic_capacity_metrics,
    irrep_balanced_capacity_loss,
)
from .electrostatic_fold_adjudication import (
    _dataset,
    encoder_width_multiplier_for_architecture,
    load_bec_response_pretraining,
    load_electronic_response_pretraining,
    load_structure_pretraining,
    make_model,
    response_active_diagnostic_indices,
)
from .electrostatic_protocol import (
    A0_ARCHITECTURES,
    DEFAULT_EARLY_STOPPING_PATIENCE_EVALUATIONS,
    STABILIZED_SELECTION_VERSION,
    compact_training_curve_row,
    development_early_stopping,
    development_selection,
    matched_material_schedule,
)
from .electrostatic_subset import load_response_subset
from .loader_runtime import loader_options
from .project_config import load_project_config
from .train import _data_commit, _git_commit, seed_everything


TASKS = ("electronic", "born", "dielectric")


def _training_schedule_tail(
    schedule: list[int],
    start_update: int,
    logical_batch_size: int,
) -> list[int]:
    """Return the exact remaining deterministic material schedule."""
    if start_update < 0 or logical_batch_size < 1:
        raise ValueError("Schedule offset arguments must be nonnegative/positive")
    offset = start_update * logical_batch_size
    if offset > len(schedule):
        raise ValueError("Schedule offset lies beyond the persisted material schedule")
    return schedule[offset:]


def _train_evaluation_due(
    block_end: int,
    total_updates: int,
    interval: int,
) -> bool:
    """Return whether the non-selecting full-train diagnostic is due."""
    if block_end < 1 or total_updates < block_end or interval < 0:
        raise ValueError("Invalid train-evaluation schedule")
    return interval == 0 or block_end == total_updates or block_end % interval == 0


def _clear_runtime_caches(module: nn.Module) -> None:
    """Drop unregistered geometry tensors before moving a tower between devices."""
    for child in module.modules():
        if hasattr(child, "_geometry_cache"):
            child._geometry_cache = None


def _replace_progress_checkpoint(source: Path, destination: Path) -> None:
    """Atomically make progress refer to one immutable evaluation checkpoint.

    A hard link avoids a second Python serialization of model and optimizer
    state.  A byte-copy fallback retains the same payload on filesystems that
    cannot hard-link the two paths.
    """
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _tower(control: nn.Module, task: str) -> nn.Module:
    if task == "electronic":
        return control.piezo_generator
    if task == "born":
        return control.born_generator
    if task == "dielectric":
        return control.dielectric_generator
    raise ValueError(f"Unknown A0 task: {task}")


def _prediction(tower: nn.Module, batch, task: str) -> torch.Tensor:
    node_features, graph_features, context = tower.encode_response_features(batch)
    if task == "electronic":
        return tower.decode_electronic_piezo(graph_features, context)
    if task == "born":
        return tower.decode_born(node_features, context, batch.batch)
    if task == "dielectric":
        return tower.decode_dielectric(graph_features)
    raise ValueError(f"Unknown A0 task: {task}")


def _loss(tower: nn.Module, batch, task: str) -> torch.Tensor:
    prediction = _prediction(tower, batch, task)
    if task == "electronic":
        return irrep_balanced_capacity_loss(prediction, batch.y_electronic_piezo)
    if task == "born":
        return born_material_balanced_loss(prediction, batch.y_born, batch.batch)
    return dielectric_material_balanced_loss(
        prediction,
        batch.y_dfpt_electronic_dielectric,
        batch.dfpt_electronic_dielectric_mask,
    )


def _gradient_audit(tower: nn.Module, batch, task: str) -> dict[str, float | int]:
    tower.train()
    parameters = [parameter for parameter in tower.parameters() if parameter.requires_grad]
    loss = _loss(tower, batch, task)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    squared = sum(
        gradient.detach().square().sum()
        for gradient in gradients
        if gradient is not None
    )
    return {
        "loss": float(loss.detach()),
        "parameter_tensors": sum(gradient is not None for gradient in gradients),
        "total_gradient_norm": float(torch.sqrt(squared)),
    }


def _evaluate_task(tower: nn.Module, loader, task: str) -> dict[str, object]:
    device = next(tower.parameters()).device
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    batch_indices: list[torch.Tensor] = []
    offset = 0
    tower.eval()
    # The reciprocal-geometry cache keys tensors by their version counters;
    # inference tensors intentionally have no version counter.
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            predictions.append(_prediction(tower, batch, task).detach().cpu())
            if task == "electronic":
                targets.append(batch.y_electronic_piezo.detach().cpu())
            elif task == "born":
                targets.append(batch.y_born.detach().cpu())
                batch_indices.append((batch.batch + offset).detach().cpu())
                offset += int(batch.num_graphs)
            else:
                targets.append(batch.y_dfpt_electronic_dielectric.detach().cpu())
                masks.append(batch.dfpt_electronic_dielectric_mask.detach().cpu())
    if task == "electronic":
        return electronic_capacity_metrics(torch.cat(predictions), torch.cat(targets))
    if task == "born":
        return born_capacity_metrics(
            torch.cat(predictions), torch.cat(targets), torch.cat(batch_indices)
        )
    return dielectric_capacity_metrics(
        torch.cat(predictions), torch.cat(targets), torch.cat(masks)
    )


def _selection_score(metrics: dict[str, dict[str, object]]) -> float:
    return float(development_selection(metrics)["raw_score"])


def _restore_a0_progress(
    control: nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    payload: dict[str, object],
    training_contract: dict[str, object],
    checkpoint_provenance: dict[str, object],
) -> dict[str, object]:
    """Validate and restore one complete A0 common-update block."""
    if payload.get("status") not in {"running", "interrupted"}:
        raise ValueError("A0 resume checkpoint is not an incomplete run")
    if payload.get("training_contract") != training_contract:
        raise ValueError("A0 resume training contract differs from the current command")
    if payload.get("checkpoint_provenance") != checkpoint_provenance:
        raise ValueError("A0 resume provenance differs from the current fold")
    completed_update = int(payload["completed_update"])
    updates = int(training_contract["updates"])
    interval = int(training_contract["eval_interval"])
    if completed_update < 0 or completed_update > updates:
        raise ValueError("A0 resume completed update lies outside the training contract")
    if completed_update != updates and completed_update % interval:
        raise ValueError("A0 resume checkpoint is not at a common update-block boundary")
    model_states = payload["model"]
    optimizer_states = payload["optimizer"]
    for task in TASKS:
        _tower(control, task).load_state_dict(model_states[task], strict=True)
        optimizers[task].load_state_dict(optimizer_states[task])
    return {
        "start_block": completed_update,
        "history": list(payload["history"]),
        "best_score": float(payload["best_score"]),
        "best_update": (
            int(payload["best_update"])
            if payload.get("best_update") is not None
            else None
        ),
        "best_states": payload["best_model"],
        "best_metrics": payload["best_metrics"],
        "initial_gradients": payload["initial_gradients"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=A0_ARCHITECTURES,
        default="a0_independent_irreps",
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--folds", type=Path,
        default=Path("data/processed/electrostatic_development_folds_v2.json"),
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--diagnostic-batch-size", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument(
        "--early-stopping-patience-evaluations",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE_EVALUATIONS,
        help=(
            "Stop after this many full-development evaluations without an eligible "
            "score improvement; 0 disables early stopping"
        ),
    )
    parser.add_argument(
        "--early-stopping-minimum-improvement",
        type=float,
        default=0.0,
        help="Minimum stabilized-score decrease counted as an improvement",
    )
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument(
        "--train-ids-file", type=Path,
        help="Preregistered balanced response-supervision subset manifest",
    )
    parser.add_argument("--development-limit", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--code-commit",
        help="Pinned 40-character source commit recorded by the execution plan",
    )
    parser.add_argument("--pretrained-encoder", type=Path, required=True)
    parser.add_argument(
        "--bec-pretrained-tower", type=Path,
        help=(
            "Strict fold-train-only BEC response-aware initializer. It may be used "
            "only with A0-PM and overwrites only born_generator after structural init."
        ),
    )
    parser.add_argument(
        "--electronic-pretrained-tower", type=Path,
        help=(
            "Strict fold-train-only electronic response-aware initializer. It may "
            "be used only with A0-PM and overwrites only piezo_generator."
        ),
    )
    parser.add_argument(
        "--resume", type=Path,
        help="Resume from a run-local block-boundary progress.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Background graph-loading workers; runtime-only and resume-safe",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per persistent worker without changing order",
    )
    parser.add_argument(
        "--graph-cache-key",
        help="Existing canonical graph-cache key; avoids recomputing the corpus hash",
    )
    parser.add_argument(
        "--train-eval-interval",
        type=int,
        default=0,
        help=(
            "Common-update interval for the non-selecting full-train diagnostic; "
            "0 preserves historical evaluation at every development checkpoint"
        ),
    )
    parser.add_argument(
        "--retain-cuda-allocator-cache",
        action="store_true",
        help="Do not clear inactive CUDA allocator blocks between disjoint towers",
    )
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"),
        default="highest", help="PyTorch float32 matmul precision policy",
    )
    args = parser.parse_args()
    if args.resume is None and args.output_dir.exists():
        raise FileExistsError(f"A0 output directory already exists: {args.output_dir}")
    if args.resume is not None:
        if not args.resume.resolve().is_relative_to(args.output_dir.resolve()):
            raise ValueError("A0 resume checkpoint must be inside --output-dir")
        if not args.output_dir.is_dir():
            raise FileNotFoundError(f"A0 resume directory is absent: {args.output_dir}")
    if args.bec_pretrained_tower is not None and args.architecture != "a0_parameter_matched_irreps":
        raise ValueError("--bec-pretrained-tower is only valid for a0_parameter_matched_irreps")
    if (
        args.electronic_pretrained_tower is not None
        and args.architecture != "a0_parameter_matched_irreps"
    ):
        raise ValueError(
            "--electronic-pretrained-tower is only valid for a0_parameter_matched_irreps"
        )
    if min(args.updates, args.batch_size, args.microbatch_size, args.eval_interval) < 1:
        raise ValueError("Update and batch arguments must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive")
    if args.train_eval_interval < 0:
        raise ValueError("--train-eval-interval cannot be negative")
    if args.train_eval_interval and args.train_eval_interval % args.eval_interval:
        raise ValueError(
            "--train-eval-interval must be a multiple of --eval-interval"
        )
    if args.early_stopping_patience_evaluations < 0:
        raise ValueError("Early-stopping patience cannot be negative")
    if (
        args.early_stopping_minimum_improvement < 0.0
        or not math.isfinite(args.early_stopping_minimum_improvement)
    ):
        raise ValueError(
            "Early-stopping minimum improvement must be finite and nonnegative"
        )
    if args.batch_size % args.microbatch_size:
        raise ValueError("Logical batch size must be divisible by microbatch size")

    config = load_project_config(args.config)
    torch.set_float32_matmul_precision(args.matmul_precision)
    config["data_commit"] = _data_commit(config["data_root"])
    code_commit = args.code_commit or _git_commit()
    if len(code_commit) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in code_commit
    ):
        raise ValueError("--code-commit must be one 40-character Git commit SHA")
    config["code_commit"] = code_commit.lower()
    config["seed"] = args.seed
    config["fold_identity"] = f"electrostatic-development-fold-{args.fold}"
    folds = json.loads(args.folds.read_text(encoding="utf-8-sig"))
    fold = next((value for value in folds["folds"] if value["fold"] == args.fold), None)
    if fold is None:
        raise ValueError(f"Fold {args.fold} is absent from {args.folds}")
    full_fold_train_ids = electrostatic_fold_train_ids(folds, args.fold)
    if args.train_ids_file is not None and args.train_limit:
        raise ValueError("--train-ids-file and --train-limit are mutually exclusive")
    subset_manifest = None
    if args.train_ids_file is not None:
        train_ids, subset_manifest = load_response_subset(
            args.train_ids_file, fold=args.fold, allowed_ids=full_fold_train_ids
        )
    else:
        train_ids = deterministic_subset(
            full_fold_train_ids, args.train_limit, args.seed + 1000
        )
    dev_ids = deterministic_subset(
        list(fold["development"]), args.development_limit, args.seed + 2000
    )
    checkpoint_provenance = build_checkpoint_provenance(
        {"train": train_ids, "val": dev_ids, "test": []},
        args.folds,
        config,
        split_kind=f"electrostatic_development_fold_{args.fold}",
    )
    checkpoint_provenance["code_commit"] = config["code_commit"]
    if args.train_ids_file is not None:
        checkpoint_provenance["response_subset_manifest"] = str(
            args.train_ids_file.resolve()
        )
        checkpoint_provenance["response_subset_material_id_sha256"] = (
            subset_manifest["material_id_sha256"]
        )
    training_contract = {
        "architecture": args.architecture,
        "fold": args.fold,
        "seed": args.seed,
        "updates": args.updates,
        "logical_batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "eval_interval": args.eval_interval,
        "train_eval_interval": args.train_eval_interval,
        "early_stopping_patience_evaluations": (
            args.early_stopping_patience_evaluations
        ),
        "early_stopping_minimum_improvement": (
            args.early_stopping_minimum_improvement
        ),
        "learning_rate": args.learning_rate,
        "encoder_width_multiplier": encoder_width_multiplier_for_architecture(
            args.architecture, config
        ),
        "train_ids": train_ids,
        "full_fold_structure_pretraining_ids": full_fold_train_ids,
        "development_ids": dev_ids,
        "bec_response_pretraining_checkpoint": (
            str(args.bec_pretrained_tower.resolve())
            if args.bec_pretrained_tower is not None
            else None
        ),
        "bec_response_pretraining_checkpoint_sha256": (
            file_sha256(args.bec_pretrained_tower)
            if args.bec_pretrained_tower is not None
            else None
        ),
        "electronic_response_pretraining_checkpoint": (
            str(args.electronic_pretrained_tower.resolve())
            if args.electronic_pretrained_tower is not None
            else None
        ),
        "electronic_response_pretraining_checkpoint_sha256": (
            file_sha256(args.electronic_pretrained_tower)
            if args.electronic_pretrained_tower is not None
            else None
        ),
        "graph_cache_key": None,
    }
    seed_everything(args.seed)
    records = load_gmtnet_records(config["data_root"])
    if args.graph_cache_key is not None:
        cache_key = args.graph_cache_key
        cache_manifest = (
            Path(config["processed_dir"])
            / "pbc_graph_cache"
            / cache_key
            / "manifest.json"
        )
        if not cache_manifest.is_file():
            raise FileNotFoundError(
                f"Requested graph cache key has no manifest: {cache_manifest}"
            )
    else:
        cache_key = graph_cache_key(
            records, float(config["cutoff"]), int(config["max_neighbors"])
        )
    training_contract["graph_cache_key"] = cache_key
    train_set = _dataset(config, records, train_ids, cache_key)
    dev_set = _dataset(config, records, dev_ids, cache_key)
    # A0 revisits the same fixed train/development panels for many common
    # update blocks.  Materialize the bounded electrostatic graph caches
    # before moving towers to CUDA so disk deserialization cannot starve the
    # GPU inside the optimizer loop.
    train_set.warm_graph_cache()
    dev_set.warm_graph_cache()
    schedule = matched_material_schedule(
        len(train_set), args.updates, args.batch_size, args.microbatch_size, args.seed
    )
    device = torch.device(args.device)
    control = make_model(args.architecture, config)
    training_contract["parameter_count"] = sum(
        parameter.numel() for parameter in control.parameters()
    )
    pretraining = load_structure_pretraining(
        control,
        args.architecture,
        args.pretrained_encoder,
        torch.device("cpu"),
        full_fold_train_ids,
        dev_ids,
        config,
    )
    bec_response_pretraining = (
        load_bec_response_pretraining(
            control,
            args.architecture,
            args.bec_pretrained_tower,
            torch.device("cpu"),
            full_fold_train_ids,
            dev_ids,
            config,
        )
        if args.bec_pretrained_tower is not None
        else None
    )
    electronic_response_pretraining = (
        load_electronic_response_pretraining(
            control,
            args.architecture,
            args.electronic_pretrained_tower,
            torch.device("cpu"),
            full_fold_train_ids,
            dev_ids,
            config,
        )
        if args.electronic_pretrained_tower is not None
        else None
    )
    optimizers = {
        task: torch.optim.AdamW(
            _tower(control, task).parameters(),
            lr=args.learning_rate,
            weight_decay=1e-6,
        )
        for task in TASKS
    }
    start_block = 0
    history: list[dict[str, object]] = []
    best_score = math.inf
    best_update = 0
    best_states: dict[str, dict[str, torch.Tensor]] | None = None
    best_metrics: dict[str, dict[str, object]] | None = None
    non_improving_evaluations = 0
    saved_initial_gradients = None
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        restored = _restore_a0_progress(
            control, optimizers, payload, training_contract, checkpoint_provenance
        )
        start_block = int(restored["start_block"])
        history = restored["history"]
        best_score = float(restored["best_score"])
        best_update = int(restored["best_update"])
        best_states = restored["best_states"]
        best_metrics = restored["best_metrics"]
        saved_initial_gradients = restored["initial_gradients"]
        non_improving_evaluations = int(
            payload.get("non_improving_evaluations", 0)
        )
    diagnostic_indices, diagnostic_materials = response_active_diagnostic_indices(
        train_set, min(args.diagnostic_batch_size, len(train_set))
    )
    diagnostic_loader = DataLoader(
        Subset(train_set, diagnostic_indices),
        batch_size=len(diagnostic_indices),
        shuffle=False,
        **loader_options(
            args.num_workers,
            cuda=device.type == "cuda",
            persistent=False,
            prefetch_factor=args.prefetch_factor,
        ),
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.eval_batch_size, shuffle=False,
        **loader_options(
            args.num_workers,
            cuda=device.type == "cuda",
            prefetch_factor=args.prefetch_factor,
        ),
    )
    train_eval_loader = DataLoader(
        train_set, batch_size=args.eval_batch_size, shuffle=False,
        **loader_options(
            args.num_workers,
            cuda=device.type == "cuda",
            prefetch_factor=args.prefetch_factor,
        ),
    )
    if args.resume is None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    initial_gradients: dict[str, object] = saved_initial_gradients or {}
    if saved_initial_gradients is None:
        for task in TASKS:
            tower = _tower(control, task).to(device)
            diagnostic_batch = next(iter(diagnostic_loader)).to(device)
            initial_gradients[task] = _gradient_audit(tower, diagnostic_batch, task)
            _clear_runtime_caches(tower)
            tower.to("cpu")
            del diagnostic_batch
            if device.type == "cuda" and not args.retain_cuda_allocator_cache:
                torch.cuda.empty_cache()

    remaining_subset = Subset(
        train_set,
        _training_schedule_tail(schedule, start_block, args.batch_size),
    )
    training_loaders = {
        task: DataLoader(
            remaining_subset,
            batch_size=args.microbatch_size,
            shuffle=False,
            **loader_options(
                args.num_workers,
                cuda=device.type == "cuda",
                persistent=True,
                prefetch_factor=args.prefetch_factor,
            ),
        )
        for task in TASKS
    }
    training_iterators = {
        task: iter(loader) for task, loader in training_loaders.items()
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    train_losses = {task: [0.0] * args.updates for task in TASKS}
    counted_flops_per_task: dict[str, int] = {}
    cuda_block_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ] = {task: [] for task in TASKS}
    data_wait_seconds = {task: 0.0 for task in TASKS}
    h2d_enqueue_seconds = {task: 0.0 for task in TASKS}
    transfer_seconds = {task: 0.0 for task in TASKS}

    stopped_early = False
    completed_update = start_block
    for block_start in range(start_block, args.updates, args.eval_interval):
        block_end = min(block_start + args.eval_interval, args.updates)
        completed_update = block_end
        block_started = time.perf_counter()
        block_metrics: dict[str, dict[str, object]] = {}
        run_train_evaluation = _train_evaluation_due(
            block_end, args.updates, args.train_eval_interval
        )
        train_block_metrics: dict[str, dict[str, object]] | None = (
            {} if run_train_evaluation else None
        )
        evaluation_seconds: dict[str, dict[str, float]] = {}
        for task in TASKS:
            transfer_started = time.perf_counter()
            tower = _tower(control, task).to(device)
            optimizer = optimizers[task]
            _optimizer_to(optimizer, device)
            transfer_seconds[task] += time.perf_counter() - transfer_started
            iterator = training_iterators[task]
            block_start_event = block_end_event = None
            if device.type == "cuda":
                block_start_event = torch.cuda.Event(enable_timing=True)
                block_end_event = torch.cuda.Event(enable_timing=True)
                block_start_event.record()
            for update in range(block_start, block_end):
                optimizer.zero_grad(set_to_none=True)
                logical_loss = 0.0
                flop_context = (
                    FlopCounterMode(display=False)
                    if task not in counted_flops_per_task
                    else nullcontext()
                )
                with flop_context:
                    for _ in range(args.batch_size // args.microbatch_size):
                        wait_started = time.perf_counter()
                        batch = next(iterator)
                        data_wait_seconds[task] += time.perf_counter() - wait_started
                        copy_started = time.perf_counter()
                        batch = batch.to(
                            device, non_blocking=device.type == "cuda"
                        )
                        h2d_enqueue_seconds[task] += (
                            time.perf_counter() - copy_started
                        )
                        tower.train()
                        loss = _loss(tower, batch, task)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Non-finite A0 {task} loss")
                        fraction = int(batch.num_graphs) / args.batch_size
                        (loss * fraction).backward()
                        logical_loss += float(loss.detach()) * fraction
                        del batch
                    optimizer.step()
                if isinstance(flop_context, FlopCounterMode):
                    counted_flops_per_task[task] = int(
                        flop_context.get_total_flops()
                    )
                train_losses[task][update] = logical_loss
            if block_start_event is not None and block_end_event is not None:
                block_end_event.record()
                cuda_block_events[task].append((block_start_event, block_end_event))
            development_evaluation_started = time.perf_counter()
            block_metrics[task] = _evaluate_task(tower, dev_loader, task)
            development_evaluation_seconds = (
                time.perf_counter() - development_evaluation_started
            )
            train_evaluation_seconds = 0.0
            if train_block_metrics is not None:
                train_evaluation_started = time.perf_counter()
                train_block_metrics[task] = _evaluate_task(
                    tower, train_eval_loader, task
                )
                train_evaluation_seconds = (
                    time.perf_counter() - train_evaluation_started
                )
            evaluation_seconds[task] = {
                "development_seconds": development_evaluation_seconds,
                "train_seconds": train_evaluation_seconds,
            }
            transfer_started = time.perf_counter()
            _clear_runtime_caches(tower)
            tower.to("cpu")
            _optimizer_to(optimizer, torch.device("cpu"))
            transfer_seconds[task] += time.perf_counter() - transfer_started
            if device.type == "cuda" and not args.retain_cuda_allocator_cache:
                torch.cuda.empty_cache()

        selection = development_selection(block_metrics)
        train_selection = (
            development_selection(train_block_metrics)
            if train_block_metrics is not None
            else None
        )
        score = float(selection["raw_score"])
        row = {
            "update": block_end,
            "development_selection_score": score,
            "development_selection": selection,
            "train_electronic_loss": train_losses["electronic"][block_end - 1],
            "train_born_loss": train_losses["born"][block_end - 1],
            "train_dielectric_loss": train_losses["dielectric"][block_end - 1],
            "metrics": block_metrics,
            "train_metrics": train_block_metrics,
            "train_selection": train_selection,
            "generalization_score_gap": (
                score - float(train_selection["raw_score"])
                if train_selection is not None
                else None
            ),
            "evaluation_runtime": {
                "per_task": evaluation_seconds,
                "total_seconds": sum(
                    sum(task_seconds.values())
                    for task_seconds in evaluation_seconds.values()
                ),
                "train_evaluation_performed": run_train_evaluation,
                "block_wall_seconds": time.perf_counter() - block_started,
            },
        }
        early_stopping = development_early_stopping(
            score=score,
            eligible=bool(selection["eligible"]),
            best_score=best_score,
            non_improving_evaluations=non_improving_evaluations,
            patience_evaluations=args.early_stopping_patience_evaluations,
            minimum_improvement=args.early_stopping_minimum_improvement,
        )
        non_improving_evaluations = int(
            early_stopping["non_improving_evaluations"]
        )
        row["early_stopping"] = early_stopping
        if bool(early_stopping["improved"]):
            best_score = score
            best_update = block_end
            best_metrics = copy.deepcopy(block_metrics)
            best_states = {
                task: {
                    name: value.detach().cpu().clone()
                    for name, value in _tower(control, task).state_dict().items()
                }
                for task in TASKS
            }
        compact_row = compact_training_curve_row(row)
        if compact_row is None:
            raise RuntimeError("A0 evaluation did not produce a compact history row")
        # Immutable update checkpoints retain full per-material metrics.  The
        # accumulated history stays compact so later checkpoint writes do not
        # repeatedly serialize all earlier material-level payloads.
        history.append(compact_row)
        evaluation_payload = {
                "schema": 1,
                "status": "running",
                "completed_update": block_end,
                "architecture": args.architecture,
                "training_contract": training_contract,
                "model": {
                    task: _tower(control, task).state_dict() for task in TASKS
                },
                "optimizer": {
                    task: optimizers[task].state_dict() for task in TASKS
                },
                "best_update": best_update,
                "best_score": best_score,
                "best_model": best_states,
                "best_metrics": best_metrics,
                "non_improving_evaluations": non_improving_evaluations,
                "history": history,
                "initial_gradients": initial_gradients,
                "checkpoint_provenance": checkpoint_provenance,
                "development_metrics": block_metrics,
                "train_metrics": train_block_metrics,
                "selection_score": score,
                "selection": selection,
                "evaluation_runtime": row["evaluation_runtime"],
                "latest_train_losses": {
                    "electronic": row["train_electronic_loss"],
                    "born": row["train_born_loss"],
                    "dielectric": row["train_dielectric_loss"],
                },
            }
        checkpoint_dir = args.output_dir / "evaluation_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / f"update_{block_end:08d}.pt"
        torch.save(evaluation_payload, checkpoint_path)
        _replace_progress_checkpoint(checkpoint_path, args.output_dir / "progress.pt")
        (args.output_dir / "progress.json").write_text(
            json.dumps({
                "schema": 1,
                "status": "running",
                "completed_update": block_end,
                **compact_row,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "training_curve.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(compact_row), flush=True)
        if bool(early_stopping["should_stop"]):
            stopped_early = True
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    optimizer_seconds_per_task = {
        task: sum(start.elapsed_time(end) for start, end in events) / 1000.0
        for task, events in cuda_block_events.items()
    }
    optimizer_seconds = (
        sum(optimizer_seconds_per_task.values())
        if device.type == "cuda"
        else seconds
    )
    counted_flops_per_common_update = sum(counted_flops_per_task.values())
    if best_states is None or best_metrics is None:
        summary = {
            "schema": 2,
            "status": "completed_no_eligible_checkpoint",
            "protocol": "resource-bounded disjoint-tower A0 Stage-A adjudication",
            "architecture": args.architecture,
            "fold": args.fold,
            "seed": args.seed,
            "checkpoint_provenance": checkpoint_provenance,
            "train_materials": len(train_set),
            "development_materials": len(dev_set),
            "frozen_validation_test_labels_read": False,
            "selection_version": STABILIZED_SELECTION_VERSION,
            "selection": (
                "no common-update checkpoint passed all directional and amplitude "
                "guardrails; no selected.pt was written"
            ),
            "history": history,
            "runtime": {
                "device": str(device),
                "seconds": seconds,
                "optimizer_seconds": optimizer_seconds,
                "optimizer_seconds_per_task": optimizer_seconds_per_task,
                "data_wait_seconds_per_task": data_wait_seconds,
                "h2d_enqueue_seconds_per_task": h2d_enqueue_seconds,
                "tower_optimizer_transfer_seconds_per_task": transfer_seconds,
                "counted_flops_per_common_update": counted_flops_per_common_update,
            },
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "progress.json").write_text(
            json.dumps({
                "schema": 1,
                "status": "completed_no_eligible_checkpoint",
                "completed_update": completed_update,
                "selected_update": None,
                "frozen_validation_test_labels_read": False,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "progress.pt").unlink(missing_ok=True)
        return
    selected_gradients: dict[str, object] = {}
    for task in TASKS:
        tower = _tower(control, task)
        tower.load_state_dict(best_states[task], strict=True)
        tower.to(device)
        diagnostic_batch = next(iter(diagnostic_loader)).to(device)
        selected_gradients[task] = _gradient_audit(tower, diagnostic_batch, task)
        _clear_runtime_caches(tower)
        tower.to("cpu")
        del diagnostic_batch
        if device.type == "cuda" and not args.retain_cuda_allocator_cache:
            torch.cuda.empty_cache()
    selected_payload = {
        "model": best_states,
        "architecture": args.architecture,
        "fold": args.fold,
        "selected_update": best_update,
        "seed": args.seed,
        "checkpoint_provenance": checkpoint_provenance,
    }
    torch.save(selected_payload, args.output_dir / "selected.pt")
    summary = {
        "schema": 2,
        "status": "complete_early_stopped" if stopped_early else "complete",
        "protocol": "resource-bounded disjoint-tower A0 Stage-A adjudication",
        "architecture": args.architecture,
        "execution": (
            "three disjoint AdamW optimizers advanced in matched update blocks; "
            "one tower and optimizer state resident on CUDA at a time"
        ),
        "fold": args.fold,
        "seed": args.seed,
        "checkpoint_provenance": checkpoint_provenance,
        "train_materials": len(train_set),
        "development_materials": len(dev_set),
        "frozen_validation_test_labels_read": False,
        "structure_pretraining": pretraining,
        "bec_response_pretraining": bec_response_pretraining,
        "electronic_response_pretraining": electronic_response_pretraining,
        "response_subset_manifest": (
            str(args.train_ids_file.resolve()) if args.train_ids_file else None
        ),
        "structure_pretraining_universe_materials": len(full_fold_train_ids),
        "requested_optimizer_updates_per_task": args.updates,
        "optimizer_updates_per_task": completed_update,
        "material_exposures_per_task": completed_update * args.batch_size,
        "effective_train_passes_per_task": (
            completed_update * args.batch_size / len(train_set)
        ),
        "logical_batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "evaluation_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "persistent_training_loaders": True,
        "train_evaluation_interval": args.train_eval_interval,
        "retain_cuda_allocator_cache": args.retain_cuda_allocator_cache,
        "selection": (
            "minimum common-update development electronic-stabilized-relative plus "
            "BEC-stabilized-relative plus electronic-dielectric-stabilized-relative score"
        ),
        "selection_version": STABILIZED_SELECTION_VERSION,
        "early_stopping": {
            "enabled": args.early_stopping_patience_evaluations > 0,
            "patience_evaluations": args.early_stopping_patience_evaluations,
            "minimum_improvement": args.early_stopping_minimum_improvement,
            "stopped_early": stopped_early,
            "completed_update": completed_update,
        },
        "selection_weights": {
            "electronic_stabilized_relative": 1.0,
            "born_stabilized_relative": 1.0,
            "electronic_dielectric_stabilized_relative": 1.0,
        },
        "evaluation_checkpoint_policy": (
            "one immutable three-tower model+optimizer+train/development-metrics "
            "checkpoint per full-development evaluation point"
        ),
        "selected_update": best_update,
        "parameter_count": sum(parameter.numel() for parameter in control.parameters()),
        "parameter_count_per_task": {
            task: sum(parameter.numel() for parameter in _tower(control, task).parameters())
            for task in TASKS
        },
        "runtime": {
            "device": str(device),
            "seconds": seconds,
            "optimizer_seconds": optimizer_seconds,
            "optimizer_seconds_per_task": optimizer_seconds_per_task,
            "data_wait_seconds_per_task": data_wait_seconds,
            "h2d_enqueue_seconds_per_task": h2d_enqueue_seconds,
            "tower_optimizer_transfer_seconds_per_task": transfer_seconds,
            "counted_flops_per_task_update": counted_flops_per_task,
            "counted_flops_per_common_update": counted_flops_per_common_update,
            "flop_count_scope": (
                "sum of torch-dispatch-supported forward, backward, and AdamW "
                "operations for one measured update of each disjoint tower"
            ),
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else None
            ),
        },
        "gradient_geometry": {
            "shared_parameter_tensors": 0,
            "shared_gradient_cosines": None,
            "diagnostic_batch_materials": len(diagnostic_indices),
            "materials": diagnostic_materials,
            "initial": initial_gradients,
            "selected": selected_gradients,
        },
        "metrics": {
            "electronic": best_metrics["electronic"],
            "born": best_metrics["born"],
            "dielectric_audit": best_metrics["dielectric"],
        },
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "progress.json").write_text(
        json.dumps({
            "schema": 1,
            "status": "complete_early_stopped" if stopped_early else "complete",
            "completed_update": completed_update,
            "selected_update": best_update,
            "frozen_validation_test_labels_read": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "progress.pt").unlink(missing_ok=True)
    print(json.dumps({"selected_update": best_update, "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
