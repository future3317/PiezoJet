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
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .build_electrostatic_development_folds import electrostatic_fold_train_ids
from .checkpoint_provenance import build_checkpoint_provenance
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
    load_structure_pretraining,
    make_model,
    response_active_diagnostic_indices,
)
from .electrostatic_protocol import (
    STABILIZED_SELECTION_VERSION,
    development_selection,
    matched_material_schedule,
)
from .electrostatic_subset import load_response_subset
from .project_config import load_project_config
from .train import _data_commit, _git_commit, seed_everything


TASKS = ("electronic", "born", "dielectric")


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
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
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
        "--resume", type=Path,
        help="Resume from a run-local block-boundary progress.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.resume is None and args.output_dir.exists():
        raise FileExistsError(f"A0 output directory already exists: {args.output_dir}")
    if args.resume is not None:
        if not args.resume.resolve().is_relative_to(args.output_dir.resolve()):
            raise ValueError("A0 resume checkpoint must be inside --output-dir")
        if not args.output_dir.is_dir():
            raise FileNotFoundError(f"A0 resume directory is absent: {args.output_dir}")
    if min(args.updates, args.batch_size, args.microbatch_size, args.eval_interval) < 1:
        raise ValueError("Update and batch arguments must be positive")
    if args.batch_size % args.microbatch_size:
        raise ValueError("Logical batch size must be divisible by microbatch size")

    config = load_project_config(args.config)
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
        "architecture": "a0_independent_irreps",
        "fold": args.fold,
        "seed": args.seed,
        "updates": args.updates,
        "logical_batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "eval_interval": args.eval_interval,
        "learning_rate": args.learning_rate,
        "train_ids": train_ids,
        "full_fold_structure_pretraining_ids": full_fold_train_ids,
        "development_ids": dev_ids,
    }
    seed_everything(args.seed)
    records = load_gmtnet_records(config["data_root"])
    cache_key = graph_cache_key(
        records, float(config["cutoff"]), int(config["max_neighbors"])
    )
    train_set = _dataset(config, records, train_ids, cache_key)
    dev_set = _dataset(config, records, dev_ids, cache_key)
    schedule = matched_material_schedule(
        len(train_set), args.updates, args.batch_size, args.microbatch_size, args.seed
    )
    device = torch.device(args.device)
    control = make_model("a0_independent_irreps", config)
    pretraining = load_structure_pretraining(
        control,
        "a0_independent_irreps",
        args.pretrained_encoder,
        torch.device("cpu"),
        full_fold_train_ids,
        dev_ids,
        config,
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
    diagnostic_indices, diagnostic_materials = response_active_diagnostic_indices(
        train_set, min(args.diagnostic_batch_size, len(train_set))
    )
    diagnostic_loader = DataLoader(
        Subset(train_set, diagnostic_indices),
        batch_size=len(diagnostic_indices),
        shuffle=False,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0
    )
    train_eval_loader = DataLoader(
        train_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0
    )
    if args.resume is None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    initial_gradients: dict[str, object] = saved_initial_gradients or {}
    if saved_initial_gradients is None:
        for task in TASKS:
            tower = _tower(control, task).to(device)
            diagnostic_batch = next(iter(diagnostic_loader)).to(device)
            initial_gradients[task] = _gradient_audit(tower, diagnostic_batch, task)
            tower.to("cpu")
            del diagnostic_batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    train_losses = {task: [0.0] * args.updates for task in TASKS}
    counted_flops_per_task: dict[str, int] = {}
    cuda_update_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ] = {task: [] for task in TASKS}

    for block_start in range(start_block, args.updates, args.eval_interval):
        block_end = min(block_start + args.eval_interval, args.updates)
        block_indices = schedule[
            block_start * args.batch_size : block_end * args.batch_size
        ]
        block_metrics: dict[str, dict[str, object]] = {}
        train_block_metrics: dict[str, dict[str, object]] = {}
        for task in TASKS:
            tower = _tower(control, task).to(device)
            optimizer = optimizers[task]
            _optimizer_to(optimizer, device)
            loader = DataLoader(
                Subset(train_set, block_indices),
                batch_size=args.microbatch_size,
                shuffle=False,
                num_workers=0,
            )
            iterator = iter(loader)
            for update in range(block_start, block_end):
                optimizer.zero_grad(set_to_none=True)
                logical_loss = 0.0
                start_event = end_event = None
                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                flop_context = (
                    FlopCounterMode(display=False)
                    if task not in counted_flops_per_task
                    else nullcontext()
                )
                with flop_context:
                    for _ in range(args.batch_size // args.microbatch_size):
                        batch = next(iterator).to(device)
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
                if start_event is not None and end_event is not None:
                    end_event.record()
                    cuda_update_events[task].append((start_event, end_event))
                train_losses[task][update] = logical_loss
            block_metrics[task] = _evaluate_task(tower, dev_loader, task)
            train_block_metrics[task] = _evaluate_task(
                tower, train_eval_loader, task
            )
            tower.to("cpu")
            _optimizer_to(optimizer, torch.device("cpu"))
            if device.type == "cuda":
                torch.cuda.empty_cache()

        selection = development_selection(block_metrics)
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
        }
        history.append(row)
        if bool(selection["eligible"]) and score < best_score:
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
        evaluation_payload = {
                "schema": 1,
                "status": "running",
                "completed_update": block_end,
                "architecture": "a0_independent_irreps",
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
                "history": history,
                "initial_gradients": initial_gradients,
                "checkpoint_provenance": checkpoint_provenance,
                "development_metrics": block_metrics,
                "train_metrics": train_block_metrics,
                "selection_score": score,
                "selection": selection,
            }
        checkpoint_dir = args.output_dir / "evaluation_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(
            evaluation_payload,
            checkpoint_dir / f"update_{block_end:08d}.pt",
        )
        torch.save(evaluation_payload, args.output_dir / "progress.pt")
        (args.output_dir / "progress.json").write_text(
            json.dumps({key: value for key, value in row.items() if key != "metrics"}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(row), flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    optimizer_seconds_per_task = {
        task: sum(start.elapsed_time(end) for start, end in events) / 1000.0
        for task, events in cuda_update_events.items()
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
            "architecture": "a0_independent_irreps",
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
                "completed_update": args.updates,
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
        tower.to("cpu")
        del diagnostic_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected_payload = {
        "model": best_states,
        "architecture": "a0_independent_irreps",
        "fold": args.fold,
        "selected_update": best_update,
        "seed": args.seed,
        "checkpoint_provenance": checkpoint_provenance,
    }
    torch.save(selected_payload, args.output_dir / "selected.pt")
    summary = {
        "schema": 2,
        "protocol": "resource-bounded disjoint-tower A0 Stage-A adjudication",
        "architecture": "a0_independent_irreps",
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
        "response_subset_manifest": (
            str(args.train_ids_file.resolve()) if args.train_ids_file else None
        ),
        "structure_pretraining_universe_materials": len(full_fold_train_ids),
        "optimizer_updates_per_task": args.updates,
        "material_exposures_per_task": args.updates * args.batch_size,
        "effective_train_passes_per_task": args.updates * args.batch_size / len(train_set),
        "logical_batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "evaluation_batch_size": args.eval_batch_size,
        "num_workers": 0,
        "selection": (
            "minimum common-update development electronic-stabilized-relative plus "
            "BEC-stabilized-relative plus electronic-dielectric-stabilized-relative score"
        ),
        "selection_version": STABILIZED_SELECTION_VERSION,
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
            "status": "complete",
            "completed_update": args.updates,
            "selected_update": best_update,
            "frozen_validation_test_labels_read": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "progress.pt").unlink(missing_ok=True)
    print(json.dumps({"selected_update": best_update, "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
