"""Formula-disjoint A0--A3 electrostatic-jet development adjudication."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .data import (
    PiezoDataset,
    deterministic_subset,
    graph_cache_key,
    load_gmtnet_records,
)
from .build_electrostatic_development_folds import electrostatic_fold_train_ids
from .electronic_capacity import (
    born_capacity_metrics,
    born_material_balanced_loss,
    electronic_capacity_metrics,
    irrep_balanced_capacity_loss,
)
from .electrostatic_protocol import ARCHITECTURES
from .model import (
    ElectromechanicalJetHead,
    IndependentElectrostaticHeads,
    NonlinearDifferentialPolarizationTower,
)
from .project_config import load_project_config
from .train import seed_everything


def _model_kwargs(config: dict[str, object]) -> dict[str, object]:
    return {
        "embedding_dim": int(config["embedding_dim"]),
        "cutoff": float(config["cutoff"]),
        "lmax": max(3, int(config.get("lmax", 3))),
        "num_blocks": int(config["num_blocks"]),
        "radial_basis": int(config["radial_basis"]),
        "radial_hidden": int(config["radial_hidden"]),
        "global_context_dim": int(config["global_context_dim"]),
        "spectral_channels": int(config["spectral_channels"]),
        "spectral_shells": int(config["spectral_shells"]),
        "polar_fluctuation_shells": int(config["polar_fluctuation_shells"]),
        "reciprocal_cutoff": float(config["reciprocal_cutoff"]),
        "attention_dim": int(config.get("global_attention_dim", 64)),
    }


def make_model(architecture: str, config: dict[str, object]) -> nn.Module:
    kwargs = _model_kwargs(config)
    if architecture == "a0_independent_irreps":
        return IndependentElectrostaticHeads(**kwargs)
    if architecture == "a1_electromechanical_jet":
        return ElectromechanicalJetHead(**kwargs)
    if architecture == "a2_nonlinear_cartesian":
        return NonlinearDifferentialPolarizationTower(
            polarization_variable="cartesian", **kwargs
        )
    if architecture == "a3_nonlinear_reduced":
        return NonlinearDifferentialPolarizationTower(
            polarization_variable="reduced", **kwargs
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def _coefficients(model: nn.Module, batch, architecture: str, create_graph: bool):
    if architecture in {"a2_nonlinear_cartesian", "a3_nonlinear_reduced"}:
        return model.coefficients(batch, create_graph=create_graph)
    return model.coefficients(batch)


def _losses(model: nn.Module, batch, architecture: str, create_graph: bool):
    prediction = _coefficients(model, batch, architecture, create_graph)
    electronic = irrep_balanced_capacity_loss(
        prediction.electronic_piezo, batch.y_electronic_piezo
    )
    born = born_material_balanced_loss(
        prediction.born_charges, batch.y_born, batch.batch
    )
    return electronic, born


def backward_training_objective(
    model: nn.Module, batch, architecture: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backpropagate one exact joint objective with bounded A0 graph memory."""
    if architecture != "a0_independent_irreps":
        electronic, born = _losses(model, batch, architecture, create_graph=True)
        loss = electronic + born
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite electrostatic fold loss")
        loss.backward()
        return electronic, born

    # A0's two towers share no parameters. Backpropagating each loss before
    # constructing the other tower's graph is exactly equivalent to backward
    # on their sum, while peak activation memory is that of one tower rather
    # than both towers together.
    electronic_prediction, _ = model.electronic_response(batch)
    electronic = irrep_balanced_capacity_loss(
        electronic_prediction, batch.y_electronic_piezo
    )
    if not torch.isfinite(electronic):
        raise FloatingPointError("Non-finite A0 electronic loss")
    electronic.backward()
    born_prediction = model.born_charges(batch)
    born = born_material_balanced_loss(
        born_prediction, batch.y_born, batch.batch
    )
    if not torch.isfinite(born):
        raise FloatingPointError("Non-finite A0 Born-charge loss")
    born.backward()
    return electronic, born


def task_gradient_geometry(
    model: nn.Module, batch, architecture: str
) -> dict[str, float | int | None]:
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if architecture == "a0_independent_irreps":
        electronic_prediction, _ = model.electronic_response(batch)
        electronic = irrep_balanced_capacity_loss(
            electronic_prediction, batch.y_electronic_piezo
        )
        electronic_grad = torch.autograd.grad(
            electronic, parameters, allow_unused=True
        )
        born_prediction = model.born_charges(batch)
        born = born_material_balanced_loss(
            born_prediction, batch.y_born, batch.batch
        )
        born_grad = torch.autograd.grad(
            born, parameters, allow_unused=True
        )
    else:
        electronic, born = _losses(
            model, batch, architecture, create_graph=True
        )
        electronic_grad = torch.autograd.grad(
            electronic, parameters, retain_graph=True, allow_unused=True
        )
        born_grad = torch.autograd.grad(
            born, parameters, allow_unused=True
        )
    electronic_total_squared = sum(
        gradient.detach().square().sum()
        for gradient in electronic_grad if gradient is not None
    )
    born_total_squared = sum(
        gradient.detach().square().sum()
        for gradient in born_grad if gradient is not None
    )
    shared = [
        (left.detach(), right.detach())
        for left, right in zip(electronic_grad, born_grad, strict=True)
        if left is not None and right is not None
    ]
    result: dict[str, float | int | None] = {
        "electronic_loss": float(electronic.detach()),
        "born_loss": float(born.detach()),
        "electronic_parameter_tensors": sum(
            gradient is not None for gradient in electronic_grad
        ),
        "born_parameter_tensors": sum(gradient is not None for gradient in born_grad),
        "electronic_total_gradient_norm": float(torch.sqrt(electronic_total_squared)),
        "born_total_gradient_norm": float(torch.sqrt(born_total_squared)),
        "shared_parameter_tensors": len(shared),
        "shared_electronic_gradient_norm": None,
        "shared_born_gradient_norm": None,
        "shared_gradient_cosine": None,
    }
    if not shared:
        return result
    left_squared = sum(left.square().sum() for left, _ in shared)
    right_squared = sum(right.square().sum() for _, right in shared)
    dot = sum((left * right).sum() for left, right in shared)
    left_norm = torch.sqrt(left_squared)
    right_norm = torch.sqrt(right_squared)
    cosine = dot / (left_norm * right_norm).clamp_min(1e-30)
    result.update({
        "shared_electronic_gradient_norm": float(left_norm),
        "shared_born_gradient_norm": float(right_norm),
        "shared_gradient_cosine": float(cosine),
    })
    return result


def response_active_diagnostic_indices(
    dataset,
    batch_size: int,
    *,
    scan_limit: int = 128,
    component_floor: float = 0.05,
) -> tuple[list[int], list[dict[str, float | str]]]:
    """Choose a deterministic norm-stratified batch of active train records."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    threshold = component_floor * math.sqrt(18.0)
    candidates: list[tuple[float, int, str]] = []
    for index in range(min(len(dataset), scan_limit)):
        graph = dataset[index]
        norm = float(torch.linalg.vector_norm(graph.y_electronic_piezo))
        if math.isfinite(norm) and norm >= threshold:
            candidates.append((norm, index, str(graph.material_id)))
    if not candidates:
        raise RuntimeError(
            f"No response-active electronic targets in the first {scan_limit} records"
        )
    candidates.sort()
    count = min(batch_size, len(candidates))
    ranks = (
        [len(candidates) - 1]
        if count == 1
        else [
            round(position * (len(candidates) - 1) / (count - 1))
            for position in range(count)
        ]
    )
    selected = [candidates[rank] for rank in ranks]
    indices = [index for _, index, _ in selected]
    audit = [
        {"material_id": material_id, "target_norm_c_per_m2": norm}
        for norm, _, material_id in selected
    ]
    return indices, audit


def load_structure_pretraining(
    model: nn.Module,
    architecture: str,
    checkpoint: Path,
    device: torch.device,
) -> dict[str, object]:
    """Load one fold-train-only PeriodicCrystalEncoder into each candidate."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != "e3nn_periodic_v1":
        raise ValueError("Structure checkpoint is not a PeriodicCrystalEncoder pretrain")
    state = payload["encoder"]
    if architecture == "a0_independent_irreps":
        model.born_generator.encoder.load_state_dict(state, strict=True)
        model.piezo_generator.encoder.load_state_dict(state, strict=True)
        encoder_copies = 2
    elif architecture == "a1_electromechanical_jet":
        model.encoder.load_state_dict(state, strict=True)
        encoder_copies = 1
    else:
        model.state.encoder.load_state_dict(state, strict=True)
        encoder_copies = 1
    return {
        "checkpoint": str(checkpoint.resolve()),
        "pretraining_provenance": payload.get("pretraining_provenance"),
        "pretraining_epoch": payload.get("epoch"),
        "pretraining_loss": payload.get("loss"),
        "encoder_copies_initialized": encoder_copies,
    }


def _dataset(config: dict[str, object], records, ids: list[str], cache_key: str):
    return PiezoDataset(
        records,
        ids,
        float(config["cutoff"]),
        int(config["max_neighbors"]),
        processed_dir=config["processed_dir"],
        cache_key=cache_key,
        project_targets=True,
        dfpt_dir=config["jarvis_dfpt_dir"],
        strain_completion_dir=None,
    )


def _evaluate(model, loader, architecture: str):
    predictions, targets = [], []
    born_predictions, born_targets, batch_indices = [], [], []
    offset = 0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = _coefficients(model, batch, architecture, create_graph=False)
            predictions.append(prediction.electronic_piezo.detach())
            targets.append(batch.y_electronic_piezo)
            born_predictions.append(prediction.born_charges.detach())
            born_targets.append(batch.y_born)
            batch_indices.append(batch.batch + offset)
            offset += int(batch.num_graphs)
    return {
        "electronic": electronic_capacity_metrics(
            torch.cat(predictions), torch.cat(targets)
        ),
        "born": born_capacity_metrics(
            torch.cat(born_predictions), torch.cat(born_targets),
            torch.cat(batch_indices),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--folds", type=Path,
        default=Path("data/processed/electrostatic_development_folds.json"),
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--development-limit", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pretrained-encoder", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1 or args.eval_interval < 1:
        raise ValueError("updates, batch size, and eval interval must be positive")

    config = load_project_config(args.config)
    folds = json.loads(args.folds.read_text(encoding="utf-8-sig"))
    fold = next((value for value in folds["folds"] if value["fold"] == args.fold), None)
    if fold is None:
        raise ValueError(f"Fold {args.fold} is absent from {args.folds}")
    train_ids = deterministic_subset(
        electrostatic_fold_train_ids(folds, args.fold),
        args.train_limit,
        args.seed + 1000,
    )
    dev_ids = deterministic_subset(
        list(fold["development"]), args.development_limit, args.seed + 2000
    )
    seed_everything(args.seed)
    records = load_gmtnet_records(config["data_root"])
    cache_key = graph_cache_key(
        records, float(config["cutoff"]), int(config["max_neighbors"])
    )
    train_set = _dataset(config, records, train_ids, cache_key)
    dev_set = _dataset(config, records, dev_ids, cache_key)
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, generator=generator,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    model = make_model(args.architecture, config).to(device)
    pretraining = (
        load_structure_pretraining(
            model, args.architecture, args.pretrained_encoder, device
        )
        if args.pretrained_encoder is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-6
    )
    iterator = iter(train_loader)
    diagnostic_indices, diagnostic_materials = response_active_diagnostic_indices(
        train_set, min(args.batch_size, len(train_set))
    )
    diagnostic_batch = next(iter(DataLoader(
        Subset(train_set, diagnostic_indices), batch_size=len(diagnostic_indices),
        shuffle=False, num_workers=0,
    ))).to(device)
    initial_gradient = task_gradient_geometry(
        model, diagnostic_batch, args.architecture
    )
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    history = []
    best_score = math.inf
    best_state = None
    best_update = None
    for update in range(1, args.updates + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = batch.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        electronic, born = backward_training_objective(
            model, batch, args.architecture
        )
        loss = electronic.detach() + born.detach()
        optimizer.step()
        row = {
            "update": update,
            "train_loss": float(loss.detach()),
            "train_electronic_loss": float(electronic.detach()),
            "train_born_loss": float(born.detach()),
        }
        if update % args.eval_interval == 0 or update == args.updates:
            metrics = _evaluate(model, dev_loader, args.architecture)
            score = (
                float(metrics["electronic"]["mean_stabilized_relative_frobenius_error"])
                + float(metrics["born"]["mean_relative_frobenius_error"])
            )
            row["development_selection_score"] = score
            if score < best_score:
                best_score, best_update = score, update
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
        history.append(row)
        if update == 1 or update % args.eval_interval == 0:
            print(json.dumps(row))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No development checkpoint was selected")
    model.load_state_dict(best_state)
    selected_metrics = _evaluate(model, dev_loader, args.architecture)
    final_gradient = task_gradient_geometry(
        model, diagnostic_batch, args.architecture
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({
        "model": best_state,
        "architecture": args.architecture,
        "fold": args.fold,
        "selected_update": best_update,
        "seed": args.seed,
    }, args.output_dir / "selected.pt")
    summary = {
        "schema": 1,
        "protocol": "formula-disjoint electrostatic A0-A3 development adjudication",
        "architecture": args.architecture,
        "fold": args.fold,
        "seed": args.seed,
        "train_materials": len(train_set),
        "development_materials": len(dev_set),
        "train_limit": args.train_limit,
        "development_limit": args.development_limit,
        "frozen_validation_test_labels_read": False,
        "initialization": (
            "fold-train-only structure-pretrained encoder; random response heads; "
            "no samples32 checkpoint"
            if pretraining is not None
            else "random response parameters; no samples32 checkpoint"
        ),
        "structure_pretraining": pretraining,
        "optimizer_updates": args.updates,
        "batch_size": args.batch_size,
        "num_workers": 0,
        "selection": "minimum development electronic-relative plus BEC-relative score",
        "selected_update": best_update,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "runtime": {
            "device": str(device),
            "seconds": seconds,
            "updates_per_second": args.updates / seconds,
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "jacobian_execution": "three fixed reverse VJPs; e3nn scripted detach has no vmap batching rule" if args.architecture.startswith(("a2", "a3")) else "direct coefficient forward",
            "a0_backward_execution": (
                "sequential exact disjoint-task backward; one tower graph resident"
                if args.architecture == "a0_independent_irreps" else None
            ),
        },
        "gradient_geometry": {
            "diagnostic_batch_materials": int(diagnostic_batch.num_graphs),
            "selection": "fixed train prefix; response-active; norm-stratified ranks",
            "materials": diagnostic_materials,
            "initial": initial_gradient,
            "selected": final_gradient,
        },
        "metrics": selected_metrics,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected_update": best_update, "metrics": selected_metrics}, indent=2))


if __name__ == "__main__":
    main()
