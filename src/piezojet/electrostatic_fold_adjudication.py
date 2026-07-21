"""Formula-disjoint first-order electrostatic-jet development adjudication."""

from __future__ import annotations

import argparse
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

from .data import (
    PiezoDataset,
    deterministic_subset,
    graph_cache_key,
    load_gmtnet_records,
)
from .checkpoint_provenance import build_checkpoint_provenance
from .build_electrostatic_development_folds import electrostatic_fold_train_ids
from .electronic_capacity import (
    born_capacity_metrics,
    born_material_balanced_loss,
    dielectric_capacity_metrics,
    dielectric_material_balanced_loss,
    electronic_capacity_metrics,
    irrep_balanced_capacity_loss,
)
from .electrostatic_protocol import (
    A0_ARCHITECTURES,
    ARCHITECTURES,
    DEFAULT_EARLY_STOPPING_PATIENCE_EVALUATIONS,
    STABILIZED_SELECTION_VERSION,
    compact_training_curve_row,
    development_early_stopping,
    development_selection,
    matched_material_schedule,
)
from .electrostatic_subset import load_response_subset
from .model import (
    ElectromechanicalJetHead,
    HierarchicalElectromechanicalJetHead,
    IndependentElectrostaticHeads,
    SoftSharedElectromechanicalJetHead,
)
from .project_config import load_project_config
from .pretraining_protocol import validate_inductive_checkpoint
from .train import _data_commit, _git_commit, seed_everything


_PRETRAINING_COMPATIBILITY_KEYS = (
    "embedding_dim",
    "cutoff",
    "max_neighbors",
    "lmax",
    "num_blocks",
    "radial_basis",
    "radial_hidden",
    "electrostatic_encoder_width_multiplier",
)


def encoder_width_multiplier_for_architecture(
    architecture: str, config: dict[str, object]
) -> float:
    """Resolve an explicit architecture-bound e3nn encoder width."""
    if architecture == "a0_parameter_matched_irreps":
        return float(config.get("a0_parameter_matched_width_multiplier", 0.56))
    return float(config.get("electrostatic_encoder_width_multiplier", 1.0))


def _model_kwargs(
    config: dict[str, object], architecture: str
) -> dict[str, object]:
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
        "encoder_width_multiplier": encoder_width_multiplier_for_architecture(
            architecture, config
        ),
    }


def make_model(architecture: str, config: dict[str, object]) -> nn.Module:
    kwargs = _model_kwargs(config, architecture)
    if architecture in A0_ARCHITECTURES:
        return IndependentElectrostaticHeads(**kwargs)
    if architecture == "a1_electromechanical_jet":
        return ElectromechanicalJetHead(**kwargs)
    if architecture == "a15_soft_shared_electromechanical_jet":
        return SoftSharedElectromechanicalJetHead(**kwargs)
    if architecture == "a16_hierarchical_electromechanical_jet":
        return HierarchicalElectromechanicalJetHead(**kwargs)
    raise ValueError(f"Unknown architecture: {architecture}")


def _coefficients(model: nn.Module, batch, architecture: str, create_graph: bool):
    return model.coefficients(batch)


def _losses(model: nn.Module, batch, architecture: str, create_graph: bool):
    prediction = _coefficients(model, batch, architecture, create_graph)
    electronic = irrep_balanced_capacity_loss(
        prediction.electronic_piezo, batch.y_electronic_piezo
    )
    born = born_material_balanced_loss(
        prediction.born_charges, batch.y_born, batch.batch
    )
    dielectric = dielectric_material_balanced_loss(
        prediction.electronic_dielectric,
        batch.y_dfpt_electronic_dielectric,
        batch.dfpt_electronic_dielectric_mask,
    )
    return electronic, born, dielectric


def backward_training_objective(
    model: nn.Module, batch, architecture: str, *, gradient_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backpropagate one joint objective, optionally as a mean microbatch.

    ``gradient_scale`` is the material fraction of one logical update.  It
    permits memory-bounded microbatch accumulation of the same mathematical
    mean objective with one AdamW update per logical batch. Different floating-
    point reduction orders are tolerance-equivalent, not bitwise identical.
    """
    if gradient_scale <= 0.0:
        raise ValueError("gradient_scale must be positive")
    if architecture not in A0_ARCHITECTURES:
        electronic, born, dielectric = _losses(
            model, batch, architecture, create_graph=True
        )
        loss = electronic + born + dielectric
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite electrostatic fold loss")
        (loss * gradient_scale).backward()
        return electronic, born, dielectric

    # A0's three towers share no parameters. Backpropagating each loss before
    # constructing the next tower's graph is exactly equivalent to backward
    # on their sum, while peak activation memory is that of one tower rather
    # than all three towers together.
    electronic_prediction = model.electronic_response(batch)
    electronic = irrep_balanced_capacity_loss(
        electronic_prediction, batch.y_electronic_piezo
    )
    if not torch.isfinite(electronic):
        raise FloatingPointError("Non-finite A0 electronic loss")
    (electronic * gradient_scale).backward()
    born_prediction = model.born_charges(batch)
    born = born_material_balanced_loss(
        born_prediction, batch.y_born, batch.batch
    )
    if not torch.isfinite(born):
        raise FloatingPointError("Non-finite A0 Born-charge loss")
    (born * gradient_scale).backward()
    dielectric_prediction = model.dielectric_response(batch)
    dielectric = dielectric_material_balanced_loss(
        dielectric_prediction,
        batch.y_dfpt_electronic_dielectric,
        batch.dfpt_electronic_dielectric_mask,
    )
    if not torch.isfinite(dielectric):
        raise FloatingPointError("Non-finite A0 dielectric loss")
    (dielectric * gradient_scale).backward()
    return electronic, born, dielectric


def task_gradient_geometry(
    model: nn.Module, batch, architecture: str
) -> dict[str, float | int | None]:
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if architecture in A0_ARCHITECTURES:
        electronic_prediction = model.electronic_response(batch)
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
        dielectric_prediction = model.dielectric_response(batch)
        dielectric = dielectric_material_balanced_loss(
            dielectric_prediction,
            batch.y_dfpt_electronic_dielectric,
            batch.dfpt_electronic_dielectric_mask,
        )
        dielectric_grad = torch.autograd.grad(
            dielectric, parameters, allow_unused=True
        )
    else:
        electronic, born, dielectric = _losses(
            model, batch, architecture, create_graph=True
        )
        electronic_grad = torch.autograd.grad(
            electronic, parameters, retain_graph=True, allow_unused=True
        )
        born_grad = torch.autograd.grad(
            born, parameters, retain_graph=True, allow_unused=True
        )
        dielectric_grad = torch.autograd.grad(
            dielectric, parameters, allow_unused=True
        )
    electronic_total_squared = sum(
        gradient.detach().square().sum()
        for gradient in electronic_grad if gradient is not None
    )
    born_total_squared = sum(
        gradient.detach().square().sum()
        for gradient in born_grad if gradient is not None
    )
    dielectric_total_squared = sum(
        gradient.detach().square().sum()
        for gradient in dielectric_grad if gradient is not None
    )
    shared = [
        (left.detach(), right.detach())
        for left, right in zip(electronic_grad, born_grad, strict=True)
        if left is not None and right is not None
    ]
    result: dict[str, float | int | None] = {
        "electronic_loss": float(electronic.detach()),
        "born_loss": float(born.detach()),
        "dielectric_loss": float(dielectric.detach()),
        "electronic_parameter_tensors": sum(
            gradient is not None for gradient in electronic_grad
        ),
        "born_parameter_tensors": sum(gradient is not None for gradient in born_grad),
        "dielectric_parameter_tensors": sum(
            gradient is not None for gradient in dielectric_grad
        ),
        "electronic_total_gradient_norm": float(torch.sqrt(electronic_total_squared)),
        "born_total_gradient_norm": float(torch.sqrt(born_total_squared)),
        "dielectric_total_gradient_norm": float(
            torch.sqrt(dielectric_total_squared)
        ),
        "shared_parameter_tensors": len(shared),
        "shared_electronic_gradient_norm": None,
        "shared_born_gradient_norm": None,
        "shared_gradient_cosine": None,
        "electronic_dielectric_shared_gradient_cosine": None,
        "born_dielectric_shared_gradient_cosine": None,
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
    def pair_cosine(left_grad, right_grad):
        pairs = [
            (left.detach(), right.detach())
            for left, right in zip(left_grad, right_grad, strict=True)
            if left is not None and right is not None
        ]
        if not pairs:
            return None
        left_norm = torch.sqrt(sum(left.square().sum() for left, _ in pairs))
        right_norm = torch.sqrt(sum(right.square().sum() for _, right in pairs))
        dot = sum((left * right).sum() for left, right in pairs)
        return float(dot / (left_norm * right_norm).clamp_min(1e-30))
    result["electronic_dielectric_shared_gradient_cosine"] = pair_cosine(
        electronic_grad, dielectric_grad
    )
    result["born_dielectric_shared_gradient_cosine"] = pair_cosine(
        born_grad, dielectric_grad
    )
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
    train_ids: list[str],
    development_ids: list[str],
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Strictly load a fold-train-only encoder into one Stage-A candidate.

    A state-dict shape match is not sufficient: pretraining on a development
    structure would make this formula-disjoint comparison transductive even
    without reading its response labels.  Validate the persisted material-ID
    provenance before copying any weights.
    """
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != "e3nn_periodic_v1":
        raise ValueError("Structure checkpoint is not a PeriodicCrystalEncoder pretrain")
    source_code_commit = payload.get("code_commit")
    downstream_code_commit = None if config is None else config.get("code_commit")
    if config is not None:
        if (
            not isinstance(source_code_commit, str)
            or len(source_code_commit) != 40
            or any(character not in "0123456789abcdefABCDEF" for character in source_code_commit)
        ):
            raise ValueError("Structure checkpoint has no valid source code commit")
        saved_config = payload.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("Structure checkpoint has no saved encoder configuration")
        expected_config = dict(config)
        expected_config["electrostatic_encoder_width_multiplier"] = (
            encoder_width_multiplier_for_architecture(architecture, config)
        )
        mismatches = {
            key: {
                "checkpoint": saved_config.get(key, 1.0)
                if key == "electrostatic_encoder_width_multiplier"
                else saved_config.get(key),
                "downstream": expected_config.get(key, 1.0)
                if key == "electrostatic_encoder_width_multiplier"
                else expected_config.get(key),
            }
            for key in _PRETRAINING_COMPATIBILITY_KEYS
            if key in expected_config
            and (
                saved_config.get(key, 1.0)
                if key == "electrostatic_encoder_width_multiplier"
                else saved_config.get(key)
            )
            != expected_config.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Structure checkpoint encoder configuration differs: {mismatches}"
            )
        contract = payload.get("pretraining_contract")
        if not isinstance(contract, dict):
            raise ValueError("Structure checkpoint has no pretraining contract")
        if (
            contract.get("objective")
            != "masked_species_plus_translation_free_coordinate_denoising"
            or contract.get("response_label_count") != 0
        ):
            raise ValueError("Structure checkpoint pretraining objective differs")
    pretraining_provenance = validate_inductive_checkpoint(
        payload, train_ids, development_ids, config
    )
    state = payload["encoder"]
    if architecture in A0_ARCHITECTURES:
        model.born_generator.encoder.load_state_dict(state, strict=True)
        model.piezo_generator.encoder.load_state_dict(state, strict=True)
        model.dielectric_generator.encoder.load_state_dict(state, strict=True)
        encoder_copies = 3
    elif architecture in {
        "a1_electromechanical_jet",
        "a15_soft_shared_electromechanical_jet",
        "a16_hierarchical_electromechanical_jet",
    }:
        model.encoder.load_state_dict(state, strict=True)
        encoder_copies = 1
    else:
        raise ValueError(f"Unsupported maintained Stage-A architecture: {architecture}")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "pretraining_provenance": pretraining_provenance,
        "pretraining_epoch": payload.get("epoch"),
        "pretraining_loss": payload.get("loss"),
        "pretraining_source_code_commit": source_code_commit,
        "downstream_code_commit": downstream_code_commit,
        "cross_commit_reuse": (
            source_code_commit is not None
            and downstream_code_commit is not None
            and source_code_commit != downstream_code_commit
        ),
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
        dfpt_profile="electrostatic",
    )


def _evaluate(model, loader, architecture: str):
    predictions, targets = [], []
    born_predictions, born_targets, batch_indices = [], [], []
    dielectric_predictions, dielectric_targets, dielectric_masks = [], [], []
    offset = 0
    device = next(model.parameters()).device
    model.eval()
    # The reciprocal-geometry cache keys tensors by their version counters;
    # inference tensors intentionally have no version counter.
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = _coefficients(model, batch, architecture, create_graph=False)
            predictions.append(prediction.electronic_piezo.detach())
            targets.append(batch.y_electronic_piezo)
            born_predictions.append(prediction.born_charges.detach())
            born_targets.append(batch.y_born)
            batch_indices.append(batch.batch + offset)
            dielectric_predictions.append(prediction.electronic_dielectric.detach())
            dielectric_targets.append(batch.y_dfpt_electronic_dielectric)
            dielectric_masks.append(batch.dfpt_electronic_dielectric_mask)
            offset += int(batch.num_graphs)
    return {
        "electronic": electronic_capacity_metrics(
            torch.cat(predictions), torch.cat(targets)
        ),
        "born": born_capacity_metrics(
            torch.cat(born_predictions), torch.cat(born_targets),
            torch.cat(batch_indices),
        ),
        "dielectric_audit": dielectric_capacity_metrics(
            torch.cat(dielectric_predictions), torch.cat(dielectric_targets),
            torch.cat(dielectric_masks),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--folds", type=Path,
        default=Path("data/processed/electrostatic_development_folds_v2.json"),
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--architecture",
        choices=tuple(
            architecture
            for architecture in ARCHITECTURES
            if architecture not in A0_ARCHITECTURES
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--microbatch-size", type=int, default=0,
        help="Physical graphs per forward/backward; 0 uses the logical batch size",
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=0,
        help="Physical graphs per development forward; 0 uses --microbatch-size",
    )
    parser.add_argument(
        "--diagnostic-batch-size", type=int, default=0,
        help="Graphs in the fixed gradient diagnostic; 0 uses --microbatch-size",
    )
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
    parser.add_argument("--pretrained-encoder", type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an exact deterministic schedule from a run-local progress.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1 or args.eval_interval < 1:
        raise ValueError("updates, batch size, and eval interval must be positive")
    if args.microbatch_size < 0 or args.eval_batch_size < 0 or args.diagnostic_batch_size < 0:
        raise ValueError("microbatch and evaluation batch sizes cannot be negative")
    if args.early_stopping_patience_evaluations < 0:
        raise ValueError("Early-stopping patience cannot be negative")
    if (
        args.early_stopping_minimum_improvement < 0.0
        or not math.isfinite(args.early_stopping_minimum_improvement)
    ):
        raise ValueError(
            "Early-stopping minimum improvement must be finite and nonnegative"
        )
    microbatch_size = args.microbatch_size or args.batch_size
    evaluation_batch_size = args.eval_batch_size or microbatch_size
    diagnostic_batch_size = args.diagnostic_batch_size or microbatch_size
    if microbatch_size > args.batch_size or args.batch_size % microbatch_size:
        raise ValueError("batch size must be divisible by microbatch size")

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
    seed_everything(args.seed)
    records = load_gmtnet_records(config["data_root"])
    cache_key = graph_cache_key(
        records, float(config["cutoff"]), int(config["max_neighbors"])
    )
    train_set = _dataset(config, records, train_ids, cache_key)
    dev_set = _dataset(config, records, dev_ids, cache_key)
    if args.batch_size > len(train_set):
        raise ValueError("Logical batch size cannot exceed the training panel")
    schedule = matched_material_schedule(
        len(train_set), args.updates, args.batch_size, microbatch_size, args.seed
    )
    device = torch.device(args.device)
    dev_loader = DataLoader(
        dev_set, batch_size=evaluation_batch_size, shuffle=False, num_workers=0,
    )
    train_eval_loader = DataLoader(
        train_set, batch_size=evaluation_batch_size, shuffle=False, num_workers=0,
    )
    if args.resume is None:
        if args.output_dir.exists():
            raise FileExistsError(f"Stage-A output directory exists: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=False)
    else:
        if not args.resume.resolve().is_relative_to(args.output_dir.resolve()):
            raise ValueError("Resume checkpoint must be inside --output-dir")
        if not args.output_dir.is_dir():
            raise FileNotFoundError(f"Resume output directory is absent: {args.output_dir}")
    model = make_model(args.architecture, config).to(device)
    pretraining = (
        load_structure_pretraining(
            model, args.architecture, args.pretrained_encoder, device,
            full_fold_train_ids, dev_ids, config,
        )
        if args.pretrained_encoder is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-6
    )
    training_contract = {
        "architecture": args.architecture,
        "fold": args.fold,
        "seed": args.seed,
        "updates": args.updates,
        "logical_batch_size": args.batch_size,
        "microbatch_size": microbatch_size,
        "eval_interval": args.eval_interval,
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
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "train_ids": train_ids,
        "full_fold_structure_pretraining_ids": full_fold_train_ids,
        "development_ids": dev_ids,
    }
    start_update = 1
    history: list[dict[str, object]] = []
    best_score = math.inf
    best_state = None
    best_update = None
    non_improving_evaluations = 0
    saved_initial_gradient = None
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if payload.get("status") not in {"running", "interrupted"}:
            raise ValueError("Resume checkpoint is not an incomplete Stage-A run")
        if payload.get("training_contract") != training_contract:
            raise ValueError("Resume training contract differs from the current command")
        if payload.get("checkpoint_provenance") != checkpoint_provenance:
            raise ValueError("Resume checkpoint provenance differs from the current fold")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        completed_update = int(payload["completed_update"])
        start_update = completed_update + 1
        history = list(payload["history"])
        best_score = float(payload["best_score"])
        best_state = payload["best_model"]
        saved_best_update = payload.get("best_update")
        best_update = (
            int(saved_best_update) if saved_best_update is not None else None
        )
        saved_initial_gradient = payload["initial_gradient"]
        non_improving_evaluations = int(
            payload.get("non_improving_evaluations", 0)
        )
    scheduled_tail = schedule[(start_update - 1) * args.batch_size :]
    train_loader = DataLoader(
        Subset(train_set, scheduled_tail),
        batch_size=microbatch_size,
        shuffle=False,
        num_workers=0,
    )
    iterator = iter(train_loader)
    diagnostic_indices, diagnostic_materials = response_active_diagnostic_indices(
        train_set, min(diagnostic_batch_size, len(train_set))
    )
    diagnostic_batch = next(iter(DataLoader(
        Subset(train_set, diagnostic_indices), batch_size=len(diagnostic_indices),
        shuffle=False, num_workers=0,
    ))).to(device)
    initial_gradient = (
        saved_initial_gradient
        if saved_initial_gradient is not None
        else task_gradient_geometry(model, diagnostic_batch, args.architecture)
    )
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    counted_flops_per_update: int | None = None
    cuda_update_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    stopped_early = False
    completed_update = start_update - 1
    for update in range(start_update, args.updates + 1):
        completed_update = update
        batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        logical_electronic = 0.0
        logical_born = 0.0
        logical_dielectric = 0.0
        start_event = end_event = None
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        flop_context = (
            FlopCounterMode(display=False)
            if counted_flops_per_update is None
            else nullcontext()
        )
        # Every physical batch has the same number of graphs (drop_last=True),
        # and the divisibility check gives the same mathematical material-mean
        # objective. Floating-point reductions need not be bitwise identical
        # to one larger forward pass.
        with flop_context:
            for microbatch in range(args.batch_size // microbatch_size):
                if microbatch:
                    batch = next(iterator)
                batch = batch.to(device)
                model.train()
                fraction = int(batch.num_graphs) / args.batch_size
                electronic, born, dielectric = backward_training_objective(
                    model, batch, args.architecture, gradient_scale=fraction,
                )
                logical_electronic += float(electronic.detach()) * fraction
                logical_born += float(born.detach()) * fraction
                logical_dielectric += float(dielectric.detach()) * fraction
                del batch
            loss = logical_electronic + logical_born + logical_dielectric
            optimizer.step()
        if isinstance(flop_context, FlopCounterMode):
            counted_flops_per_update = int(flop_context.get_total_flops())
        if start_event is not None and end_event is not None:
            end_event.record()
            cuda_update_events.append((start_event, end_event))
        row = {
            "update": update,
            "train_loss": loss,
            "train_electronic_loss": logical_electronic,
            "train_born_loss": logical_born,
            "train_dielectric_loss": logical_dielectric,
        }
        if update % args.eval_interval == 0 or update == args.updates:
            evaluation_started = time.perf_counter()
            metrics = _evaluate(model, dev_loader, args.architecture)
            development_evaluation_seconds = time.perf_counter() - evaluation_started
            train_evaluation_started = time.perf_counter()
            train_metrics = _evaluate(model, train_eval_loader, args.architecture)
            train_evaluation_seconds = time.perf_counter() - train_evaluation_started
            selection = development_selection(metrics)
            train_selection = development_selection(train_metrics)
            score = float(selection["raw_score"])
            row["development_selection_score"] = score
            row["development_selection"] = selection
            row["development_metrics"] = metrics
            row["train_metrics"] = train_metrics
            row["train_selection"] = train_selection
            row["generalization_score_gap"] = score - float(
                train_selection["raw_score"]
            )
            row["evaluation_runtime"] = {
                "development_seconds": development_evaluation_seconds,
                "train_seconds": train_evaluation_seconds,
                "total_seconds": (
                    development_evaluation_seconds + train_evaluation_seconds
                ),
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
                best_score, best_update = score, update
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            evaluation_payload = {
                "schema": 1,
                "status": "running",
                "completed_update": update,
                "training_contract": training_contract,
                "checkpoint_provenance": checkpoint_provenance,
                "model": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "optimizer": optimizer.state_dict(),
                "history": [*history, row],
                "best_score": best_score,
                "best_update": best_update,
                "best_model": best_state,
                "non_improving_evaluations": non_improving_evaluations,
                "initial_gradient": initial_gradient,
                "development_metrics": metrics,
                "train_metrics": train_metrics,
                "selection_score": score,
                "selection": selection,
            }
            checkpoint_dir = args.output_dir / "evaluation_checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(
                evaluation_payload,
                checkpoint_dir / f"update_{update:08d}.pt",
            )
            torch.save(evaluation_payload, args.output_dir / "progress.pt")
            (args.output_dir / "progress.json").write_text(
                json.dumps({
                    "schema": 1,
                    "status": "running",
                    "completed_update": update,
                    "best_update": best_update,
                    "best_score": best_score,
                    "early_stopping": early_stopping,
                    "selection_version": STABILIZED_SELECTION_VERSION,
                    "frozen_validation_test_labels_read": False,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            curve_history = [*history, row]
            (args.output_dir / "training_curve.json").write_text(
                json.dumps(
                    [
                        compact
                        for history_row in curve_history
                        if (compact := compact_training_curve_row(history_row))
                        is not None
                    ],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        history.append(row)
        if update == 1 or update % args.eval_interval == 0:
            print(json.dumps(row))
        if row.get("early_stopping", {}).get("should_stop", False):
            stopped_early = True
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    optimizer_seconds = (
        sum(start.elapsed_time(end) for start, end in cuda_update_events) / 1000.0
        if cuda_update_events
        else seconds
    )
    if best_state is None:
        summary = {
            "schema": 2,
            "status": "completed_no_eligible_checkpoint",
            "protocol": (
                "formula-disjoint three-task first-order electrostatic-jet "
                "development adjudication"
            ),
            "architecture": args.architecture,
            "fold": args.fold,
            "seed": args.seed,
            "checkpoint_provenance": checkpoint_provenance,
            "train_materials": len(train_set),
            "development_materials": len(dev_set),
            "frozen_validation_test_labels_read": False,
            "selection_version": STABILIZED_SELECTION_VERSION,
            "selection": (
                "no checkpoint passed all directional and amplitude guardrails; "
                "no selected.pt was written"
            ),
            "history": history,
            "runtime": {
                "device": str(device),
                "seconds": seconds,
                "optimizer_seconds": optimizer_seconds,
                "counted_flops_per_update": counted_flops_per_update,
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
        return
    model.load_state_dict(best_state)
    selected_metrics = _evaluate(model, dev_loader, args.architecture)
    final_gradient = task_gradient_geometry(
        model, diagnostic_batch, args.architecture
    )
    torch.save({
        "model": best_state,
        "architecture": args.architecture,
        "fold": args.fold,
        "selected_update": best_update,
        "seed": args.seed,
        "checkpoint_provenance": checkpoint_provenance,
    }, args.output_dir / "selected.pt")
    summary = {
        "schema": 2,
        "status": "complete_early_stopped" if stopped_early else "complete",
        "protocol": (
            "formula-disjoint three-task first-order electrostatic-jet "
            "development adjudication"
        ),
        "architecture": args.architecture,
        "fold": args.fold,
        "seed": args.seed,
        "checkpoint_provenance": checkpoint_provenance,
        "train_materials": len(train_set),
        "development_materials": len(dev_set),
        "train_limit": args.train_limit,
        "response_subset_manifest": (
            str(args.train_ids_file.resolve()) if args.train_ids_file else None
        ),
        "structure_pretraining_universe_materials": len(full_fold_train_ids),
        "development_limit": args.development_limit,
        "frozen_validation_test_labels_read": False,
        "initialization": (
            "fold-train-only structure-pretrained encoder; random response heads; "
            "no samples32 checkpoint"
            if pretraining is not None
            else "random response parameters; no samples32 checkpoint"
        ),
        "structure_pretraining": pretraining,
        "requested_optimizer_updates": args.updates,
        "optimizer_updates": completed_update,
        "material_exposures": completed_update * args.batch_size,
        "effective_train_passes": completed_update * args.batch_size / len(train_set),
        "logical_batch_size": args.batch_size,
        "microbatch_size": microbatch_size,
        "microbatches_per_update": args.batch_size // microbatch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "diagnostic_batch_size": int(diagnostic_batch.num_graphs),
        "num_workers": 0,
        "selection": (
            "minimum development electronic-stabilized-relative plus "
            "BEC-stabilized-relative plus "
            "electronic-dielectric-relative score"
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
            "one immutable model+optimizer+train/development-metrics checkpoint "
            "per full-development evaluation point"
        ),
        "selected_update": best_update,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "runtime": {
            "device": str(device),
            "seconds": seconds,
            "optimizer_seconds": optimizer_seconds,
            "counted_flops_per_update": counted_flops_per_update,
            "flop_count_scope": (
                "torch-dispatch-supported forward, backward, and AdamW operations "
                "for one measured logical update"
            ),
            "updates_per_second": completed_update / seconds,
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "jacobian_execution": "direct first-order coefficient forward",
            "a0_backward_execution": (
                "sequential exact three-disjoint-task backward; one tower graph resident"
                if args.architecture in A0_ARCHITECTURES else None
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
    print(json.dumps({"selected_update": best_update, "metrics": selected_metrics}, indent=2))


if __name__ == "__main__":
    main()
