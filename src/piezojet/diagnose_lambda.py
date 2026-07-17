"""Controlled diagnostics for atom-coordinate internal-strain tensors.

This module deliberately avoids the macroscopic piezoelectric training path.
It answers four narrower questions on the strictly symmetry-completed DFPT
subset: can the model memorize Lambda, can it recover a teacher-generated
Lambda, does held-out performance grow with complete-supervision count, and
which engineering-Voigt strain channels fail.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from .data import PiezoDataset, create_or_load_splits, graph_cache_key, load_gmtnet_records
from .model import AtomCoordinateResponsePotential, model_from_config
from .project_config import load_project_config
from .train import device_from_config, seed_everything


VOIGT_CHANNELS = ("xx", "yy", "zz", "yz", "xz", "xy")


@dataclass(frozen=True)
class LossWeights:
    full: float = 1.0
    full_direction: float = 2.0
    response: float = 1.0
    response_direction: float = 0.25


def _strict_ids(completion_dir: Path) -> set[str]:
    """Return only schema-validated, accepted completion records."""
    accepted: set[str] = set()
    for path in completion_dir.glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema") == 2
            and payload.get("jid") == path.stem
            and bool(payload.get("audit", {}).get("accepted", False))
        ):
            accepted.add(path.stem)
    if not accepted:
        raise ValueError(f"No accepted strict Lambda completions in {completion_dir}")
    return accepted


def _coupling(response: AtomCoordinateResponsePotential, value: torch.Tensor) -> torch.Tensor:
    """Unique engineering-Voigt strain components, [atom, displacement, 6]."""
    return response._coupling_voigt(value)


def _ionic_from_true_factors(batch, coupling: torch.Tensor, response) -> torch.Tensor:
    """Evaluate Z*^T D_delta(Phi) Lambda using only DFPT Z* and Phi."""
    values, offset = [], 0
    cells = batch.cell.reshape(-1, 3, 3)
    for graph_index in range(batch.ptr.numel() - 1):
        start, stop = int(batch.ptr[graph_index]), int(batch.ptr[graph_index + 1])
        atoms = stop - start
        count = 9 * atoms * atoms
        blocks = batch.dfpt_force_constants_flat[offset : offset + count].reshape(atoms, atoms, 3, 3)
        offset += count
        charge = batch.y_born[start:stop].reshape(3 * atoms, 3)
        volume = torch.linalg.det(cells[graph_index]).abs().clamp_min(torch.finfo(cells.dtype).eps)
        relaxed = response.apply_optical_operator(
            blocks, coupling[start:stop].reshape(3 * atoms, 6), solve_policy="regularized"
        )
        values.append(response.PIEZO_C_PER_M2 * (charge.T @ relaxed) / volume)
    if offset != batch.dfpt_force_constants_flat.numel():
        raise ValueError("Ragged force-constant targets do not match graph pointers")
    return torch.stack(values)


def _response_context(batch, response) -> list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Precompute immutable DFPT response factors for a fixed diagnostic batch."""
    entries, offset = [], 0
    cells = batch.cell.reshape(-1, 3, 3)
    for graph_index in range(batch.ptr.numel() - 1):
        start, stop = int(batch.ptr[graph_index]), int(batch.ptr[graph_index + 1])
        atoms = stop - start
        count = 9 * atoms * atoms
        blocks = batch.dfpt_force_constants_flat[offset : offset + count].reshape(atoms, atoms, 3, 3)
        offset += count
        entries.append((
            start, stop,
            blocks,
            batch.y_born[start:stop].reshape(3 * atoms, 3),
            torch.linalg.det(cells[graph_index]).abs().clamp_min(torch.finfo(cells.dtype).eps),
        ))
    if offset != batch.dfpt_force_constants_flat.numel():
        raise ValueError("Ragged force-constant targets do not match graph pointers")
    return entries


def _ionic_from_context(coupling: torch.Tensor, response, context) -> torch.Tensor:
    return torch.stack([
        response.PIEZO_C_PER_M2 * (
            charge.T @ response.apply_optical_operator(
                blocks, coupling[start:stop].reshape(3 * (stop - start), 6), solve_policy="regularized"
            )
        ) / volume
        for start, stop, blocks, charge, volume in context
    ])


def _per_graph_cosine(prediction: torch.Tensor, target: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    values = []
    for index in range(ptr.numel() - 1):
        start, stop = int(ptr[index]), int(ptr[index + 1])
        predicted = prediction[start:stop].reshape(-1)
        expected = target[start:stop].reshape(-1)
        values.append(torch.dot(predicted, expected) / (torch.linalg.vector_norm(predicted) * torch.linalg.vector_norm(expected)).clamp_min(1e-12))
    return torch.stack(values)


def _lambda_objective(prediction: torch.Tensor, target: torch.Tensor, batch, response, weights: LossWeights, response_context=None, target_response=None) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = _coupling(response, prediction)
    target = _coupling(response, target)
    scale = target.abs().mean().clamp_min(1e-5)
    full = F.smooth_l1_loss(predicted / scale, target / scale)
    full_direction = 1.0 - _per_graph_cosine(predicted, target, batch.ptr).mean()
    if response_context is None:
        response_context = _response_context(batch, response)
    predicted_response = _ionic_from_context(predicted, response, response_context)
    if target_response is None:
        target_response = _ionic_from_context(target, response, response_context)
    response_scale = target_response.abs().mean(dim=(1, 2)).clamp_min(0.05).view(-1, 1, 1)
    response_loss = F.smooth_l1_loss(predicted_response / response_scale, target_response / response_scale)
    response_cosines = F.cosine_similarity(
        predicted_response.reshape(predicted_response.shape[0], -1),
        target_response.reshape(target_response.shape[0], -1), dim=-1, eps=1e-12,
    )
    # Direction is meaningful only above the same 0.05 C/m^2 resolution floor
    # used by production ionic supervision; do not let cancellation noise set it.
    active = torch.linalg.vector_norm(target_response.reshape(target_response.shape[0], -1), dim=-1) >= 0.05
    response_direction = 1.0 - response_cosines[active].mean() if active.any() else response_loss * 0.0
    loss = (
        weights.full * full
        + weights.full_direction * full_direction
        + weights.response * response_loss
        + weights.response_direction * response_direction
    )
    return loss, {
        "full": float(full.detach()), "full_direction": float(full_direction.detach()),
        "response": float(response_loss.detach()), "response_direction": float(response_direction.detach()),
    }


def _metric(prediction: torch.Tensor, target: torch.Tensor, floor: float) -> dict[str, float]:
    prediction = prediction.detach().to(torch.float64).reshape(-1)
    target = target.detach().to(torch.float64).reshape(-1)
    residual = prediction - target
    target_norm, prediction_norm = torch.linalg.vector_norm(target), torch.linalg.vector_norm(prediction)
    denominator = target_norm.clamp_min(floor * target.numel() ** 0.5)
    return {
        "mae": float(residual.abs().mean()),
        "rmse": float(residual.square().mean().sqrt()),
        "cosine": float(torch.dot(prediction, target) / (prediction_norm * target_norm).clamp_min(1e-12)),
        "amplitude_ratio": float(prediction_norm / denominator),
        "stabilized_relative_frobenius_error": float(torch.linalg.vector_norm(residual) / denominator),
    }


@torch.no_grad()
def _evaluate(model, dataset, device: torch.device, target_field: str) -> dict[str, object]:
    model.eval()
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch = next(iter(loader)).to(device)
    prediction = model.predict_factors(batch).internal_strain
    target = getattr(batch, target_field)
    predicted_coupling, target_coupling = _coupling(model.response, prediction), _coupling(model.response, target)
    predicted_response = _ionic_from_true_factors(batch, predicted_coupling, model.response)
    target_response = _ionic_from_true_factors(batch, target_coupling, model.response)
    channels = {}
    for channel, index in zip(VOIGT_CHANNELS, range(6)):
        channels[channel] = {
            "lambda": _metric(predicted_coupling[..., index], target_coupling[..., index], 0.01),
            "oracle_ionic_response": _metric(predicted_response[..., index], target_response[..., index], 0.05),
            "components": int(target_coupling[..., index].numel()),
        }
    return {
        "materials": len(dataset),
        "lambda": _metric(predicted_coupling, target_coupling, 0.01),
        "oracle_ionic_response": _metric(predicted_response, target_response, 0.05),
        "per_strain_channel": channels,
    }


def _train(model, dataset, target_field: str, device: torch.device, steps: int, learning_rate: float, seed: int, weights: LossWeights) -> tuple[list[dict[str, float]], dict[str, object]]:
    if steps < 1:
        raise ValueError("Diagnostic training steps must be positive")
    seed_everything(seed)
    model.to(device)
    model.train()
    # No augmentation, dropout, validation selection, or macroscopic-response loss.
    # This is intentionally one fixed full batch: the diagnostic has no data
    # augmentation and measures exact memorization, so repeatedly collating
    # identical variable-size PBC graphs only obscures the optimization time.
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False))).to(device)
    target = getattr(batch, target_field)
    response_context = _response_context(batch, model.response)
    target_response = _ionic_from_context(_coupling(model.response, target), model.response, response_context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        factors = model.predict_factors(batch)
        loss, components = _lambda_objective(
            factors.internal_strain, target, batch, model.response, weights,
            response_context=response_context, target_response=target_response,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Lambda diagnostic loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            history.append({"step": step, "loss": float(loss.detach()), **components})
    return history, _evaluate(model, dataset, device, target_field)


def _attach_synthetic_targets(dataset, teacher, device: torch.device) -> None:
    """Attach frozen teacher Lambda tensors to graph objects for reproducible batching."""
    teacher.eval().to(device)
    for index in range(len(dataset)):
        graph = dataset[index]
        batch = DataLoader([graph], batch_size=1)
        graph_batch = next(iter(batch)).to(device)
        graph.synthetic_internal_strain = teacher.predict_factors(graph_batch).internal_strain.detach().cpu()


def _dataset(records, ids, cfg):
    return PiezoDataset(
        records, ids, float(cfg["cutoff"]), int(cfg["max_neighbors"]),
        processed_dir=cfg["processed_dir"], cache_key=graph_cache_key(records, cfg["cutoff"], cfg["max_neighbors"]),
        dfpt_dir=cfg.get("jarvis_dfpt_dir"), strain_completion_dir=cfg.get("jarvis_strain_completion_dir"),
    )


def _markdown(report: dict[str, object]) -> str:
    memory = report["memorization"]["metrics"]
    synthetic = report["synthetic_recovery"]["metrics"]
    lines = [
        "# Strict-completion Lambda diagnostics", "",
        "All runs use true DFPT Born charges and force constants in the response-active objective. "
        "They have no augmentation, no validation early stopping, and no total-piezo loss.", "",
        "## Memorization", "",
        f"- Materials: {memory['materials']}",
        f"- Full-Lambda cosine: {memory['lambda']['cosine']:.4f}; MAE: {memory['lambda']['mae']:.6g} eV/Angstrom",
        f"- Oracle ionic cosine: {memory['oracle_ionic_response']['cosine']:.4f}; amplitude ratio: {memory['oracle_ionic_response']['amplitude_ratio']:.4f}", "",
        "## Synthetic recoverability", "",
        f"- Full-Lambda cosine: {synthetic['lambda']['cosine']:.4f}",
        f"- Oracle ionic cosine: {synthetic['oracle_ionic_response']['cosine']:.4f}", "",
        "## Nested learning curve (fixed four-material global holdout)", "",
        "| Complete Lambda train materials | Held-out Lambda cosine | Held-out oracle ionic cosine |",
        "| ---: | ---: | ---: |",
    ]
    for row in report["learning_curve"]:
        metrics = row["heldout_metrics"]
        lines.append(f"| {row['train_materials']} | {metrics['lambda']['cosine']:.4f} | {metrics['oracle_ionic_response']['cosine']:.4f} |")
    lines.extend(["", "## Per-strain-channel (memorization)", "", "| Channel | Lambda cosine | Oracle ionic cosine |", "| --- | ---: | ---: |"])
    for channel in VOIGT_CHANNELS:
        values = memory["per_strain_channel"][channel]
        lines.append(f"| {channel} | {values['lambda']['cosine']:.4f} | {values['oracle_ionic_response']['cosine']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lambda_diagnostics_v1"))
    parser.add_argument("--memory-steps", type=int, default=600)
    parser.add_argument("--synthetic-steps", type=int, default=600)
    parser.add_argument("--curve-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument(
        "--curve-only", action="store_true",
        help="Run only a matched-initialization 5/10/15/19 learning curve and write learning_curve.json.",
    )
    args = parser.parse_args()
    if min(args.memory_steps, args.synthetic_steps, args.curve_steps) < 1 or args.learning_rate <= 0:
        raise ValueError("Steps and learning rate must be positive")
    cfg = load_project_config(args.config)
    completion_dir = Path(cfg["jarvis_strain_completion_dir"])
    accepted = _strict_ids(completion_dir)
    records = load_gmtnet_records(cfg["data_root"])
    splits = create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
    train_ids = [jid for jid in splits["train"] if jid in accepted]
    heldout_ids = [jid for name in ("val", "test") for jid in splits[name] if jid in accepted]
    if len(train_ids) < 19 or len(heldout_ids) < 1:
        raise ValueError(f"Expected strict train/holdout data, got {len(train_ids)} train and {len(heldout_ids)} holdout")
    # Stable, published nested subsets rather than a seed-dependent filesystem order.
    ranked = sorted(train_ids, key=lambda jid: hashlib.sha256(f"{args.seed}:{jid}".encode()).hexdigest())
    train_set = _dataset(records, ranked, cfg)
    heldout_set = _dataset(records, heldout_ids, cfg)
    if not all(bool(train_set[index].internal_strain_full_mask) for index in range(len(train_set))):
        raise RuntimeError("A selected strict-training graph lacks full Lambda")
    device = device_from_config(str(cfg.get("device", "auto")))
    args.output.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(cfg)
    resolved.update({"diagnostic_seed": args.seed, "device": str(device), "strict_train_ids": ranked, "strict_holdout_ids": heldout_ids})
    (args.output / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    weights = LossWeights()

    def matched_curve() -> list[dict[str, object]]:
        entries = []
        # Same initialization is essential: otherwise four random starts are
        # confounded with the number of complete Lambda labels.  Full batches
        # and the fixed seed also make optimization order identical.
        curve_seed = args.seed + 1000
        for size in (5, 10, 15, len(ranked)):
            subset_ids = ranked[:size]
            subset = _dataset(records, subset_ids, cfg)
            seed_everything(curve_seed)
            model = model_from_config(cfg)
            history, train_metrics = _train(
                model, subset, "dfpt_internal_strain_full", device,
                args.curve_steps, args.learning_rate, curve_seed, weights,
            )
            entries.append({
                "train_materials": size, "train_ids": subset_ids, "history": history,
                "train_metrics": train_metrics,
                "heldout_metrics": _evaluate(model, heldout_set, device, "dfpt_internal_strain_full"),
            })
        return entries

    if args.curve_only:
        curve = matched_curve()
        partial = {
            "schema": 1, "voigt_order": list(VOIGT_CHANNELS),
            "strict_completion": {"accepted_total": len(accepted), "train": len(train_ids), "heldout": len(heldout_ids)},
            "protocol": {"matched_initialization_seed": args.seed + 1000, "curve_steps": args.curve_steps, "loss_weights": weights.__dict__},
            "learning_curve": curve,
        }
        (args.output / "learning_curve.json").write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
        print(args.output / "learning_curve.json")
        return

    seed_everything(args.seed)
    memory_model = model_from_config(cfg)
    memory_history, memory_metrics = _train(memory_model, train_set, "dfpt_internal_strain_full", device, args.memory_steps, args.learning_rate, args.seed, weights)

    seed_everything(args.seed + 1)
    teacher = model_from_config(cfg)
    _attach_synthetic_targets(train_set, teacher, device)
    seed_everything(args.seed + 2)
    student = model_from_config(cfg)
    synthetic_history, synthetic_metrics = _train(student, train_set, "synthetic_internal_strain", device, args.synthetic_steps, args.learning_rate, args.seed + 2, weights)

    curve = matched_curve()

    report = {
        "schema": 1,
        "voigt_order": list(VOIGT_CHANNELS),
        "strict_completion": {"accepted_total": len(accepted), "train": len(train_ids), "heldout": len(heldout_ids)},
        "protocol": {
            "target": "strict symmetry-completed internal strain", "true_factors": "DFPT Z* and Phi",
            "optical_operator": "continuous signed regularized", "loss_weights": weights.__dict__,
            "no_augmentation": True, "no_validation_early_stopping": True, "no_total_piezo_loss": True,
        },
        "memorization": {"history": memory_history, "metrics": memory_metrics},
        "synthetic_recovery": {"history": synthetic_history, "metrics": synthetic_metrics},
        "learning_curve": curve,
    }
    (args.output / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "diagnostic.md").write_text(_markdown(report), encoding="utf-8")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = ([{"experiment": "memorization", **row} for row in memory_history]
                + [{"experiment": "synthetic", **row} for row in synthetic_history]
                + [{"experiment": f"curve_{entry['train_materials']}", **row} for entry in curve for row in entry["history"]])
        writer = csv.DictWriter(handle, fieldnames=("experiment", "step", "loss", "full", "full_direction", "response", "response_direction"))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output / "diagnostic.md")


if __name__ == "__main__":
    main()
