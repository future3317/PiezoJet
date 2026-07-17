"""Numerically adjudicate matched-direct versus PiezoJet macro training.

This is an E0 protocol control, not a performance experiment.  The direct
model receives an exact copy of the PiezoJet macro tower, optimizer state, and
every minibatch.  Any trajectory difference therefore indicates an
implementation mismatch that must be fixed before comparing validation TRS.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .baselines import direct_cartesian_baseline_from_config
from .data import PiezoDataset, graph_cache_key, load_gmtnet_records
from .model import model_from_config
from .project_config import load_project_config
from .tensor_ops import piezo_scale
from .train import (
    full_loss,
    load_explicit_splits,
    response_bin_weights,
    seed_everything,
)


def _aligned_parameters(piezojet, direct):
    macro = list(piezojet.macro_encoder.parameters()) + list(
        piezojet.macro_total_head.parameters()
    )
    control = list(direct.encoder.parameters()) + list(direct.head.parameters())
    if len(macro) != len(control):
        raise RuntimeError("Macro/direct parameter layouts differ")
    return list(zip(macro, control))


def _max_parameter_difference(pairs) -> float:
    return max(
        (float((left.detach() - right.detach()).abs().max()) for left, right in pairs),
        default=0.0,
    )


def _max_gradient_difference(pairs) -> float:
    differences: list[float] = []
    for left, right in pairs:
        if (left.grad is None) != (right.grad is None):
            return float("inf")
        if left.grad is not None:
            differences.append(float((left.grad - right.grad).abs().max()))
    return max(differences, default=0.0)


def _optimizer_tensor_difference(left, right) -> float:
    left_state, right_state = left.state_dict(), right.state_dict()
    if left_state["param_groups"] != right_state["param_groups"]:
        return float("inf")
    differences: list[float] = []
    for key in left_state["state"]:
        if key not in right_state["state"]:
            return float("inf")
        for name, value in left_state["state"][key].items():
            other = right_state["state"][key][name]
            if torch.is_tensor(value):
                differences.append(float((value - other).abs().max()))
            elif value != other:
                return float("inf")
    return max(differences, default=0.0)


def run_exact_clone_control(
    config: dict[str, object],
    splits_file: Path,
    *,
    updates: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    if updates < 1 or batch_size < 1:
        raise ValueError("updates and batch_size must be positive")
    seed = int(config["seed"])
    seed_everything(seed)
    records = load_gmtnet_records(config["data_root"])
    splits = load_explicit_splits(
        splits_file, {str(record["JARVIS_ID"]) for record in records}
    )
    cache_key = graph_cache_key(
        records, float(config["cutoff"]), int(config["max_neighbors"])
    )
    dataset = PiezoDataset(
        records,
        splits["train"],
        float(config["cutoff"]),
        int(config["max_neighbors"]),
        processed_dir=config["processed_dir"],
        cache_key=cache_key,
        project_targets=True,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    scale = piezo_scale(
        torch.cat([dataset[index].y_voigt for index in range(len(dataset))])
    ).to(device)
    bins = response_bin_weights(
        torch.stack([dataset[index].y.squeeze(0) for index in range(len(dataset))])
    ).to(device)

    piezojet = model_from_config(config).to(device)
    direct = direct_cartesian_baseline_from_config(config).to(device)
    pretrained = torch.load(
        Path(str(config["pretrained_encoder"])), map_location=device, weights_only=False
    )
    piezojet.macro_encoder.load_state_dict(pretrained["encoder"], strict=True)
    direct.encoder.load_state_dict(pretrained["encoder"], strict=True)
    direct.head.load_state_dict(
        copy.deepcopy(piezojet.macro_total_head.state_dict()), strict=True
    )
    pairs = _aligned_parameters(piezojet, direct)
    initial_parameter_difference = _max_parameter_difference(pairs)

    macro_optimizer = torch.optim.AdamW(
        [left for left, _ in pairs],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    direct_optimizer = torch.optim.AdamW(
        [right for _, right in pairs],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    direct_optimizer.load_state_dict(copy.deepcopy(macro_optimizer.state_dict()))

    piezojet.macro_encoder.train()
    piezojet.macro_total_head.train()
    direct.train()
    rows: list[dict[str, float | int]] = []
    iterator = iter(loader)
    for update in range(1, updates + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device, non_blocking=device.type == "cuda")
        macro_optimizer.zero_grad(set_to_none=True)
        direct_optimizer.zero_grad(set_to_none=True)
        macro_prediction = piezojet.predict_macro_total(batch)
        direct_prediction = direct(batch)
        macro_loss = full_loss(macro_prediction, batch.y, scale, bins)
        direct_loss = full_loss(direct_prediction, batch.y, scale, bins)
        macro_loss.backward()
        direct_loss.backward()
        row: dict[str, float | int] = {
            "update": update,
            "prediction_max_abs_difference": float(
                (macro_prediction - direct_prediction).detach().abs().max()
            ),
            "loss_abs_difference": float((macro_loss - direct_loss).detach().abs()),
            "gradient_max_abs_difference": _max_gradient_difference(pairs),
        }
        macro_optimizer.step()
        direct_optimizer.step()
        row["parameter_max_abs_difference"] = _max_parameter_difference(pairs)
        row["optimizer_state_max_abs_difference"] = _optimizer_tensor_difference(
            macro_optimizer, direct_optimizer
        )
        rows.append(row)

    maxima = {
        key: max(float(row[key]) for row in rows)
        for key in (
            "prediction_max_abs_difference",
            "loss_abs_difference",
            "gradient_max_abs_difference",
            "parameter_max_abs_difference",
            "optimizer_state_max_abs_difference",
        )
    }
    tolerance = 1e-7
    return {
        "schema": 1,
        "protocol": "E0_exact_clone_macro_direct",
        "selection": "macro train IDs only; frozen validation/test not read",
        "seed": seed,
        "updates": updates,
        "batch_size": batch_size,
        "runtime_device": str(device),
        "initial_parameter_max_abs_difference": initial_parameter_difference,
        "numerical_tolerance": tolerance,
        "maxima": maxima,
        "passed": initial_parameter_difference <= tolerance
        and all(value <= tolerance for value in maxima.values()),
        "updates_detail": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_project_config(args.config)
    result = run_exact_clone_control(
        config,
        args.splits_file,
        updates=args.updates,
        batch_size=args.batch_size,
        device=torch.device(args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["maxima"], indent=2))
    if not result["passed"]:
        raise SystemExit("E0 exact-clone control failed")


if __name__ == "__main__":
    main()
