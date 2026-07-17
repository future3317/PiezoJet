"""Evaluate the matched direct baseline once on a frozen explicit split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .baselines import direct_cartesian_baseline_from_config, e3nn_direct_baseline_from_config
from .checkpoint_provenance import build_checkpoint_provenance, validate_checkpoint_provenance
from .data import PiezoDataset, graph_cache_key, load_gmtnet_records
from .metrics import material_bootstrap_confidence_interval, response_tensor_skill, tensor_metrics
from .train import device_from_config, load_explicit_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20270715)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    family = str(checkpoint.get("model_family", ""))
    builders = {
        "matched_direct_cartesian_piezo": direct_cartesian_baseline_from_config,
        "matched_direct_e3nn_piezo": e3nn_direct_baseline_from_config,
    }
    if family not in builders:
        raise ValueError("Checkpoint is not a supported matched direct baseline")
    cfg = checkpoint["config"]
    device = device_from_config(args.device)
    records = load_gmtnet_records(cfg["data_root"])
    splits = load_explicit_splits(args.splits_file, {str(record["JARVIS_ID"]) for record in records})
    expected_provenance = build_checkpoint_provenance(
        splits,
        args.splits_file,
        cfg,
        split_kind=str(checkpoint.get("checkpoint_provenance", {}).get("split_kind", "")),
    )
    validate_checkpoint_provenance(checkpoint, expected_provenance)
    cache_key = graph_cache_key(records, float(cfg["cutoff"]), int(cfg["max_neighbors"]))
    dataset = PiezoDataset(
        records, splits[args.split], float(cfg["cutoff"]), int(cfg["max_neighbors"]),
        processed_dir=cfg["processed_dir"], cache_key=cache_key,
    )
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    model = builders[family](cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu())
            targets.append(batch.y.cpu())
    prediction, target = torch.cat(predictions), torch.cat(targets)
    prediction = 0.5 * (prediction + prediction.transpose(-1, -2))
    target = 0.5 * (target + target.transpose(-1, -2))
    scale = float(checkpoint["piezo_scale"])
    payload = {
        "schema": 2,
        "model_family": family,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "formula_disjoint": True,
        "material_count": len(dataset),
        "material_ids": [str(record["JARVIS_ID"]) for record in dataset.records],
        "response_tensor_skill": response_tensor_skill(prediction, target),
        "response_tensor_skill_bootstrap_95": material_bootstrap_confidence_interval(
            list(prediction),
            list(target),
            lambda values, labels: response_tensor_skill(
                torch.stack(values), torch.stack(labels)
            )["tensor_response_skill_vs_zero"],
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        ),
        "tensor_metrics": tensor_metrics(prediction, target, scale * 0.05),
        "frozen_test_used_for_selection": False,
        "resampling_material_rows": [
            {
                "material_id": str(dataset.records[index]["JARVIS_ID"]),
                "total_prediction": prediction[index].reshape(-1).tolist(),
                "total_target": target[index].reshape(-1).tolist(),
            }
            for index in range(len(dataset))
        ],
        "resampling_contract": (
            "complete materials are the resampling unit; rows support paired "
            "hierarchical seed/material intervals only"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
