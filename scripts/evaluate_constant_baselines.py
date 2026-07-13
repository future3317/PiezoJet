"""Evaluate zero and train-mean piezo baselines on the fixed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from piezojet.data import PiezoDataset, create_or_load_splits, graph_cache_key, load_gmtnet_records
from piezojet.metrics import response_tensor_skill


def _metrics(prediction: torch.Tensor, target: torch.Tensor, floor: float) -> dict[str, float | int | dict[str, float | int]]:
    difference = prediction - target
    sample_error = torch.linalg.vector_norm(difference.reshape(difference.shape[0], -1), dim=-1)
    target_norm = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=-1)
    return {
        "cartesian_component_mae": float(difference.abs().mean()),
        "sample_frobenius_mae": float(sample_error.mean()),
        "normalized_frobenius_error": float((sample_error / target_norm.clamp_min(floor)).mean()),
        "response_tensor_metrics": response_tensor_skill(prediction, target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records = load_gmtnet_records(config["data_root"])
    splits = create_or_load_splits(records, config["processed_dir"], int(config["seed"]))
    cache_key = graph_cache_key(records, config["cutoff"], config["max_neighbors"])
    train_set = PiezoDataset(records, splits["train"], config["cutoff"], config["max_neighbors"], processed_dir=config["processed_dir"], cache_key=cache_key)
    test_set = PiezoDataset(records, splits["test"], config["cutoff"], config["max_neighbors"], processed_dir=config["processed_dir"], cache_key=cache_key)
    train = torch.cat([train_set[index].y for index in range(len(train_set))])
    test = torch.cat([test_set[index].y for index in range(len(test_set))])
    train_voigt = torch.cat([train_set[index].y_voigt for index in range(len(train_set))])
    scale = torch.sqrt(train_voigt.square().mean())
    floor = float(scale * 0.05 * (18.0 ** 0.5))
    result = {
        "split_sizes": {name: len(ids) for name, ids in splits.items()},
        "equivariance_norm_floor": floor,
        "zero": _metrics(torch.zeros_like(test), test, floor),
        "train_mean": _metrics(train.mean(dim=0, keepdim=True).expand_as(test), test, floor),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
