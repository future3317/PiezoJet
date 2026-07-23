"""Development-only amplitude versus shape audit for the electronic tower.

This module deliberately does not alter a model or a training objective.  It
loads one selected, development-trained electronic tower and separates three
questions on a small held-out *development* slice:

* how much error is explained by a single scalar amplitude mismatch;
* whether independent per-irrep scalar calibration removes the residual; and
* what an oracle per-material norm rescaling would leave as a pure shape error.

The calibration slice is disjoint from the reported audit slice.  The oracle
quantity is explicitly diagnostic and must not be used for checkpoint
selection or production inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .data import deterministic_subset, graph_cache_key, load_gmtnet_records
from .electronic_capacity import electronic_capacity_metrics
from .electrostatic_fold_adjudication import (
    _dataset,
    make_model,
)
from .electrostatic_a0_fold_adjudication import _prediction, _tower
from .electrostatic_protocol import A0_ARCHITECTURES
from .project_config import load_project_config
from .tensor_ops import PIEZO_IRREP_SLICES, piezo_from_irreps, piezo_to_irreps
from .train import _data_commit, _git_commit


def fit_scale(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Least-squares scalar mapping prediction -> target, without clipping."""
    prediction = prediction.to(torch.float64)
    target = target.to(torch.float64)
    denominator = prediction.square().sum()
    if float(denominator) <= 1e-30:
        return prediction.new_zeros(())
    return (prediction * target).sum() / denominator


def fit_irrep_scales(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Fit one least-squares scalar for each orthonormal piezo irrep block."""
    predicted = piezo_to_irreps(prediction).to(torch.float64)
    expected = piezo_to_irreps(target).to(torch.float64)
    return {
        name: fit_scale(predicted[..., block], expected[..., block])
        for name, block in PIEZO_IRREP_SLICES.items()
    }


def apply_irrep_scales(
    prediction: torch.Tensor, scales: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Apply irrep scales in orthonormal coordinates and return Cartesian form."""
    coordinates = piezo_to_irreps(prediction).to(torch.float64).clone()
    for name, block in PIEZO_IRREP_SLICES.items():
        coordinates[..., block] *= scales[name]
    # The inverse conversion is intentionally imported lazily to keep this
    # module's pure diagnostic helpers cheap to import in tests.
    return piezo_from_irreps(coordinates)


def oracle_shape_prediction(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Rescale each prediction to the target norm (diagnostic upper bound)."""
    predicted = piezo_to_irreps(prediction).to(torch.float64)
    expected = piezo_to_irreps(target).to(torch.float64)
    predicted_norm = torch.linalg.vector_norm(predicted, dim=-1, keepdim=True)
    target_norm = torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
    normalized = predicted / predicted_norm.clamp_min(1e-30)
    return piezo_from_irreps(normalized * target_norm)


def _collect_predictions(tower, dataset, device, batch_size: int):
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    tower.eval().to(device)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(_prediction(tower, batch, "electronic").cpu())
            targets.append(batch.y_electronic_piezo.cpu())
    return torch.cat(predictions), torch.cat(targets)


def _metric_summary(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    metrics = electronic_capacity_metrics(prediction, target)
    return {
        "materials": metrics["materials"],
        "stabilized_relative_frobenius_error": metrics[
            "mean_stabilized_relative_frobenius_error"
        ],
        "active_relative_frobenius_error": metrics[
            "mean_active_relative_frobenius_error"
        ],
        "active_cosine": metrics["mean_active_cosine"],
        "active_amplitude_ratio": metrics["mean_active_amplitude_ratio"],
        "stabilized_amplitude_ratio": metrics["mean_stabilized_amplitude_ratio"],
        "per_irrep": metrics["per_irrep"],
    }


def run_diagnostic(
    prediction: torch.Tensor,
    target: torch.Tensor,
    calibration_count: int,
) -> dict[str, object]:
    """Run leakage-safe scalar, irrep, and oracle-shape comparisons."""
    if prediction.shape[0] != target.shape[0] or calibration_count <= 0:
        raise ValueError("Prediction/target rows and calibration_count are invalid")
    if calibration_count >= prediction.shape[0]:
        raise ValueError("Calibration slice must leave a non-empty audit slice")
    calibration_prediction = prediction[:calibration_count]
    calibration_target = target[:calibration_count]
    audit_prediction = prediction[calibration_count:]
    audit_target = target[calibration_count:]
    global_scale = fit_scale(
        piezo_to_irreps(calibration_prediction), piezo_to_irreps(calibration_target)
    )
    irrep_scales = fit_irrep_scales(calibration_prediction, calibration_target)
    baseline = _metric_summary(audit_prediction, audit_target)
    global_scaled = _metric_summary(audit_prediction * global_scale, audit_target)
    irrep_scaled = _metric_summary(
        apply_irrep_scales(audit_prediction, irrep_scales), audit_target
    )
    oracle = _metric_summary(
        oracle_shape_prediction(audit_prediction, audit_target), audit_target
    )
    return {
        "calibration_materials": calibration_count,
        "audit_materials": int(prediction.shape[0] - calibration_count),
        "calibration_global_scale": float(global_scale),
        "calibration_irrep_scales": {
            name: float(value) for name, value in irrep_scales.items()
        },
        "audit_baseline": baseline,
        "audit_global_scalar_calibration": global_scaled,
        "audit_irrep_scalar_calibration": irrep_scaled,
        "audit_oracle_per_material_norm": oracle,
        "interpretation": {
            "global_scalar": "deployable post-hoc amplitude calibration diagnostic; fit only on calibration slice",
            "irrep_scalar": "shape-preserving per-irrep amplitude diagnostic; fit only on calibration slice",
            "oracle_per_material_norm": "non-deployable shape-only upper bound; uses audit target norms",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--folds", type=Path,
        default=Path("data/processed/electrostatic_development_folds_v2.json"),
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=A0_ARCHITECTURES, default="a0_parameter_matched_irreps")
    parser.add_argument("--audit-materials", type=int, default=128)
    parser.add_argument("--calibration-materials", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph-cache-key")
    args = parser.parse_args()
    if args.architecture != "a0_parameter_matched_irreps":
        raise ValueError("The diagnostic currently requires the selected A0-PM tower")
    if args.audit_materials < 2 or not 0 < args.calibration_materials < args.audit_materials:
        raise ValueError("Require 2 <= audit-materials and 0 < calibration-materials < audit-materials")

    config = load_project_config(args.config)
    config["data_commit"] = _data_commit(config["data_root"])
    folds = json.loads(args.folds.read_text(encoding="utf-8-sig"))
    fold = next((value for value in folds["folds"] if value["fold"] == args.fold), None)
    if fold is None:
        raise ValueError(f"Fold {args.fold} is absent from {args.folds}")
    all_ids = deterministic_subset(list(fold["development"]), args.audit_materials, args.seed + 2000)
    if len(all_ids) != args.audit_materials:
        raise ValueError("Requested diagnostic slice exceeds development-fold size")
    records = load_gmtnet_records(config["data_root"])
    cache_key = args.graph_cache_key or graph_cache_key(
        records, float(config["cutoff"]), int(config["max_neighbors"])
    )
    cache_manifest = Path(config["processed_dir"]) / "pbc_graph_cache" / cache_key / "manifest.json"
    if not cache_manifest.is_file():
        raise FileNotFoundError(f"Graph cache manifest is absent: {cache_manifest}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != args.architecture or int(payload.get("fold", -1)) != args.fold:
        raise ValueError("Selected checkpoint architecture/fold does not match diagnostic")
    control = make_model(args.architecture, config)
    _tower(control, "electronic").load_state_dict(payload["model"]["electronic"], strict=True)
    dataset = _dataset(config, records, all_ids, cache_key, cache_graphs=False)
    prediction, target = _collect_predictions(
        _tower(control, "electronic"), dataset, torch.device(args.device), args.eval_batch_size
    )
    result = run_diagnostic(prediction, target, args.calibration_materials)
    result["schema"] = 1
    result["protocol"] = "development-only electronic amplitude/scale-shape diagnostic"
    result["architecture"] = args.architecture
    result["fold"] = args.fold
    result["seed"] = args.seed
    result["material_ids"] = all_ids
    result["calibration_ids"] = all_ids[:args.calibration_materials]
    result["audit_ids"] = all_ids[args.calibration_materials:]
    result["checkpoint"] = str(args.checkpoint.resolve())
    result["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result["data_manifest"] = str(args.folds.resolve())
    result["graph_cache_key"] = cache_key
    result["frozen_validation_test_labels_read"] = False
    result["code_commit"] = _git_commit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {"prediction": prediction, "target": target, "material_ids": all_ids},
        args.output.with_suffix(".pt"),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
