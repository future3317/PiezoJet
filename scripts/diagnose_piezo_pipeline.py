"""End-to-end, split-safe diagnosis for the PiezoJet piezoelectric pipeline.

This is deliberately an evaluation utility, not a training mode.  It audits
raw labels, fixed-split balance, cached PBC graph health, a checkpoint's
per-signal behavior, and a stratified crystallographic-symmetry sample.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.loader import DataLoader

from piezojet.data import PiezoDataset, create_or_load_splits, formula, graph_cache_key, load_gmtnet_records
from piezojet.metrics import response_tensor_skill
from piezojet.model import model_from_config
from piezojet.project_config import load_project_config
from piezojet.tensor_ops import cartesian_to_piezo_voigt, piezo_voigt_to_cartesian, rotate_piezo, source_voigt_to_canonical
from piezojet.train import device_from_config


SIGNAL_BINS = (
    ("exact_zero", 0.0, 0.0),
    ("weak_0_0.05", 0.0, 0.05),
    ("moderate_0.05_0.5", 0.05, 0.5),
    ("high_0.5_1", 0.5, 1.0),
    ("very_high_ge_1", 1.0, float("inf")),
)


def _targets(records_by_id: dict[str, dict[str, Any]], ids: list[str]) -> torch.Tensor:
    return torch.stack([
        piezo_voigt_to_cartesian(source_voigt_to_canonical(torch.tensor(records_by_id[material_id]["piezoelectric_C_m2"], dtype=torch.float32)))
        for material_id in ids
    ])


def _summary(values: torch.Tensor) -> dict[str, float]:
    quantiles = torch.quantile(values, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]))
    return {
        "mean": float(values.mean()), "q00": float(quantiles[0]), "q25": float(quantiles[1]),
        "q50": float(quantiles[2]), "q75": float(quantiles[3]), "q90": float(quantiles[4]),
        "q99": float(quantiles[5]), "max": float(quantiles[6]),
    }


def _signal_mask(max_component: torch.Tensor, low: float, high: float, name: str) -> torch.Tensor:
    if name == "exact_zero":
        return max_component == 0
    if high == float("inf"):
        return max_component >= low
    return (max_component > low) & (max_component < high)


def _label_profile(target: torch.Tensor) -> dict[str, Any]:
    voigt = cartesian_to_piezo_voigt(target)
    max_component = voigt.abs().amax(dim=(-2, -1))
    cartesian_norm = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=-1)
    squared_norm = cartesian_norm.square()
    total_squared_norm = squared_norm.sum().clamp_min(torch.finfo(target.dtype).eps)
    top_1_percent = max(1, round(0.01 * target.shape[0]))
    return {
        "records": int(target.shape[0]),
        "zero_tensor_fraction": float((max_component == 0).float().mean()),
        "high_response_fraction_ge_0.5": float((max_component >= 0.5).float().mean()),
        "very_high_fraction_ge_1": float((max_component >= 1.0).float().mean()),
        "max_component_C_m2": _summary(max_component),
        "cartesian_frobenius_norm_C_m2": _summary(cartesian_norm),
        "cartesian_squared_target_mass_share": {
            "largest_1": float(torch.topk(squared_norm, 1).values.sum() / total_squared_norm),
            "largest_10": float(torch.topk(squared_norm, min(10, target.shape[0])).values.sum() / total_squared_norm),
            "largest_1_percent": float(torch.topk(squared_norm, top_1_percent).values.sum() / total_squared_norm),
        },
    }


def _prediction_profile(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    target_voigt, prediction_voigt = cartesian_to_piezo_voigt(target), cartesian_to_piezo_voigt(prediction)
    max_component = target_voigt.abs().amax(dim=(-2, -1))
    target_norm = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=-1)
    prediction_norm = torch.linalg.vector_norm(prediction.reshape(prediction.shape[0], -1), dim=-1)
    error_norm = torch.linalg.vector_norm((prediction - target).reshape(target.shape[0], -1), dim=-1)
    squared_error = (prediction - target).reshape(target.shape[0], -1).square().mean(dim=-1)
    denominator = (prediction_norm * target_norm).clamp_min(torch.finfo(target.dtype).eps)
    cosine = (prediction.reshape(target.shape[0], -1) * target.reshape(target.shape[0], -1)).sum(dim=-1) / denominator
    strata: dict[str, Any] = {}
    for name, low, high in SIGNAL_BINS:
        mask = _signal_mask(max_component, low, high, name)
        if not mask.any():
            strata[name] = {"count": 0}
            continue
        active = target_norm[mask] > 0
        strata[name] = {
            "count": int(mask.sum()),
            "fraction": float(mask.float().mean()),
            "target_norm_mean": float(target_norm[mask].mean()),
            "prediction_norm_mean": float(prediction_norm[mask].mean()),
            "amplitude_ratio_mean": float((prediction_norm[mask][active] / target_norm[mask][active]).mean()) if active.any() else None,
            "directional_cosine_mean": float(cosine[mask][active].mean()) if active.any() else None,
            "relative_frobenius_error_mean": float((error_norm[mask][active] / target_norm[mask][active]).mean()) if active.any() else None,
            "mse_loss_share": float(squared_error[mask].sum() / squared_error.sum().clamp_min(torch.finfo(target.dtype).eps)),
        }
    norm_centered = target_norm - target_norm.mean()
    pred_centered = prediction_norm - prediction_norm.mean()
    norm_correlation = (norm_centered * pred_centered).mean() / (norm_centered.square().mean().sqrt() * pred_centered.square().mean().sqrt()).clamp_min(torch.finfo(target.dtype).eps)
    return {
        "response_tensor_metrics": response_tensor_skill(prediction, target),
        "prediction_norm_C_m2": _summary(prediction_norm),
        "target_prediction_norm_pearson": float(norm_correlation),
        "predicted_high_response_fraction_ge_0.5": float((prediction_voigt.abs().amax(dim=(-2, -1)) >= 0.5).float().mean()),
        "signal_strata": strata,
    }


def _graph_profile(dataset: PiezoDataset) -> dict[str, Any]:
    atoms, edges, isolated_graphs, max_degrees, saturated_nodes, saturated_graphs = [], [], 0, [], 0, 0
    for index in range(len(dataset)):
        graph = dataset[index]
        degree = torch.bincount(graph.edge_index[1], minlength=graph.num_nodes)
        atoms.append(graph.num_nodes)
        edges.append(graph.edge_index.shape[1])
        isolated_graphs += int((degree == 0).any())
        max_degrees.append(int(degree.max()))
        saturated_nodes += int((degree >= dataset.max_neighbors).sum())
        saturated_graphs += int((degree >= dataset.max_neighbors).any())
    return {
        "graphs": len(dataset),
        "isolated_node_graphs": isolated_graphs,
        "neighbor_cap": dataset.max_neighbors,
        "nodes_at_neighbor_cap_fraction": saturated_nodes / max(sum(atoms), 1),
        "graphs_with_any_node_at_neighbor_cap_fraction": saturated_graphs / max(len(dataset), 1),
        "atoms_per_graph": _summary(torch.tensor(atoms, dtype=torch.float32)),
        "edges_per_graph": _summary(torch.tensor(edges, dtype=torch.float32)),
        "max_in_degree": _summary(torch.tensor(max_degrees, dtype=torch.float32)),
    }


def _symmetry_profile(records: list[dict[str, Any]], samples: int) -> dict[str, Any]:
    if samples <= 0:
        return {"samples": 0}
    targets = _targets({str(record["JARVIS_ID"]): record for record in records}, [str(record["JARVIS_ID"]) for record in records])
    signal = cartesian_to_piezo_voigt(targets).abs().amax(dim=(-2, -1))
    # Evenly spaced ranks cover the exact-zero peak and the long high-response tail.
    selected = torch.linspace(0, len(records) - 1, min(samples, len(records))).round().long()
    ranked = torch.argsort(signal)
    residuals, centro_norms, invalid, operations_counts, residual_rows = [], [], 0, [], []
    for index in ranked[selected].tolist():
        record, target = records[index], targets[index : index + 1]
        atoms = record["atoms"]
        try:
            structure = Structure(atoms["lattice_mat"], atoms["elements"], atoms["coords"], coords_are_cartesian=False)
            operations = SpacegroupAnalyzer(structure, symprec=1e-5).get_point_group_operations(cartesian=True)
            matrices = [torch.tensor(op.rotation_matrix, dtype=target.dtype) for op in operations]
            transformed = torch.cat([rotate_piezo(target, matrix) for matrix in matrices])
            denominator = torch.linalg.vector_norm(target.reshape(1, -1), dim=-1).clamp_min(0.5)
            residual = float((torch.linalg.vector_norm((transformed - target).reshape(len(operations), -1), dim=-1) / denominator).mean())
            residuals.append(residual)
            residual_rows.append({
                "material_id": str(record["JARVIS_ID"]), "point_group_label_residual": residual,
                "max_component_C_m2": float(signal[index]), "operation_count": len(operations),
            })
            operations_counts.append(len(operations))
            if any(torch.allclose(matrix, -torch.eye(3, dtype=target.dtype), atol=1e-5) for matrix in matrices):
                centro_norms.append(float(torch.linalg.vector_norm(target)))
        except Exception:
            invalid += 1
    result: dict[str, Any] = {
        "samples": len(residuals), "failed_symmetry_analysis": invalid,
        "mean_point_group_label_residual": float(torch.tensor(residuals).mean()) if residuals else None,
        "p95_point_group_label_residual": float(torch.quantile(torch.tensor(residuals), 0.95)) if residuals else None,
        "fraction_residual_gt_1e_4": float((torch.tensor(residuals) > 1e-4).float().mean()) if residuals else None,
        "point_group_operation_count": _summary(torch.tensor(operations_counts, dtype=torch.float32)) if operations_counts else None,
        "centrosymmetric_samples": len(centro_norms),
        "centrosymmetric_target_norm": _summary(torch.tensor(centro_norms)) if centro_norms else None,
        "largest_label_residuals": sorted(residual_rows, key=lambda row: row["point_group_label_residual"], reverse=True)[:10],
    }
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# PiezoJet data-to-model diagnostic", "", "This report is an evaluation-only audit of the persisted seed-42 split and the supplied checkpoint.", ""]
    lines += ["## Headline", "", f"- Test tensor-response skill vs zero: `{report['model']['test']['response_tensor_metrics']['tensor_response_skill_vs_zero']:.4f}`.", f"- Train tensor-response skill vs zero: `{report['model']['train']['response_tensor_metrics']['tensor_response_skill_vs_zero']:.4f}`.", f"- Cached graphs with isolated nodes: `{report['graphs']['all_records']['isolated_node_graphs']}` / `{report['graphs']['all_records']['graphs']}`.", f"- Stratified label point-group residual: `{report['symmetry'].get('mean_point_group_label_residual', 'not requested')}`.", ""]
    lines += ["## Signal-stratified model behavior", "", "| Split | Stratum | Count | Target norm | Predicted norm | Amplitude ratio | Relative error | Loss share |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("train", "val", "test"):
        for name, metrics in report["model"][split]["signal_strata"].items():
            if not metrics["count"]:
                continue
            ratio = "--" if metrics["amplitude_ratio_mean"] is None else f"{metrics['amplitude_ratio_mean']:.4f}"
            relative = "--" if metrics["relative_frobenius_error_mean"] is None else f"{metrics['relative_frobenius_error_mean']:.4f}"
            lines.append(f"| {split} | {name} | {metrics['count']} | {metrics['target_norm_mean']:.4f} | {metrics['prediction_norm_mean']:.4f} | {ratio} | {relative} | {metrics['mse_loss_share']:.2%} |")
    lines += ["", "## Interpretation boundary", "", "This audit identifies failure modes and data risks. It does not establish a causal fix; any change to the loss, architecture, or split must be evaluated as a new controlled experiment.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--symmetry-samples", type=int, default=128)
    args = parser.parse_args()
    cfg = load_project_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    records = load_gmtnet_records(cfg["data_root"])
    records_by_id = {str(record["JARVIS_ID"]): record for record in records}
    splits = create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
    cache_key = graph_cache_key(records, cfg["cutoff"], cfg["max_neighbors"])
    datasets = {name: PiezoDataset(records, ids, cfg["cutoff"], cfg["max_neighbors"], processed_dir=cfg["processed_dir"], cache_key=cache_key) for name, ids in splits.items()}
    # PiezoDataset preserves raw-record order rather than shuffled split-ID
    # order.  Build targets in the same order as inference batches.
    target_by_split = {name: torch.cat([dataset[index].y for index in range(len(dataset))]) for name, dataset in datasets.items()}
    device = device_from_config(cfg["device"])
    model = model_from_config(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    prediction_by_split: dict[str, torch.Tensor] = {}
    direct_by_split: dict[str, torch.Tensor] = {}
    operator_by_split: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, dataset in datasets.items():
            loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
            direct_batches, operator_batches, prediction_batches = [], [], []
            for batch in loader:
                batch = batch.to(device)
                features = model.encode(batch)
                direct = model.head(features, batch.batch)
                _, operator = model.global_context(
                    batch, batch.batch, model.local_polar_mode(features), return_operator=True
                )
                prediction = model.predict_components(batch).tensor
                direct_batches.append(direct.cpu())
                operator_batches.append(operator.cpu())
                prediction_batches.append(prediction.cpu())
            direct_by_split[name] = torch.cat(direct_batches)
            operator_by_split[name] = torch.cat(operator_batches)
            prediction_by_split[name] = torch.cat(prediction_batches)
    train_mean = target_by_split["train"].mean(dim=0, keepdim=True)
    raw_ids = [str(record["JARVIS_ID"]) for record in records]
    all_dataset = PiezoDataset(records, raw_ids, cfg["cutoff"], cfg["max_neighbors"], processed_dir=cfg["processed_dir"], cache_key=cache_key)
    all_target = torch.cat([all_dataset[index].y for index in range(len(all_dataset))])
    raw_formula_counts = Counter(formula(record) for record in records)
    formula_overlap = {name: len(set(formula(records_by_id[material_id]) for material_id in ids) & set(formula(records_by_id[other_id]) for other in splits if other != name for other_id in splits[other])) for name, ids in splits.items()}
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint), "device": str(device), "split_sizes": {name: len(ids) for name, ids in splits.items()},
        "data": {
            "valid_records": len(records), "unique_material_ids": len(set(raw_ids)), "duplicate_material_ids": len(raw_ids) - len(set(raw_ids)),
            "unique_formulas": len(raw_formula_counts), "formulas_with_multiple_records": sum(count > 1 for count in raw_formula_counts.values()),
            "formula_overlap_across_other_splits": formula_overlap,
            "all_records": _label_profile(all_target), "by_split": {name: _label_profile(target) for name, target in target_by_split.items()},
        },
        "graphs": {"all_records": _graph_profile(all_dataset)},
        "model": {},
        "symmetry": _symmetry_profile(records, args.symmetry_samples),
    }
    for name in splits:
        direct, operator, prediction, target = direct_by_split[name], operator_by_split[name], prediction_by_split[name], target_by_split[name]
        direct_norm = torch.linalg.vector_norm(direct.reshape(direct.shape[0], -1), dim=-1)
        operator_norm = torch.linalg.vector_norm(operator.reshape(operator.shape[0], -1), dim=-1)
        shape_norm = torch.linalg.vector_norm((direct + operator).reshape(direct.shape[0], -1), dim=-1)
        final_norm = torch.linalg.vector_norm(prediction.reshape(prediction.shape[0], -1), dim=-1)
        report["model"][name] = _prediction_profile(prediction, target) | {
            "direct_head_response_tensor_metrics": response_tensor_skill(direct, target),
            "collective_response_operator": {
                "direct_norm_mean": float(direct_norm.mean()), "final_norm_mean": float(final_norm.mean()),
                "operator_norm_mean": float(operator_norm.mean()),
                "pre_factorization_shape_norm_mean": float(shape_norm.mean()),
                "operator_to_direct_mean_norm_ratio": float(operator_norm.mean() / direct_norm.mean().clamp_min(torch.finfo(direct.dtype).eps)),
            },
            "train_mean_response_tensor_metrics": response_tensor_skill(train_mean.expand_as(target), target),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "diagnostic.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
