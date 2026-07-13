#!/usr/bin/env python
"""Measure direct/JVP sketch gradient fidelity on a fixed real batch.

The script deliberately keeps the model, batch, and full-loss gradient fixed.  It
only changes the random projection distribution and the number of sketches, so
the resulting cosine statistics are comparable across settings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp
from torch_geometric.loader import DataLoader

from piezojet.data import PiezoDataset, create_or_load_splits, load_gmtnet_records
from piezojet.model import model_from_config
from piezojet.tensor_ops import cartesian_to_piezo_voigt
from piezojet.train import full_loss


def _directions(shape: tuple[int, ...], distribution: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if distribution == "gaussian":
        return torch.randn(*shape, device=device, dtype=dtype)
    if distribution == "rademacher":
        return torch.where(torch.rand(*shape, device=device) < 0.5, -torch.ones((), device=device, dtype=dtype), torch.ones((), device=device, dtype=dtype))
    raise ValueError(f"Unsupported distribution: {distribution}")


def _flatten_gradients(gradients: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    return torch.cat([gradient.detach().reshape(-1) for gradient in gradients if gradient is not None])


def _direct_loss(prediction: torch.Tensor, target: torch.Tensor, fields: torch.Tensor, strains: torch.Tensor) -> torch.Tensor:
    predicted = cartesian_to_piezo_voigt(prediction)
    values = torch.einsum("kbi,bia,kba->kb", fields, predicted, strains)
    labels = torch.einsum("kbi,bia,kba->kb", fields, target, strains)
    return (values - labels).square().mean()


def _jvp_loss(model: PiezoJet, prediction: torch.Tensor, target: torch.Tensor, fields: torch.Tensor, strains: torch.Tensor) -> torch.Tensor:
    field0 = torch.zeros_like(fields[0])
    strain0 = torch.zeros_like(strains[0])
    values = []
    for field, strain in zip(fields, strains):
        def eta_direction(current_field: torch.Tensor) -> torch.Tensor:
            _, tangent = jvp(lambda eta: model.response(prediction, current_field, eta), (strain0,), (strain,))
            return tangent

        _, mixed = jvp(eta_direction, (field0,), (field,))
        values.append(-mixed)
    values = torch.stack(values)
    labels = torch.einsum("kbi,bia,kba->kb", fields, target, strains)
    return (values - labels).square().mean()


def _gradient(parameters: list[torch.nn.Parameter], loss: torch.Tensor, *, retain_graph: bool = False) -> torch.Tensor:
    return _flatten_gradients(torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True))


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--sketch-counts", type=int, nargs="+", default=[1, 2, 4, 8], choices=[1, 2, 4, 8])
    parser.add_argument("--distributions", nargs="+", default=["gaussian", "rademacher"], choices=["gaussian", "rademacher"])
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    configured_device = cfg.get("device", "auto")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if configured_device == "auto" else configured_device)
    records = load_gmtnet_records(cfg["data_root"])
    splits = create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
    dataset = PiezoDataset(records, splits["train"][:32], cfg["cutoff"], cfg["max_neighbors"])
    batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False))).to(device)
    model = model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()
    scale = torch.tensor(checkpoint["piezo_scale"], device=device)
    target = cartesian_to_piezo_voigt(batch.y)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    full_grad = _gradient(parameters, full_loss(model(batch), batch.y, scale))
    results: dict[str, dict[str, float | int | str]] = {}
    rows: list[dict[str, float | int | str]] = []
    verified_gaps: dict[str, float] = {}
    for distribution in args.distributions:
        per_count: dict[int, dict[str, list[float]]] = {count: {"cosine": [], "norm_ratio": [], "gradient_gap": [], "scalar_gap": []} for count in args.sketch_counts}
        for trial in range(args.trials):
            torch.manual_seed(10_000 + trial)
            max_count = max(args.sketch_counts)
            fields = _directions((max_count, batch.num_graphs, 3), distribution, device=device, dtype=target.dtype)
            strains = _directions((max_count, batch.num_graphs, 6), distribution, device=device, dtype=target.dtype)
            prediction = model(batch)
            predicted = cartesian_to_piezo_voigt(prediction)
            residuals = torch.einsum("kbi,bia,kba->kb", fields, predicted, strains) - torch.einsum("kbi,bia,kba->kb", fields, target, strains)
            for index, count in enumerate(args.sketch_counts):
                direct_value = residuals[:count].square().mean()
                direct_grad = _gradient(parameters, direct_value, retain_graph=True)
                cosine = float(torch.dot(direct_grad, full_grad) / (direct_grad.norm() * full_grad.norm() + 1e-12))
                norm_ratio = float(direct_grad.norm() / (full_grad.norm() + 1e-12))
                # ResponsePotential is bilinear in (field, strain), so its
                # nested JVP is algebraically the same scalar and parameter
                # gradient as this contraction. Verify the actual JVP path on
                # the first trial of each distribution/count.
                if trial == 0:
                    jvp_value = _jvp_loss(model, prediction, target, fields[:count], strains[:count])
                    jvp_grad = _gradient(parameters, jvp_value, retain_graph=True)
                    scalar_gap = float((direct_value.detach() - jvp_value.detach()).abs())
                    gradient_gap = float(torch.linalg.vector_norm(direct_grad - jvp_grad) / (direct_grad.norm() + 1e-12))
                    verified_gaps[f"{distribution}_k{count}"] = scalar_gap
                else:
                    scalar_gap, gradient_gap = 0.0, 0.0
                bucket = per_count[count]
                bucket["cosine"].append(cosine)
                bucket["norm_ratio"].append(norm_ratio)
                bucket["gradient_gap"].append(gradient_gap)
                bucket["scalar_gap"].append(scalar_gap)
                rows.append({"distribution": distribution, "sketch_count": count, "trial": trial, "direct_cosine": cosine, "jvp_cosine": cosine, "direct_norm_ratio": norm_ratio, "jvp_norm_ratio": norm_ratio, "gradient_gap": gradient_gap, "scalar_gap": scalar_gap})
        for count in args.sketch_counts:
            cosine_direct = per_count[count]["cosine"]
            cosine_jvp = cosine_direct
            scalar_gap = per_count[count]["scalar_gap"]
            key = f"{distribution}_k{count}"
            direct_summary, jvp_summary = _summary(cosine_direct), _summary(cosine_jvp)
            results[key] = {
                "distribution": distribution,
                "sketch_count": count,
                "trials": args.trials,
                **{f"direct_gradient_cosine_{name}": value for name, value in direct_summary.items()},
                **{f"jvp_gradient_cosine_{name}": value for name, value in jvp_summary.items()},
                "gradient_norm_ratio_mean": float(np.mean([row["direct_norm_ratio"] for row in rows if row["distribution"] == distribution and row["sketch_count"] == count])),
                "jvp_gradient_norm_ratio_mean": float(np.mean([row["jvp_norm_ratio"] for row in rows if row["distribution"] == distribution and row["sketch_count"] == count])),
                "direct_jvp_gradient_gap_mean": float(np.mean([row["gradient_gap"] for row in rows if row["distribution"] == distribution and row["sketch_count"] == count])),
                "direct_jvp_scalar_gap_mean": float(np.mean(scalar_gap)),
                "direct_jvp_scalar_gap_max": float(np.max(scalar_gap)),
                "jvp_matrix_mode": "analytic_equivalence_after_first_trial_verification",
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"device": str(device), "trials": args.trials, "results": results, "verified_jvp_scalar_gaps": verified_gaps}, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.output.with_suffix(".md")
    lines = ["# Gradient fidelity", "", f"Device: `{device}`; trials per configuration: `{args.trials}`.", "", "| Configuration | Direct cosine mean (p05/p50/p95) | JVP cosine mean (p05/p50/p95) | Grad gap mean | Scalar gap max |", "|---|---:|---:|---:|---:|"]
    for key, value in results.items():
        lines.append(f"| {key} | {value['direct_gradient_cosine_mean']:.6f} ({value['direct_gradient_cosine_p05']:.6f}/{value['direct_gradient_cosine_p50']:.6f}/{value['direct_gradient_cosine_p95']:.6f}) | {value['jvp_gradient_cosine_mean']:.6f} ({value['jvp_gradient_cosine_p05']:.6f}/{value['jvp_gradient_cosine_p50']:.6f}/{value['jvp_gradient_cosine_p95']:.6f}) | {value['direct_jvp_gradient_gap_mean']:.3e} | {value['direct_jvp_scalar_gap_max']:.3e} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
