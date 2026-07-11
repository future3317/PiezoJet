#!/usr/bin/env python
"""Measure direct/JVP sketch gradient fidelity on a fixed real batch.

The script deliberately keeps the model, batch, and full-loss gradient fixed.  It
only changes the random projection distribution and the number of sketches, so
the resulting cosine statistics are comparable across settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp
from torch_geometric.loader import DataLoader

from piezojet.data import PiezoDataset, create_or_load_splits, load_gmtnet_records
from piezojet.model import PiezoJet
from piezojet.tensor_ops import cartesian_to_piezo_voigt
from piezojet.train import full_loss


def _directions(shape: tuple[int, ...], distribution: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if distribution == "gaussian":
        return torch.randn(*shape, device=device, dtype=dtype)
    if distribution == "rademacher":
        return torch.where(torch.rand(*shape, device=device) < 0.5, -torch.ones((), device=device, dtype=dtype), torch.ones((), device=device, dtype=dtype))
    raise ValueError(f"Unsupported distribution: {distribution}")


def _flatten_grad(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters() if parameter.grad is not None])


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


def _gradient(model: PiezoJet, loss: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss.backward()
    return _flatten_grad(model).clone()


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
    model = PiezoJet(embedding_dim=cfg["embedding_dim"], cutoff=cfg["cutoff"], lmax=cfg["lmax"], num_blocks=cfg["num_blocks"], radial_basis=cfg["radial_basis"], radial_hidden=cfg["radial_hidden"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()
    scale = torch.tensor(checkpoint["piezo_scale"], device=device)
    target = cartesian_to_piezo_voigt(batch.y)
    full_grad = _gradient(model, full_loss(model(batch), batch.y, scale))
    results: dict[str, dict[str, float | int | str]] = {}
    for distribution in args.distributions:
        for count in args.sketch_counts:
            cosine_direct, cosine_jvp, scalar_gap = [], [], []
            for trial in range(args.trials):
                torch.manual_seed(10_000 + trial)
                fields = _directions((count, batch.num_graphs, 3), distribution, device=device, dtype=target.dtype)
                strains = _directions((count, batch.num_graphs, 6), distribution, device=device, dtype=target.dtype)
                direct_grad = _gradient(model, _direct_loss(model(batch), target, fields, strains))
                jvp_grad = _gradient(model, _jvp_loss(model, model(batch), target, fields, strains))
                cosine_direct.append(float(torch.dot(direct_grad, full_grad) / (direct_grad.norm() * full_grad.norm() + 1e-12)))
                cosine_jvp.append(float(torch.dot(jvp_grad, full_grad) / (jvp_grad.norm() * full_grad.norm() + 1e-12)))
                direct_value = _direct_loss(model(batch), target, fields, strains)
                jvp_value = _jvp_loss(model, model(batch), target, fields, strains)
                scalar_gap.append(float((direct_value.detach() - jvp_value.detach()).abs()))
            key = f"{distribution}_k{count}"
            results[key] = {
                "distribution": distribution,
                "sketch_count": count,
                "trials": args.trials,
                "direct_gradient_cosine_mean": float(np.mean(cosine_direct)),
                "direct_gradient_cosine_std": float(np.std(cosine_direct)),
                "direct_gradient_cosine_p05": float(np.quantile(cosine_direct, 0.05)),
                "direct_gradient_cosine_p95": float(np.quantile(cosine_direct, 0.95)),
                "jvp_gradient_cosine_mean": float(np.mean(cosine_jvp)),
                "jvp_gradient_cosine_std": float(np.std(cosine_jvp)),
                "jvp_gradient_cosine_p05": float(np.quantile(cosine_jvp, 0.05)),
                "jvp_gradient_cosine_p95": float(np.quantile(cosine_jvp, 0.95)),
                "direct_jvp_scalar_gap_max": float(np.max(scalar_gap)),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"device": str(device), "trials": args.trials, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
