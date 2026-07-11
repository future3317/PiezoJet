"""Tensor metrics used by M3/M4 reports."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from .tensor_ops import cartesian_to_piezo_voigt


def tensor_metrics(prediction: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, float]:
    pred = cartesian_to_piezo_voigt(prediction)
    truth = cartesian_to_piezo_voigt(target)
    diff = pred - truth
    frob = torch.linalg.vector_norm(diff, dim=(-2, -1))
    truth_norm = torch.linalg.vector_norm(truth, dim=(-2, -1))
    tau = torch.tensor(threshold, device=truth.device, dtype=truth.dtype)
    return {
        "component_mae": float(diff.abs().mean()),
        "frob_rmse": float(torch.sqrt(diff.square().mean())),
        "frob_mae": float((frob / math.sqrt(18)).mean()),
        "relative_error_tau": float((frob / truth_norm.clamp_min(tau)).mean()),
        "max_component_mae": float((pred.abs().amax(dim=(-2, -1)) - truth.abs().amax(dim=(-2, -1))).abs().mean()),
    }


def centro_fp(prediction: torch.Tensor, centrosymmetric: torch.Tensor) -> dict[str, float]:
    pred = cartesian_to_piezo_voigt(prediction)
    norms = torch.linalg.vector_norm(pred[centrosymmetric], dim=(-2, -1))
    if norms.numel() == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan"), "gt_1e-4": float("nan"), "gt_1e-3": float("nan"), "gt_1e-2": float("nan")}
    return {"mean": float(norms.mean()), "median": float(norms.median()), "p90": float(torch.quantile(norms, 0.9)), "max": float(norms.max()), "gt_1e-4": float((norms > 1e-4).float().mean()), "gt_1e-3": float((norms > 1e-3).float().mean()), "gt_1e-2": float((norms > 1e-2).float().mean())}


def stratified_metrics(prediction: torch.Tensor, target: torch.Tensor, target_norm: torch.Tensor, centrosymmetric: torch.Tensor, threshold: float) -> dict[str, dict[str, float]]:
    groups = {"all": torch.ones(target.shape[0], dtype=torch.bool, device=target.device), "centrosymmetric": centrosymmetric, "non_centrosymmetric": ~centrosymmetric}
    norm = target_norm.detach()
    quantiles = torch.quantile(norm.cpu(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.9, 0.99])).to(norm.device)
    zero = norm <= torch.finfo(norm.dtype).eps
    groups["zero"] = zero
    groups["nonzero_0_5"] = (norm > torch.finfo(norm.dtype).eps) & (norm <= quantiles[0])
    groups["q05_25"] = (norm > quantiles[0]) & (norm <= quantiles[1])
    groups["q25_50"] = (norm > quantiles[1]) & (norm <= quantiles[2])
    groups["q50_75"] = (norm > quantiles[2]) & (norm <= quantiles[3])
    groups["q75_90"] = (norm > quantiles[3]) & (norm <= quantiles[4])
    groups["q90_99"] = (norm > quantiles[4]) & (norm <= quantiles[5])
    groups["top_1"] = norm > quantiles[5]
    result = {}
    for name, mask in groups.items():
        result[name] = tensor_metrics(prediction[mask], target[mask], threshold) if mask.any() else {}
    result["centro_fp"] = centro_fp(prediction, centrosymmetric)
    result["non_centrosymmetric_summary"] = result["non_centrosymmetric"]
    return result
