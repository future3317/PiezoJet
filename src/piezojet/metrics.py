"""Tensor metrics used by M3/M4 reports."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Iterable

import torch

from .tensor_ops import cartesian_to_piezo_voigt


def stabilized_relative_residual(actual: torch.Tensor, expected: torch.Tensor, floor: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return absolute and scale-stabilized relative tensor residuals.

    Exact equivariant models can legitimately predict a near-zero tensor (in
    particular for centrosymmetric crystals).  Dividing by that prediction
    norm turns floating-point roundoff into an arbitrarily large ``relative``
    error, so use a task-scale floor and report the absolute residual too.
    """
    absolute = torch.linalg.vector_norm((actual - expected).reshape(actual.shape[0], -1), dim=-1)
    reference = torch.maximum(
        torch.linalg.vector_norm(expected.reshape(expected.shape[0], -1), dim=-1),
        torch.as_tensor(floor, dtype=absolute.dtype, device=absolute.device),
    )
    return absolute, absolute / reference


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


def response_tensor_skill(
    prediction: torch.Tensor,
    target: torch.Tensor,
    signal_scale: float = 0.5,
    high_response_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Evaluate response-bearing tensors without rewarding an all-zero model.

    ``signal_scale`` is expressed in C/m^2.  It is fixed *before* evaluating
    the test split and is used only to make the continuous signal weight
    saturate.  With ``0.5`` it is anchored to the high-piezoelectric screening
    scale used in the JARVIS DFPT study, rather than fitted to model outputs.

    The primary ``tensor_response_skill_vs_zero`` is based on a weighted,
    scale-stabilized Frobenius residual and calibrated against the all-zero
    predictor on the same labels.  It is exactly zero for that predictor, one
    for a perfect predictor, and negative if a model is worse than zero.  Both
    the residual and its signal weights use Cartesian Frobenius norms, so they
    are invariant to a common rotation of prediction and target.

    The high-response diagnostics deliberately use the JARVIS screening
    convention ``max |e_ij| >= 0.5 C/m^2``.  They answer the separate
    materials-discovery question: does the model reproduce the magnitude and
    orientation of tensors that are large enough to matter?
    """
    if signal_scale <= 0 or high_response_threshold <= 0:
        raise ValueError("Signal scales must be positive")
    if prediction.shape != target.shape or prediction.shape[-3:] != (3, 3, 3):
        raise ValueError("Prediction and target must have matching [..., 3, 3, 3] shapes")

    error_norm = torch.linalg.vector_norm((prediction - target).reshape(target.shape[0], -1), dim=-1)
    target_norm = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=-1)
    tau = torch.as_tensor(signal_scale, dtype=target.dtype, device=target.device)
    # Bounded, monotone signal weighting: exactly-zero tensors have zero
    # evaluation weight, while genuinely responsive tensors dominate smoothly.
    weight = target_norm.square() / (target_norm.square() + tau.square())
    residual = error_norm / (target_norm + tau)
    zero_residual = target_norm / (target_norm + tau)
    weighted_relative_error = (weight * residual).sum() / weight.sum().clamp_min(torch.finfo(target.dtype).eps)
    zero_weighted_relative_error = (weight * zero_residual).sum() / weight.sum().clamp_min(torch.finfo(target.dtype).eps)

    target_voigt = cartesian_to_piezo_voigt(target)
    high_response = target_voigt.abs().amax(dim=(-2, -1)) >= high_response_threshold
    result: dict[str, float | int] = {
        "signal_scale_C_m2": float(signal_scale),
        "high_response_threshold_C_m2": float(high_response_threshold),
        "signal_weighted_relative_frobenius_error": float(weighted_relative_error),
        "zero_signal_weighted_relative_frobenius_error": float(zero_weighted_relative_error),
        "tensor_response_skill_vs_zero": float(1.0 - weighted_relative_error / zero_weighted_relative_error.clamp_min(torch.finfo(target.dtype).eps)),
        "high_response_count": int(high_response.sum()),
    }
    if not high_response.any():
        result.update({
            "high_response_relative_frobenius_error": float("nan"),
            "high_response_directional_cosine": float("nan"),
            "high_response_amplitude_ratio": float("nan"),
        })
        return result

    active_prediction = prediction[high_response].reshape(int(high_response.sum()), -1)
    active_target = target[high_response].reshape(int(high_response.sum()), -1)
    active_prediction_norm = torch.linalg.vector_norm(active_prediction, dim=-1)
    active_target_norm = torch.linalg.vector_norm(active_target, dim=-1)
    cosine = (active_prediction * active_target).sum(dim=-1) / (active_prediction_norm * active_target_norm).clamp_min(torch.finfo(target.dtype).eps)
    active_error = torch.linalg.vector_norm(active_prediction - active_target, dim=-1)
    result.update({
        "high_response_relative_frobenius_error": float((active_error / active_target_norm).mean()),
        "high_response_directional_cosine": float(cosine.mean()),
        "high_response_amplitude_ratio": float((active_prediction_norm / active_target_norm).mean()),
    })
    return result


def material_bootstrap_confidence_interval(
    predictions: Iterable[torch.Tensor],
    targets: Iterable[torch.Tensor],
    statistic: Callable[[list[torch.Tensor], list[torch.Tensor]], float],
    *,
    resamples: int = 2000,
    seed: int = 20270715,
) -> dict[str, float | int | str]:
    """Percentile confidence interval by resampling complete materials.

    Tensor components within a crystal are correlated.  The material, rather
    than the component, is therefore the only valid resampling unit here.
    This post-evaluation helper contains no checkpoint-selection logic.
    """
    prediction_rows = [value.detach().cpu() for value in predictions]
    target_rows = [value.detach().cpu() for value in targets]
    if len(prediction_rows) != len(target_rows) or not prediction_rows:
        raise ValueError("Bootstrap requires equal non-empty material lists")
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    point = float(statistic(prediction_rows, target_rows))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.randint(len(prediction_rows), (resamples, len(prediction_rows)), generator=generator)
    values = torch.empty(resamples, dtype=torch.float64)
    for row, indices in enumerate(draws):
        selected = indices.tolist()
        values[row] = float(statistic(
            [prediction_rows[index] for index in selected],
            [target_rows[index] for index in selected],
        ))
    if not torch.isfinite(values).all():
        raise FloatingPointError("Bootstrap statistic produced a non-finite draw")
    interval = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=values.dtype))
    return {
        "point_estimate": point,
        "lower_95": float(interval[0]),
        "upper_95": float(interval[1]),
        "resamples": resamples,
        "seed": seed,
        "resampling_unit": "material",
        "interval": "percentile_95",
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
