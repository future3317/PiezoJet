"""Maintained first-order electrostatic-jet candidates and selection policy."""

from __future__ import annotations

import math
from typing import Any

import torch

ARCHITECTURES = (
    "a0_independent_irreps",
    "a0_parameter_matched_irreps",
    "a1_electromechanical_jet",
    "a15_soft_shared_electromechanical_jet",
    "a16_hierarchical_electromechanical_jet",
)

A0_ARCHITECTURES = (
    "a0_independent_irreps",
    "a0_parameter_matched_irreps",
)

CURRENT_ADJUDICATION_ARCHITECTURES = (
    "a0_independent_irreps",
    "a0_parameter_matched_irreps",
    "a1_electromechanical_jet",
    "a16_hierarchical_electromechanical_jet",
)

STABILIZED_SELECTION_VERSION = "electrostatic_stabilized_v2"
ELECTRONIC_AMPLITUDE_GUARDRAIL = 0.05
DEFAULT_EARLY_STOPPING_PATIENCE_EVALUATIONS = 2


def development_early_stopping(
    *,
    score: float,
    eligible: bool,
    best_score: float,
    non_improving_evaluations: int,
    patience_evaluations: int,
    minimum_improvement: float,
) -> dict[str, int | float | bool]:
    """Advance guardrail-aware development early stopping state.

    Early stopping starts only after an eligible checkpoint exists.  Thus a
    collapsed initialization cannot terminate a run before it has had a chance
    to cross the directional and amplitude guardrails.  A patience of zero
    explicitly disables early stopping.
    """
    if patience_evaluations < 0:
        raise ValueError("Early-stopping patience cannot be negative")
    if minimum_improvement < 0.0 or not math.isfinite(minimum_improvement):
        raise ValueError("Early-stopping minimum improvement must be finite and nonnegative")
    if non_improving_evaluations < 0:
        raise ValueError("Non-improving evaluation count cannot be negative")
    improved = bool(
        eligible
        and math.isfinite(score)
        and score < best_score - minimum_improvement
    )
    has_eligible_best = math.isfinite(best_score) or improved
    if improved:
        next_non_improving = 0
    elif has_eligible_best:
        next_non_improving = non_improving_evaluations + 1
    else:
        next_non_improving = 0
    should_stop = bool(
        patience_evaluations > 0
        and has_eligible_best
        and next_non_improving >= patience_evaluations
    )
    return {
        "improved": improved,
        "non_improving_evaluations": next_non_improving,
        "patience_evaluations": patience_evaluations,
        "minimum_improvement": minimum_improvement,
        "should_stop": should_stop,
    }


def compact_training_curve_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Extract one plotting-ready row without per-material evaluation payloads."""
    selection = row.get("development_selection")
    if not isinstance(selection, dict):
        return None
    train_selection = row.get("train_selection")
    compact = {
        key: row[key]
        for key in (
            "update",
            "train_loss",
            "train_electronic_loss",
            "train_born_loss",
            "train_dielectric_loss",
            "development_selection_score",
            "generalization_score_gap",
        )
        if key in row
    }
    compact.update({
        "development_eligible": bool(selection["eligible"]),
        "development_components": selection["components"],
        "development_guardrails": selection["guardrails"],
        "guardrail_failures": selection["guardrail_failures"],
        "train_selection_score": (
            float(train_selection["raw_score"])
            if isinstance(train_selection, dict)
            else None
        ),
        "early_stopping": row.get("early_stopping"),
        "evaluation_runtime": row.get("evaluation_runtime"),
    })
    return compact


def development_selection(
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return the preregistered three-task score and non-maskable guardrails.

    The scalar score uses stabilized material-level relative errors.  It may
    rank only checkpoints that have positive electronic/BEC directional skill
    and whose active electronic response has not collapsed below five percent
    of the stabilized target amplitude.  A failed checkpoint remains fully
    auditable but is never silently selected through a fallback score.
    """
    electronic = metrics["electronic"]
    born = metrics["born"]
    dielectric = metrics.get("dielectric_audit", metrics.get("dielectric"))
    if dielectric is None:
        raise KeyError("Development metrics have no dielectric block")
    components = {
        "electronic_stabilized_relative": float(
            electronic["mean_stabilized_relative_frobenius_error"]
        ),
        "born_stabilized_relative": float(
            born["mean_stabilized_relative_frobenius_error"]
        ),
        "electronic_dielectric_stabilized_relative": float(
            dielectric["mean_stabilized_relative_frobenius_error"]
        ),
    }
    raw_score = sum(components.values())
    electronic_cosine = float(electronic["mean_active_cosine"])
    born_cosine_value = born.get("mean_nonzero_cosine")
    born_cosine = (
        float(born_cosine_value) if born_cosine_value is not None else float("nan")
    )
    electronic_amplitude = float(electronic["mean_active_amplitude_ratio"])
    failures: list[str] = []
    if not math.isfinite(raw_score):
        failures.append("nonfinite_stabilized_score")
    if not math.isfinite(electronic_cosine) or electronic_cosine <= 0.0:
        failures.append("nonpositive_electronic_active_cosine")
    if not math.isfinite(born_cosine) or born_cosine <= 0.0:
        failures.append("nonpositive_bec_nonzero_cosine")
    if (
        not math.isfinite(electronic_amplitude)
        or electronic_amplitude < ELECTRONIC_AMPLITUDE_GUARDRAIL
    ):
        failures.append("electronic_active_amplitude_collapse")
    return {
        "version": STABILIZED_SELECTION_VERSION,
        "raw_score": raw_score,
        "eligible": not failures,
        "guardrail_failures": failures,
        "components": components,
        "guardrails": {
            "electronic_active_cosine": electronic_cosine,
            "bec_nonzero_cosine": born_cosine,
            "electronic_active_amplitude_ratio": electronic_amplitude,
            "minimum_electronic_active_amplitude_ratio": (
                ELECTRONIC_AMPLITUDE_GUARDRAIL
            ),
        },
        "exact_zero_bec_absolute_leakage_e": born.get(
            "mean_exact_zero_prediction_norm_e"
        ),
    }


def matched_material_schedule(
    materials: int,
    updates: int,
    logical_batch_size: int,
    microbatch_size: int,
    seed: int,
) -> list[int]:
    """Deterministic shuffled complete-microbatch material schedule."""
    if materials < logical_batch_size:
        raise ValueError("Logical batch size cannot exceed the training panel")
    if logical_batch_size % microbatch_size:
        raise ValueError("Logical batch size must be divisible by microbatch size")
    usable = materials - materials % microbatch_size
    required = updates * logical_batch_size
    generator = torch.Generator().manual_seed(seed)
    schedule: list[int] = []
    while len(schedule) < required:
        schedule.extend(torch.randperm(materials, generator=generator)[:usable].tolist())
    return schedule[:required]
