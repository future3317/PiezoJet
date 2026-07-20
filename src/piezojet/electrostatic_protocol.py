"""Maintained first-order electrostatic-jet candidates and selection policy."""

from __future__ import annotations

import math
from typing import Any

import torch

ARCHITECTURES = (
    "a0_independent_irreps",
    "a1_electromechanical_jet",
    "a15_soft_shared_electromechanical_jet",
)

STABILIZED_SELECTION_VERSION = "electrostatic_stabilized_v2"
ELECTRONIC_AMPLITUDE_GUARDRAIL = 0.05


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
