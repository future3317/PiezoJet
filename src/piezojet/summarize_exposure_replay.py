"""Summarize the preregistered exposure replay without test-driven selection."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .evaluate_dfpt import FACTOR_FLOORS, ionic_aggregate_metrics
from .metrics import response_tensor_skill


def _tensor(row: dict[str, object], key: str) -> torch.Tensor:
    return torch.tensor(row[key], dtype=torch.float64).reshape(3, 3, 3)


def _physical_statistics(rows: list[dict[str, object]]) -> dict[str, float]:
    total_prediction = torch.stack([_tensor(row, "total_prediction") for row in rows])
    total_target = torch.stack([_tensor(row, "total_target") for row in rows])
    ionic_prediction = [_tensor(row, "ionic_prediction") for row in rows]
    ionic_target = [_tensor(row, "ionic_target") for row in rows]
    ionic = ionic_aggregate_metrics(
        ionic_prediction, ionic_target, FACTOR_FLOORS["ionic_piezo"]
    )
    return {
        "total_trs": float(
            response_tensor_skill(total_prediction, total_target)[
                "tensor_response_skill_vs_zero"
            ]
        ),
        "ionic_cosine_macro_material": float(
            ionic["ionic_cosine_macro_material"]
        ),
        "ionic_cosine_active_only": float(ionic["ionic_cosine_active_only"]),
        "ionic_amplitude_ratio_macro": float(ionic["ionic_amplitude_ratio_macro"]),
        "ionic_mae_skill_vs_zero_macro": float(
            ionic["ionic_mae_skill_vs_zero_macro"]
        ),
        "ionic_active_materials": float(ionic["ionic_active_materials"]),
    }


def _direct_total_trs(rows: list[dict[str, object]]) -> float:
    prediction = torch.stack([_tensor(row, "total_prediction") for row in rows])
    target = torch.stack([_tensor(row, "total_target") for row in rows])
    return float(
        response_tensor_skill(prediction, target)["tensor_response_skill_vs_zero"]
    )


def hierarchical_seed_material_interval(
    rows_by_seed: dict[int, list[dict[str, object]]],
    statistic: Callable[[list[dict[str, object]]], float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Resample seeds, then complete materials within each sampled seed."""
    seed_ids = sorted(rows_by_seed)
    if not seed_ids or resamples < 1:
        raise ValueError("hierarchical bootstrap requires seeds and resamples")
    generator = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for replicate in range(resamples):
        selected_seeds = generator.choice(seed_ids, size=len(seed_ids), replace=True)
        seed_values = []
        for selected_seed in selected_seeds:
            rows = rows_by_seed[int(selected_seed)]
            indices = generator.integers(0, len(rows), size=len(rows))
            seed_values.append(statistic([rows[int(index)] for index in indices]))
        values[replicate] = float(np.mean(seed_values))
    point_values = [statistic(rows_by_seed[seed_id]) for seed_id in seed_ids]
    return {
        "point_estimate_seed_mean": float(np.mean(point_values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "resamples": resamples,
        "seed_resampling": "with replacement",
        "material_resampling_within_seed": "with replacement",
    }


def paired_macro_difference_interval(
    physical_by_seed: dict[int, list[dict[str, object]]],
    direct_by_seed: dict[int, list[dict[str, object]]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Paired physical-macro minus direct-control TRS interval."""
    seed_ids = sorted(set(physical_by_seed) & set(direct_by_seed))
    generator = np.random.default_rng(seed)

    def paired_seed_value(seed_id: int, sampled_ids: list[str]) -> float:
        physical = {str(row["material_id"]): row for row in physical_by_seed[seed_id]}
        direct = {str(row["material_id"]): row for row in direct_by_seed[seed_id]}
        return _physical_statistics([physical[item] for item in sampled_ids])[
            "total_trs"
        ] - _direct_total_trs([direct[item] for item in sampled_ids])

    point_values = []
    for seed_id in seed_ids:
        ids = [str(row["material_id"]) for row in physical_by_seed[seed_id]]
        if set(ids) != {
            str(row["material_id"]) for row in direct_by_seed[seed_id]
        }:
            raise ValueError(f"Physical/direct material IDs differ for seed {seed_id}")
        point_values.append(paired_seed_value(seed_id, ids))
    values = np.empty(resamples, dtype=np.float64)
    for replicate in range(resamples):
        selected_seeds = generator.choice(seed_ids, size=len(seed_ids), replace=True)
        differences = []
        for selected_seed in selected_seeds:
            seed_id = int(selected_seed)
            ids = [str(row["material_id"]) for row in physical_by_seed[seed_id]]
            sampled = generator.choice(ids, size=len(ids), replace=True).tolist()
            differences.append(paired_seed_value(seed_id, sampled))
        values[replicate] = float(np.mean(differences))
    return {
        "point_estimate_seed_mean": float(np.mean(point_values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "resamples": resamples,
        "interpretation": "physical macro tower TRS minus matched direct-total TRS; negative control only",
    }


def _mean_sd(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else float("nan"),
        "seeds": len(values),
    }


def _selected_conditioning_metrics(run_dir: Path, epoch: int) -> dict[str, float]:
    path = run_dir / "metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = next(row for row in rows if int(row["epoch"]) == epoch)
    return {
        key.removeprefix("train_strict_stream_conditioning_").removesuffix(
            "_loss"
        ): float(value)
        for key, value in selected.items()
        if key.startswith("train_strict_stream_conditioning_") and value != ""
    }


def summarize(
    output_root: Path,
    passes: list[int],
    seeds: list[int],
    *,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    report: dict[str, object] = {
        "schema": 1,
        "registered_passes": passes,
        "registered_seeds": seeds,
        "selection": "validation loss only; all preregistered test points reported",
        "experiment_separation": {
            "physical": "branch/strict labels test direct U_{eta,delta} ionic learning",
            "macro": "total-only labels test the isolated macro predictor; this is not a factorization advantage test",
        },
        "points": {},
    }
    metric_names = (
        "total_trs",
        "ionic_cosine_macro_material",
        "ionic_cosine_active_only",
        "ionic_amplitude_ratio_macro",
        "ionic_mae_skill_vs_zero_macro",
    )
    for pass_count in passes:
        physical_by_seed: dict[int, list[dict[str, object]]] = {}
        direct_by_seed: dict[int, list[dict[str, object]]] = {}
        individual: dict[str, object] = {}
        for seed in seeds:
            physical_dir = output_root / "physical" / f"passes{pass_count}_seed{seed}"
            direct_dir = output_root / "direct" / f"passes{pass_count}_seed{seed}"
            physical = json.loads(
                (physical_dir / "dfpt_test.json").read_text(encoding="utf-8")
            )
            direct = json.loads(
                (direct_dir / "test.json").read_text(encoding="utf-8")
            )
            physical_rows = physical["resampling_material_rows"]
            direct_rows = direct["resampling_material_rows"]
            physical_by_seed[seed] = physical_rows
            direct_by_seed[seed] = direct_rows
            physical_stats = _physical_statistics(physical_rows)
            physical_stats.update(
                true_bec_u_cross_covariance_cosine=float(
                    physical["strict_symmetry_completed_lambda_oracle"]
                    ["response_active_alignment"]["mean"]
                    ["true_charge_cross_covariance_directional_cosine"]
                ),
                true_bec_u_cross_covariance_amplitude_ratio=float(
                    physical["strict_symmetry_completed_lambda_oracle"]
                    ["response_active_alignment"]["mean"]
                    ["true_charge_cross_covariance_amplitude_ratio"]
                ),
                response_weighted_log_stiffness_bias=float(
                    physical["stability"]["response_weighted_log_stiffness_bias"]
                    ["mean"]
                ),
            )
            physical_stats["conditioning"] = _selected_conditioning_metrics(
                physical_dir, int(physical["checkpoint_epoch"])
            )
            individual[str(seed)] = {
                "physical": physical_stats,
                "macro_direct_control_total_trs": _direct_total_trs(direct_rows),
            }
        aggregate = {
            name: _mean_sd(
                [float(individual[str(seed)]["physical"][name]) for seed in seeds]
            )
            for name in metric_names
        }
        intervals = {
            name: hierarchical_seed_material_interval(
                physical_by_seed,
                lambda rows, metric=name: _physical_statistics(rows)[metric],
                resamples=resamples,
                seed=bootstrap_seed + pass_count,
            )
            for name in metric_names
        }
        macro_control_values = [
            float(individual[str(seed)]["macro_direct_control_total_trs"])
            for seed in seeds
        ]
        report["points"][str(pass_count)] = {
            "individual_seeds": individual,
            "physical_seed_mean_sd": aggregate,
            "physical_hierarchical_bootstrap_95": intervals,
            "macro_direct_control_seed_mean_sd": _mean_sd(macro_control_values),
            "paired_physical_macro_minus_direct_control_trs_bootstrap_95": (
                paired_macro_difference_interval(
                    physical_by_seed,
                    direct_by_seed,
                    resamples=resamples,
                    seed=bootstrap_seed + 1000 + pass_count,
                )
            ),
            "ionic_mae_skill_verdict": (
                "positive"
                if intervals["ionic_mae_skill_vs_zero_macro"]["lower_95"] > 0
                else "negative"
                if intervals["ionic_mae_skill_vs_zero_macro"]["upper_95"] < 0
                else "inconclusive"
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--passes", default="1,5,10,20")
    parser.add_argument("--seeds", default="42,7,1729")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20270716)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.output_root,
        [int(value) for value in args.passes.split(",")],
        [int(value) for value in args.seeds.split(",")],
        resamples=args.resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
