"""Physical-unit evaluation on a formula-disjoint JARVIS DFPT subset.

Unlike :mod:`piezojet.evaluate`, this entry point evaluates the variable-size
atom-coordinate factors used by the relaxed response.  It never pads phonon
coordinates and scores only the internal-strain blocks actually printed by
VASP.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from .data import PiezoDataset, create_or_load_splits, graph_cache_key, load_gmtnet_records
from .metrics import response_tensor_skill
from .model import AtomCoordinateResponsePotential, model_from_config
from .tensor_ops import piezo_voigt_to_cartesian
from .train import device_from_config, restrict_splits_to_material_ids


FACTOR_FLOORS = {
    "born_charge": 0.1,  # elementary charge
    "force_constant": 0.01,  # eV / Angstrom^2
    "internal_strain": 0.01,  # eV / Angstrom
    "ionic_piezo": 0.05,  # C / m^2
    "electronic_piezo": 0.05,  # C / m^2
    "total_piezo": 0.05,  # C / m^2
    "dielectric": 0.1,  # relative permittivity
}

FACTOR_UNITS = {
    "born_charge": "e",
    "force_constant": "eV/Angstrom^2",
    "internal_strain": "eV/Angstrom",
    "ionic_piezo": "C/m^2",
    "electronic_piezo": "C/m^2",
    "total_piezo": "C/m^2",
    "dielectric": "relative",
}


def _translation_projector(atoms: int, reference: torch.Tensor) -> torch.Tensor:
    translation = reference.new_zeros(3 * atoms, 3)
    for axis in range(3):
        translation[axis::3, axis] = atoms ** -0.5
    return torch.eye(3 * atoms, dtype=reference.dtype, device=reference.device) - translation @ translation.T


def clean_force_constant_target(target_flat: torch.Tensor, atoms: int) -> torch.Tensor:
    """Apply the same symmetry and translational projection as training."""
    blocks = target_flat.reshape(atoms, atoms, 3, 3)
    matrix = blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
    matrix = 0.5 * (matrix + matrix.T)
    projector = _translation_projector(atoms, matrix)
    cleaned = projector @ matrix @ projector
    return cleaned.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)


def force_constant_matrix(blocks: torch.Tensor) -> torch.Tensor:
    atoms = blocks.shape[0]
    return blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)


def optical_eigensystem(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenpairs with the three ASR translations removed, sorted by energy."""
    matrix = force_constant_matrix(blocks).to(torch.float64)
    values, vectors = torch.linalg.eigh(matrix)
    if values.numel() <= 3:
        return values.new_empty(0), vectors.new_empty(values.numel(), 0)
    optical_indices = torch.argsort(values.abs())[3:]
    optical_indices = optical_indices[torch.argsort(values[optical_indices])]
    return values[optical_indices], vectors[:, optical_indices]


def soft_mode_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    predicted_born: torch.Tensor,
    target_born: torch.Tensor,
    mode_count: int = 3,
) -> dict[str, float | int]:
    """Response-relevant low-optical-mode diagnostics without mode padding."""
    pred_values, pred_vectors = optical_eigensystem(prediction)
    true_values, true_vectors = optical_eigensystem(target)
    count = min(mode_count, pred_values.numel(), true_values.numel())
    if count == 0:
        return {
            "soft_mode_count": 0,
            "minimum_true_optical_eigenvalue": float("nan"),
            "minimum_predicted_optical_eigenvalue": float("nan"),
            "lowest_optical_eigenvalue_mae": float("nan"),
            "soft_mode_sign_accuracy": float("nan"),
            "soft_mode_subspace_overlap": float("nan"),
            "mode_effective_charge_norm_mae": float("nan"),
        }
    pred_values, true_values = pred_values[:count], true_values[:count]
    pred_soft, true_soft = pred_vectors[:, :count], true_vectors[:, :count]
    overlap = (true_soft.T @ pred_soft).square().sum() / count
    target_charge = target_born.reshape(-1, 3).to(torch.float64)
    predicted_charge = predicted_born.reshape(-1, 3).to(torch.float64)
    # Project both charges on the target soft coordinates.  Comparing norms
    # removes the arbitrary sign of each target eigenvector.
    true_mode_charge = torch.linalg.vector_norm(true_soft.T @ target_charge, dim=-1)
    pred_mode_charge = torch.linalg.vector_norm(true_soft.T @ predicted_charge, dim=-1)
    return {
        "soft_mode_count": count,
        "minimum_true_optical_eigenvalue": float(true_values.min()),
        "minimum_predicted_optical_eigenvalue": float(pred_values.min()),
        "lowest_optical_eigenvalue_mae": float((pred_values - true_values).abs().mean()),
        "soft_mode_sign_accuracy": float((torch.sign(pred_values) == torch.sign(true_values)).to(torch.float64).mean()),
        "soft_mode_subspace_overlap": float(overlap),
        "mode_effective_charge_norm_mae": float((pred_mode_charge - true_mode_charge).abs().mean()),
    }


def replace_printed_internal_strain(
    prediction: torch.Tensor,
    target_flat: torch.Tensor,
    ions: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Replace only genuinely observed Lambda blocks; never invent the rest."""
    output = prediction.clone()
    target = target_flat.reshape(-1, 3, 3)
    output[ions, directions] = 0.5 * (target + target.transpose(-1, -2))
    return output


def ionic_piezo_from_factors(
    response: AtomCoordinateResponsePotential,
    born: torch.Tensor,
    force_constants: torch.Tensor,
    internal_strain: torch.Tensor,
    volume: torch.Tensor | float,
    solve_policy: str = "auto",
    regularization: float | None = None,
) -> torch.Tensor:
    """Compute the ionic tensor for an explicit oracle factor combination."""
    operator = response.optical_operator(force_constants, solve_policy, regularization)
    coupling = response._coupling_voigt(internal_strain).reshape(-1, 6)
    charge = born.reshape(-1, 3)
    volume_tensor = torch.as_tensor(volume, dtype=born.dtype, device=born.device)
    piezo_voigt = response.PIEZO_C_PER_M2 * charge.T @ operator @ coupling / volume_tensor
    return piezo_voigt_to_cartesian(piezo_voigt)


def selected_internal_strain(
    prediction: torch.Tensor,
    target_flat: torch.Tensor,
    ions: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select only source-observed ion/direction blocks and symmetrize labels."""
    target = target_flat.reshape(-1, 3, 3)
    if target.shape[0] != ions.numel() or ions.numel() != directions.numel():
        raise ValueError("Internal-strain block metadata is inconsistent")
    selected = prediction[ions, directions]
    return selected, 0.5 * (target + target.transpose(-1, -2))


def pair_metrics(prediction: torch.Tensor, target: torch.Tensor, floor: float) -> dict[str, float | int]:
    """Physical component errors and a zero-predictor comparison for one array."""
    prediction = prediction.detach().to(torch.float64).reshape(-1)
    target = target.detach().to(torch.float64).reshape(-1)
    if prediction.shape != target.shape or target.numel() == 0:
        raise ValueError("Metric arrays must be equally shaped and non-empty")
    residual = prediction - target
    mae = residual.abs().mean()
    zero_mae = target.abs().mean()
    residual_norm = torch.linalg.vector_norm(residual)
    target_norm = torch.linalg.vector_norm(target)
    prediction_norm = torch.linalg.vector_norm(prediction)
    denominator = target_norm.clamp_min(float(floor) * target.numel() ** 0.5)
    return {
        "components": target.numel(),
        "component_mae": float(mae),
        "component_rmse": float(residual.square().mean().sqrt()),
        "frobenius_error": float(residual_norm),
        "stabilized_relative_frobenius_error": float(residual_norm / denominator),
        "zero_component_mae": float(zero_mae),
        "mae_skill_vs_zero": float(1.0 - mae / zero_mae.clamp_min(torch.finfo(torch.float64).eps)),
        "stabilized_amplitude_ratio": float(prediction_norm / denominator),
        "directional_cosine": float(
            torch.dot(prediction, target)
            / (prediction_norm * target_norm).clamp_min(torch.finfo(torch.float64).eps)
        ),
    }


@dataclass
class FactorAccumulator:
    name: str
    predictions: list[torch.Tensor] = field(default_factory=list)
    targets: list[torch.Tensor] = field(default_factory=list)
    material_metrics: list[dict[str, float | int]] = field(default_factory=list)

    def add(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
        prediction = prediction.detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        metrics = pair_metrics(prediction, target, FACTOR_FLOORS[self.name])
        self.predictions.append(prediction)
        self.targets.append(target)
        self.material_metrics.append(metrics)
        return metrics

    def summary(self) -> dict[str, float | int | str]:
        if not self.predictions:
            raise ValueError(f"No observations collected for {self.name}")
        micro = pair_metrics(
            torch.cat(self.predictions), torch.cat(self.targets), FACTOR_FLOORS[self.name]
        )
        macro_keys = (
            "component_mae",
            "component_rmse",
            "stabilized_relative_frobenius_error",
            "zero_component_mae",
            "stabilized_amplitude_ratio",
            "directional_cosine",
        )
        macro_mae = sum(float(row["component_mae"]) for row in self.material_metrics) / len(self.material_metrics)
        macro_zero_mae = sum(float(row["zero_component_mae"]) for row in self.material_metrics) / len(self.material_metrics)
        return {
            "unit": FACTOR_UNITS[self.name],
            "materials": len(self.material_metrics),
            **{f"micro_{key}": value for key, value in micro.items()},
            **{
                f"macro_material_{key}": sum(float(row[key]) for row in self.material_metrics)
                / len(self.material_metrics)
                for key in macro_keys
            },
            "macro_material_mae_skill_vs_zero": 1.0 - macro_mae / max(
                macro_zero_mae, torch.finfo(torch.float64).eps
            ),
        }


def _read_material_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    ids = [str(value) for value in values]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Material-ID file must contain a non-empty list of unique IDs")
    return ids


def _mean_rows(rows: Iterable[dict[str, float | int]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--material-ids-file", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--soft-mode-count", type=int, default=3)
    parser.add_argument(
        "--delta-grid",
        default="1e-4,3e-4,1e-3,3e-3,1e-2",
        help="Comma-separated signed-Green regularization scales in eV/Angstrom^2",
    )
    args = parser.parse_args()
    if args.soft_mode_count <= 0:
        raise ValueError("--soft-mode-count must be positive")
    delta_grid = [float(value) for value in args.delta_grid.split(",")]
    if not delta_grid or any(value <= 0 for value in delta_grid):
        raise ValueError("--delta-grid must contain positive values")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    device = device_from_config(str(cfg["device"]))
    records = load_gmtnet_records(cfg["data_root"])
    global_splits = create_or_load_splits(records, cfg["processed_dir"], int(cfg["seed"]))
    ids_path = args.material_ids_file or Path(str(cfg.get("material_ids_file", "")))
    if not ids_path.is_file():
        raise FileNotFoundError("A persisted audited material-ID file is required for DFPT evaluation")
    selected_ids = _read_material_ids(ids_path)
    splits = restrict_splits_to_material_ids(global_splits, selected_ids, "global")
    split_ids = splits[args.split]
    if not split_ids:
        raise ValueError(f"The audited DFPT {args.split} split is empty")

    cache_key = graph_cache_key(records, float(cfg["cutoff"]), int(cfg["max_neighbors"]))
    dataset = PiezoDataset(
        records,
        split_ids,
        float(cfg["cutoff"]),
        int(cfg["max_neighbors"]),
        processed_dir=cfg["processed_dir"],
        cache_key=cache_key,
        dfpt_dir=cfg.get("jarvis_dfpt_dir"),
    )
    model = model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    factor_names = (
        "born_charge",
        "force_constant",
        "internal_strain",
        "ionic_piezo",
        "electronic_piezo",
        "total_piezo",
        "dielectric",
    )
    accumulators = {name: FactorAccumulator(name) for name in factor_names}
    rows: list[dict[str, float | int | str]] = []
    total_predictions, total_targets, electronic_predictions = [], [], []
    oracle_names = (
        "pred_all_auto",
        "true_z_true_phi_pred_lambda_auto",
        "pred_z_true_phi_pred_lambda_auto",
        "true_z_pred_phi_pred_lambda_auto",
        "true_z_true_phi_observed_lambda_auto",
        "true_z_true_phi_pred_lambda_regularized",
        "true_z_true_phi_pred_lambda_exact",
    )
    oracle_values: dict[str, dict[str, list[torch.Tensor]]] = {
        stratum: {name: [] for name in oracle_names}
        for stratum in ("all", "stable", "unstable")
    }
    oracle_targets: dict[str, list[torch.Tensor]] = {
        stratum: [] for stratum in ("all", "stable", "unstable")
    }
    delta_values: dict[str, dict[float, list[torch.Tensor]]] = {
        stratum: {delta: [] for delta in delta_grid}
        for stratum in ("all", "stable", "unstable")
    }
    delta_targets: dict[str, list[torch.Tensor]] = {
        stratum: [] for stratum in ("all", "stable", "unstable")
    }

    with torch.inference_mode():
        for index in range(len(dataset)):
            graph = dataset[index].clone()
            material_id = str(dataset.records[index]["JARVIS_ID"])
            if not bool(graph.force_constant_mask) or not bool(graph.ionic_piezo_mask) or not bool(graph.born_mask.all()):
                raise ValueError(f"Missing verified DFPT factors for {material_id}")
            if not bool(graph.dielectric_mask):
                raise ValueError(f"Missing same-record dielectric target for {material_id}")
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
            graph.ptr = torch.tensor([0, graph.num_nodes], dtype=torch.long)
            graph = graph.to(device)
            components = model.predict_components(graph)

            force_target = clean_force_constant_target(graph.dfpt_force_constants_flat, graph.num_nodes)
            force_prediction = components.force_constants_flat.reshape(graph.num_nodes, graph.num_nodes, 3, 3)
            internal_prediction, internal_target = selected_internal_strain(
                components.internal_strain,
                graph.dfpt_internal_strain_flat,
                graph.dfpt_internal_strain_ions,
                graph.dfpt_internal_strain_directions,
            )
            total_target = graph.y.squeeze(0)
            ionic_target = graph.y_ionic_piezo.squeeze(0)
            electronic_target = total_target - ionic_target
            values = {
                "born_charge": (components.born_charges, graph.y_born),
                "force_constant": (force_prediction, force_target),
                "internal_strain": (internal_prediction, internal_target),
                "ionic_piezo": (components.ionic_piezo.squeeze(0), ionic_target),
                "electronic_piezo": (components.electronic_piezo.squeeze(0), electronic_target),
                "total_piezo": (components.tensor.squeeze(0), total_target),
                "dielectric": (components.dielectric.squeeze(0), graph.y_dielectric.squeeze(0)),
            }
            row: dict[str, float | int | str] = {
                "material_id": material_id,
                "atoms": graph.num_nodes,
                "printed_internal_strain_blocks": int(graph.dfpt_internal_strain_count.item()),
            }
            true_born = graph.y_born
            predicted_lambda = components.internal_strain
            observed_lambda = replace_printed_internal_strain(
                predicted_lambda,
                graph.dfpt_internal_strain_flat,
                graph.dfpt_internal_strain_ions,
                graph.dfpt_internal_strain_directions,
            )
            volume = torch.linalg.det(graph.cell.reshape(-1, 3, 3)[0]).abs()
            true_optical, _ = optical_eigensystem(force_target)
            minimum_true = float(true_optical.min()) if true_optical.numel() else float("inf")
            stable = minimum_true > model.response.optical_stability_cutoff
            stratum = "stable" if stable else "unstable"
            row["stability_stratum"] = stratum
            mode_row = soft_mode_metrics(
                force_prediction,
                force_target,
                components.born_charges,
                true_born,
                args.soft_mode_count,
            )
            row.update(mode_row)

            oracle = {
                "pred_all_auto": components.ionic_piezo.squeeze(0),
                "true_z_true_phi_pred_lambda_auto": ionic_piezo_from_factors(
                    model.response, true_born, force_target, predicted_lambda, volume, "auto"
                ),
                "pred_z_true_phi_pred_lambda_auto": ionic_piezo_from_factors(
                    model.response, components.born_charges, force_target, predicted_lambda, volume, "auto"
                ),
                "true_z_pred_phi_pred_lambda_auto": ionic_piezo_from_factors(
                    model.response, true_born, force_prediction, predicted_lambda, volume, "auto"
                ),
                "true_z_true_phi_observed_lambda_auto": ionic_piezo_from_factors(
                    model.response, true_born, force_target, observed_lambda, volume, "auto"
                ),
                "true_z_true_phi_pred_lambda_regularized": ionic_piezo_from_factors(
                    model.response, true_born, force_target, predicted_lambda, volume, "regularized"
                ),
            }
            try:
                oracle["true_z_true_phi_pred_lambda_exact"] = ionic_piezo_from_factors(
                    model.response, true_born, force_target, predicted_lambda, volume, "exact"
                )
            except RuntimeError:
                oracle["true_z_true_phi_pred_lambda_exact"] = torch.full_like(ionic_target, torch.nan)

            true_operator = model.response.optical_operator(force_target, "auto")
            predicted_operator = model.response.optical_operator(force_prediction, "auto")
            coupling = model.response._coupling_voigt(predicted_lambda).reshape(3 * graph.num_nodes, 6)
            response_weighted_phi = (
                model.response.PIEZO_C_PER_M2
                * true_born.reshape(-1, 3).T
                @ (predicted_operator - true_operator)
                @ coupling
                / volume
            )
            row["response_weighted_force_constant_mae_c_per_m2"] = float(response_weighted_phi.abs().mean())
            for name, value in oracle.items():
                metric = pair_metrics(value, ionic_target, FACTOR_FLOORS["ionic_piezo"])
                row[f"oracle_{name}_mae"] = metric["component_mae"]
                for target_stratum in ("all", stratum):
                    oracle_values[target_stratum][name].append(value.detach().cpu())
            for target_stratum in ("all", stratum):
                oracle_targets[target_stratum].append(ionic_target.detach().cpu())

            for delta in delta_grid:
                value = ionic_piezo_from_factors(
                    model.response,
                    true_born,
                    force_target,
                    predicted_lambda,
                    volume,
                    "regularized",
                    delta,
                )
                for target_stratum in ("all", stratum):
                    delta_values[target_stratum][delta].append(value.detach().cpu())
            for target_stratum in ("all", stratum):
                delta_targets[target_stratum].append(ionic_target.detach().cpu())
            for name, (prediction, target) in values.items():
                metrics = accumulators[name].add(prediction, target)
                for key, value in metrics.items():
                    row[f"{name}_{key}"] = value
            rows.append(row)
            total_predictions.append(components.tensor.cpu())
            total_targets.append(graph.y.cpu())
            electronic_predictions.append(components.electronic_piezo.cpu())

    total_prediction = torch.cat(total_predictions)
    total_target = torch.cat(total_targets)
    electronic_prediction = torch.cat(electronic_predictions)
    oracle_summary: dict[str, object] = {}
    delta_summary: dict[str, object] = {}
    for stratum in ("all", "stable", "unstable"):
        if not oracle_targets[stratum]:
            oracle_summary[stratum] = {"materials": 0}
            delta_summary[stratum] = {"materials": 0}
            continue
        target = torch.stack(oracle_targets[stratum])
        oracle_summary[stratum] = {
            "materials": target.shape[0],
            "experiments": {
                name: pair_metrics(
                    torch.stack(values), target, FACTOR_FLOORS["ionic_piezo"]
                )
                for name, values in oracle_values[stratum].items()
                if values and torch.isfinite(torch.stack(values)).all()
            },
        }
        delta_target = torch.stack(delta_targets[stratum])
        delta_summary[stratum] = {
            "materials": delta_target.shape[0],
            "regularized_true_z_true_phi_pred_lambda": {
                str(delta): pair_metrics(
                    torch.stack(delta_values[stratum][delta]),
                    delta_target,
                    FACTOR_FLOORS["ionic_piezo"],
                )
                for delta in delta_grid
            },
        }

    summary: dict[str, object] = {
        "schema": 2,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "formula_disjoint": True,
        "material_count": len(dataset),
        "material_ids": [str(record["JARVIS_ID"]) for record in dataset.records],
        "factor_denominator_floors": FACTOR_FLOORS,
        "factors": {name: accumulator.summary() for name, accumulator in accumulators.items()},
        "total_response_skill": response_tensor_skill(total_prediction, total_target),
        "electronic_only_response_skill": response_tensor_skill(electronic_prediction, total_target),
        "zero_response_skill": response_tensor_skill(torch.zeros_like(total_target), total_target),
        "coverage": {
            "atoms": sum(int(row["atoms"]) for row in rows),
            "printed_internal_strain_blocks": sum(int(row["printed_internal_strain_blocks"]) for row in rows),
            "mean_printed_blocks_per_material": _mean_rows(rows, "printed_internal_strain_blocks"),
        },
        "stability": {
            "cutoff_eV_per_A2": model.response.optical_stability_cutoff,
            "stable_materials": sum(row["stability_stratum"] == "stable" for row in rows),
            "unstable_materials": sum(row["stability_stratum"] == "unstable" for row in rows),
        },
        "soft_mode_metrics": {
            key: _mean_rows(rows, key)
            for key in (
                "lowest_optical_eigenvalue_mae",
                "soft_mode_sign_accuracy",
                "soft_mode_subspace_overlap",
                "mode_effective_charge_norm_mae",
                "response_weighted_force_constant_mae_c_per_m2",
            )
        },
        "oracle_factor_replacement": oracle_summary,
        "delta_sensitivity": delta_summary,
        "unavailable_oracles": {
            "true_lambda": (
                "The public archives expose only symmetry-inequivalent OUTCAR internal-strain "
                "blocks, not a complete atom-coordinate Lambda. Missing blocks are not padded, "
                "zero-filled, or symmetry-fabricated."
            ),
            "modewise_strain_coupling_target": "Requires the unavailable complete true Lambda.",
            "true_factor_exact_upper_bound": "Requires the unavailable complete true Lambda.",
        },
    }
    output = args.output or args.checkpoint.parent / f"dfpt_factor_evaluation_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
