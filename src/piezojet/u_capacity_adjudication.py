"""Same-ID displacement-response capacity adjudication.

This diagnostic never reads frozen validation/test IDs.  It freezes the DFPT
``Z*``, ``Phi``, and ``Lambda`` labels and trains only a structure-to-``U``
model, separating representation capacity from factor-learning error.  Fresh
output directories are mandatory so negative results remain auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .data import PiezoDataset, graph_cache_key, load_gmtnet_records
from .model import (
    AtomCoordinateResponsePotential,
    CartesianInternalStrainHead,
    CartesianLocalEnvironmentEncoder,
    CartesianPolarReadout,
    CrystalGlobalContext,
    GlobalDisplacementResponseHead,
    OctupoleGlobalDisplacementResponseHead,
)
from .project_config import load_project_config


class UCapacityModel(nn.Module):
    """One controlled local/global ``U`` capacity model."""

    def __init__(self, config: dict, architecture: str, first_order_auxiliary: bool):
        super().__init__()
        if architecture not in {"local", "global", "global_l3"}:
            raise ValueError("architecture must be local, global, or global_l3")
        self.architecture = architecture
        encoder_kwargs = {
            key: config[key]
            for key in (
                "embedding_dim", "cutoff", "num_blocks", "radial_basis",
                "radial_hidden", "cartesian_channels",
            )
        }
        self.encoder = CartesianLocalEnvironmentEncoder(**encoder_kwargs)
        scalar_dim, channels = self.encoder.scalar_dim, self.encoder.channels
        if architecture == "local":
            self.head = CartesianInternalStrainHead(scalar_dim, channels)
            self.auxiliary_head = (
                CartesianInternalStrainHead(scalar_dim, channels)
                if first_order_auxiliary else None
            )
            self.local_polar = None
            self.global_context = None
        else:
            context_dim = int(config["global_context_dim"])
            self.local_polar = CartesianPolarReadout(scalar_dim, channels)
            self.global_context = CrystalGlobalContext(
                context_dim,
                int(config["spectral_channels"]),
                int(config["spectral_shells"]),
                int(config["polar_fluctuation_shells"]),
                float(config["reciprocal_cutoff"]),
            )
            head_kwargs = dict(
                scalar_dim=scalar_dim,
                channels=channels,
                context_dim=context_dim,
                attention_dim=int(config.get("displacement_attention_dim", 64)),
                cross_rank=int(config.get("displacement_cross_rank", 24)),
            )
            if architecture == "global_l3":
                head_kwargs.update(
                    radial_basis=int(config["radial_basis"]),
                    radial_hidden=int(config["radial_hidden"]),
                    cutoff=float(config["cutoff"]),
                )
                head_type = OctupoleGlobalDisplacementResponseHead
            else:
                head_type = GlobalDisplacementResponseHead
            self.head = head_type(**head_kwargs)
            self.auxiliary_head = (
                head_type(**head_kwargs)
                if first_order_auxiliary else None
            )

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.encoder(batch)
        if self.architecture == "local":
            real = self.head(features, batch.batch)
            imaginary = (
                None if self.auxiliary_head is None
                else self.auxiliary_head(features, batch.batch)
            )
            return real, imaginary
        assert self.local_polar is not None and self.global_context is not None
        local_polar = self.local_polar(features)
        context, operator = self.global_context(
            batch, batch.batch, local_polar, return_operator=True
        )
        if self.architecture == "global_l3":
            real = self.head(features, batch, context, operator)
        else:
            real = self.head(features, batch.batch, context, operator)
        imaginary = (
            None if self.auxiliary_head is None
            else (
                self.auxiliary_head(features, batch, context, operator)
                if self.architecture == "global_l3"
                else self.auxiliary_head(features, batch.batch, context, operator)
            )
        )
        return real, imaginary


def _matrix_to_response_tensor(matrix: torch.Tensor, atoms: int) -> torch.Tensor:
    values = matrix.reshape(atoms, 3, 6)
    tensor = matrix.new_zeros(atoms, 3, 3, 3)
    tensor[..., 0, 0] = values[..., 0]
    tensor[..., 1, 1] = values[..., 1]
    tensor[..., 2, 2] = values[..., 2]
    tensor[..., 1, 2] = tensor[..., 2, 1] = values[..., 3]
    tensor[..., 0, 2] = tensor[..., 2, 0] = values[..., 4]
    tensor[..., 0, 1] = tensor[..., 1, 0] = values[..., 5]
    return tensor


def _true_graph_tensors(batch, response: AtomCoordinateResponsePotential) -> list[dict[str, torch.Tensor]]:
    force_offset = 0
    result: list[dict[str, torch.Tensor]] = []
    for graph_index in range(batch.num_graphs):
        start, stop = int(batch.ptr[graph_index]), int(batch.ptr[graph_index + 1])
        atoms = stop - start
        values = 9 * atoms * atoms
        if not bool(batch.force_constant_mask[graph_index]):
            raise ValueError("Capacity adjudication requires a true Phi for every graph")
        if not bool(batch.internal_strain_full_mask[graph_index]):
            raise ValueError("Capacity adjudication requires a strict Lambda for every graph")
        blocks = batch.dfpt_force_constants_flat[
            force_offset : force_offset + values
        ].reshape(atoms, atoms, 3, 3)
        force_offset += values
        matrix = response._matrix_from_blocks(blocks)
        coupling = response._coupling_voigt(
            batch.dfpt_internal_strain_full[start:stop]
        ).reshape(3 * atoms, 6)
        target_u = response.apply_optical_operator(
            blocks, coupling, solve_policy="regularized"
        )
        result.append(
            {
                "phi": matrix,
                "lambda": coupling,
                "u": target_u,
                "u_tensor": _matrix_to_response_tensor(target_u, atoms),
            }
        )
    if force_offset != batch.dfpt_force_constants_flat.numel():
        raise ValueError("Ragged true Phi payload did not match graph boundaries")
    return result


def _cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sum(left * right) / (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right) + eps
    )


def _losses(
    predicted: torch.Tensor,
    imaginary: torch.Tensor | None,
    batch,
    truths: list[dict[str, torch.Tensor]],
    response: AtomCoordinateResponsePotential,
    consistency: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    direct_terms, ionic_terms, consistency_terms = [], [], []
    predicted_ionic = response.ionic_piezo_from_displacement_response(
        batch.y_born, predicted, batch
    )
    true_u_tensor = torch.cat([item["u_tensor"] for item in truths], dim=0)
    target_ionic = response.ionic_piezo_from_displacement_response(
        batch.y_born, true_u_tensor, batch
    )
    for graph_index, truth in enumerate(truths):
        start, stop = int(batch.ptr[graph_index]), int(batch.ptr[graph_index + 1])
        atoms = stop - start
        u = response._coupling_voigt(predicted[start:stop]).reshape(3 * atoms, 6)
        target_u = truth["u"]
        direct_terms.append(
            (u - target_u).square().sum() / target_u.square().sum().clamp_min(1e-8)
        )
        ionic_prediction = predicted_ionic[graph_index]
        ionic_target = target_ionic[graph_index]
        target_norm = torch.linalg.vector_norm(ionic_target).clamp_min(1e-8)
        # The capacity objective must remain finite and order-one at the
        # physically common near-zero initialization.  A raw log-amplitude
        # penalty has an artificial singularity there and previously made the
        # true-BEC contraction dominate the direct U target by roughly 50x.
        # Relative squared error already couples direction and amplitude and
        # has value one for the zero prediction.
        ionic_terms.append(
            (ionic_prediction - ionic_target).square().sum()
            / target_norm.square()
        )
        if consistency == "none":
            continue
        phi, coupling = truth["phi"], truth["lambda"]
        if consistency == "squared_normal":
            residual = phi @ (phi @ u) + response.optical_regularization**2 * u
            residual = residual - phi @ coupling
            scale = torch.linalg.vector_norm(phi @ coupling).square().clamp_min(1e-8)
            consistency_terms.append(residual.square().sum() / scale)
        elif consistency == "first_order":
            if imaginary is None:
                raise ValueError("first_order consistency requires an auxiliary V head")
            v = response._coupling_voigt(imaginary[start:stop]).reshape(3 * atoms, 6)
            delta = response.optical_regularization
            residual_real = phi @ u - delta * v - coupling
            residual_imaginary = phi @ v + delta * u
            scale = coupling.square().sum().clamp_min(1e-8)
            consistency_terms.append(
                (residual_real.square().sum() + residual_imaginary.square().sum()) / scale
            )
        else:
            raise ValueError(f"Unsupported consistency: {consistency}")
    direct = torch.stack(direct_terms).mean() + 0.25 * torch.stack(ionic_terms).mean()
    constraint = (
        predicted.sum() * 0.0
        if not consistency_terms else torch.stack(consistency_terms).mean()
    )
    return direct, constraint, {
        "u_relative_mse": torch.stack(direct_terms).mean(),
        "ionic_relative_mse": torch.stack(ionic_terms).mean(),
    }


def _gradient_vector(loss: torch.Tensor, parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    parameters = tuple(parameters)
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    flat = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(flat) if flat else loss.new_zeros(1)


@torch.no_grad()
def _metrics(
    model: UCapacityModel,
    batch,
    truths: list[dict[str, torch.Tensor]],
    response: AtomCoordinateResponsePotential,
) -> dict[str, float | int]:
    model.eval()
    predicted, _ = model(batch)
    true_tensor = torch.cat([item["u_tensor"] for item in truths], dim=0)
    predicted_ionic = response.ionic_piezo_from_displacement_response(
        batch.y_born, predicted, batch
    )
    true_ionic = response.ionic_piezo_from_displacement_response(
        batch.y_born, true_tensor, batch
    )
    u_relative, u_cosine, ionic_relative, ionic_cosine, ionic_amplitude = [], [], [], [], []
    active_cosine, active_amplitude = [], []
    active_floor = 0.05 * math.sqrt(18.0)
    for graph_index, truth in enumerate(truths):
        start, stop = int(batch.ptr[graph_index]), int(batch.ptr[graph_index + 1])
        atoms = stop - start
        observed = response._coupling_voigt(predicted[start:stop]).reshape(3 * atoms, 6)
        target = truth["u"]
        target_norm = torch.linalg.vector_norm(target).clamp_min(1e-12)
        u_relative.append(torch.linalg.vector_norm(observed - target) / target_norm)
        u_cosine.append(_cosine(observed, target))
        ionic_target = true_ionic[graph_index]
        ionic_observed = predicted_ionic[graph_index]
        norm = torch.linalg.vector_norm(ionic_target).clamp_min(1e-12)
        amplitude = torch.linalg.vector_norm(ionic_observed) / norm
        ionic_relative.append(torch.linalg.vector_norm(ionic_observed - ionic_target) / norm)
        ionic_cosine.append(_cosine(ionic_observed, ionic_target))
        ionic_amplitude.append(amplitude)
        if float(norm) >= active_floor:
            active_cosine.append(ionic_cosine[-1])
            active_amplitude.append(amplitude)

    def mean(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean()) if values else float("nan")

    return {
        "materials": batch.num_graphs,
        "u_relative_frobenius_error": mean(u_relative),
        "u_cosine": mean(u_cosine),
        "true_bec_ionic_relative_frobenius_error": mean(ionic_relative),
        "true_bec_ionic_cosine": mean(ionic_cosine),
        "true_bec_ionic_amplitude_ratio": mean(ionic_amplitude),
        "active_materials": len(active_cosine),
        "active_true_bec_ionic_cosine": mean(active_cosine),
        "active_true_bec_ionic_amplitude_ratio": mean(active_amplitude),
    }


def run(args: argparse.Namespace) -> dict:
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"Fresh output directory required: {output}")
    output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    config = load_project_config(args.config)
    ids = json.loads(args.material_ids_file.read_text(encoding="utf-8-sig"))
    if len(ids) != 32:
        raise ValueError("Capacity adjudication is restricted to the declared 32 IDs")
    records = load_gmtnet_records(config["data_root"])
    dataset = PiezoDataset(
        records,
        [str(value) for value in ids],
        float(config["cutoff"]),
        int(config["max_neighbors"]),
        processed_dir=config["processed_dir"],
        cache_key=graph_cache_key(
            records, float(config["cutoff"]), int(config["max_neighbors"])
        ),
        dfpt_dir=config["jarvis_dfpt_dir"],
        strain_completion_dir=config["jarvis_strain_completion_dir"],
        elastic_targets_path=config.get("elastic_targets_path"),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    device = torch.device(args.device)
    batch = batch.to(device)
    response = AtomCoordinateResponsePotential(
        optical_regularization=float(config["optical_regularization"]),
        optical_stability_cutoff=float(config["optical_stability_cutoff"]),
        optical_solve_policy="regularized",
    ).to(device)
    truths = _true_graph_tensors(batch, response)
    model = UCapacityModel(
        config, args.architecture, args.consistency == "first_order"
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    u_parameters = tuple(model.head.parameters())
    rows: list[dict[str, float | int]] = []
    best_metric = float("inf")
    best_epoch = 0
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        predicted, imaginary = model(batch)
        direct, constraint, components = _losses(
            predicted, imaginary, batch, truths, response, args.consistency
        )
        direct_gradient = _gradient_vector(direct, u_parameters)
        if args.consistency == "none":
            constraint_gradient = torch.zeros_like(direct_gradient)
        else:
            constraint_gradient = _gradient_vector(constraint, u_parameters)
        direct_norm = torch.linalg.vector_norm(direct_gradient)
        constraint_norm = torch.linalg.vector_norm(constraint_gradient)
        gradient_cosine = _cosine(direct_gradient, constraint_gradient)
        if args.consistency != "none":
            effective_weight = min(
                args.consistency_weight,
                args.max_consistency_gradient_ratio
                * float(direct_norm.detach())
                / max(float(constraint_norm.detach()), 1e-12),
            )
        else:
            effective_weight = args.consistency_weight
        loss = direct + effective_weight * constraint
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "direct_loss": float(direct.detach()),
            "constraint_loss": float(constraint.detach()),
            "effective_constraint_weight": effective_weight,
            "u_head_direct_gradient_norm": float(direct_norm.detach()),
            "u_head_constraint_gradient_norm": float(constraint_norm.detach()),
            "u_head_gradient_cosine": float(gradient_cosine.detach()),
            **{name: float(value.detach()) for name, value in components.items()},
        }
        rows.append(row)
        if row["direct_loss"] < best_metric:
            best_metric = float(row["direct_loss"])
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, output / "best.pt")
        if epoch == 1 or epoch % args.report_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch} loss={row['loss']:.6g} direct={row['direct_loss']:.6g} "
                f"constraint={row['constraint_loss']:.6g} weight={effective_weight:.3g}"
            )
    saved = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    final_metrics = _metrics(model, batch, truths, response)
    gate = {
        "u_relative_frobenius_error_below_0_20": final_metrics["u_relative_frobenius_error"] < 0.20,
        "u_cosine_above_0_95": final_metrics["u_cosine"] > 0.95,
        "active_ionic_cosine_above_0_95": final_metrics["active_true_bec_ionic_cosine"] > 0.95,
        "active_ionic_amplitude_in_0_8_1_2": 0.8 < final_metrics["active_true_bec_ionic_amplitude_ratio"] < 1.2,
    }
    summary = {
        "schema": 2,
        "diagnostic": "true_factor_u_capacity_adjudication",
        "selection": "declared strict-train samples32 only; frozen validation/test not read",
        "architecture": args.architecture,
        "consistency": args.consistency,
        "epochs": args.epochs,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.perf_counter() - start_time,
        "graph_cache_schema": 4,
        "neighbor_policy": "distance-shell-complete target budget",
        "objective": (
            "mean material-relative U squared error + 0.25 mean true-BEC "
            "ionic relative squared error"
        ),
        "consistency_gradient_policy": (
            "actual U-head consistency/direct gradient ratio capped at "
            f"{args.max_consistency_gradient_ratio:g}"
        ),
        "metrics": final_metrics,
        "capacity_gate": gate,
        "capacity_gate_passed": all(gate.values()),
    }
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--material-ids-file", type=Path,
        default=Path("data/processed/capacity_probe_ids/samples32_ids.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("local", "global", "global_l3"), required=True
    )
    parser.add_argument(
        "--consistency", choices=("none", "squared_normal", "first_order"), required=True
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--max-consistency-gradient-ratio", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs <= 0 or args.report_every <= 0:
        raise ValueError("epochs and report_every must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
