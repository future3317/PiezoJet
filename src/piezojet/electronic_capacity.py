"""Train-only same-ID capacity probe for the electronic piezo branch."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from .data import PiezoDataset, graph_cache_key, load_gmtnet_records
from .model import (
    NonlinearDifferentialPolarizationTower,
    CartesianLocalEnvironmentEncoder,
    CartesianPiezoTensorHead,
    CartesianPolarReadout,
    CrystalGlobalContext,
    ElectromechanicalJetPrediction,
    ElectromechanicalJetHead,
)
from .project_config import load_project_config
from .tensor_ops import (
    PIEZO_IRREP_SLICES,
    cartesian_to_piezo_voigt,
    piezo_to_irreps,
    voigt_to_symmetric_matrix,
)
from .train import seed_everything


def _read_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("material_ids", payload.get("ids"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Capacity material-ID file must contain a non-empty list")
    return [str(value) for value in payload]


class CurrentElectronicCapacityModel(nn.Module):
    """The maintained electronic encoder/readout, isolated from other tasks."""

    def __init__(self, config: dict[str, object]):
        super().__init__()
        encoder_kwargs = dict(
            embedding_dim=int(config["embedding_dim"]),
            cutoff=float(config["cutoff"]),
            num_blocks=int(config["num_blocks"]),
            radial_basis=int(config["radial_basis"]),
            radial_hidden=int(config["radial_hidden"]),
            cartesian_channels=int(config.get("cartesian_channels", 48)),
        )
        self.encoder = CartesianLocalEnvironmentEncoder(**encoder_kwargs)
        self.head = CartesianPiezoTensorHead(
            self.encoder.scalar_dim, self.encoder.channels
        )
        self.local_polar = CartesianPolarReadout(
            self.encoder.scalar_dim, self.encoder.channels
        )
        self.global_context = CrystalGlobalContext(
            int(config["global_context_dim"]),
            int(config["spectral_channels"]),
            int(config["spectral_shells"]),
            int(config["polar_fluctuation_shells"]),
            float(config["reciprocal_cutoff"]),
        )

    def forward(self, batch) -> torch.Tensor:
        features = self.encoder(batch)
        local_polar = self.local_polar(features)
        _, spectral_operator = self.global_context(
            batch, batch.batch, local_polar, return_operator=True
        )
        tensor = self.head(features, batch.batch) + spectral_operator
        return 0.5 * (tensor + tensor.transpose(-1, -2))


def electromechanical_jet_from_config(
    config: dict[str, object],
) -> ElectromechanicalJetHead:
    return ElectromechanicalJetHead(
        embedding_dim=int(config["embedding_dim"]),
        cutoff=float(config["cutoff"]),
        lmax=max(3, int(config.get("lmax", 3))),
        num_blocks=int(config["num_blocks"]),
        radial_basis=int(config["radial_basis"]),
        radial_hidden=int(config["radial_hidden"]),
        global_context_dim=int(config["global_context_dim"]),
        spectral_channels=int(config["spectral_channels"]),
        spectral_shells=int(config["spectral_shells"]),
        polar_fluctuation_shells=int(config["polar_fluctuation_shells"]),
        reciprocal_cutoff=float(config["reciprocal_cutoff"]),
        attention_dim=int(config.get("global_attention_dim", 64)),
    )


def nonlinear_polarization_from_config(
    config: dict[str, object],
) -> NonlinearDifferentialPolarizationTower:
    return NonlinearDifferentialPolarizationTower(
        polarization_variable=str(config["polarization_variable"]),
        embedding_dim=int(config["embedding_dim"]),
        cutoff=float(config["cutoff"]),
        lmax=max(3, int(config.get("lmax", 3))),
        num_blocks=int(config["num_blocks"]),
        radial_basis=int(config["radial_basis"]),
        radial_hidden=int(config["radial_hidden"]),
        global_context_dim=int(config["global_context_dim"]),
        spectral_channels=int(config["spectral_channels"]),
        spectral_shells=int(config["spectral_shells"]),
        polar_fluctuation_shells=int(config["polar_fluctuation_shells"]),
        reciprocal_cutoff=float(config["reciprocal_cutoff"]),
        attention_dim=int(config.get("global_attention_dim", 64)),
    )


def _response_coefficients(
    model: nn.Module,
    batch,
    architecture: str,
    *,
    create_graph: bool,
) -> ElectromechanicalJetPrediction:
    if architecture in {"nonlinear_cartesian", "nonlinear_reduced"}:
        return model.coefficients(batch, create_graph=create_graph)
    return model.coefficients(batch)


def _capacity_batches(dataset, batch_size: int, device: torch.device) -> list:
    effective = len(dataset) if batch_size <= 0 else min(batch_size, len(dataset))
    return [
        value.to(device)
        for value in DataLoader(
            dataset, batch_size=effective, shuffle=False, num_workers=0
        )
    ]


def _evaluate_capacity_model(
    model: nn.Module,
    batches: list,
    architecture: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    born_predictions: list[torch.Tensor] = []
    born_targets: list[torch.Tensor] = []
    global_batch_indices: list[torch.Tensor] = []
    graph_offset = 0
    model.eval()
    context = (
        torch.no_grad()
        if architecture in {"nonlinear_cartesian", "nonlinear_reduced"}
        else torch.inference_mode()
    )
    with context:
        for batch in batches:
            if architecture == "current":
                prediction = model(batch)
                coefficients = None
            else:
                coefficients = _response_coefficients(
                    model, batch, architecture, create_graph=False
                )
                prediction = coefficients.electronic_piezo
            predictions.append(prediction.detach())
            targets.append(batch.y_electronic_piezo)
            if coefficients is not None:
                born_predictions.append(coefficients.born_charges.detach())
                born_targets.append(batch.y_born)
                global_batch_indices.append(batch.batch + graph_offset)
            graph_offset += int(batch.num_graphs)
    return (
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(born_predictions) if born_predictions else None,
        torch.cat(born_targets) if born_targets else None,
        torch.cat(global_batch_indices) if global_batch_indices else None,
    )


def irrep_balanced_capacity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    floor: float = 0.05,
) -> torch.Tensor:
    """Give each material and each of the four irreducible blocks one vote."""
    predicted = piezo_to_irreps(prediction)
    expected = piezo_to_irreps(target)
    losses: list[torch.Tensor] = []
    for block in PIEZO_IRREP_SLICES.values():
        residual = predicted[..., block] - expected[..., block]
        target_norm = torch.linalg.vector_norm(expected[..., block], dim=-1)
        denominator = target_norm.clamp_min(floor * expected[..., block].shape[-1] ** 0.5)
        losses.append(
            residual.square().sum(dim=-1) / denominator.square()
        )
    return torch.stack(losses, dim=-1).mean()


def electronic_capacity_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    floor: float = 0.05,
) -> dict[str, object]:
    predicted = piezo_to_irreps(prediction.detach()).to(torch.float64)
    expected = piezo_to_irreps(target.detach()).to(torch.float64)
    residual = predicted - expected
    target_norm = torch.linalg.vector_norm(expected, dim=-1)
    predicted_norm = torch.linalg.vector_norm(predicted, dim=-1)
    floor_norm = floor * expected.shape[-1] ** 0.5
    active = target_norm >= floor_norm
    cosine = (predicted * expected).sum(dim=-1) / (
        predicted_norm * target_norm
    ).clamp_min(1e-30)
    relative = torch.linalg.vector_norm(residual, dim=-1) / target_norm.clamp_min(
        floor_norm
    )
    amplitude = predicted_norm / target_norm.clamp_min(floor_norm)
    raw_relative = torch.linalg.vector_norm(residual, dim=-1) / target_norm.clamp_min(1e-30)
    per_irrep: dict[str, dict[str, float | int]] = {}
    for name, block in PIEZO_IRREP_SLICES.items():
        block_prediction = predicted[..., block]
        block_target = expected[..., block]
        block_prediction_norm = torch.linalg.vector_norm(block_prediction, dim=-1)
        block_target_norm = torch.linalg.vector_norm(block_target, dim=-1)
        block_floor = floor * block_target.shape[-1] ** 0.5
        block_active = block_target_norm >= block_floor
        block_cosine = (block_prediction * block_target).sum(dim=-1) / (
            block_prediction_norm * block_target_norm
        ).clamp_min(1e-30)
        block_relative = torch.linalg.vector_norm(
            block_prediction - block_target, dim=-1
        ) / block_target_norm.clamp_min(block_floor)
        per_irrep[name] = {
            "active_materials": int(block_active.sum()),
            "mean_stabilized_relative_error": float(block_relative.mean()),
            "mean_active_cosine": (
                float(block_cosine[block_active].mean())
                if bool(block_active.any())
                else float("nan")
            ),
        }
    return {
        "materials": int(target.shape[0]),
        "active_materials": int(active.sum()),
        "active_threshold_norm_c_per_m2": floor_norm,
        "mean_stabilized_relative_frobenius_error": float(relative.mean()),
        "mean_active_relative_frobenius_error": (
            float(relative[active].mean()) if bool(active.any()) else float("nan")
        ),
        "mean_active_cosine": (
            float(cosine[active].mean()) if bool(active.any()) else float("nan")
        ),
        "mean_stabilized_amplitude_ratio": float(amplitude.mean()),
        "per_material": [
            {
                "target_norm_c_per_m2": float(target_norm[index]),
                "prediction_norm_c_per_m2": float(predicted_norm[index]),
                "active": bool(active[index]),
                "raw_relative_frobenius_error": float(raw_relative[index]),
                "stabilized_relative_frobenius_error": float(relative[index]),
                "cosine": float(cosine[index]),
                "stabilized_amplitude_ratio": float(amplitude[index]),
            }
            for index in range(target.shape[0])
        ],
        "per_irrep": per_irrep,
    }


def born_material_balanced_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch_index: torch.Tensor,
    floor_per_component: float = 0.1,
) -> torch.Tensor:
    """One normalized full-tensor vote per material for neutral BECs."""
    graphs = int(batch_index.max()) + 1
    residual_energy = torch.zeros(graphs, dtype=target.dtype, device=target.device)
    target_energy = torch.zeros_like(residual_energy)
    component_count = torch.zeros_like(residual_energy)
    residual_energy.index_add_(
        0, batch_index, (prediction - target).square().sum(dim=(-1, -2))
    )
    target_energy.index_add_(0, batch_index, target.square().sum(dim=(-1, -2)))
    component_count.index_add_(
        0, batch_index, torch.full_like(batch_index, 9, dtype=target.dtype)
    )
    denominator = target_energy.clamp_min(
        floor_per_component**2 * component_count
    )
    return (residual_energy / denominator).mean()


def born_capacity_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch_index: torch.Tensor,
) -> dict[str, object]:
    graphs = int(batch_index.max()) + 1
    rows: list[dict[str, float | int]] = []
    for graph_index in range(graphs):
        mask = batch_index == graph_index
        predicted = prediction[mask].detach().to(torch.float64).reshape(-1)
        expected = target[mask].detach().to(torch.float64).reshape(-1)
        predicted_norm = torch.linalg.vector_norm(predicted)
        target_norm = torch.linalg.vector_norm(expected)
        residual_norm = torch.linalg.vector_norm(predicted - expected)
        cosine = (predicted * expected).sum() / (
            predicted_norm * target_norm
        ).clamp_min(1e-30)
        acoustic = prediction[mask].detach().to(torch.float64).sum(dim=0)
        rows.append(
            {
                "atoms": int(mask.sum()),
                "target_norm_e": float(target_norm),
                "prediction_norm_e": float(predicted_norm),
                "relative_frobenius_error": float(
                    residual_norm / target_norm.clamp_min(1e-30)
                ),
                "cosine": float(cosine),
                "acoustic_sum_norm_e": float(torch.linalg.vector_norm(acoustic)),
            }
        )
    return {
        "materials": graphs,
        "mean_relative_frobenius_error": sum(
            float(row["relative_frobenius_error"]) for row in rows
        ) / graphs,
        "mean_cosine": sum(float(row["cosine"]) for row in rows) / graphs,
        "mean_acoustic_sum_norm_e": sum(
            float(row["acoustic_sum_norm_e"]) for row in rows
        ) / graphs,
        "per_material": rows,
    }


def response_jet_probe_loss(
    prediction: ElectromechanicalJetPrediction,
    target: ElectromechanicalJetPrediction,
    batch,
    probes: int = 3,
    floor_c_per_m2: float = 0.05,
) -> torch.Tensor:
    r"""Unbiased displacement/strain action supervision for the response jet.

    Standard-normal translation-free displacement and engineering-strain
    probes satisfy identity covariance on their physical subspaces. Therefore
    expected squared action error equals the corresponding operator Frobenius
    error. Material normalization is computed from the full target operator,
    not from a possibly cancelling individual probe response.
    """
    if probes < 1:
        raise ValueError("response-jet probe count must be positive")
    graphs = prediction.electronic_piezo.shape[0]
    dtype, device = batch.pos.dtype, batch.pos.device
    zero_displacement = torch.zeros_like(batch.pos)
    zero_strain = torch.zeros(graphs, 3, 3, dtype=dtype, device=device)
    displacement_error = torch.zeros(graphs, dtype=dtype, device=device)
    strain_error = torch.zeros_like(displacement_error)
    for _ in range(probes):
        displacement = torch.randn_like(batch.pos)
        displacement = displacement - scatter(
            displacement, batch.batch, dim=0, dim_size=graphs, reduce="mean"
        )[batch.batch]
        predicted_u = ElectromechanicalJetHead.polarization_increment_from_coefficients(
            prediction, displacement, zero_strain, batch
        )
        target_u = ElectromechanicalJetHead.polarization_increment_from_coefficients(
            target, displacement, zero_strain, batch
        )
        displacement_error = displacement_error + (predicted_u - target_u).square().sum(dim=-1)

        eta6 = torch.randn(graphs, 6, dtype=dtype, device=device)
        strain = voigt_to_symmetric_matrix(eta6)
        predicted_eta = ElectromechanicalJetHead.polarization_increment_from_coefficients(
            prediction, zero_displacement, strain, batch
        )
        target_eta = ElectromechanicalJetHead.polarization_increment_from_coefficients(
            target, zero_displacement, strain, batch
        )
        strain_error = strain_error + (predicted_eta - target_eta).square().sum(dim=-1)
    displacement_error = displacement_error / probes
    strain_error = strain_error / probes

    cells = batch.cell.reshape(graphs, 3, 3)
    volume = torch.linalg.det(cells).abs().clamp_min(torch.finfo(dtype).eps)
    born_energy = scatter(
        target.born_charges.square().sum(dim=(-1, -2)),
        batch.batch,
        dim=0,
        dim_size=graphs,
        reduce="sum",
    )
    born_energy = born_energy * (ElectromechanicalJetHead.PIEZO_C_PER_M2 / volume).square()
    electronic_energy = cartesian_to_piezo_voigt(
        target.electronic_piezo
    ).square().sum(dim=(-1, -2))
    floor = 3.0 * floor_c_per_m2**2
    return 0.5 * (
        displacement_error / born_energy.clamp_min(floor)
        + strain_error / electronic_energy.clamp_min(floor)
    ).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--material-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument(
        "--architecture",
        choices=(
            "current", "electromechanical_jet",
            "nonlinear_cartesian", "nonlinear_reduced",
        ),
        default="current",
    )
    parser.add_argument("--bec-weight", type=float, default=1.0)
    parser.add_argument("--jet-weight", type=float, default=0.0)
    parser.add_argument("--jet-probes", type=int, default=3)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument(
        "--train-batch-size", type=int, default=0,
        help="Materials per gradient-accumulation microbatch; 0 uses the full cohort",
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if (
        args.epochs < 1 or args.learning_rate <= 0 or args.log_interval < 1
        or args.bec_weight < 0 or args.jet_weight < 0 or args.jet_probes < 1
        or args.checkpoint_interval < 1 or args.train_batch_size < 0
    ):
        raise ValueError("epochs, learning rate, and log interval must be positive")
    response_architectures = (
        "electromechanical_jet", "nonlinear_cartesian", "nonlinear_reduced"
    )
    if args.architecture not in response_architectures and (
        args.bec_weight != 1.0 or args.jet_weight != 0.0
    ):
        raise ValueError(
            "BEC/jet weights are configurable only for response-generator architectures"
        )
    config = load_project_config(args.config)
    config["seed"] = args.seed
    seed_everything(args.seed)
    device = torch.device(args.device)
    ids = _read_ids(args.material_ids_file)
    records = load_gmtnet_records(config["data_root"])
    cache_key = graph_cache_key(
        records, float(config["cutoff"]), int(config["max_neighbors"])
    )
    dataset = PiezoDataset(
        records,
        ids,
        float(config["cutoff"]),
        int(config["max_neighbors"]),
        processed_dir=config["processed_dir"],
        cache_key=cache_key,
        project_targets=True,
        dfpt_dir=config["jarvis_dfpt_dir"],
        strain_completion_dir=config["jarvis_strain_completion_dir"],
    )
    for index in range(len(dataset)):
        graph = dataset[index]
        if not bool(graph.dfpt_branch_mask):
            raise ValueError(
                f"Same-ID electronic capacity requires a verified branch label: {ids[index]}"
            )
    batches = _capacity_batches(dataset, args.train_batch_size, device)
    batch = batches[0]
    if args.architecture == "current":
        model = CurrentElectronicCapacityModel(config)
    elif args.architecture in {"nonlinear_cartesian", "nonlinear_reduced"}:
        config["polarization_variable"] = (
            "reduced" if args.architecture == "nonlinear_reduced" else "cartesian"
        )
        model = nonlinear_polarization_from_config(config)
    else:
        model = electromechanical_jet_from_config(config)
    model = model.to(device)
    if device.type == "cuda" and (
        not batch.pos.is_cuda or not next(model.parameters()).is_cuda
    ):
        raise RuntimeError("CUDA capacity run has CPU-resident batch or parameters")
    initialization = "random_e3nn_l3"
    if args.architecture == "current":
        pretrained = torch.load(
            Path(str(config["pretrained_encoder"])), map_location=device, weights_only=False
        )
        model.encoder.load_state_dict(pretrained["encoder"], strict=True)
        initialization = str(config["pretrained_encoder"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-6
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    history: list[dict[str, float | int]] = []
    prior_training_seconds = 0.0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("architecture") != args.architecture:
            raise ValueError("Resume checkpoint architecture does not match CLI")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        history = list(saved.get("history", []))
        prior_training_seconds = float(saved.get("training_seconds", 0.0))
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        if start_epoch > args.epochs:
            raise ValueError("Resume checkpoint already exceeds requested epochs")
    # Populate non-learned geometry caches and initialize CUDA kernels before
    # measuring optimizer throughput.  No parameters or running statistics are
    # changed by this inference-only warmup.
    model.eval()
    # ``no_grad`` (rather than inference_mode) is intentional: cached geometry
    # is reused by the subsequent backward pass and therefore must remain a
    # normal tensor even though it never requires gradients.
    with torch.no_grad():
        if args.architecture == "current":
            model(batch)
        else:
            _response_coefficients(
                model, batch, args.architecture, create_graph=False
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_electronic_loss = 0.0
        epoch_bec_loss = 0.0
        epoch_jet_loss = 0.0
        for training_batch in batches:
            target = training_batch.y_electronic_piezo
            target_born = training_batch.y_born
            if args.architecture == "current":
                electronic_prediction = model(training_batch)
                coefficient_prediction = None
            else:
                coefficient_prediction = _response_coefficients(
                    model, training_batch, args.architecture, create_graph=True
                )
                electronic_prediction = coefficient_prediction.electronic_piezo
            electronic_loss = irrep_balanced_capacity_loss(
                electronic_prediction, target
            )
            loss = electronic_loss
            bec_loss = electronic_loss.new_zeros(())
            jet_loss = electronic_loss.new_zeros(())
            if args.architecture in response_architectures:
                assert coefficient_prediction is not None
                true_coefficients = ElectromechanicalJetPrediction(
                    target_born, target, None
                )
                bec_loss = born_material_balanced_loss(
                    coefficient_prediction.born_charges,
                    target_born,
                    training_batch.batch,
                )
                if args.jet_weight > 0:
                    jet_loss = response_jet_probe_loss(
                        coefficient_prediction,
                        true_coefficients,
                        training_batch,
                        args.jet_probes,
                    )
                loss = (
                    loss + args.bec_weight * bec_loss
                    + args.jet_weight * jet_loss
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite electronic capacity loss")
            material_weight = int(training_batch.num_graphs) / len(dataset)
            (material_weight * loss).backward()
            epoch_loss += material_weight * float(loss.detach())
            epoch_electronic_loss += material_weight * float(electronic_loss.detach())
            epoch_bec_loss += material_weight * float(bec_loss.detach())
            epoch_jet_loss += material_weight * float(jet_loss.detach())
        optimizer.step()
        row = {
            "epoch": epoch,
            "loss": epoch_loss,
            "electronic_loss": epoch_electronic_loss,
            "bec_loss": epoch_bec_loss,
            "jet_loss": epoch_jet_loss,
        }
        history.append(row)
        if epoch % args.checkpoint_interval == 0 and epoch < args.epochs:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = prior_training_seconds + time.perf_counter() - training_started
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "architecture": args.architecture,
                    "history": history,
                    "training_seconds": elapsed,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                    ),
                    "config": config,
                },
                args.output_dir / "latest.pt",
            )
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            model.eval()
            evaluated, evaluated_target, _, _, _ = _evaluate_capacity_model(
                model, batches, args.architecture
            )
            metrics = electronic_capacity_metrics(evaluated, evaluated_target)
            print(
                f"epoch={epoch} loss={epoch_loss:.6g} "
                f"rel={float(metrics['mean_stabilized_relative_frobenius_error']):.6g} "
                f"cos={float(metrics['mean_active_cosine']):.6g}"
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = (
        prior_training_seconds + time.perf_counter() - training_started
    )
    (
        final_prediction, final_target, final_born, final_born_target,
        final_batch_index,
    ) = _evaluate_capacity_model(model, batches, args.architecture)
    final_metrics = electronic_capacity_metrics(final_prediction, final_target)
    born_metrics = (
        born_capacity_metrics(
            final_born, final_born_target, final_batch_index
        )
        if args.architecture in response_architectures and final_born is not None
        else None
    )
    torch.save(
        {"model": model.state_dict(), "epoch": args.epochs, "config": config},
        args.output_dir / "final.pt",
    )
    has_active_targets = int(final_metrics["active_materials"]) > 0
    strong_pass = (
        float(final_metrics["mean_active_relative_frobenius_error"]) < 0.2
        and float(final_metrics["mean_active_cosine"]) > 0.95
        if has_active_targets
        else None
    )
    practical_pass = (
        float(final_metrics["mean_active_relative_frobenius_error"]) <= 0.25
        and float(final_metrics["mean_active_cosine"]) >= 0.90
        if has_active_targets
        else None
    )
    born_strong_pass = (
        float(born_metrics["mean_relative_frobenius_error"]) < 0.2
        and float(born_metrics["mean_cosine"]) > 0.95
        if born_metrics is not None
        else None
    )
    born_practical_pass = (
        float(born_metrics["mean_relative_frobenius_error"]) <= 0.25
        and float(born_metrics["mean_cosine"]) >= 0.90
        if born_metrics is not None
        else None
    )
    summary = {
        "schema": 1,
        "protocol": (
            "J1_literal_nonlinear_differential_polarization_with_response_jet"
            if args.architecture in {"nonlinear_cartesian", "nonlinear_reduced"} and args.jet_weight > 0
            else "J1_first_order_electromechanical_jet_with_redundant_probes"
            if args.architecture == "electromechanical_jet" and args.jet_weight > 0
            else {
                "current": "E2_current_electronic_same_id_capacity",
                "electromechanical_jet": (
                    "A1_exact_first_order_electromechanical_jet_same_id_capacity"
                ),
                "nonlinear_cartesian": (
                    "A2_literal_nonlinear_cartesian_polarization_same_id_capacity"
                ),
                "nonlinear_reduced": (
                    "A3_literal_nonlinear_reduced_polarization_same_id_capacity"
                ),
            }[args.architecture]
        ),
        "selection": "declared strict-train capacity IDs only; frozen validation/test not read",
        "architecture": args.architecture,
        "initialization": initialization,
        "material_ids": ids,
        "epochs": args.epochs,
        "selected_epoch": args.epochs,
        "checkpoint_selection": "none; fixed-epoch same-ID capacity endpoint",
        "resumed_from": str(args.resume) if args.resume is not None else None,
        "seed": args.seed,
        "history": history,
        "runtime_device": str(device),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-6,
        },
        "runtime": {
            "optimizer_seconds": training_seconds,
            "optimizer_steps_per_second": args.epochs / training_seconds,
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else None
            ),
            "num_workers": 0,
            "training_microbatch_materials": (
                len(dataset) if args.train_batch_size <= 0
                else min(args.train_batch_size, len(dataset))
            ),
            "gradient_accumulation_microbatches": len(batches),
            "gradient_accumulation_reduction": "material-count-weighted exact cohort mean",
            "batch_device": str(batch.pos.device),
            "parameter_device": str(next(model.parameters()).device),
            "fixed_geometry_cache": args.architecture not in {"nonlinear_cartesian", "nonlinear_reduced"},
            "differentiable_geometry_cache": (
                "disabled_to_avoid_retaining_second_order_graphs"
                if args.architecture in {"nonlinear_cartesian", "nonlinear_reduced"}
                else "not_applicable"
            ),
            "reciprocal_execution": "padded batched GEMM/einsum",
        },
        "loss": "four-irrep material-balanced normalized squared error",
        "loss_weights": {
            "electronic": 1.0,
            "bec": args.bec_weight if args.architecture in response_architectures else 0.0,
            "response_jet": args.jet_weight if args.architecture in response_architectures else 0.0,
            "jet_probes_per_displacement_and_strain": args.jet_probes,
        },
        "metrics": final_metrics,
        "born_metrics": born_metrics,
        "gate": {
            "strong_threshold": "active relative error <0.2 and active cosine >0.95",
            "practical_threshold": "active relative error <=0.25 and active cosine >=0.90",
            "strong_pass": strong_pass,
            "practical_pass": practical_pass,
            "status": (
                "evaluated_on_active_targets"
                if has_active_targets
                else "not_applicable_no_active_targets"
            ),
            "born_strong_threshold": "mean relative error <0.2 and mean cosine >0.95",
            "born_practical_threshold": "mean relative error <=0.25 and mean cosine >=0.90",
            "born_strong_pass": born_strong_pass,
            "born_practical_pass": born_practical_pass,
            "joint_strong_pass": (
                bool(strong_pass and born_strong_pass)
                if born_strong_pass is not None and strong_pass is not None
                else None
            ),
            "joint_practical_pass": (
                bool(practical_pass and born_practical_pass)
                if born_practical_pass is not None and practical_pass is not None
                else None
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()
