"""Read-only capacity-decomposition audits for the physical factor stack.

The first command, ``batch-invariance``, is intentionally a diagnostic rather
than a training mode.  It verifies that variable-size batching does not alter
factor outputs, response action, composite loss, or parameter-group gradients
on a declared same-ID capacity subset.  It never loads a frozen test ID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from .data import PiezoDataset, graph_cache_key, load_gmtnet_records
from .evaluate_dfpt import optical_eigensystem
from .model import AtomCoordinateResponsePotential, PiezoJet, model_from_config
from .operator_losses import (
    born_charge_probe_loss,
    born_oracle_piezo_loss,
    internal_strain_probe_loss,
    ionic_elastic_response_loss,
    low_mode_operator_action_losses,
    mixed_force_constant_probe_loss,
    phi_oracle_normal_equation_loss,
)
from .projectors import translation_projector
from .project_config import load_project_config
from .strain_completion import (
    CartesianSymmetryOperation,
    cartesian_space_group_operations,
    transform_internal_tensor,
)
from .tensor_ops import piezo_voigt_to_cartesian
from .train import (
    born_loss,
    dielectric_loss,
    displacement_macro_piezo_loss,
    displacement_response_target_loss,
    force_constant_loss,
    full_internal_strain_loss,
)


def _relative_tensor_difference(reference: torch.Tensor, observed: torch.Tensor) -> dict[str, float]:
    difference = observed.detach().to(torch.float64) - reference.detach().to(torch.float64)
    denominator = torch.linalg.vector_norm(reference.detach().to(torch.float64)).clamp_min(1e-12)
    return {
        "relative_frobenius": float(torch.linalg.vector_norm(difference) / denominator),
        "max_abs": float(difference.abs().max()),
        "reference_norm": float(denominator),
    }


def _scale_aware_comparison(
    reference: dict[str, dict[str, torch.Tensor]],
    observed: dict[str, dict[str, torch.Tensor]],
    names: list[str],
    metric: str,
) -> dict[str, float]:
    """Report absolute error and reference scale beside a relative maximum."""
    rows = [
        _relative_tensor_difference(reference[name][metric], observed[name][metric])
        for name in names
    ]
    absolute_norms = [
        float(torch.linalg.vector_norm(
            observed[name][metric].detach().to(torch.float64)
            - reference[name][metric].detach().to(torch.float64)
        ))
        for name in names
    ]
    return {
        "max_relative_frobenius": max(row["relative_frobenius"] for row in rows),
        "max_absolute_frobenius": max(absolute_norms),
        "minimum_reference_norm": min(row["reference_norm"] for row in rows),
        "maximum_reference_norm": max(row["reference_norm"] for row in rows),
    }


def _graph_slices(batch, components) -> dict[str, dict[str, torch.Tensor]]:
    """Extract factor and response-action tensors in material-ID keyed form."""
    ptr = batch.ptr
    ionic = batch._response_for_audit.ionic_piezo_from_displacement_response(  # type: ignore[attr-defined]
        batch.y_born, components.displacement_response, batch
    )
    result: dict[str, dict[str, torch.Tensor]] = {}
    force_offset = 0
    for index, material_id in enumerate(batch.material_id):
        start, stop = int(ptr[index]), int(ptr[index + 1])
        atoms, values = stop - start, 9 * (stop - start) ** 2
        force = components.force_constants_flat[force_offset : force_offset + values].reshape(
            atoms, atoms, 3, 3
        )
        force_offset += values
        true_lambda = components.internal_strain.new_zeros(3 * atoms, 6)
        true_charge = batch.y_born[start:stop].reshape(3 * atoms, 3)
        if bool(batch.internal_strain_full_mask[index]):
            true_lambda = components.internal_strain.new_zeros(3 * atoms, 6)
            true_lambda = batch.dfpt_internal_strain_full[start:stop]
            true_lambda = torch.stack(
                (
                    true_lambda[..., 0, 0], true_lambda[..., 1, 1], true_lambda[..., 2, 2],
                    true_lambda[..., 1, 2], true_lambda[..., 0, 2], true_lambda[..., 0, 1],
                ), dim=-1,
            ).reshape(3 * atoms, 6)
        action_rhs = torch.cat((true_lambda, true_charge), dim=1)
        action = components.force_constants_flat.new_zeros(3 * atoms, action_rhs.shape[1])
        if bool(batch.internal_strain_full_mask[index]):
            # Linear Hessian-vector products are the maintained Phi action.
            # They do not differentiate through a resolvent or inverse.
            matrix = batch._response_for_audit._matrix_from_blocks(force)  # type: ignore[attr-defined]
            action = matrix @ action_rhs
        result[str(material_id)] = {
            "born": components.born_charges[start:stop],
            "phi": force,
            "lambda": components.internal_strain[start:stop],
            "u": components.displacement_response[start:stop],
            "true_bec_ionic": ionic[index],
            "phi_action": action,
        }
    return result


def _attach_response(batch, model: PiezoJet):
    """Keep the extraction helper pure with respect to public batch fields."""
    batch._response_for_audit = model.response
    return batch


def _capacity_loss_components(model: PiezoJet, batch) -> dict[str, torch.Tensor]:
    """Equal-weight factor loss used solely to compare batch and single gradients."""
    components = model.predict_components(batch)
    born = born_loss(components.born_charges, batch.y_born, batch.born_mask, batch.batch)
    force = force_constant_loss(
        components.force_constants_flat, batch.dfpt_force_constants_flat, batch.ptr, batch.force_constant_mask
    )
    full_lambda = full_internal_strain_loss(
        components.internal_strain, batch.dfpt_internal_strain_full,
        batch.internal_strain_full_mask, batch.batch,
    )
    u = displacement_response_target_loss(
        components.displacement_response, batch.dfpt_force_constants_flat,
        batch.dfpt_internal_strain_full, batch.internal_strain_full_mask,
        batch.force_constant_mask, batch.ptr, model.response,
    )
    ionic, _ = displacement_macro_piezo_loss(
        components.displacement_response, batch.y_born, batch.y_ionic_piezo,
        batch.ionic_piezo_mask, batch, model.response,
    )
    low_action, low_leak = low_mode_operator_action_losses(
        components.force_constants_flat, batch.dfpt_force_constants_flat,
        batch.ptr, batch.force_constant_mask, mode_count=6,
    )
    phi_probe = mixed_force_constant_probe_loss(
        components.force_constants_flat, batch.dfpt_force_constants_flat,
        batch.dfpt_internal_strain_full, batch.ptr, batch.force_constant_mask,
        batch.internal_strain_full_mask, model.response,
        material_ids=batch.material_id,
    )
    lambda_probe = internal_strain_probe_loss(
        components.internal_strain, batch.dfpt_internal_strain_full,
        batch.internal_strain_full_mask, batch.batch,
        material_ids=batch.material_id,
    )
    bec_probe = born_charge_probe_loss(
        components.born_charges, batch.y_born, batch.born_mask, batch.batch,
        material_ids=batch.material_id,
    )
    oracle_mask = batch.internal_strain_full_mask.reshape(-1) & batch.ionic_piezo_mask.reshape(-1)
    bec_oracle = born_oracle_piezo_loss(
        components.born_charges, batch.dfpt_force_constants_flat,
        batch.dfpt_internal_strain_full, batch.y_ionic_piezo, oracle_mask,
        batch.force_constant_mask, batch.ptr, batch.cell.reshape(-1, 3, 3),
        model.response,
    )
    phi_oracle = phi_oracle_normal_equation_loss(
        components.force_constants_flat, batch.dfpt_force_constants_flat,
        batch.dfpt_internal_strain_full, batch.internal_strain_full_mask,
        batch.force_constant_mask, batch.ptr, model.response,
    )
    ionic_elastic = ionic_elastic_response_loss(
        components.elastic_softening, batch.dfpt_force_constants_flat,
        batch.dfpt_internal_strain_full, batch.internal_strain_full_mask,
        batch.force_constant_mask, batch.ptr, batch.cell.reshape(-1, 3, 3),
        model.response,
    )
    ionic_dielectric = dielectric_loss(
        components.ionic_dielectric,
        batch.y_dfpt_ionic_dielectric,
        batch.dfpt_ionic_dielectric_mask,
    )
    return {
        "born": born,
        "force_constant": force,
        "full_lambda": full_lambda,
        "u_target": u,
        "true_bec_ionic": ionic,
        "low_mode_action": low_action,
        "low_mode_leak": low_leak,
        "phi_probe": phi_probe,
        "lambda_probe": lambda_probe,
        "bec_probe": bec_probe,
        "bec_oracle": bec_oracle,
        "phi_oracle_normal": phi_oracle,
        "ionic_elastic": ionic_elastic,
        "ionic_dielectric": ionic_dielectric,
    }


def _capacity_composite_loss(model: PiezoJet, batch) -> torch.Tensor:
    return sum(_capacity_loss_components(model, batch).values())


def _parameter_groups(model: PiezoJet) -> dict[str, list[torch.nn.Parameter]]:
    """Explicit gradient routes for independent quadratic-energy coefficients."""
    return {
        "shared_encoder": list(model.encoder.parameters()),
        "phi_stiffness_head": list(model.response_factors.edge_stiffness.parameters()),
        "lambda_strain_head": list(model.response_factors.cross_derivative_head.parameters()),
        "shared_phi_lambda_conditioners": list(model.local_polar_mode.parameters()) + list(model.global_context.parameters()),
        "bec_head": list(model.born_head.parameters()),
        "u_head": list(model.displacement_response_head.parameters()),
    }


def _group_gradients(model: PiezoJet, groups: dict[str, list[torch.nn.Parameter]]) -> dict[str, list[torch.Tensor]]:
    return {
        name: [torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]
        for name, parameters in groups.items()
    }


def _component_gradient_norms(
    losses: dict[str, torch.Tensor],
    groups: dict[str, list[torch.nn.Parameter]],
) -> dict[str, dict[str, float]]:
    """Record exact loss-to-parameter-group routes before any weighting."""
    output: dict[str, dict[str, float]] = {}
    for loss_name, loss in losses.items():
        output[loss_name] = {}
        for group_name, parameters in groups.items():
            gradients = torch.autograd.grad(
                loss, parameters, retain_graph=True, allow_unused=True
            )
            squared = loss.new_zeros(())
            for gradient in gradients:
                if gradient is not None:
                    squared = squared + gradient.square().sum()
            output[loss_name][group_name] = float(squared.sqrt().detach())
    return output


def _mean_group_gradients(values: list[dict[str, list[torch.Tensor]]]) -> dict[str, list[torch.Tensor]]:
    return {
        name: [torch.stack([item[name][index] for item in values], dim=0).mean(dim=0) for index in range(len(values[0][name]))]
        for name in values[0]
    }


def _compare_group_gradients(reference: dict[str, list[torch.Tensor]], observed: dict[str, list[torch.Tensor]]) -> dict[str, dict[str, float]]:
    output = {}
    for name, gradients in reference.items():
        ref = torch.cat([value.reshape(-1).to(torch.float64) for value in gradients])
        got = torch.cat([value.reshape(-1).to(torch.float64) for value in observed[name]])
        denominator = torch.linalg.vector_norm(ref).clamp_min(1e-12)
        output[name] = {
            "reference_norm": float(denominator),
            "relative_l2_difference": float(torch.linalg.vector_norm(got - ref) / denominator),
            "cosine": float(torch.dot(ref, got) / (denominator * torch.linalg.vector_norm(got).clamp_min(1e-12))),
        }
    return output


def _load_graphs(config: dict[str, Any], ids_path: Path) -> list:
    ids = json.loads(ids_path.read_text(encoding="utf-8-sig"))
    if not isinstance(ids, list) or not ids:
        raise ValueError("material IDs must be a nonempty JSON list")
    records = load_gmtnet_records(config["data_root"])
    dataset = PiezoDataset(
        records, [str(value) for value in ids], config["cutoff"], config["max_neighbors"],
        processed_dir=config["processed_dir"],
        cache_key=graph_cache_key(records, config["cutoff"], config["max_neighbors"]),
        dfpt_dir=config["jarvis_dfpt_dir"],
        strain_completion_dir=config["jarvis_strain_completion_dir"],
        dfpt_total_consistency_absolute_tolerance=config["dfpt_total_consistency_absolute_tolerance_c_per_m2"],
        dfpt_total_consistency_relative_tolerance=config["dfpt_total_consistency_relative_tolerance"],
    )
    return [dataset[index] for index in range(len(dataset))]


def run_batch_invariance(
    config_path: Path,
    checkpoint_path: Path | None,
    ids_path: Path,
    output: Path,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    config = load_project_config(config_path)
    graphs = _load_graphs(config, ids_path)
    runtime = torch.device(device)
    torch.manual_seed(seed)
    if runtime.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = model_from_config(config).to(runtime)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=runtime, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    model.eval()  # Capacity audit disables stochastic training-time behavior.
    groups = _parameter_groups(model)

    def forward(graph_list: list, gradients: bool = False, routes: bool = False):
        batch = _attach_response(Batch.from_data_list(graph_list).to(runtime), model)
        if gradients:
            model.zero_grad(set_to_none=True)
            loss_components = _capacity_loss_components(model, batch)
            route_norms = _component_gradient_norms(loss_components, groups) if routes else None
            loss = sum(loss_components.values())
            loss.backward()
            return (
                float(loss.detach()), _group_gradients(model, groups),
                {name: float(value.detach()) for name, value in loss_components.items()},
                route_norms,
            )
        with torch.no_grad():
            components = model.predict_components(batch)
            return _graph_slices(batch, components)

    singles = {
        str(graph.material_id): forward([graph])[str(graph.material_id)]
        for graph in graphs
    }
    batched = forward(graphs)
    output_differences = {
        metric: max(_relative_tensor_difference(singles[name][metric], batched[name][metric])["relative_frobenius"] for name in singles)
        for metric in ("born", "phi", "lambda", "u", "true_bec_ionic", "phi_action")
    }
    output_scale_audit = {
        metric: _scale_aware_comparison(singles, batched, list(singles), metric)
        for metric in ("born", "phi", "lambda", "u", "true_bec_ionic", "phi_action")
    }
    permutation = list(reversed(graphs))
    permuted = forward(permutation)
    permutation_differences = {
        metric: max(_relative_tensor_difference(batched[name][metric], permuted[name][metric])["relative_frobenius"] for name in singles)
        for metric in ("born", "phi", "lambda", "u", "true_bec_ionic", "phi_action")
    }
    permutation_scale_audit = {
        metric: _scale_aware_comparison(batched, permuted, list(singles), metric)
        for metric in ("born", "phi", "lambda", "u", "true_bec_ionic", "phi_action")
    }
    duplicate = forward([graphs[0], graphs[0]])[str(graphs[0].material_id)]
    duplicate_differences = {
        metric: _relative_tensor_difference(singles[str(graphs[0].material_id)][metric], duplicate[metric])
        for metric in ("born", "phi", "lambda", "u", "true_bec_ionic", "phi_action")
    }
    size_differences: dict[str, dict[str, float]] = {}
    for size in (2, 4, len(graphs)):
        observed: dict[str, dict[str, torch.Tensor]] = {}
        for start in range(0, len(graphs), size):
            observed.update(forward(graphs[start : start + size]))
        size_differences[str(size)] = {
            metric: max(_relative_tensor_difference(singles[name][metric], observed[name][metric])["relative_frobenius"] for name in singles)
            for metric in ("born", "phi", "lambda", "u", "true_bec_ionic", "phi_action")
        }
    single_rows = [forward([graph], gradients=True) for graph in graphs]
    single_losses = [row[0] for row in single_rows]
    single_gradients = [row[1] for row in single_rows]
    single_components = [row[2] for row in single_rows]
    batch_loss, batch_gradients, batch_components, component_routes = forward(
        graphs, gradients=True, routes=True
    )
    mean_loss = sum(single_losses) / len(single_losses)
    payload = {
        "schema": 2,
        "diagnostic": "batch_decomposition_invariance",
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else f"fresh_initialization_seed_{seed}",
        "material_ids": [str(graph.material_id) for graph in graphs],
        "capacity_mode": {"model_eval": True, "augmentation": "disabled", "dropout": "disabled"},
        "interpretation": "Same-ID implementation diagnostic only; no frozen test IDs are loaded or inspected.",
        "forward_max_relative_frobenius_vs_single": output_differences,
        "forward_scale_audit_vs_single": output_scale_audit,
        "permutation_max_relative_frobenius": permutation_differences,
        "permutation_scale_audit": permutation_scale_audit,
        "duplicate_graph_relative_frobenius": duplicate_differences,
        "batch_size_max_relative_frobenius_vs_single": size_differences,
        "loss": {
            "batch": batch_loss, "mean_single": mean_loss, "absolute_difference": abs(batch_loss - mean_loss),
            "components": {
                name: {
                    "batch": value,
                    "mean_single": sum(row[name] for row in single_components) / len(single_components),
                    "absolute_difference": abs(value - sum(row[name] for row in single_components) / len(single_components)),
                }
                for name, value in batch_components.items()
            },
        },
        "gradient_vs_mean_single": _compare_group_gradients(_mean_group_gradients(list(single_gradients)), batch_gradients),
        "unweighted_component_gradient_norms": component_routes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _apply_atom_permutation(value: torch.Tensor, operation: CartesianSymmetryOperation, atom_axes: tuple[int, ...]) -> torch.Tensor:
    """Place source-atom entries at their symmetry-mapped target indices."""
    output = torch.empty_like(value)
    permutation = operation.permutation.to(device=value.device)
    if atom_axes == (0,):
        output[permutation] = value
    elif atom_axes == (0, 1):
        output[permutation[:, None], permutation[None, :]] = value
    else:  # The audit only has one- and two-atom-index physical factors.
        raise ValueError(f"Unsupported atom axes: {atom_axes}")
    return output


def _transform_born(value: torch.Tensor, operation: CartesianSymmetryOperation) -> torch.Tensor:
    rotation = operation.rotation.to(dtype=value.dtype, device=value.device)
    rotated = torch.einsum("ab,cd,nbd->nac", rotation, rotation, value)
    return _apply_atom_permutation(rotated, operation, (0,))


def _transform_phi(value: torch.Tensor, operation: CartesianSymmetryOperation) -> torch.Tensor:
    rotation = operation.rotation.to(dtype=value.dtype, device=value.device)
    rotated = torch.einsum("ab,cd,nmbd->nmac", rotation, rotation, value)
    return _apply_atom_permutation(rotated, operation, (0, 1))


def _transform_rank3_atom_tensor(value: torch.Tensor, operation: CartesianSymmetryOperation) -> torch.Tensor:
    """Transform ``[atom, vector, strain, strain]`` tensors."""
    return transform_internal_tensor(value, operation)


def _reynolds_projection(
    value: torch.Tensor,
    operations: list[CartesianSymmetryOperation],
    transform,
) -> torch.Tensor:
    # The source and learned tensors are ordinarily float32.  Point-group
    # averaging is a numerical audit, so promote before applying rotations;
    # otherwise an identity operation can spuriously lower the cosine ceiling
    # by float32 norm-rounding alone.
    value = value.to(torch.float64)
    return torch.stack([transform(value, operation) for operation in operations], dim=0).mean(dim=0)


def _symmetry_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    operations: list[CartesianSymmetryOperation],
    transform,
) -> dict[str, float]:
    """Report source symmetry residual and the invariant-subspace cosine ceiling.

    The ceiling is ``||P target||/||target||`` because a crystal-symmetric
    prediction belongs to the Reynolds range.  It prevents source rounding or
    symmetry noise from being misread as learnable error.
    """
    target_projected = _reynolds_projection(target, operations, transform)
    prediction_projected = _reynolds_projection(prediction, operations, transform)
    target_norm = torch.linalg.vector_norm(target.to(torch.float64)).clamp_min(1e-30)
    prediction_norm = torch.linalg.vector_norm(prediction.to(torch.float64)).clamp_min(1e-30)
    return {
        "target_reynolds_relative_residual": float(torch.linalg.vector_norm(target - target_projected) / target_norm),
        "prediction_reynolds_relative_residual": float(torch.linalg.vector_norm(prediction - prediction_projected) / prediction_norm),
        "theoretical_maximum_cosine_from_target_symmetry": float(
            (torch.linalg.vector_norm(target_projected) / target_norm).clamp(max=1.0)
        ),
    }


def _factor_metrics(prediction: torch.Tensor, target: torch.Tensor, *, component_floor: float) -> dict[str, float | bool]:
    prediction = prediction.to(torch.float64)
    target = target.to(torch.float64)
    target_norm = torch.linalg.vector_norm(target)
    prediction_norm = torch.linalg.vector_norm(prediction)
    error_norm = torch.linalg.vector_norm(prediction - target)
    active_threshold = component_floor * (target.numel() ** 0.5)
    return {
        "target_frobenius_norm": float(target_norm),
        "prediction_frobenius_norm": float(prediction_norm),
        "error_frobenius_norm": float(error_norm),
        "relative_frobenius_error": float(error_norm / target_norm.clamp_min(1e-30)),
        "cosine": float(torch.sum(prediction * target) / (prediction_norm * target_norm).clamp_min(1e-30)),
        "amplitude_ratio": float(prediction_norm / target_norm.clamp_min(1e-30)),
        "active_threshold_frobenius": float(active_threshold),
        "active_target": bool(target_norm >= active_threshold),
    }


def _clean_true_phi(blocks: torch.Tensor) -> torch.Tensor:
    atoms = blocks.shape[0]
    matrix = AtomCoordinateResponsePotential._matrix_from_blocks(blocks)
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    projector, _ = translation_projector(atoms, matrix)
    return AtomCoordinateResponsePotential._blocks_from_matrix(projector @ matrix @ projector, atoms)


def _record_metadata(records: list[dict[str, Any]], material_id: str) -> dict[str, Any]:
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    record = next(record for record in records if str(record["JARVIS_ID"]) == material_id)
    atoms = record["atoms"]
    structure = Structure(atoms["lattice_mat"], atoms["elements"], atoms["coords"], coords_are_cartesian=False)
    analyzer = SpacegroupAnalyzer(structure, symprec=1e-5)
    return {
        "record": record,
        "crystal_system": str(analyzer.get_crystal_system()),
        "space_group_number": int(analyzer.get_space_group_number()),
        "space_group_symbol": str(analyzer.get_space_group_symbol()),
    }


def run_per_material_audit(config_path: Path, checkpoint_path: Path, ids_path: Path, output: Path, device: str = "cpu") -> dict[str, Any]:
    """Audit same-ID capacity factors without loading validation or frozen-test rows."""
    config = load_project_config(config_path)
    material_ids = [str(value) for value in json.loads(ids_path.read_text(encoding="utf-8-sig"))]
    graphs = _load_graphs(config, ids_path)
    records = load_gmtnet_records(config["data_root"])
    runtime = torch.device(device)
    model = model_from_config(config).to(runtime)
    checkpoint = torch.load(checkpoint_path, map_location=runtime, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for graph in graphs:
            material_id = str(graph.material_id)
            metadata = _record_metadata(records, material_id)
            operations = cartesian_space_group_operations(metadata["record"])
            batch = _attach_response(Batch.from_data_list([graph]).to(runtime), model)
            components = model.predict_components(batch)
            atoms = int(graph.num_nodes)
            predicted_phi = components.force_constants_flat.reshape(atoms, atoms, 3, 3)
            target_phi = _clean_true_phi(graph.dfpt_force_constants_flat.reshape(atoms, atoms, 3, 3).to(runtime))
            predicted_lambda = components.internal_strain
            target_lambda = graph.dfpt_internal_strain_full.to(runtime)
            predicted_u = components.displacement_response
            target_lambda_voigt = model.response._coupling_voigt(target_lambda).reshape(3 * atoms, 6)
            target_u = model.response.apply_optical_operator(target_phi, target_lambda_voigt, solve_policy="regularized")
            target_u = piezo_voigt_to_cartesian(target_u.reshape(atoms, 3, 6))
            predicted_ionic = model.response.ionic_piezo_from_displacement_response(batch.y_born, components.displacement_response, batch)[0]
            target_ionic = batch.y_ionic_piezo[0]
            eigenvalues, _ = optical_eigensystem(target_phi.detach().cpu())
            scale = {
                "phi_target_frobenius": float(torch.linalg.vector_norm(target_phi)),
                "lambda_target_frobenius": float(torch.linalg.vector_norm(target_lambda)),
                "u_target_frobenius": float(torch.linalg.vector_norm(target_u)),
                "phi_prediction_frobenius": float(torch.linalg.vector_norm(predicted_phi)),
                "lambda_prediction_frobenius": float(torch.linalg.vector_norm(predicted_lambda)),
                "u_prediction_frobenius": float(torch.linalg.vector_norm(predicted_u)),
            }
            rows.append({
                "material_id": material_id,
                "atom_count": atoms,
                "directed_edge_count": int(graph.edge_index.shape[1]),
                "crystal_system": metadata["crystal_system"],
                "space_group_number": metadata["space_group_number"],
                "space_group_symbol": metadata["space_group_symbol"],
                "space_group_operations": len(operations),
                "stability": {
                    "minimum_true_optical_eigenvalue_eV_per_A2": float(eigenvalues.min()) if eigenvalues.numel() else None,
                    "stratum": "stable" if bool((eigenvalues > model.response.optical_stability_cutoff).all()) else ("soft_positive" if bool((eigenvalues > 0).all()) else "unstable"),
                },
                "scales": scale,
                "factors": {
                    "born": {
                        **_factor_metrics(components.born_charges, batch.y_born, component_floor=0.1),
                        **_symmetry_metrics(components.born_charges, batch.y_born, operations, _transform_born),
                    },
                    "phi": {
                        **_factor_metrics(predicted_phi, target_phi, component_floor=0.01),
                        **_symmetry_metrics(predicted_phi, target_phi, operations, _transform_phi),
                    },
                    "lambda": {
                        **_factor_metrics(predicted_lambda, target_lambda, component_floor=0.01),
                        **_symmetry_metrics(predicted_lambda, target_lambda, operations, _transform_rank3_atom_tensor),
                    },
                    "u": {
                        **_factor_metrics(predicted_u, target_u, component_floor=0.001),
                        **_symmetry_metrics(predicted_u, target_u, operations, _transform_rank3_atom_tensor),
                    },
                    "true_bec_ionic": _factor_metrics(predicted_ionic, target_ionic, component_floor=0.05),
                },
            })
    payload = {
        "schema": 1,
        "diagnostic": "capacity_decomposition_per_material_audit",
        "checkpoint": str(checkpoint_path),
        "material_ids": material_ids,
        "capacity_mode": {"model_eval": True, "augmentation": "disabled", "dropout": "disabled"},
        "interpretation": "Same-ID capacity audit only. It uses only the declared material IDs and does not read frozen validation/test labels.",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def summarize_per_material_audits(inputs: list[Path], output: Path) -> dict[str, Any]:
    """Aggregate all and active-only factor metrics without hiding zero targets."""
    result: dict[str, Any] = {
        "schema": 1,
        "diagnostic": "capacity_decomposition_per_material_summary",
        "interpretation": (
            "Cosine is never interpreted alone. All rows retain norm, relative-error, and amplitude values; "
            "active-only summaries exclude targets below the declared factor-resolution floor."
        ),
        "audits": {},
    }
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        factors: dict[str, Any] = {}
        for factor in ("born", "phi", "lambda", "u", "true_bec_ionic"):
            factor_rows = [row["factors"][factor] for row in rows]
            groups = {"all": factor_rows, "active_target": [row for row in factor_rows if row["active_target"]]}
            summary: dict[str, Any] = {}
            for name, group in groups.items():
                values = {
                    metric: [float(row[metric]) for row in group]
                    for metric in (
                        "target_frobenius_norm", "prediction_frobenius_norm", "error_frobenius_norm",
                        "relative_frobenius_error", "cosine", "amplitude_ratio",
                    )
                }
                summary[name] = {
                    "materials": len(group),
                    **{f"mean_{metric}": float(sum(value) / len(value)) if value else None for metric, value in values.items()},
                    **{f"median_{metric}": float(torch.tensor(value).median()) if value else None for metric, value in values.items()},
                }
                if factor != "true_bec_ionic" and group:
                    for metric in (
                        "target_reynolds_relative_residual", "prediction_reynolds_relative_residual",
                        "theoretical_maximum_cosine_from_target_symmetry",
                    ):
                        summary[name][f"mean_{metric}"] = float(sum(float(row[metric]) for row in group) / len(group))
            factors[factor] = summary
        strata = {name: sum(row["stability"]["stratum"] == name for row in rows) for name in ("stable", "soft_positive", "unstable")}
        result["audits"][path.stem] = {
            "input": str(path), "materials": len(rows), "stability_strata": strata, "factors": factors,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _synthetic_factor_losses(prediction, target, ptr: torch.Tensor) -> dict[str, torch.Tensor]:
    """Dimensionless, material-balanced student-to-frozen-teacher losses."""
    losses: dict[str, list[torch.Tensor]] = {name: [] for name in ("born", "phi", "lambda", "u")}
    phi_offset = 0
    for graph_index in range(ptr.numel() - 1):
        start, stop = int(ptr[graph_index]), int(ptr[graph_index + 1])
        atoms = stop - start
        phi_values = 9 * atoms * atoms
        pairs = {
            "born": (prediction.born_charges[start:stop], target["born"][start:stop]),
            "phi": (
                prediction.force_constants_flat[phi_offset : phi_offset + phi_values],
                target["phi"][phi_offset : phi_offset + phi_values],
            ),
            "lambda": (prediction.internal_strain[start:stop], target["lambda"][start:stop]),
            "u": (prediction.displacement_response[start:stop], target["u"][start:stop]),
        }
        phi_offset += phi_values
        for name, (estimated, expected) in pairs.items():
            scale = expected.detach().square().mean().clamp_min(1e-10)
            losses[name].append((estimated - expected).square().mean() / scale)
    return {name: torch.stack(value).mean() for name, value in losses.items()}


def _synthetic_gradient_routes(
    model: PiezoJet, losses: dict[str, torch.Tensor], groups: dict[str, list[torch.nn.Parameter]]
) -> dict[str, dict[str, float]]:
    """State exactly which production parameter groups receive each synthetic loss."""
    output: dict[str, dict[str, float]] = {}
    parameters = [parameter for values in groups.values() for parameter in values]
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, values in groups.items():
        offsets[name] = (cursor, cursor + len(values))
        cursor += len(values)
    for loss_name, loss in losses.items():
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        output[loss_name] = {}
        for group_name, (start, stop) in offsets.items():
            squared = sum(
                (gradient.square().sum() if gradient is not None else loss.new_zeros(()))
                for gradient in gradients[start:stop]
            )
            output[loss_name][group_name] = float(squared.sqrt().detach())
    return output


def run_synthetic_teacher_capacity(
    config_path: Path,
    checkpoint_path: Path,
    ids_path: Path,
    output_root: Path,
    *,
    steps: int = 500,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit a fresh student to frozen current-model factors on 1/8/32 train IDs.

    This is a model-class/optimizer control, not DFPT supervision: the teacher
    is evaluated once in ``eval`` mode and then discarded from the loss graph.
    Consequently an inability to fit cannot be blamed on source convention,
    strict completion, resolvent action, or normal-equation weighting.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    config = load_project_config(config_path)
    all_ids = [str(value) for value in json.loads(ids_path.read_text(encoding="utf-8-sig"))]
    if len(all_ids) < 32:
        raise ValueError("Synthetic teacher ladder needs an ordered 32-material strict-train ID list")
    runtime = torch.device(device)
    teacher = model_from_config(config).to(runtime)
    checkpoint = torch.load(checkpoint_path, map_location=runtime, weights_only=False)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    outputs: dict[str, Any] = {}
    for count in (1, 8, 32):
        subset = all_ids[:count]
        subset_path = output_root / f"samples{count}_ids.json"
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_path.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
        graphs = _load_graphs(config, subset_path)
        batch = _attach_response(Batch.from_data_list(graphs).to(runtime), teacher)
        with torch.no_grad():
            teacher_prediction = teacher.predict_components(batch)
            target = {
                "born": teacher_prediction.born_charges.detach(),
                "phi": teacher_prediction.force_constants_flat.detach(),
                "lambda": teacher_prediction.internal_strain.detach(),
                "u": teacher_prediction.displacement_response.detach(),
            }
        student = model_from_config(config).to(runtime)
        student.train()
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
        groups = _parameter_groups(student)
        initial_routes: dict[str, dict[str, float]] | None = None
        history: list[dict[str, float]] = []
        checkpoints = {0, steps - 1} | set(range(max(1, steps // 10) - 1, steps, max(1, steps // 10)))
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = student.predict_components(batch)
            losses = _synthetic_factor_losses(prediction, target, batch.ptr)
            if initial_routes is None:
                initial_routes = _synthetic_gradient_routes(student, losses, groups)
            total = sum(losses.values())
            total.backward()
            optimizer.step()
            if step in checkpoints:
                history.append({"step": step + 1, "total": float(total.detach()), **{name: float(value.detach()) for name, value in losses.items()}})
        student.eval()
        with torch.no_grad():
            prediction = student.predict_components(batch)
        rows: list[dict[str, Any]] = []
        phi_offset = 0
        for index, material_id in enumerate(subset):
            start, stop = int(batch.ptr[index]), int(batch.ptr[index + 1])
            atoms = stop - start
            phi_values = 9 * atoms * atoms
            rows.append({
                "material_id": material_id,
                "born": _factor_metrics(prediction.born_charges[start:stop], target["born"][start:stop], component_floor=0.0),
                "phi": _factor_metrics(prediction.force_constants_flat[phi_offset : phi_offset + phi_values], target["phi"][phi_offset : phi_offset + phi_values], component_floor=0.0),
                "lambda": _factor_metrics(prediction.internal_strain[start:stop], target["lambda"][start:stop], component_floor=0.0),
                "u": _factor_metrics(prediction.displacement_response[start:stop], target["u"][start:stop], component_floor=0.0),
            })
            phi_offset += phi_values
        outputs[f"samples{count}"] = {
            "material_ids": subset,
            "steps": steps,
            "learning_rate": learning_rate,
            "loss_history": history,
            "initial_gradient_routes": initial_routes,
            "rows": rows,
        }
    payload = {
        "schema": 1,
        "diagnostic": "frozen_current_model_synthetic_teacher_capacity",
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_mode": "eval_frozen",
        "student": "fresh production-model-class initialization",
        "capacity_mode": {"augmentation": "disabled", "dropout": "disabled", "normal_equation": "off", "resolvent_action": "off"},
        "interpretation": "Synthetic same-ID model-class/optimization control only; no DFPT labels or frozen validation/test IDs are read.",
        "probes": outputs,
    }
    output = output_root / "synthetic_teacher_capacity.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    batch = subparsers.add_parser("batch-invariance")
    batch.add_argument("--config", type=Path, required=True)
    batch.add_argument(
        "--checkpoint", type=Path,
        help="Optional current-schema checkpoint; omit to audit a fresh seeded initialization",
    )
    batch.add_argument("--material-ids-file", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--device", default="cpu")
    batch.add_argument("--seed", type=int, default=42)
    audit = subparsers.add_parser("per-material-audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--material-ids-file", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--device", default="cpu")
    summary = subparsers.add_parser("summarize-per-material")
    summary.add_argument("--inputs", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    synthetic = subparsers.add_parser("synthetic-teacher")
    synthetic.add_argument("--config", type=Path, required=True)
    synthetic.add_argument("--checkpoint", type=Path, required=True)
    synthetic.add_argument("--material-ids-file", type=Path, required=True)
    synthetic.add_argument("--output-root", type=Path, required=True)
    synthetic.add_argument("--steps", type=int, default=500)
    synthetic.add_argument("--learning-rate", type=float, default=1e-3)
    synthetic.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "batch-invariance":
        print(json.dumps(run_batch_invariance(
            args.config, args.checkpoint, args.material_ids_file, args.output,
            args.device, args.seed,
        ), indent=2))
    elif args.command == "per-material-audit":
        print(json.dumps(run_per_material_audit(args.config, args.checkpoint, args.material_ids_file, args.output, args.device), indent=2))
    elif args.command == "summarize-per-material":
        print(json.dumps(summarize_per_material_audits(args.inputs, args.output), indent=2))
    elif args.command == "synthetic-teacher":
        print(json.dumps(run_synthetic_teacher_capacity(
            args.config, args.checkpoint, args.material_ids_file, args.output_root,
            steps=args.steps, learning_rate=args.learning_rate, device=args.device,
        ), indent=2))


if __name__ == "__main__":
    main()
