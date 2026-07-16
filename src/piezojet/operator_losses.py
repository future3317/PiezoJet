"""Gauge-safe response-operator supervision for atom-coordinate factors.

These losses use true DFPT directions as fixed probes.  They never match an
individual predicted eigenvector, never materialize a predicted inverse, and
never route gradients through a true factor.  The functions are kept separate
from the trainer so their numerical and gradient contracts can be tested in
isolation.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Sequence

import torch

from .model import AtomCoordinateResponsePotential
from .elastic_dielectric_ops import elastic_voigt_to_cartesian
from .tensor_ops import piezo_voigt_to_cartesian


def _matrix(blocks: torch.Tensor) -> torch.Tensor:
    matrix = AtomCoordinateResponsePotential._matrix_from_blocks(blocks)
    return 0.5 * (matrix + matrix.transpose(0, 1))


def _clean_true_matrix(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = _matrix(blocks)
    atoms = int(blocks.shape[0])
    basis = AtomCoordinateResponsePotential._optical_basis(atoms, matrix)
    if basis.shape[1] == 0:
        return matrix.new_zeros(matrix.shape), basis
    return basis @ (basis.transpose(0, 1) @ matrix @ basis) @ basis.transpose(0, 1), basis


def _floor_tensor(reference: torch.Tensor, floor: float | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(floor, dtype=reference.dtype, device=reference.device)


def _normalized_squared(
    prediction: torch.Tensor,
    target: torch.Tensor,
    floor: float | torch.Tensor = 1e-8,
) -> torch.Tensor:
    denominator = target.square().sum().clamp_min(_floor_tensor(target, floor).square())
    return (prediction - target).square().sum() / denominator


def _normalized_energy(
    value: torch.Tensor,
    reference: torch.Tensor,
    floor: float | torch.Tensor = 1e-8,
) -> torch.Tensor:
    denominator = reference.square().sum().clamp_min(_floor_tensor(reference, floor).square())
    return value.square().sum() / denominator


def _normalized_pseudo_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    floor: float | torch.Tensor = 1e-8,
) -> torch.Tensor:
    return torch.sqrt(1.0 + _normalized_squared(prediction, target, floor)) - 1.0


def _energy_pseudo_huber(
    value: torch.Tensor,
    reference: torch.Tensor,
    floor: float | torch.Tensor = 1e-8,
) -> torch.Tensor:
    return torch.sqrt(1.0 + _normalized_energy(value, reference, floor)) - 1.0


def _spectral_action_floor(
    true_operator: torch.Tensor,
    optical_basis: torch.Tensor,
    columns: int,
    fraction: float = 0.01,
) -> torch.Tensor:
    """Return a material-relative absolute floor for operator actions.

    Relative normalization by ``||Phi X||`` is ill-conditioned when ``X``
    contains a genuine soft or symmetry-inactive direction.  The floor is one
    percent of the median nonzero true optical spectral scale per unit probe,
    multiplied by ``sqrt(columns)`` so it has the Frobenius units of the
    complete action.  This is invariant to atom count, batching, and rotations.
    """
    if columns <= 0 or optical_basis.shape[1] == 0:
        return true_operator.new_tensor(1e-8)
    reduced = optical_basis.transpose(0, 1) @ true_operator @ optical_basis
    spectrum = torch.linalg.eigvalsh(0.5 * (reduced + reduced.transpose(0, 1))).abs()
    active = spectrum[spectrum > 1e-8]
    material_scale = active.median() if active.numel() else spectrum.new_tensor(1e-8)
    return (
        material_scale.clamp_min(1e-8)
        * float(fraction)
        * float(columns) ** 0.5
    )


def _material_seed(material_id: str | None, atoms: int, salt: str) -> int:
    text = f"{material_id or 'unknown'}:{atoms}:{salt}"
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:8], "little") % (2**31)


def _random_columns(rows: int, columns: int, reference: torch.Tensor, seed: int) -> torch.Tensor:
    if columns <= 0:
        return reference.new_empty(rows, 0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    return values.to(dtype=reference.dtype, device=reference.device)


def _unit_columns(values: torch.Tensor, tolerance: float = 1e-12) -> torch.Tensor:
    if values.numel() == 0:
        return values
    norms = torch.linalg.vector_norm(values, dim=0)
    keep = norms > tolerance
    return values[:, keep] / norms[keep].unsqueeze(0) if bool(keep.any()) else values[:, :0]


def _true_and_predicted_blocks(
    prediction_flat: torch.Tensor,
    target_flat: torch.Tensor,
    node_ptr: torch.Tensor,
    mask: torch.Tensor,
):
    """Yield ragged predicted/all and true/masked blocks with checked offsets."""
    mask = mask.reshape(-1).to(dtype=torch.bool)
    prediction_offset = target_offset = 0
    for graph_index in range(node_ptr.numel() - 1):
        atoms = int(node_ptr[graph_index + 1] - node_ptr[graph_index])
        values = 9 * atoms * atoms
        predicted = prediction_flat[prediction_offset : prediction_offset + values].reshape(atoms, atoms, 3, 3)
        prediction_offset += values
        target = None
        if bool(mask[graph_index]):
            target = target_flat[target_offset : target_offset + values].reshape(atoms, atoms, 3, 3)
            target_offset += values
        yield graph_index, atoms, predicted, target
    if prediction_offset != prediction_flat.numel():
        raise ValueError("Predicted force constants did not match graph boundaries")
    if target_offset != target_flat.numel():
        raise ValueError("True force constants did not match labelled graph boundaries")


def low_mode_operator_action_losses(
    prediction_flat: torch.Tensor,
    target_flat: torch.Tensor,
    node_ptr: torch.Tensor,
    mask: torch.Tensor,
    mode_count: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direct low-mode action and hard-subspace leakage losses.

    True optical eigenvectors are ordered by ``|lambda|`` because these are
    the response-active directions of the signed continuous regularized
    operator.  Only the invariant projector/subspace is used; predicted
    eigenvectors are never selected or matched.
    """
    if mode_count < 1:
        raise ValueError("mode_count must be positive")
    action_losses: list[torch.Tensor] = []
    leakage_losses: list[torch.Tensor] = []
    for _, _, predicted_blocks, target_blocks in _true_and_predicted_blocks(
        prediction_flat, target_flat, node_ptr, mask
    ):
        if target_blocks is None:
            continue
        predicted = _matrix(predicted_blocks)
        target, optical_basis = _clean_true_matrix(target_blocks)
        if optical_basis.shape[1] == 0:
            continue
        eigenvalues, reduced_vectors = torch.linalg.eigh(optical_basis.transpose(0, 1) @ target @ optical_basis)
        order = torch.argsort(eigenvalues.abs())[: min(mode_count, eigenvalues.numel())]
        low_vectors = (optical_basis @ reduced_vectors[:, order]).detach()
        target_action = target @ low_vectors
        predicted_action = predicted @ low_vectors
        floor = _spectral_action_floor(target, optical_basis, low_vectors.shape[1])
        action_losses.append(_normalized_pseudo_huber(predicted_action, target_action, floor))
        projected_back = low_vectors @ (low_vectors.transpose(0, 1) @ predicted_action)
        leakage_losses.append(
            _energy_pseudo_huber(predicted_action - projected_back, target_action, floor)
        )
    zero = prediction_flat.sum() * 0.0
    action = torch.stack(action_losses).mean() if action_losses else zero
    leakage = torch.stack(leakage_losses).mean() if leakage_losses else zero
    return action, leakage


def mixed_force_constant_probe_loss(
    prediction_flat: torch.Tensor,
    target_flat: torch.Tensor,
    true_internal_strain: torch.Tensor,
    node_ptr: torch.Tensor,
    force_mask: torch.Tensor,
    strict_mask: torch.Tensor,
    response: AtomCoordinateResponsePotential,
    material_ids: Sequence[str] | None = None,
    random_count: int = 4,
    low_mode_count: int = 3,
    displacement_count: int = 3,
) -> torch.Tensor:
    """Supervise direct ``Phi X`` action on 40/30/30 probe families.

    On non-strict records the unavailable ``U*`` family is omitted rather
    than fabricated; random-optical and true-low-mode families retain their
    relative 4:3 weighting.  Strict records add columns from
    ``D_delta(Phi_true) Lambda_true``.  Each family is normalized separately,
    so the declared mixture is not changed by units or column count.
    """
    if min(random_count, low_mode_count, displacement_count) < 0:
        raise ValueError("probe counts must be non-negative")
    strict_mask = strict_mask.reshape(-1).to(dtype=torch.bool)
    losses: list[torch.Tensor] = []
    for graph_index, atoms, predicted_blocks, target_blocks in _true_and_predicted_blocks(
        prediction_flat, target_flat, node_ptr, force_mask
    ):
        if target_blocks is None:
            continue
        predicted = _matrix(predicted_blocks)
        target, basis = _clean_true_matrix(target_blocks)
        if basis.shape[1] == 0:
            continue
        identifier = None if material_ids is None else str(material_ids[graph_index])
        reduced_random = _random_columns(
            basis.shape[1], random_count, target,
            _material_seed(identifier, atoms, "phi-random-optical"),
        )
        random_probes = _unit_columns(basis @ reduced_random)
        eigenvalues, reduced_vectors = torch.linalg.eigh(basis.transpose(0, 1) @ target @ basis)
        order = torch.argsort(eigenvalues.abs())[: min(low_mode_count, eigenvalues.numel())]
        low_probes = _unit_columns(basis @ reduced_vectors[:, order])
        families: list[tuple[float, torch.Tensor]] = [(0.4, random_probes), (0.3, low_probes)]
        if bool(strict_mask[graph_index]) and displacement_count:
            start, stop = int(node_ptr[graph_index]), int(node_ptr[graph_index + 1])
            coupling = response._coupling_voigt(true_internal_strain[start:stop].detach()).reshape(3 * atoms, 6)
            u_true = response.apply_optical_operator(target_blocks.detach(), coupling, solve_policy="regularized")
            if u_true.shape[1] > displacement_count:
                # Select the dominant strain-response directions without
                # assigning a physical phonon-mode meaning to the SVD rank.
                left, _, _ = torch.linalg.svd(u_true, full_matrices=False)
                u_true = left[:, :displacement_count]
            families.append((0.3, _unit_columns(u_true)))
        available = [(weight, probes) for weight, probes in families if probes.shape[1] > 0]
        weight_sum = sum(weight for weight, _ in available)
        if weight_sum:
            losses.append(sum(
                (weight / weight_sum) * _normalized_pseudo_huber(
                    predicted @ probes,
                    target @ probes,
                    _spectral_action_floor(target, basis, probes.shape[1]),
                )
                for weight, probes in available
            ))
    return torch.stack(losses).mean() if losses else prediction_flat.sum() * 0.0


def internal_strain_probe_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    graph_mask: torch.Tensor,
    batch_index: torch.Tensor,
    material_ids: Sequence[str] | None = None,
    probe_count: int = 6,
) -> torch.Tensor:
    """Supervise ``Lambda eta`` on deterministic random unit strain probes."""
    graph_mask = graph_mask.reshape(-1).to(dtype=torch.bool)
    losses: list[torch.Tensor] = []
    for graph_index in torch.nonzero(graph_mask, as_tuple=False).reshape(-1):
        index = int(graph_index)
        selected = batch_index == graph_index
        atoms = int(selected.sum())
        identifier = None if material_ids is None else str(material_ids[index])
        eta = _unit_columns(_random_columns(6, probe_count, prediction, _material_seed(identifier, atoms, "lambda-strain")))
        predicted_matrix = AtomCoordinateResponsePotential._coupling_voigt(prediction[selected]).reshape(3 * atoms, 6)
        target_matrix = AtomCoordinateResponsePotential._coupling_voigt(target[selected].detach()).reshape(3 * atoms, 6)
        losses.append(_normalized_pseudo_huber(predicted_matrix @ eta, target_matrix @ eta))
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def born_charge_probe_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    batch_index: torch.Tensor,
    material_ids: Sequence[str] | None = None,
    probe_count: int = 3,
) -> torch.Tensor:
    """Supervise ``Z* E`` with material-balanced deterministic field probes."""
    node_mask = node_mask.reshape(-1).to(dtype=torch.bool)
    graphs = int(batch_index.max()) + 1
    losses: list[torch.Tensor] = []
    for graph_index in range(graphs):
        selected = node_mask & (batch_index == graph_index)
        if not bool(selected.any()):
            continue
        atoms = int(selected.sum())
        identifier = None if material_ids is None else str(material_ids[graph_index])
        field = _unit_columns(_random_columns(3, probe_count, prediction, _material_seed(identifier, atoms, "born-field")))
        true = target[selected].detach()
        true = true - true.mean(dim=0, keepdim=True)
        losses.append(_normalized_pseudo_huber(prediction[selected] @ field, true @ field))
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def born_oracle_piezo_loss(
    predicted_born: torch.Tensor,
    true_force_constants_flat: torch.Tensor,
    true_internal_strain: torch.Tensor,
    target_ionic_piezo: torch.Tensor,
    strict_mask: torch.Tensor,
    force_mask: torch.Tensor,
    node_ptr: torch.Tensor,
    cells: torch.Tensor,
    response: AtomCoordinateResponsePotential,
) -> torch.Tensor:
    """Isolate BEC response error using true ``Phi``, true ``Lambda`` and ``U*``."""
    strict_mask = strict_mask.reshape(-1).to(dtype=torch.bool)
    force_mask = force_mask.reshape(-1).to(dtype=torch.bool)
    losses: list[torch.Tensor] = []
    true_offset = 0
    for graph_index in range(node_ptr.numel() - 1):
        start, stop = int(node_ptr[graph_index]), int(node_ptr[graph_index + 1])
        atoms, values = stop - start, 9 * (stop - start) ** 2
        blocks = None
        if bool(force_mask[graph_index]):
            blocks = true_force_constants_flat[true_offset : true_offset + values].reshape(atoms, atoms, 3, 3)
            true_offset += values
        if not bool(strict_mask[graph_index]):
            continue
        if blocks is None:
            raise ValueError("BEC oracle response requires true force constants")
        coupling = response._coupling_voigt(true_internal_strain[start:stop].detach()).reshape(3 * atoms, 6)
        u_true = response.apply_optical_operator(blocks.detach(), coupling, solve_policy="regularized")
        charge = predicted_born[start:stop].reshape(3 * atoms, 3)
        volume = torch.linalg.det(cells[graph_index]).abs().clamp_min(torch.finfo(cells.dtype).eps)
        predicted = response.PIEZO_C_PER_M2 * charge.transpose(0, 1) @ u_true / volume
        target = target_ionic_piezo[graph_index]
        predicted_cartesian = piezo_voigt_to_cartesian(predicted)
        denominator = target.square().sum().clamp_min(target.new_tensor(0.05).square() * target.numel())
        losses.append(torch.sqrt(1.0 + (predicted_cartesian - target).square().sum() / denominator) - 1.0)
    if true_offset != true_force_constants_flat.numel():
        raise ValueError("True force constants did not match BEC-oracle boundaries")
    return torch.stack(losses).mean() if losses else predicted_born.sum() * 0.0


def phi_oracle_normal_equation_loss(
    prediction_flat: torch.Tensor,
    true_force_constants_flat: torch.Tensor,
    true_internal_strain: torch.Tensor,
    strict_mask: torch.Tensor,
    force_mask: torch.Tensor,
    node_ptr: torch.Tensor,
    response: AtomCoordinateResponsePotential,
) -> torch.Tensor:
    """Isolate ``Phi`` with true ``Lambda`` and true regularized ``U*``.

    Residuals are resolved in true optical modes, normalized by their local
    equation scale, and averaged equally across four fixed ``|lambda|/delta``
    regions.  This prevents numerous hard modes from drowning the soft-mode
    residual while avoiding a predicted eigendecomposition.
    """
    strict_mask = strict_mask.reshape(-1).to(dtype=torch.bool)
    force_mask = force_mask.reshape(-1).to(dtype=torch.bool)
    delta = float(response.optical_regularization)
    delta2 = delta * delta
    losses: list[torch.Tensor] = []
    for graph_index, atoms, predicted_blocks, target_blocks in _true_and_predicted_blocks(
        prediction_flat, true_force_constants_flat, node_ptr, force_mask
    ):
        if not bool(strict_mask[graph_index]):
            continue
        if target_blocks is None:
            raise ValueError("Phi oracle normal equation requires true force constants")
        start, stop = int(node_ptr[graph_index]), int(node_ptr[graph_index + 1])
        coupling = response._coupling_voigt(true_internal_strain[start:stop].detach()).reshape(3 * atoms, 6)
        u_true = response.apply_optical_operator(target_blocks.detach(), coupling, solve_policy="regularized")
        predicted = _matrix(predicted_blocks)
        true, basis = _clean_true_matrix(target_blocks)
        if basis.shape[1] == 0:
            continue
        eigenvalues, reduced_vectors = torch.linalg.eigh(basis.transpose(0, 1) @ true @ basis)
        modes = basis @ reduced_vectors
        residual = predicted @ (predicted @ u_true) + delta2 * u_true - predicted @ coupling
        residual_modes = modes.transpose(0, 1) @ residual
        u_modes = modes.transpose(0, 1) @ u_true
        lambda_modes = modes.transpose(0, 1) @ coupling
        scale = (
            (eigenvalues.square() + delta2) * torch.linalg.vector_norm(u_modes, dim=1)
            + eigenvalues.abs() * torch.linalg.vector_norm(lambda_modes, dim=1)
        )
        # A mode with essentially zero Lambda and U projection is not response
        # active; dividing by its roundoff-level equation scale would create a
        # catastrophic gradient unrelated to the observable.  Use a declared
        # one-percent material spectral floor, then robustify the normalized
        # residual.  This changes optimization conditioning, not the target.
        positive = scale[scale > 1e-8]
        material_floor = (
            0.01 * positive.median() if positive.numel()
            else scale.new_tensor(1e-8)
        )
        scale = scale.clamp_min(material_floor.clamp_min(1e-8))
        normalized_energy = residual_modes.square().sum(dim=1) / scale.square()
        normalized_penalty = torch.sqrt(1.0 + normalized_energy) - 1.0
        absolute = eigenvalues.abs()
        regions = (
            absolute < delta,
            (absolute >= delta) & (absolute < 3.0 * delta),
            (absolute >= 3.0 * delta) & (absolute < 10.0 * delta),
            absolute >= 10.0 * delta,
        )
        region_losses = [normalized_penalty[region].mean() for region in regions if bool(region.any())]
        if region_losses:
            losses.append(torch.stack(region_losses).mean())
    return torch.stack(losses).mean() if losses else prediction_flat.sum() * 0.0


def ionic_elastic_response_loss(
    predicted_softening_gpa: torch.Tensor,
    true_force_constants_flat: torch.Tensor,
    true_internal_strain: torch.Tensor,
    strict_mask: torch.Tensor,
    force_mask: torch.Tensor,
    node_ptr: torch.Tensor,
    cells: torch.Tensor,
    response: AtomCoordinateResponsePotential,
) -> torch.Tensor:
    """Supervise ``Lambda^T D_delta(Phi) Lambda`` from strict true factors.

    The target requires no external elastic label and is explicitly named a
    regularized ionic softening, not a measured or relaxed total stiffness.
    Cartesian fourth-rank norms preserve the engineering-shear convention.
    """
    strict_mask = strict_mask.reshape(-1).to(dtype=torch.bool)
    force_mask = force_mask.reshape(-1).to(dtype=torch.bool)
    losses: list[torch.Tensor] = []
    true_offset = 0
    for graph_index in range(node_ptr.numel() - 1):
        start, stop = int(node_ptr[graph_index]), int(node_ptr[graph_index + 1])
        atoms, values = stop - start, 9 * (stop - start) ** 2
        blocks = None
        if bool(force_mask[graph_index]):
            blocks = true_force_constants_flat[true_offset : true_offset + values].reshape(atoms, atoms, 3, 3)
            true_offset += values
        if not bool(strict_mask[graph_index]):
            continue
        if blocks is None:
            raise ValueError("Ionic elastic response requires true force constants")
        coupling = response._coupling_voigt(true_internal_strain[start:stop].detach()).reshape(3 * atoms, 6)
        relaxed = response.apply_optical_operator(blocks.detach(), coupling, solve_policy="regularized")
        volume = torch.linalg.det(cells[graph_index]).abs().clamp_min(torch.finfo(cells.dtype).eps)
        target = response.EV_PER_A3_TO_GPA * coupling.transpose(0, 1) @ relaxed / volume
        target = 0.5 * (target + target.transpose(0, 1))
        predicted_cartesian = elastic_voigt_to_cartesian(predicted_softening_gpa[graph_index])
        target_cartesian = elastic_voigt_to_cartesian(target)
        losses.append(_normalized_pseudo_huber(predicted_cartesian, target_cartesian, floor=0.1))
    if true_offset != true_force_constants_flat.numel():
        raise ValueError("True force constants did not match ionic-elastic boundaries")
    return torch.stack(losses).mean() if losses else predicted_softening_gpa.sum() * 0.0
