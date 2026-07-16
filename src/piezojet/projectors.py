"""Shared Cartesian subspace projectors used by training and evaluation."""

from __future__ import annotations

import torch


def translation_projector(
    atoms: int, reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Cartesian optical projector and normalized translations.

    Coordinates are flattened atom-major as ``(kappa, alpha)``.  Returning
    the translations alongside the projector avoids subtly divergent local
    reimplementations in training, response propagation, and diagnostics.
    """
    if atoms < 1:
        raise ValueError("A Cartesian projector requires at least one atom")
    size = 3 * atoms
    translations = reference.new_zeros(size, 3)
    for axis in range(3):
        translations[axis::3, axis] = atoms ** -0.5
    projector = torch.eye(size, dtype=reference.dtype, device=reference.device)
    return projector - translations @ translations.transpose(0, 1), translations
