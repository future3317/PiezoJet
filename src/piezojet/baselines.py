"""Small M3 baselines with explicit interpretation boundaries."""

from __future__ import annotations

import torch
from torch import nn

from .model import (
    CartesianLocalEnvironmentEncoder,
    CartesianPiezoTensorHead,
    PeriodicCrystalEncoder,
    PiezoTensorHead,
)


class DirectCartesianPiezoBaseline(nn.Module):
    """Matched direct-tensor baseline for the atom-coordinate encoder.

    It uses precisely PiezoJet's Cartesian local encoder and the same
    strain-symmetric equivariant tensor readout, but removes Born charges,
    force constants, internal strain, reciprocal response propagation, and all
    factor losses.  Thus a comparison isolates the empirical value of the
    factorized/observable-response path rather than conflating it with a
    stronger geometric encoder or a different tensor convention.
    """

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = CartesianLocalEnvironmentEncoder(**encoder_kwargs)
        self.head = CartesianPiezoTensorHead(
            self.encoder.scalar_dim, self.encoder.channels
        )

    def forward(self, batch) -> torch.Tensor:
        features = self.encoder(batch)
        tensor = self.head(features, batch.batch)
        # Match the production macro tower's final numerical invariant
        # projection exactly.  The head is symmetric by construction, but the
        # shared projection removes even roundoff-level protocol differences.
        return 0.5 * (tensor + tensor.transpose(-1, -2))


class E3nnDirectPiezoBaseline(nn.Module):
    """PBC e3nn direct-tensor control with steerable CG message passing."""

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = PeriodicCrystalEncoder(**encoder_kwargs)
        self.head = PiezoTensorHead(self.encoder.hidden_irreps)

    def forward(self, batch) -> torch.Tensor:
        return self.head(self.encoder(batch), batch.batch)


def direct_cartesian_baseline_from_config(config: dict[str, object]) -> DirectCartesianPiezoBaseline:
    """Build the matched baseline with the production encoder hyperparameters."""
    return DirectCartesianPiezoBaseline(
        embedding_dim=int(config["embedding_dim"]),
        cutoff=float(config["cutoff"]),
        num_blocks=int(config["num_blocks"]),
        radial_basis=int(config["radial_basis"]),
        radial_hidden=int(config["radial_hidden"]),
        cartesian_channels=int(config.get("cartesian_channels", 48)),
    )


def e3nn_direct_baseline_from_config(config: dict[str, object]) -> E3nnDirectPiezoBaseline:
    """Build the PBC e3nn control on the same graph convention and cutoff."""
    return E3nnDirectPiezoBaseline(
        embedding_dim=int(config["embedding_dim"]),
        cutoff=float(config["cutoff"]),
        lmax=int(config.get("lmax", 3)),
        num_blocks=int(config["num_blocks"]),
        radial_basis=int(config["radial_basis"]),
        radial_hidden=int(config["radial_hidden"]),
        width_multiplier=float(
            config.get("electrostatic_encoder_width_multiplier", 1.0)
        ),
    )
