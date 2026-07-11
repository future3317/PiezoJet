"""Small M3 baselines with explicit interpretation boundaries."""

from __future__ import annotations

import torch
from torch import nn

from .tensor_ops import piezo_voigt_to_cartesian


class ZeroBaseline(nn.Module):
    def forward(self, batch) -> torch.Tensor:
        return batch.y.new_zeros((batch.num_graphs, 3, 3, 3))


class MeanBaseline(nn.Module):
    def __init__(self, train_mean_voigt: torch.Tensor):
        super().__init__()
        self.register_buffer("mean_voigt", train_mean_voigt.detach().clone())

    def forward(self, batch) -> torch.Tensor:
        return piezo_voigt_to_cartesian(self.mean_voigt).expand(batch.num_graphs, -1, -1, -1)


class CompositionOnlyBaseline(nn.Module):
    """Atomic-count MLP; it intentionally has no directional/geometric input."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(118, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 18))

    def forward(self, batch) -> torch.Tensor:
        features = batch.z.new_zeros((batch.num_graphs, 118), dtype=torch.float32)
        features.index_put_((batch.batch, batch.z.clamp_max(118) - 1), torch.ones_like(batch.z, dtype=torch.float32), accumulate=True)
        features = features / features.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return piezo_voigt_to_cartesian(self.network(features).reshape(batch.num_graphs, 3, 6))


class DirectScalarBaseline(nn.Module):
    """Reuses the periodic encoder but emits unconstrained 18 scalar components."""

    def __init__(self, encoder, hidden_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.readout = nn.Sequential(nn.Linear(encoder.hidden_irreps.dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 18))

    def forward(self, batch) -> torch.Tensor:
        from torch_geometric.utils import scatter
        node = self.encoder(batch)
        graph = scatter(node, batch.batch, dim=0, dim_size=batch.num_graphs, reduce="mean")
        return self.readout(graph).reshape(batch.num_graphs, 3, 3, 2).repeat_interleave(2, dim=-1)
