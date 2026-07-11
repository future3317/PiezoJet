"""Small O(3)-equivariant periodic encoder and response potential."""

from __future__ import annotations

import torch
from e3nn import o3
from e3nn.nn import Gate
from torch import nn
from torch_geometric.utils import scatter

from .tensor_ops import PIEZO_TYPE, piezo_from_irreps, voigt_to_symmetric_matrix


def _radial_basis(distance: torch.Tensor, cutoff: float, count: int) -> torch.Tensor:
    centers = torch.linspace(0, cutoff, count, device=distance.device, dtype=distance.dtype)
    width = cutoff / max(count - 1, 1)
    envelope = (1 - distance / cutoff).clamp_min(0).square()
    return torch.exp(-((distance.unsqueeze(-1) - centers) / width).square()) * envelope.unsqueeze(-1)


class _MessageBlock(nn.Module):
    def __init__(self, irreps: o3.Irreps, sh_irreps: o3.Irreps, radial_basis: int, radial_hidden: int):
        super().__init__()
        self.gate = _gate_for(irreps)
        self.tp = o3.FullyConnectedTensorProduct(
            irreps, sh_irreps, self.gate.irreps_in, internal_weights=False, shared_weights=False
        )
        self.radial = nn.Sequential(nn.Linear(radial_basis, radial_hidden), nn.SiLU(), nn.Linear(radial_hidden, self.tp.weight_numel))
        self.residual = o3.Linear(irreps, self.gate.irreps_in)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor, sh: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        messages = self.tp(features[source], sh, self.radial(radial))
        aggregate = scatter(messages, target, dim=0, dim_size=features.shape[0], reduce="mean")
        return self.gate(self.residual(features) + aggregate)


def _gate_for(irreps_out: o3.Irreps) -> Gate:
    """Create the specified scalar-plus-gated non-scalar e3nn activation."""
    scalars = o3.Irreps("64x0e + 16x0o")
    gated = o3.Irreps("24x1e + 24x1o + 12x2e + 12x2o + 6x3e + 6x3o")
    if irreps_out != scalars + gated:
        raise ValueError("PiezoJet gate layout must match the fixed hidden irreps")
    gates = o3.Irreps("84x0e")
    gate = Gate(scalars, [torch.nn.functional.silu, torch.tanh], gates, [torch.sigmoid], gated)
    if gate.irreps_out != irreps_out:
        raise RuntimeError(f"Gate output mismatch: {gate.irreps_out} != {irreps_out}")
    return gate


class PeriodicCrystalEncoder(nn.Module):
    """Three PBC-aware e3nn message-passing blocks with l<=3 harmonics."""

    def __init__(
        self,
        embedding_dim: int = 32,
        cutoff: float = 5.0,
        lmax: int = 3,
        num_blocks: int = 3,
        radial_basis: int = 12,
        radial_hidden: int = 64,
    ):
        super().__init__()
        self.cutoff, self.radial_basis = cutoff, radial_basis
        self.embedding = nn.Embedding(119, embedding_dim)
        self.input_irreps = o3.Irreps(f"{embedding_dim}x0e")
        self.hidden_irreps = o3.Irreps("64x0e + 16x0o + 24x1e + 24x1o + 12x2e + 12x2o + 6x3e + 6x3o")
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.initial_gate = _gate_for(self.hidden_irreps)
        self.initial_tp = o3.FullyConnectedTensorProduct(
            self.input_irreps, self.sh_irreps, self.initial_gate.irreps_in, internal_weights=False, shared_weights=False
        )
        self.initial_radial = nn.Sequential(
            nn.Linear(radial_basis, radial_hidden), nn.SiLU(), nn.Linear(radial_hidden, self.initial_tp.weight_numel)
        )
        self.blocks = nn.ModuleList(
            _MessageBlock(self.hidden_irreps, self.sh_irreps, radial_basis, radial_hidden) for _ in range(num_blocks)
        )

    def forward(self, batch) -> torch.Tensor:
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]] + batch.edge_shift
        distance = torch.linalg.vector_norm(vectors, dim=-1)
        if torch.any(distance > self.cutoff + 1e-5):
            raise ValueError("Graph contains an edge beyond the configured cutoff")
        sh = o3.spherical_harmonics(self.sh_irreps, vectors, normalize=True, normalization="component")
        radial = _radial_basis(distance, self.cutoff, self.radial_basis)
        source, target = batch.edge_index
        initial = self.initial_tp(self.embedding(batch.z)[source], sh, self.initial_radial(radial))
        features = self.initial_gate(scatter(initial, target, dim=0, dim_size=batch.z.shape[0], reduce="mean"))
        for block in self.blocks:
            features = block(features, batch.edge_index, sh, radial)
        return features


class PiezoTensorHead(nn.Module):
    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in, PIEZO_TYPE)

    def forward(self, node_features: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        count = int(batch_index.max()) + 1
        graph_features = scatter(node_features, batch_index, dim=0, dim_size=count, reduce="mean")
        return piezo_from_irreps(self.linear(graph_features))


class ResponsePotential(nn.Module):
    """Phi(x, E, eta)=-E_i e_ijk(x) eta_jk using one Voigt conversion."""

    def forward(self, piezo_cart: torch.Tensor, field: torch.Tensor, eta6: torch.Tensor) -> torch.Tensor:
        if field.shape[-1] != 3 or eta6.shape[-1] != 6:
            raise ValueError("Expected field [...,3] and eta6 [...,6]")
        strain = voigt_to_symmetric_matrix(eta6)
        return -torch.einsum("bi,bijk,bjk->b", field, piezo_cart, strain)


class PiezoJet(nn.Module):
    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = PeriodicCrystalEncoder(**encoder_kwargs)
        self.head = PiezoTensorHead(self.encoder.hidden_irreps)
        self.response = ResponsePotential()

    def forward(self, batch) -> torch.Tensor:
        return self.head(self.encoder(batch), batch.batch)

    def potential(self, batch, field: torch.Tensor, eta6: torch.Tensor) -> torch.Tensor:
        return self.response(self(batch), field, eta6)
