"""Small O(3)-equivariant periodic encoder and response potential."""

from __future__ import annotations

from itertools import permutations
from typing import Mapping

import torch
from e3nn import o3
from e3nn.nn import Gate
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter

from .tensor_ops import PIEZO_TYPE, cartesian_to_piezo_voigt, piezo_from_irreps, piezo_voigt_to_cartesian, voigt_to_symmetric_matrix


def _radial_basis(distance: torch.Tensor, centers: torch.Tensor, cutoff: float) -> torch.Tensor:
    centers = centers.to(dtype=distance.dtype)
    width = cutoff / max(centers.numel() - 1, 1)
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
        self.register_buffer("radial_centers", torch.linspace(0, cutoff, radial_basis), persistent=False)
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

    def forward(self, batch, atomic_numbers: torch.Tensor | None = None) -> torch.Tensor:
        atomic_numbers = batch.z if atomic_numbers is None else atomic_numbers
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]] + batch.edge_shift
        distance = torch.linalg.vector_norm(vectors, dim=-1)
        # Denoising pretraining perturbs atoms inside a fixed neighbor list.
        # A small tolerance retains those edges while the cutoff envelope gives
        # them vanishing weight; substantially malformed graphs still fail.
        if torch.any(distance > self.cutoff + 0.25):
            raise ValueError("Graph contains an edge beyond the configured cutoff")
        sh = o3.spherical_harmonics(self.sh_irreps, vectors, normalize=True, normalization="component")
        radial = _radial_basis(distance, self.radial_centers, self.cutoff)
        source, target = batch.edge_index
        initial = self.initial_tp(self.embedding(atomic_numbers)[source], sh, self.initial_radial(radial))
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


class CrystalGlobalContext(nn.Module):
    """Translation- and O(3)-invariant periodic crystal context.

    The local encoder only observes a finite real-space neighborhood.  This
    module supplements it with composition, lattice invariants, and a learned
    reciprocal-space power spectrum.  For reciprocal integer vectors h,
    ``|sum_i a(Z_i) exp(2 pi i h.f_i)|^2`` is invariant to the choice of cell
    origin and to global rotations.  Radial averaging over reciprocal-vector
    norms makes the descriptor a continuous crystal-level conditioner.
    """

    def __init__(self, context_dim: int = 128, spectral_channels: int = 16, spectral_shells: int = 8):
        super().__init__()
        values = torch.tensor([-1, 0, 1], dtype=torch.long)
        grid = torch.stack(torch.meshgrid(values, values, values, indexing="ij"), dim=-1).reshape(-1, 3)
        self.register_buffer("reciprocal_indices", grid[(grid != 0).any(dim=-1)], persistent=False)
        self.register_buffer("spectral_centers", torch.linspace(0.5, 5.5, spectral_shells), persistent=False)
        self.spectral_shells, self.spectral_channels = spectral_shells, spectral_channels
        self.species_features = nn.Embedding(119, spectral_channels)
        # The final scalar per reciprocal shell is the norm-squared spectrum
        # of learned local polar vectors.  It captures collective alignment or
        # cancellation of polar motifs across the whole periodic crystal.
        input_dim = 119 + 5 + spectral_shells * (spectral_channels + 1)
        self.network = nn.Sequential(
            nn.Linear(input_dim, context_dim), nn.SiLU(), nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim), nn.SiLU(), nn.LayerNorm(context_dim),
        )

    @staticmethod
    def _as_cells(cell: torch.Tensor | None, graphs: int) -> torch.Tensor | None:
        if cell is None:
            return None
        if cell.ndim == 2:
            cell = cell.unsqueeze(0)
        if cell.shape != (graphs, 3, 3):
            raise ValueError(f"Expected graph cells [{graphs},3,3], got {tuple(cell.shape)}")
        return cell

    def forward(self, batch, batch_index: torch.Tensor, polar_vectors: torch.Tensor | None = None) -> torch.Tensor:
        graphs = int(batch_index.max()) + 1
        dtype, device = batch.pos.dtype, batch.pos.device
        counts = scatter(torch.ones_like(batch_index, dtype=dtype), batch_index, dim=0, dim_size=graphs, reduce="sum")
        composition = scatter(F.one_hot(batch.z, num_classes=119).to(dtype), batch_index, dim=0, dim_size=graphs, reduce="sum")
        composition = composition / counts.unsqueeze(-1).clamp_min(1.0)
        cell = self._as_cells(getattr(batch, "cell", None), graphs)
        chemical_spectrum = torch.zeros(graphs, self.spectral_shells * self.spectral_channels, dtype=dtype, device=device)
        polar_spectrum = torch.zeros(graphs, self.spectral_shells, dtype=dtype, device=device)
        lattice = torch.zeros(graphs, 5, dtype=dtype, device=device)
        frac = getattr(batch, "frac", None)
        if cell is not None:
            volume = torch.linalg.det(cell).abs().clamp_min(torch.finfo(dtype).eps)
            gram = cell @ cell.transpose(-1, -2)
            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(torch.finfo(dtype).eps)
            reduced_eigenvalues = eigenvalues / volume.pow(2.0 / 3.0).unsqueeze(-1)
            lattice = torch.cat((reduced_eigenvalues.log(), volume.log().unsqueeze(-1), (counts / volume).log().unsqueeze(-1)), dim=-1)
            if frac is not None:
                reciprocal = 2.0 * torch.pi * torch.einsum("hi,bij->bhj", self.reciprocal_indices.to(dtype), torch.linalg.inv(cell))
                scaled_norm = torch.linalg.vector_norm(reciprocal, dim=-1) * volume.pow(1.0 / 3.0).unsqueeze(-1)
                radial = torch.exp(-((scaled_norm.unsqueeze(-1) - self.spectral_centers.to(dtype)) / 0.75).square())
                phase = 2.0 * torch.pi * (frac @ self.reciprocal_indices.to(dtype).transpose(0, 1))
                species = self.species_features(batch.z).to(dtype)
                real = scatter(torch.cos(phase).unsqueeze(-1) * species.unsqueeze(1), batch_index, dim=0, dim_size=graphs, reduce="sum")
                imag = scatter(torch.sin(phase).unsqueeze(-1) * species.unsqueeze(1), batch_index, dim=0, dim_size=graphs, reduce="sum")
                power = (real.square() + imag.square()) / counts.square().view(graphs, 1, 1).clamp_min(1.0)
                chemical_spectrum = (torch.einsum("bhs,bhc->bsc", radial, power) / radial.sum(dim=1).clamp_min(1e-8).unsqueeze(-1)).reshape(graphs, -1)
                if polar_vectors is not None:
                    polar_real = scatter(torch.cos(phase).unsqueeze(-1) * polar_vectors.unsqueeze(1), batch_index, dim=0, dim_size=graphs, reduce="sum")
                    polar_imag = scatter(torch.sin(phase).unsqueeze(-1) * polar_vectors.unsqueeze(1), batch_index, dim=0, dim_size=graphs, reduce="sum")
                    polar_power = (polar_real.square() + polar_imag.square()).sum(dim=-1) / counts.square().view(graphs, 1).clamp_min(1.0)
                    polar_spectrum = torch.einsum("bhs,bh->bs", radial, polar_power) / radial.sum(dim=1).clamp_min(1e-8)
        return self.network(torch.cat((composition, lattice, chemical_spectrum, polar_spectrum), dim=-1))


class ContinuousWeightedFrameRefiner(nn.Module):
    """Continuously average tensor refinements over all lattice-axis frames.

    A polar factor of each row permutation of the lattice gives a frame that
    rotates with the crystal.  Every expert predicts in its own local frame;
    the softmax-weighted, rotated-back average is therefore O(3)-equivariant.
    Summing all six permutations removes any arbitrary ordering of lattice
    vectors, while polar factors avoid eigenvector sign/swap discontinuities.
    """

    def __init__(self, context_dim: int, hidden_dim: int = 192):
        super().__init__()
        self.register_buffer("permutations", torch.tensor(list(permutations(range(3))), dtype=torch.long), persistent=False)
        descriptor_dim, local_dim = 9, 18
        self.weight_network = nn.Sequential(nn.Linear(context_dim + descriptor_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.delta_network = nn.Sequential(nn.Linear(context_dim + descriptor_dim + local_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, local_dim))
        # Begin from the direct equivariant head.  The frame path is learned as
        # a residual rather than destabilizing the initial response tensor.
        nn.init.zeros_(self.delta_network[-1].weight)
        nn.init.zeros_(self.delta_network[-1].bias)
        nn.init.zeros_(self.weight_network[-1].weight)
        nn.init.zeros_(self.weight_network[-1].bias)

    def _frames_and_descriptors(self, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = torch.stack([cell[:, permutation, :] for permutation in self.permutations.tolist()], dim=1)
        metric = candidates.transpose(-1, -2) @ candidates
        values, vectors = torch.linalg.eigh(metric)
        inverse_root = (vectors * values.clamp_min(torch.finfo(cell.dtype).eps).rsqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
        frames = candidates @ inverse_root
        volume_scale = torch.linalg.det(cell).abs().clamp_min(torch.finfo(cell.dtype).eps).pow(2.0 / 3.0).view(-1, 1, 1, 1)
        descriptor = (candidates @ candidates.transpose(-1, -2) / volume_scale).reshape(cell.shape[0], candidates.shape[1], -1)
        return frames, descriptor

    def forward(self, base: torch.Tensor, context: torch.Tensor, cell: torch.Tensor | None) -> torch.Tensor:
        if cell is None:
            return base
        if cell.ndim == 2:
            cell = cell.unsqueeze(0)
        frames, descriptor = self._frames_and_descriptors(cell)
        graphs, frame_count = frames.shape[:2]
        local_base = torch.einsum(
            "nfia,nfjb,nfkc,nfabc->nfijk", frames, frames, frames,
            base.unsqueeze(1).expand(-1, frame_count, -1, -1, -1),
        )
        local_voigt = cartesian_to_piezo_voigt(local_base)
        expanded_context = context.unsqueeze(1).expand(-1, frame_count, -1)
        logits = self.weight_network(torch.cat((expanded_context, descriptor), dim=-1)).squeeze(-1)
        delta = self.delta_network(torch.cat((expanded_context, descriptor, local_voigt.reshape(graphs, frame_count, -1)), dim=-1)).reshape(graphs, frame_count, 3, 6)
        local_refined = piezo_voigt_to_cartesian(local_voigt + delta)
        to_global = frames.transpose(-1, -2)
        global_refined = torch.einsum("nfia,nfjb,nfkc,nfabc->nfijk", to_global, to_global, to_global, local_refined)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(weights[..., None, None, None] * global_refined, dim=1)


class ResponsePotential(nn.Module):
    """Phi(x, E, eta)=-E_i e_ijk(x) eta_jk using one Voigt conversion."""

    def forward(self, piezo_cart: torch.Tensor, field: torch.Tensor, eta6: torch.Tensor) -> torch.Tensor:
        if field.shape[-1] != 3 or eta6.shape[-1] != 6:
            raise ValueError("Expected field [...,3] and eta6 [...,6]")
        strain = voigt_to_symmetric_matrix(eta6)
        return -torch.einsum("bi,bijk,bjk->b", field, piezo_cart, strain)


class PiezoJet(nn.Module):
    def __init__(self, global_context_dim: int = 128, spectral_channels: int = 16, spectral_shells: int = 8, **encoder_kwargs):
        super().__init__()
        self.encoder = PeriodicCrystalEncoder(**encoder_kwargs)
        self.head = PiezoTensorHead(self.encoder.hidden_irreps)
        self.local_polar_mode = o3.Linear(self.encoder.hidden_irreps, o3.Irreps("1x1o"))
        self.global_context = CrystalGlobalContext(global_context_dim, spectral_channels, spectral_shells)
        self.frame_refiner = ContinuousWeightedFrameRefiner(global_context_dim)
        self.response = ResponsePotential()

    def encode(self, batch, atomic_numbers: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(batch, atomic_numbers)

    def forward(self, batch) -> torch.Tensor:
        features = self.encode(batch)
        direct = self.head(features, batch.batch)
        context = self.global_context(batch, batch.batch, self.local_polar_mode(features))
        return self.frame_refiner(direct, context, getattr(batch, "cell", None))

    def potential(self, batch, field: torch.Tensor, eta6: torch.Tensor) -> torch.Tensor:
        return self.response(self(batch), field, eta6)


def model_from_config(config: Mapping[str, object]) -> PiezoJet:
    """Construct the single production PiezoJet architecture from a run config."""
    return PiezoJet(
        embedding_dim=int(config["embedding_dim"]), cutoff=float(config["cutoff"]), lmax=int(config["lmax"]),
        num_blocks=int(config["num_blocks"]), radial_basis=int(config["radial_basis"]), radial_hidden=int(config["radial_hidden"]),
        global_context_dim=int(config.get("global_context_dim", 128)), spectral_channels=int(config.get("spectral_channels", 16)),
        spectral_shells=int(config.get("spectral_shells", 8)),
    )
