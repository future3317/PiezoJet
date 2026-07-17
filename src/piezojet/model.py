"""Small O(3)-equivariant periodic encoder and response potential."""

from __future__ import annotations

from typing import Mapping, NamedTuple

import math

import torch
from e3nn import o3
from e3nn.nn import Gate
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter, to_dense_batch

from .tensor_ops import (
    PIEZO_TYPE,
    cartesian_to_piezo_voigt,
    piezo_from_irreps,
    piezo_voigt_to_cartesian,
    voigt_to_symmetric_matrix,
)
from .projectors import translation_projector
from .elastic_dielectric_ops import elastic_cartesian_to_voigt


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


class CartesianNodeFeatures(NamedTuple):
    """Channelized local Cartesian environment at every atom.

    Scalars are invariant; vectors and traceless symmetric matrices transform
    as rank-one and rank-two Cartesian tensors, respectively.
    """

    scalar: torch.Tensor
    vector: torch.Tensor
    quadrupole: torch.Tensor


class _InvariantMessageBlock(nn.Module):
    """PBC chemical message passing with no directional hidden features."""

    def __init__(self, hidden_dim: int, radial_basis: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + radial_basis, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.update = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, scalar: torch.Tensor, edge_index: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        message = self.message(torch.cat((scalar[source], scalar[target], radial), dim=-1))
        aggregate = scatter(message, target, dim=0, dim_size=scalar.shape[0], reduce="mean")
        return self.norm(scalar + self.update(aggregate))


class CartesianLocalEnvironmentEncoder(nn.Module):
    """CEIT-style local Cartesian environment with channel-space mixing.

    High-rank geometric information is never propagated through a spherical
    harmonic/Clebsch--Gordan stack.  Instead, an invariant chemical backbone
    produces edge coefficients, which weight vector and quadrupole Cartesian
    bases.  A learnable channel interaction matrix then realizes local
    many-body coupling while preserving exact O(3) transformation laws.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        cutoff: float = 5.0,
        num_blocks: int = 3,
        radial_basis: int = 12,
        radial_hidden: int = 64,
        cartesian_channels: int = 48,
    ):
        super().__init__()
        self.cutoff, self.radial_basis, self.channels = cutoff, radial_basis, cartesian_channels
        # Public contract for downstream readout heads.  Keep this independent
        # of the private layout of ``self.input`` so encoder refactors cannot
        # silently select the wrong layer.
        self.scalar_dim = radial_hidden
        self.register_buffer("radial_centers", torch.linspace(0, cutoff, radial_basis), persistent=False)
        self.register_buffer("identity", torch.eye(3), persistent=False)
        self.embedding = nn.Embedding(119, embedding_dim)
        self.input = nn.Sequential(nn.Linear(embedding_dim, radial_hidden), nn.SiLU(), nn.Linear(radial_hidden, radial_hidden), nn.SiLU())
        self.blocks = nn.ModuleList(_InvariantMessageBlock(radial_hidden, radial_basis) for _ in range(num_blocks))
        edge_dim = 2 * radial_hidden + radial_basis
        self.vector_weights = nn.Sequential(nn.Linear(edge_dim, radial_hidden), nn.SiLU(), nn.Linear(radial_hidden, cartesian_channels))
        self.quadrupole_weights = nn.Sequential(nn.Linear(edge_dim, radial_hidden), nn.SiLU(), nn.Linear(radial_hidden, cartesian_channels))
        # CEIT's many-body interaction is deliberately learned in channel
        # space.  Identity initialization preserves independent local modes at
        # the start of optimization.
        self.channel_interaction = nn.Parameter(torch.eye(cartesian_channels))
        self.degree_power = nn.Parameter(torch.tensor(0.5))

    def forward(self, batch, atomic_numbers: torch.Tensor | None = None) -> CartesianNodeFeatures:
        atomic_numbers = batch.z if atomic_numbers is None else atomic_numbers
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]] + batch.edge_shift
        distance = torch.linalg.vector_norm(vectors, dim=-1)
        if torch.any(distance > self.cutoff + 0.25):
            raise ValueError("Graph contains an edge beyond the configured cutoff")
        radial = _radial_basis(distance, self.radial_centers, self.cutoff)
        scalar = self.input(self.embedding(atomic_numbers))
        for block in self.blocks:
            scalar = block(scalar, batch.edge_index, radial)
        source, target = batch.edge_index
        edge_context = torch.cat((scalar[source], scalar[target], radial), dim=-1)
        direction = vectors / distance.clamp_min(torch.finfo(vectors.dtype).eps).unsqueeze(-1)
        vector = scatter(
            self.vector_weights(edge_context).unsqueeze(-1) * direction.unsqueeze(1), target,
            dim=0, dim_size=batch.num_nodes, reduce="sum",
        )
        quadrupole_basis = direction.unsqueeze(-1) * direction.unsqueeze(-2) - self.identity.to(dtype=vectors.dtype) / 3
        quadrupole = scatter(
            self.quadrupole_weights(edge_context).unsqueeze(-1).unsqueeze(-1) * quadrupole_basis.unsqueeze(1), target,
            dim=0, dim_size=batch.num_nodes, reduce="sum",
        )
        degree = scatter(torch.ones_like(distance), target, dim=0, dim_size=batch.num_nodes, reduce="sum")
        normalization = degree.clamp_min(1.0).pow(-self.degree_power.clamp(0.0, 1.0))
        vector = vector * normalization[:, None, None]
        quadrupole = quadrupole * normalization[:, None, None, None]
        vector = torch.einsum("cd,ndj->ncj", self.channel_interaction, vector)
        quadrupole = torch.einsum("cd,ndjk->ncjk", self.channel_interaction, quadrupole)
        return CartesianNodeFeatures(scalar, vector, quadrupole)


class CartesianPolarReadout(nn.Module):
    """Invariantly gated channel sum yielding the local polar motif."""

    def __init__(self, scalar_dim: int, channels: int):
        super().__init__()
        self.gates = nn.Sequential(nn.Linear(scalar_dim, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, channels))

    def forward(self, features: CartesianNodeFeatures) -> torch.Tensor:
        return torch.einsum("nc,nci->ni", self.gates(features.scalar), features.vector)


class CartesianPiezoTensorHead(nn.Module):
    """Assemble a symmetric rank-three polar tensor from Cartesian bases."""

    def __init__(self, scalar_dim: int, channels: int):
        super().__init__()
        self.coefficients = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, 4 * channels)
        )
        self.register_buffer("identity", torch.eye(3), persistent=False)

    def forward(self, features: CartesianNodeFeatures, batch_index: torch.Tensor) -> torch.Tensor:
        vector, quadrupole = features.vector, features.quadrupole
        identity = self.identity.to(dtype=vector.dtype)
        vector_identity = torch.einsum("nci,jk->ncijk", vector, identity)
        identity_vector = 0.5 * (
            torch.einsum("ij,nck->ncijk", identity, vector) + torch.einsum("ik,ncj->ncijk", identity, vector)
        )
        vector_quadrupole = torch.einsum("nci,ncjk->ncijk", vector, quadrupole)
        quadrupole_vector = 0.5 * (
            torch.einsum("ncij,nck->ncijk", quadrupole, vector) + torch.einsum("ncik,ncj->ncijk", quadrupole, vector)
        )
        bases = torch.stack((vector_identity, identity_vector, vector_quadrupole, quadrupole_vector), dim=2)
        coefficients = self.coefficients(features.scalar).reshape(features.scalar.shape[0], vector.shape[1], 4)
        node_tensor = (coefficients[..., None, None, None] * bases).sum(dim=(1, 2))
        return scatter(node_tensor, batch_index, dim=0, dim_size=int(batch_index.max()) + 1, reduce="mean")


class CartesianMacroDielectricHead(nn.Module):
    """Independent O(3)-equivariant SPD total-dielectric head."""

    def __init__(self, scalar_dim: int, channels: int):
        super().__init__()
        self.coefficients = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim), nn.SiLU(),
            nn.Linear(scalar_dim, 1 + channels),
        )
        self.register_buffer("identity", torch.eye(3), persistent=False)

    def forward(self, features: CartesianNodeFeatures, batch_index: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficients(features.scalar)
        identity = self.identity.to(dtype=features.scalar.dtype)
        symmetric = (
            coefficients[:, 0, None, None] * identity
            + torch.einsum("nc,ncij->nij", coefficients[:, 1:], features.quadrupole)
        )
        graphs = int(batch_index.max()) + 1
        symmetric = scatter(symmetric, batch_index, dim=0, dim_size=graphs, reduce="mean")
        # Every SPD relative dielectric >= I has an equivariant symmetric
        # square-root representation. No physical-factor feature is consumed.
        return identity + symmetric @ symmetric.transpose(-1, -2)


class CartesianMacroElasticHead(nn.Module):
    """Independent O(3)-equivariant positive total-stiffness head.

    The isotropic bulk/shear projectors cover high-symmetry crystals even when
    every graph quadrupole vanishes. Structure-conditioned symmetric modes add
    a positive semidefinite anisotropic stiffness in Cartesian tensor space.
    """

    def __init__(self, scalar_dim: int, channels: int, modes: int = 16):
        super().__init__()
        self.modes = int(modes)
        self.moduli = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim), nn.SiLU(),
            nn.Linear(scalar_dim, 2 + modes),
        )
        self.identity_mix = nn.Parameter(torch.zeros(modes))
        self.quadrupole_mix = nn.Parameter(torch.randn(modes, channels) / channels**0.5)
        self.register_buffer("identity", torch.eye(3), persistent=False)

    def forward(self, features: CartesianNodeFeatures, batch_index: torch.Tensor) -> torch.Tensor:
        graphs = int(batch_index.max()) + 1
        scalar = scatter(features.scalar, batch_index, dim=0, dim_size=graphs, reduce="mean")
        quadrupole = scatter(features.quadrupole, batch_index, dim=0, dim_size=graphs, reduce="mean")
        parameters = F.softplus(self.moduli(scalar))
        bulk, shear, weights = parameters[:, 0], parameters[:, 1], parameters[:, 2:]
        identity = self.identity.to(dtype=features.scalar.dtype)
        delta_delta = torch.einsum("ij,kl->ijkl", identity, identity)
        identity_symmetric = 0.5 * (
            torch.einsum("ik,jl->ijkl", identity, identity)
            + torch.einsum("il,jk->ijkl", identity, identity)
        )
        deviatoric = identity_symmetric - delta_delta / 3.0
        stiffness = (
            bulk[:, None, None, None, None] * delta_delta
            + 2.0 * shear[:, None, None, None, None] * deviatoric
        )
        modes = (
            self.identity_mix[None, :, None, None] * identity
            + torch.einsum("mc,bcij->bmij", self.quadrupole_mix, quadrupole)
        )
        stiffness = stiffness + torch.einsum("bm,bmij,bmkl->bijkl", weights, modes, modes)
        return elastic_cartesian_to_voigt(stiffness)


class CartesianBornChargeHead(nn.Module):
    """Node-resolved Born effective charges in an explicitly Cartesian basis.

    A Born tensor maps a polar displacement to a polar polarization.  The
    isotropic, symmetric-traceless, and antisymmetric Cartesian bases below
    span a general second-rank tensor without selecting a crystal frame.
    """

    def __init__(self, scalar_dim: int, channels: int):
        super().__init__()
        self.coefficients = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, 1 + 2 * channels)
        )
        self.register_buffer("identity", torch.eye(3), persistent=False)
        self.register_buffer(
            "levi_civita",
            torch.tensor((((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
                          ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                          ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
                         dtype=torch.float32),
            persistent=False,
        )

    def forward(self, features: CartesianNodeFeatures) -> torch.Tensor:
        coefficients = self.coefficients(features.scalar)
        channels = features.vector.shape[1]
        isotropic = coefficients[:, 0, None, None] * self.identity.to(dtype=features.scalar.dtype)
        symmetric = torch.einsum("nc,ncij->nij", coefficients[:, 1 : 1 + channels], features.quadrupole)
        antisymmetric = torch.einsum(
            "nc,nck,ijk->nij", coefficients[:, 1 + channels :], features.vector, self.levi_civita.to(dtype=features.scalar.dtype)
        )
        return isotropic + symmetric + antisymmetric


def _graph_ptr(batch_index: torch.Tensor) -> torch.Tensor:
    """Return graph boundaries without requiring a PyG ``Batch.ptr`` field."""
    graphs = int(batch_index.max()) + 1
    counts = torch.bincount(batch_index, minlength=graphs)
    return torch.cat((counts.new_zeros(1), counts.cumsum(0)))


class CartesianInternalStrainHead(nn.Module):
    """Atom-coordinate strain forces ``Lambda[kappa, alpha, j, k]``.

    The displacement/force index is polar and the last two strain indices are
    symmetric.  Subtracting the graph mean imposes zero net force under a
    homogeneous strain, the internal-strain acoustic sum rule.
    """

    def __init__(self, scalar_dim: int, channels: int):
        super().__init__()
        self.coefficients = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, 4 * channels)
        )
        self.register_buffer("identity", torch.eye(3), persistent=False)

    def forward(self, features: CartesianNodeFeatures, batch_index: torch.Tensor) -> torch.Tensor:
        vector, quadrupole = features.vector, features.quadrupole
        identity = self.identity.to(dtype=vector.dtype)
        vector_identity = torch.einsum("nca,jk->ncajk", vector, identity)
        identity_vector = 0.5 * (
            torch.einsum("aj,nck->ncajk", identity, vector)
            + torch.einsum("ak,ncj->ncajk", identity, vector)
        )
        vector_quadrupole = torch.einsum("nca,ncjk->ncajk", vector, quadrupole)
        quadrupole_vector = 0.5 * (
            torch.einsum("ncaj,nck->ncajk", quadrupole, vector)
            + torch.einsum("ncak,ncj->ncajk", quadrupole, vector)
        )
        bases = torch.stack((vector_identity, identity_vector, vector_quadrupole, quadrupole_vector), dim=2)
        coefficients = self.coefficients(features.scalar).reshape(features.scalar.shape[0], vector.shape[1], 4)
        internal_strain = (coefficients[..., None, None, None] * bases).sum(dim=(1, 2))
        graphs = int(batch_index.max()) + 1
        mean = scatter(internal_strain, batch_index, dim=0, dim_size=graphs, reduce="mean")
        return internal_strain - mean[batch_index]


class GlobalDisplacementResponseHead(nn.Module):
    """Nonlocal equivariant ``U_{eta,delta}`` readout.

    The regularized displacement response is a dense Green-action field even
    when ``Phi`` is sparse.  This head therefore augments local Cartesian node
    tensors with permutation-equivariant, all-to-all invariant attention,
    graph-level reciprocal context, and the graph-level polar strain operator.
    Channel-factorized cross mixing supplies ``V_c tensor T_c'`` terms without
    materializing a quadratic number of channel pairs.  The final graph mean
    subtraction is the exact translation projection.
    """

    def __init__(
        self,
        scalar_dim: int,
        channels: int,
        context_dim: int,
        attention_dim: int = 64,
        cross_rank: int = 24,
    ):
        super().__init__()
        if attention_dim <= 0 or cross_rank <= 0:
            raise ValueError("attention_dim and cross_rank must be positive")
        self.attention_dim = int(attention_dim)
        self.cross_rank = int(cross_rank)
        self.query = nn.Linear(scalar_dim, attention_dim, bias=False)
        self.key = nn.Linear(scalar_dim, attention_dim, bias=False)
        self.vector_value_mix = nn.Parameter(torch.eye(channels))
        self.quadrupole_value_mix = nn.Parameter(torch.eye(channels))
        self.vector_nonlocal_gate = nn.Parameter(torch.tensor(0.1))
        self.quadrupole_nonlocal_gate = nn.Parameter(torch.tensor(0.1))
        self.vector_rank_mix = nn.Parameter(
            torch.randn(cross_rank, channels) / channels**0.5
        )
        self.quadrupole_rank_mix = nn.Parameter(
            torch.randn(cross_rank, channels) / channels**0.5
        )
        hidden = max(128, scalar_dim, context_dim)
        self.coefficients = nn.Sequential(
            nn.Linear(scalar_dim + context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4 * cross_rank + 1),
        )
        self.register_buffer("identity", torch.eye(3), persistent=False)

    def _nonlocal_features(
        self,
        features: CartesianNodeFeatures,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vector_values = torch.einsum(
            "cd,ndj->ncj", self.vector_value_mix, features.vector
        )
        quadrupole_values = torch.einsum(
            "cd,ndjk->ncjk", self.quadrupole_value_mix, features.quadrupole
        )
        queries, keys = self.query(features.scalar), self.key(features.scalar)
        graphs = int(batch_index.max()) + 1
        scale = self.attention_dim**-0.5
        # PyG batches store every material in one contiguous node segment.
        # Padding those segments once turns O(B) small Python-launched matrix
        # products into two batched GEMMs.  The key mask makes padding exactly
        # invisible to the per-material softmax.
        dense_queries, mask = to_dense_batch(
            queries, batch_index, batch_size=graphs
        )
        dense_keys, key_mask = to_dense_batch(
            keys, batch_index, batch_size=graphs
        )
        if not torch.equal(mask, key_mask):
            raise RuntimeError("Global attention query/key batching mismatch")
        scores = torch.bmm(dense_queries, dense_keys.transpose(1, 2)) * scale
        scores = scores.masked_fill(~mask[:, None, :], -torch.inf)
        attention = torch.softmax(scores, dim=-1)

        vector_shape = vector_values.shape[1:]
        quadrupole_shape = quadrupole_values.shape[1:]
        values = torch.cat(
            (
                vector_values.flatten(1),
                quadrupole_values.flatten(1),
            ),
            dim=-1,
        )
        dense_values, value_mask = to_dense_batch(
            values, batch_index, batch_size=graphs
        )
        if not torch.equal(mask, value_mask):
            raise RuntimeError("Global attention value batching mismatch")
        attended = torch.bmm(attention, dense_values)[mask]
        vector_width = vector_values[0].numel()
        vector_global = attended[:, :vector_width].reshape(
            -1, *vector_shape
        )
        quadrupole_global = attended[:, vector_width:].reshape(
            -1, *quadrupole_shape
        )
        vector = features.vector + torch.tanh(self.vector_nonlocal_gate) * vector_global
        quadrupole = (
            features.quadrupole
            + torch.tanh(self.quadrupole_nonlocal_gate) * quadrupole_global
        )
        return vector, quadrupole

    def forward(
        self,
        features: CartesianNodeFeatures,
        batch_index: torch.Tensor,
        context: torch.Tensor,
        spectral_operator: torch.Tensor,
    ) -> torch.Tensor:
        graphs = int(batch_index.max()) + 1
        if context.shape[0] != graphs or spectral_operator.shape != (graphs, 3, 3, 3):
            raise ValueError("Global U head received inconsistent graph context")
        vector, quadrupole = self._nonlocal_features(features, batch_index)
        vector = torch.einsum("rc,nci->nri", self.vector_rank_mix, vector)
        quadrupole = torch.einsum(
            "rc,ncij->nrij", self.quadrupole_rank_mix, quadrupole
        )
        identity = self.identity.to(dtype=vector.dtype)
        vector_identity = torch.einsum("nra,jk->nrajk", vector, identity)
        identity_vector = 0.5 * (
            torch.einsum("aj,nrk->nrajk", identity, vector)
            + torch.einsum("ak,nrj->nrajk", identity, vector)
        )
        vector_quadrupole = torch.einsum(
            "nra,nrjk->nrajk", vector, quadrupole
        )
        quadrupole_vector = 0.5 * (
            torch.einsum("nraj,nrk->nrajk", quadrupole, vector)
            + torch.einsum("nrak,nrj->nrajk", quadrupole, vector)
        )
        bases = torch.stack(
            (vector_identity, identity_vector, vector_quadrupole, quadrupole_vector),
            dim=2,
        )
        conditioned = torch.cat((features.scalar, context[batch_index]), dim=-1)
        parameters = self.coefficients(conditioned)
        coefficients = parameters[:, :-1].reshape(
            features.scalar.shape[0], self.cross_rank, 4
        )
        response = (
            coefficients[..., None, None, None] * bases
        ).sum(dim=(1, 2))
        response = response + parameters[:, -1, None, None, None] * spectral_operator[
            batch_index
        ]
        mean = scatter(response, batch_index, dim=0, dim_size=graphs, reduce="mean")
        return response - mean[batch_index]


class OctupoleGlobalDisplacementResponseHead(GlobalDisplacementResponseHead):
    """Global U head with an explicit odd-parity ``l=3`` Cartesian channel.

    Products of learned vector and quadrupole channels contain an ``l=3``
    component in principle, but that product can vanish at high-symmetry
    sites.  A directly aggregated symmetric-traceless edge octupole supplies
    the missing independent channel while retaining O(3), atom-permutation,
    periodic-edge, and translation covariance.  It is a capacity candidate,
    not a conditional fallback selected by material or prediction quality.
    """

    def __init__(
        self,
        scalar_dim: int,
        channels: int,
        context_dim: int,
        attention_dim: int = 64,
        cross_rank: int = 24,
        radial_basis: int = 12,
        radial_hidden: int = 64,
        cutoff: float = 5.0,
    ):
        super().__init__(
            scalar_dim, channels, context_dim,
            attention_dim=attention_dim, cross_rank=cross_rank,
        )
        self.cutoff = float(cutoff)
        self.register_buffer(
            "octupole_radial_centers",
            torch.linspace(0.0, self.cutoff, radial_basis),
            persistent=False,
        )
        self.octupole_weights = nn.Sequential(
            nn.Linear(2 * scalar_dim + radial_basis, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, cross_rank),
        )
        hidden = max(128, scalar_dim, context_dim)
        self.coefficients = nn.Sequential(
            nn.Linear(scalar_dim + context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 5 * cross_rank + 1),
        )

    def _octupole_features(
        self,
        features: CartesianNodeFeatures,
        batch,
    ) -> torch.Tensor:
        source, target = batch.edge_index
        vectors = batch.pos[source] - batch.pos[target] + batch.edge_shift
        distance = torch.linalg.vector_norm(vectors, dim=-1)
        if torch.any(distance > self.cutoff + 0.25):
            raise ValueError("Graph contains an edge beyond the configured cutoff")
        direction = vectors / distance.clamp_min(
            torch.finfo(vectors.dtype).eps
        ).unsqueeze(-1)
        identity = self.identity.to(dtype=vectors.dtype)
        triple = torch.einsum("ei,ej,ek->eijk", direction, direction, direction)
        traces = (
            torch.einsum("ei,jk->eijk", direction, identity)
            + torch.einsum("ej,ik->eijk", direction, identity)
            + torch.einsum("ek,ij->eijk", direction, identity)
        ) / 5.0
        stf = triple - traces
        radial = _radial_basis(
            distance, self.octupole_radial_centers, self.cutoff
        )
        edge_context = torch.cat(
            (features.scalar[source], features.scalar[target], radial), dim=-1
        )
        weights = self.octupole_weights(edge_context)
        octupole = scatter(
            weights[..., None, None, None] * stf[:, None],
            target,
            dim=0,
            dim_size=features.scalar.shape[0],
            reduce="sum",
        )
        degree = scatter(
            torch.ones_like(distance), target, dim=0,
            dim_size=features.scalar.shape[0], reduce="sum",
        )
        return octupole / degree.clamp_min(1.0).sqrt()[:, None, None, None, None]

    def forward(
        self,
        features: CartesianNodeFeatures,
        batch,
        context: torch.Tensor,
        spectral_operator: torch.Tensor,
    ) -> torch.Tensor:
        batch_index = batch.batch
        graphs = int(batch_index.max()) + 1
        if context.shape[0] != graphs or spectral_operator.shape != (graphs, 3, 3, 3):
            raise ValueError("Global U head received inconsistent graph context")
        vector, quadrupole = self._nonlocal_features(features, batch_index)
        vector = torch.einsum("rc,nci->nri", self.vector_rank_mix, vector)
        quadrupole = torch.einsum(
            "rc,ncij->nrij", self.quadrupole_rank_mix, quadrupole
        )
        identity = self.identity.to(dtype=vector.dtype)
        vector_identity = torch.einsum("nra,jk->nrajk", vector, identity)
        identity_vector = 0.5 * (
            torch.einsum("aj,nrk->nrajk", identity, vector)
            + torch.einsum("ak,nrj->nrajk", identity, vector)
        )
        vector_quadrupole = torch.einsum("nra,nrjk->nrajk", vector, quadrupole)
        quadrupole_vector = 0.5 * (
            torch.einsum("nraj,nrk->nrajk", quadrupole, vector)
            + torch.einsum("nrak,nrj->nrajk", quadrupole, vector)
        )
        octupole = self._octupole_features(features, batch)
        bases = torch.stack(
            (
                vector_identity, identity_vector, vector_quadrupole,
                quadrupole_vector, octupole,
            ),
            dim=2,
        )
        conditioned = torch.cat((features.scalar, context[batch_index]), dim=-1)
        parameters = self.coefficients(conditioned)
        coefficients = parameters[:, :-1].reshape(
            features.scalar.shape[0], self.cross_rank, 5
        )
        response = (
            coefficients[..., None, None, None] * bases
        ).sum(dim=(1, 2))
        response = response + parameters[:, -1, None, None, None] * spectral_operator[
            batch_index
        ]
        mean = scatter(response, batch_index, dim=0, dim_size=graphs, reduce="mean")
        return response - mean[batch_index]

class IndependentQuadraticResponseHead(nn.Module):
    """Independent coefficients of one atom-coordinate quadratic energy.

    Directed periodic bonds predict a symmetric signed stiffness ``K_e`` for
    ``Phi``.  A separate equivariant, acoustic-projected head predicts the
    mixed derivative ``Lambda``.  Together with the affine strain curvature,
    the coefficients define the single scalar energy

    ``1/2 u^T Phi u - u^T Lambda eta + 1/2 eta^T C_aff eta``.

    Thus Maxwell/integrability is retained without imposing the extra and
    generally unjustified model-class restriction ``Lambda=B^T K S``.  Signed
    stiffnesses retain unstable optical modes.  Reciprocal/global context
    conditions ``K_e`` so collective information remains in the ionic path.
    """

    def __init__(
        self,
        scalar_dim: int,
        channels: int,
        radial_basis: int,
        cutoff: float,
        context_dim: int,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.register_buffer("radial_centers", torch.linspace(0, cutoff, radial_basis), persistent=False)
        self.register_buffer("identity", torch.eye(3), persistent=False)
        self.register_buffer(
            "unit_strains",
            voigt_to_symmetric_matrix(torch.eye(6)),
            persistent=False,
        )
        hidden = max(64, scalar_dim)
        self.edge_stiffness = nn.Sequential(
            nn.Linear(2 * scalar_dim + radial_basis + context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),
        )
        self.cross_derivative_head = CartesianInternalStrainHead(scalar_dim, channels)

    def forward(
        self,
        features: CartesianNodeFeatures,
        local_polar: torch.Tensor,
        context: torch.Tensor,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, target = batch.edge_index
        vectors = batch.pos[source] - batch.pos[target] + batch.edge_shift
        distance = torch.linalg.vector_norm(vectors, dim=-1)
        direction = vectors / distance.clamp_min(torch.finfo(vectors.dtype).eps).unsqueeze(-1)
        radial = _radial_basis(distance, self.radial_centers, self.cutoff)
        symmetric_scalar = features.scalar[source] + features.scalar[target]
        contrast = (features.scalar[source] - features.scalar[target]).abs()
        edge_graph = batch.batch[target]
        edge_context = torch.cat((symmetric_scalar, contrast, radial, context[edge_graph]), dim=-1)
        coefficients = self.edge_stiffness(edge_context)
        identity = self.identity.to(dtype=vectors.dtype)
        rr = direction.unsqueeze(-1) * direction.unsqueeze(-2)
        polar = 0.5 * (local_polar[source] + local_polar[target])
        polar_r = 0.5 * (
            polar.unsqueeze(-1) * direction.unsqueeze(-2)
            + direction.unsqueeze(-1) * polar.unsqueeze(-2)
        )
        pp = polar.unsqueeze(-1) * polar.unsqueeze(-2)
        stiffness = (
            coefficients[:, 0, None, None] * identity
            + coefficients[:, 1, None, None] * rr
            + coefficients[:, 2, None, None] * polar_r
            + coefficients[:, 3, None, None] * pp
        )
        stiffness = 0.5 * (stiffness + stiffness.transpose(-1, -2))
        strain_map = torch.einsum(
            "mij,ej->eim", self.unit_strains.to(dtype=vectors.dtype), vectors
        )
        stiffness_strain = stiffness @ strain_map
        internal_strain = self.cross_derivative_head(features, batch.batch)
        ptr = _graph_ptr(batch.batch)
        force_flat, strain_curvature = [], []
        for graph_index in range(ptr.numel() - 1):
            start, stop = int(ptr[graph_index]), int(ptr[graph_index + 1])
            atoms = stop - start
            edge_mask = edge_graph == graph_index
            local_source = source[edge_mask] - start
            local_target = target[edge_mask] - start
            local_stiffness = stiffness[edge_mask]
            local_stiffness_strain = stiffness_strain[edge_mask]

            blocks = local_stiffness.new_zeros(atoms, atoms, 3, 3)
            blocks.index_put_((local_source, local_source), local_stiffness, accumulate=True)
            blocks.index_put_((local_target, local_target), local_stiffness, accumulate=True)
            blocks.index_put_((local_source, local_target), -local_stiffness, accumulate=True)
            blocks.index_put_((local_target, local_source), -local_stiffness, accumulate=True)
            blocks = 0.5 * (blocks + blocks.permute(1, 0, 3, 2))

            local_map = strain_map[edge_mask]
            local_curvature = torch.einsum(
                "eai,eaj->ij", local_map, local_stiffness_strain
            )
            force_flat.append(blocks.reshape(-1))
            strain_curvature.append(
                0.5 * (local_curvature + local_curvature.transpose(0, 1))
            )
        return torch.cat(force_flat), internal_strain, torch.stack(strain_curvature)


class LinearResponseBackground(nn.Module):
    """Stable isotropic clamped-ion elastic and dielectric background blocks."""

    def __init__(self, context_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(context_dim, context_dim), nn.SiLU(), nn.Linear(context_dim, 3))
        self.register_buffer("identity", torch.eye(3), persistent=False)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lam, shear, susceptibility = F.softplus(self.network(context)).unbind(-1)
        graphs = context.shape[0]
        elastic = context.new_zeros(graphs, 6, 6)
        elastic[:, :3, :3] = lam[:, None, None]
        diagonal = torch.arange(3, device=context.device)
        elastic[:, diagonal, diagonal] = (lam + 2.0 * shear).unsqueeze(-1).expand(-1, 3)
        elastic[:, 3, 3], elastic[:, 4, 4], elastic[:, 5, 5] = shear, shear, shear
        # GMTNet/JARVIS labels are relative permittivities epsilon_r, not
        # susceptibilities chi_r.  The vacuum contribution is therefore the
        # identity and the learned non-negative scalar is the electronic
        # susceptibility: epsilon_r^el = I + chi_r^el.
        identity = self.identity.to(dtype=context.dtype)
        dielectric = (1.0 + susceptibility)[:, None, None] * identity
        return elastic, dielectric

class AtomCoordinatePrediction(NamedTuple):
    """Responses propagated through the physical ``3N-3`` optical space."""

    tensor: torch.Tensor
    macro_dielectric: torch.Tensor
    macro_elastic: torch.Tensor
    physical_tensor: torch.Tensor
    electronic_piezo: torch.Tensor
    # Maintained ionic prediction from the independent displacement-response
    # field U_eta.  It is not constructed by inverting a predicted observable
    # map or by propagating the predicted Lambda.
    ionic_piezo: torch.Tensor
    # Independent Phi/Lambda propagation retained as a physical diagnostic.
    factorized_ionic_piezo: torch.Tensor
    displacement_response: torch.Tensor
    born_charges: torch.Tensor
    force_constants_flat: torch.Tensor
    internal_strain: torch.Tensor
    # This is empty in the normal training/inference path.  A full Green
    # operator is quadratic in the atom count and is only materialized by an
    # explicit diagnostic request (``return_optical_operator=True``).
    optical_operator_flat: torch.Tensor
    shared_clamped_elastic: torch.Tensor
    elastic_background: torch.Tensor
    dielectric_background: torch.Tensor
    ionic_dielectric: torch.Tensor
    elastic_softening: torch.Tensor
    dielectric: torch.Tensor
    elastic: torch.Tensor


class AtomCoordinateFactors(NamedTuple):
    """Directly supervised atom-coordinate factors before the response solve."""

    born_charges: torch.Tensor
    force_constants_flat: torch.Tensor
    internal_strain: torch.Tensor
    strain_curvature: torch.Tensor


class CrystalGlobalContext(nn.Module):
    """Invariant context plus a cell-basis-invariant tensorial response operator.

    A reciprocal shell is enumerated by its *physical* scaled length
    ``|2 pi h L^{-T}| V^(1/3)`` rather than by a fixed cube of integer labels.
    Consequently an integral change of lattice basis only re-indexes the same
    shell.  In addition to scalar power spectra, the module constructs the
    origin-invariant polar cross-spectrum ``Re[P_h S_h^*]`` and contracts it
    with ``g_hat ⊗ g_hat`` to yield a polar rank-three response operator.
    """

    def __init__(
        self,
        context_dim: int = 128,
        spectral_channels: int = 16,
        spectral_shells: int = 8,
        polar_fluctuation_shells: int = 8,
        reciprocal_cutoff: float = 7.0,
    ):
        super().__init__()
        self.register_buffer("spectral_centers", torch.linspace(0.5, 5.5, spectral_shells), persistent=False)
        self.register_buffer("fluctuation_centers", torch.linspace(0.5, 5.0, polar_fluctuation_shells), persistent=False)
        self.spectral_shells, self.spectral_channels = spectral_shells, spectral_channels
        self.polar_fluctuation_shells = polar_fluctuation_shells
        self.reciprocal_cutoff = float(reciprocal_cutoff)
        self.species_features = nn.Embedding(119, spectral_channels)
        # This scalar reference structure factor fixes the phase of the
        # learned local polar mode without selecting an origin or a frame.
        self.phase_reference = nn.Embedding(119, 1)
        self.operator_kernel = nn.Linear(spectral_shells, 1, bias=False)
        self.uniform_operator_weight = nn.Parameter(torch.tensor(0.01))
        nn.init.normal_(self.operator_kernel.weight, std=0.01)
        # The final scalar per reciprocal shell is the norm-squared spectrum
        # of learned local polar vectors.  It captures collective alignment or
        # cancellation of polar motifs across the whole periodic crystal.
        # Raw Gram-matrix eigenvalues are not invariant under an integral
        # change of lattice basis.  The physical reciprocal shell below
        # already encodes lattice shape, so only volume and number density are
        # retained as explicit scalar lattice information.
        input_dim = 119 + 2 + spectral_shells * (spectral_channels + 1) + polar_fluctuation_shells
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

    def _physical_shell(self, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enumerate every reciprocal vector inside the scaled physical cutoff.

        For ``B=2πL^{-T}V^(1/3)``, ``|hB| >= sigma_min(B)|h|`` supplies a
        finite, representation-independent integer bound.  Filtering by the
        physical norm then makes the returned shell exactly invariant under
        any unimodular integer change of lattice basis (up to roundoff).
        """
        dtype, device = cell.dtype, cell.device
        volume = torch.linalg.det(cell).abs().clamp_min(torch.finfo(dtype).eps)
        reciprocal_basis = 2.0 * torch.pi * torch.linalg.inv(cell).transpose(-1, -2) * volume.pow(1.0 / 3.0)
        smallest = torch.linalg.svdvals(reciprocal_basis).amin().clamp_min(torch.finfo(dtype).eps)
        limit = max(1, int(math.ceil(self.reciprocal_cutoff / float(smallest.detach().cpu()))))
        values = torch.arange(-limit, limit + 1, dtype=dtype, device=device)
        integer_indices = torch.stack(torch.meshgrid(values, values, values, indexing="ij"), dim=-1).reshape(-1, 3)
        scaled_reciprocal = integer_indices @ reciprocal_basis
        scaled_norm = torch.linalg.vector_norm(scaled_reciprocal, dim=-1)
        keep = (scaled_norm > torch.finfo(dtype).eps) & (scaled_norm <= self.reciprocal_cutoff)
        integer_indices, scaled_reciprocal, scaled_norm = integer_indices[keep], scaled_reciprocal[keep], scaled_norm[keep]
        return integer_indices, scaled_reciprocal / scaled_norm[:, None], scaled_norm

    def forward(
        self,
        batch,
        batch_index: torch.Tensor,
        polar_vectors: torch.Tensor | None = None,
        *,
        return_operator: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        graphs = int(batch_index.max()) + 1
        dtype, device = batch.pos.dtype, batch.pos.device
        counts = scatter(torch.ones_like(batch_index, dtype=dtype), batch_index, dim=0, dim_size=graphs, reduce="sum")
        composition = scatter(F.one_hot(batch.z, num_classes=119).to(dtype), batch_index, dim=0, dim_size=graphs, reduce="sum")
        composition = composition / counts.unsqueeze(-1).clamp_min(1.0)
        cell = self._as_cells(getattr(batch, "cell", None), graphs)
        chemical_spectrum = torch.zeros(graphs, self.spectral_shells * self.spectral_channels, dtype=dtype, device=device)
        polar_spectrum = torch.zeros(graphs, self.spectral_shells, dtype=dtype, device=device)
        fluctuation_spectrum = torch.zeros(graphs, self.polar_fluctuation_shells, dtype=dtype, device=device)
        tensorial_operator = torch.zeros(graphs, 3, 3, 3, dtype=dtype, device=device)
        lattice = torch.zeros(graphs, 2, dtype=dtype, device=device)
        frac = getattr(batch, "frac", None)
        if cell is not None:
            volume = torch.linalg.det(cell).abs().clamp_min(torch.finfo(dtype).eps)
            lattice = torch.stack((volume.log(), (counts / volume).log()), dim=-1)
            if frac is not None:
                species = self.species_features(batch.z).to(dtype)
                references = self.phase_reference(batch.z).squeeze(-1).to(dtype)
                for graph_index in range(graphs):
                    node_mask = batch_index == graph_index
                    indices, directions, scaled_norm = self._physical_shell(cell[graph_index])
                    # Very short conventional cells can have no nonzero
                    # reciprocal vector below a finite physical cutoff.  The
                    # corresponding shell contribution is exactly zero, not
                    # an undefined empty reduction.
                    if indices.numel() == 0:
                        continue
                    radial = torch.exp(-((scaled_norm.unsqueeze(-1) - self.spectral_centers.to(dtype)) / 0.75).square())
                    phase = 2.0 * torch.pi * (frac[node_mask] @ indices.transpose(0, 1))
                    cosine, sine = torch.cos(phase), torch.sin(phase)
                    normalizer = counts[graph_index].square().clamp_min(1.0)
                    chemical_real = torch.einsum("nh,nc->hc", cosine, species[node_mask])
                    chemical_imag = torch.einsum("nh,nc->hc", sine, species[node_mask])
                    chemical_power = (chemical_real.square() + chemical_imag.square()) / normalizer
                    chemical_spectrum[graph_index] = (radial.transpose(0, 1) @ chemical_power / radial.sum(dim=0).clamp_min(1e-8)[:, None]).reshape(-1)
                    if polar_vectors is not None:
                        polar_real = torch.einsum("nh,ni->hi", cosine, polar_vectors[node_mask]) / counts[graph_index].clamp_min(1.0)
                        polar_imag = torch.einsum("nh,ni->hi", sine, polar_vectors[node_mask]) / counts[graph_index].clamp_min(1.0)
                        polar_power = polar_real.square().sum(dim=-1) + polar_imag.square().sum(dim=-1)
                        polar_spectrum[graph_index] = radial.transpose(0, 1) @ polar_power / radial.sum(dim=0).clamp_min(1e-8)
                        reference_real = torch.einsum("nh,n->h", cosine, references[node_mask]) / counts[graph_index].clamp_min(1.0)
                        reference_imag = torch.einsum("nh,n->h", sine, references[node_mask]) / counts[graph_index].clamp_min(1.0)
                        # Re[P_h S_h^*] retains the polar direction while the
                        # shared phase removes origin dependence.
                        cross_polar = polar_real * reference_real[:, None] + polar_imag * reference_imag[:, None]
                        shell_weight = self.operator_kernel(radial).squeeze(-1)
                        tensorial_operator[graph_index] = torch.einsum(
                            "h,hi,hj,hk->ijk", shell_weight, cross_polar, directions, directions
                        ) / float(indices.shape[0])
                        uniform = polar_vectors[node_mask].mean(dim=0)
                        uniform_shape = torch.einsum("i,j,k->ijk", uniform, uniform, uniform)
                        tensorial_operator[graph_index] = tensorial_operator[graph_index] + self.uniform_operator_weight * uniform_shape / uniform.square().sum().clamp_min(1e-8)
                if polar_vectors is not None:
                    # The reciprocal operator is complemented by a local
                    # fluctuation statistic: coherent and cancelling motifs
                    # are not conflated merely because their net mode is zero.
                    centered = polar_vectors - scatter(polar_vectors, batch_index, dim=0, dim_size=graphs, reduce="mean")[batch_index]
                    source, target = batch.edge_index
                    edge_vectors = batch.pos[source] - batch.pos[target] + batch.edge_shift
                    edge_distance = torch.linalg.vector_norm(edge_vectors, dim=-1)
                    local_kernel = torch.exp(-((edge_distance.unsqueeze(-1) - self.fluctuation_centers.to(dtype)) / 0.6).square())
                    edge_graph = batch_index[target]
                    correlation = (centered[source] * centered[target]).sum(dim=-1, keepdim=True)
                    numerator = scatter(correlation * local_kernel, edge_graph, dim=0, dim_size=graphs, reduce="sum")
                    denominator = scatter(local_kernel, edge_graph, dim=0, dim_size=graphs, reduce="sum")
                    fluctuation_spectrum = numerator / denominator.clamp_min(1e-8)
        context = self.network(torch.cat((composition, lattice, chemical_spectrum, polar_spectrum, fluctuation_spectrum), dim=-1))
        return (context, tensorial_operator) if return_operator else context


class ResponsePotential(nn.Module):
    """Phi(x, E, eta)=-E_i e_ijk(x) eta_jk using one Voigt conversion."""

    def forward(self, piezo_cart: torch.Tensor, field: torch.Tensor, eta6: torch.Tensor) -> torch.Tensor:
        if field.shape[-1] != 3 or eta6.shape[-1] != 6:
            raise ValueError("Expected field [...,3] and eta6 [...,6]")
        strain = voigt_to_symmetric_matrix(eta6)
        return -torch.einsum("bi,bijk,bjk->b", field, piezo_cart, strain)


class AtomCoordinateResponsePotential(nn.Module):
    """Generate responses through the atom-coordinate optical subspace.

    ``Z[kappa, alpha, i]`` follows VASP's force/displacement-row convention,
    ``Phi[kappa, alpha, kappa', beta]`` is in eV/Angstrom^2, and
    ``Lambda[kappa, alpha, j, k]`` is in eV/Angstrom per unit strain.  The
    conversion constants therefore produce C/m^2, relative permittivity, and
    GPa for the lattice-mediated response blocks.
    """

    PIEZO_C_PER_M2 = 16.02176634
    DIELECTRIC_RELATIVE = 180.9512817
    EV_PER_A3_TO_GPA = 160.2176634
    def __init__(
        self,
        optical_regularization: float = 1e-3,
        optical_stability_cutoff: float = 1e-4,
        optical_solve_policy: str = "regularized",
    ):
        super().__init__()
        if optical_regularization <= 0:
            raise ValueError("optical_regularization must be positive")
        if optical_stability_cutoff <= 0:
            raise ValueError("optical_stability_cutoff must be positive")
        if optical_solve_policy not in {"exact", "regularized"}:
            raise ValueError("optical_solve_policy must be exact or regularized")
        self.optical_regularization = float(optical_regularization)
        self.optical_stability_cutoff = float(optical_stability_cutoff)
        self.optical_solve_policy = optical_solve_policy

    @staticmethod
    def _coupling_voigt(coupling: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (coupling[..., 0, 0], coupling[..., 1, 1], coupling[..., 2, 2], coupling[..., 1, 2], coupling[..., 0, 2], coupling[..., 0, 1]),
            dim=-1,
        )

    @staticmethod
    def _matrix_from_blocks(blocks: torch.Tensor) -> torch.Tensor:
        atoms = blocks.shape[0]
        return blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)

    @staticmethod
    def _blocks_from_matrix(matrix: torch.Tensor, atoms: int) -> torch.Tensor:
        return matrix.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)

    def internal_quadratic_energy(
        self,
        force_constants: torch.Tensor,
        internal_strain: torch.Tensor,
        strain_curvature: torch.Tensor,
        displacement: torch.Tensor,
        eta6: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the explicit scalar energy generating ``Phi/Lambda/C``.

        ``Lambda`` is VASP's printed force derivative ``dF/deta``.  Therefore
        the mixed energy is ``-u^T Lambda eta``.  The coefficients may be
        predicted by independent equivariant heads; equality of mixed
        derivatives follows from this scalar assembly and does not require
        the extra factorization ``Lambda=B^T K S``.
        """
        atoms = int(force_constants.shape[0])
        if force_constants.shape != (atoms, atoms, 3, 3):
            raise ValueError("force_constants must have shape [N,N,3,3]")
        if internal_strain.shape != (atoms, 3, 3, 3):
            raise ValueError("internal_strain must have shape [N,3,3,3]")
        if strain_curvature.shape != (6, 6):
            raise ValueError("strain_curvature must have shape [6,6]")
        if displacement.numel() != 3 * atoms or eta6.shape != (6,):
            raise ValueError("displacement and eta6 must have shapes [N,3]/[3N] and [6]")
        u = displacement.reshape(3 * atoms)
        matrix = self._matrix_from_blocks(force_constants)
        coupling = self._coupling_voigt(internal_strain).reshape(3 * atoms, 6)
        curvature = 0.5 * (strain_curvature + strain_curvature.transpose(0, 1))
        return (
            0.5 * torch.einsum("i,ij,j->", u, matrix, u)
            - torch.einsum("i,ij,j->", u, coupling, eta6)
            + 0.5 * torch.einsum("i,ij,j->", eta6, curvature, eta6)
        )

    @staticmethod
    def _finalize_operator(matrix: torch.Tensor, atoms: int, dtype: torch.dtype) -> torch.Tensor:
        """Symmetrize and re-project after a possible float64-to-float32 cast."""
        matrix = (0.5 * (matrix + matrix.transpose(0, 1))).to(dtype)
        projector, _ = translation_projector(atoms, matrix)
        matrix = projector @ matrix @ projector
        return 0.5 * (matrix + matrix.transpose(0, 1))

    @staticmethod
    def _optical_basis(atoms: int, reference: torch.Tensor) -> torch.Tensor:
        """Orthonormal Cartesian basis for displacements orthogonal to translations.

        A Helmert basis is deterministic, exactly orthogonal in exact
        arithmetic, and independent of a coordinate-frame choice.  Its span
        is the physical optical subspace; changing its columns merely applies
        a similarity transform to the reduced solve.  We deliberately solve
        in this ``3N-3`` basis rather than lifting translations by an arbitrary
        energy-scale penalty in full Cartesian coordinates.
        """
        if atoms < 1:
            raise ValueError("A force-constant matrix must contain at least one atom")
        if atoms == 1:
            return reference.new_empty(3, 0)
        indices = torch.arange(atoms - 1, device=reference.device)
        # Form the normalization in the solve dtype.  Taking ``sqrt`` of the
        # integer index product first silently creates a float32 tensor in
        # PyTorch, which limits a nominal float64 optical solve to ~1e-7 basis
        # orthogonality.
        indices_float = indices.to(dtype=reference.dtype)
        denominator = torch.sqrt((indices_float + 1) * (indices_float + 2))
        helmert = reference.new_zeros(atoms, atoms - 1)
        rows = torch.arange(atoms, device=reference.device).unsqueeze(1)
        columns = indices.unsqueeze(0)
        helmert = torch.where(rows <= columns, denominator.reciprocal().unsqueeze(0), helmert)
        helmert[indices + 1, indices] = -(indices_float + 1) / denominator
        identity = torch.eye(3, dtype=reference.dtype, device=reference.device)
        return torch.kron(helmert, identity)

    def _reduced_operator(
        self,
        force_constants: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
        """Return ``(Q, Q^T Phi Q, output_dtype)`` in a stable solve dtype."""
        atoms = force_constants.shape[0]
        matrix = self._matrix_from_blocks(force_constants)
        output_dtype = matrix.dtype
        if matrix.dtype in (torch.float16, torch.bfloat16, torch.float32):
            matrix = matrix.to(torch.float64)
        basis = self._optical_basis(atoms, matrix)
        return basis, basis.transpose(0, 1) @ matrix @ basis, output_dtype

    def _optical_eigenvalues(self, force_constants: torch.Tensor) -> torch.Tensor:
        """Return eigenvalues on the exact ``3N-3`` optical subspace."""
        _, reduced, _ = self._reduced_operator(force_constants)
        return torch.linalg.eigvalsh(reduced) if reduced.numel() else reduced.new_empty(0)

    def apply_optical_operator(
        self,
        force_constants: torch.Tensor,
        rhs: torch.Tensor,
        solve_policy: str | None = None,
        regularization: float | None = None,
    ) -> torch.Tensor:
        """Apply the declared optical response operator without materializing it.

        ``rhs`` has shape ``[3N, K]``.  The exact path returns the stationary
        optical displacement ``Q (Q^T Phi Q)^{-1} Q^T rhs``.  The regularized
        path computes the signed resolvent with one complex reduced solve,
        ``Re[(Phi_o + i delta I)^{-1}]``.  Both paths project translations by
        construction, so there is no auxiliary translation penalty or squared
        condition number from a normal-equation solve.
        """
        policy = self.optical_solve_policy if solve_policy is None else solve_policy
        if policy not in {"exact", "regularized"}:
            raise ValueError("solve_policy must be exact or regularized")
        matrix = self._matrix_from_blocks(force_constants)
        if rhs.ndim != 2 or rhs.shape[0] != matrix.shape[0]:
            raise ValueError("rhs must have shape [3N, K] for the supplied force constants")
        basis, reduced, output_dtype = self._reduced_operator(force_constants)
        if reduced.numel() == 0:
            return rhs.new_zeros(rhs.shape)
        rhs_solve = rhs.to(dtype=reduced.dtype)
        rhs_optical = basis.transpose(0, 1) @ rhs_solve
        if policy == "exact":
            eigenvalues = torch.linalg.eigvalsh(reduced)
            if bool((eigenvalues.min() <= self.optical_stability_cutoff).item()):
                raise RuntimeError(
                    "Exact optical inverse is restricted to a positive stable optical spectrum"
                )
            reduced_solution = torch.linalg.solve(reduced, rhs_optical)
        else:
            delta = self.optical_regularization if regularization is None else float(regularization)
            if delta <= 0:
                raise ValueError("regularization must be positive")
            complex_dtype = torch.complex128 if reduced.dtype == torch.float64 else torch.complex64
            identity = torch.eye(reduced.shape[0], dtype=complex_dtype, device=reduced.device)
            shifted = reduced.to(complex_dtype) + (1j * delta) * identity
            reduced_solution = torch.linalg.solve(shifted, rhs_optical.to(complex_dtype)).real
        return (basis @ reduced_solution).to(dtype=output_dtype)

    def exact_optical_inverse(self, force_constants: torch.Tensor) -> torch.Tensor:
        """Exact inverse on a nonsingular optical subspace.

        This diagnostic-only convenience API materializes the operator from
        applications to Cartesian unit vectors.  The solve itself remains in
        the reduced optical basis; translations are never lifted by a
        numerical penalty. It is intentionally restricted to a positive
        true-DFPT stable spectrum.
        """
        atoms = force_constants.shape[0]
        matrix = self._matrix_from_blocks(force_constants)
        identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        inverse = self.apply_optical_operator(force_constants, identity, solve_policy="exact")
        return self._finalize_operator(inverse, atoms, matrix.dtype)

    def signed_regularized_optical_green(
        self,
        force_constants: torch.Tensor,
        regularization: float | None = None,
    ) -> torch.Tensor:
        """Signed regularized optical Green operator.

        This applies ``lambda / (lambda**2 + delta**2)``.  It is a bounded
        response generator for soft or unstable structures, not the stationary
        solution of the unregularized quadratic potential.
        """
        atoms = force_constants.shape[0]
        matrix = self._matrix_from_blocks(force_constants)
        identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        inverse = self.apply_optical_operator(
            force_constants, identity, solve_policy="regularized", regularization=regularization
        )
        return self._finalize_operator(inverse, atoms, matrix.dtype)

    def optical_operator(
        self,
        force_constants: torch.Tensor,
        solve_policy: str | None = None,
        regularization: float | None = None,
    ) -> torch.Tensor:
        """Choose the declared exact or regularized optical response policy."""
        policy = self.optical_solve_policy if solve_policy is None else solve_policy
        if policy not in {"exact", "regularized"}:
            raise ValueError("solve_policy must be exact or regularized")
        if policy == "exact":
            return self.exact_optical_inverse(force_constants)
        return self.signed_regularized_optical_green(force_constants, regularization)

    def ionic_piezo_from_displacement_response(
        self,
        born_charges: torch.Tensor,
        displacement_response: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        """Contract an independent internal-displacement response with BEC.

        ``displacement_response[kappa,a,j,k]`` is the production regularized
        coordinate ``U_{eta,delta}`` in Cartesian coordinates, with symmetric
        strain indices and zero mean over atoms.  It is not an equilibrium
        ``du/deta`` for an unstable reference.  This path contains no inverse,
        pseudoinverse, SVD cutoff, or chart-dependent gradient.
        """
        graphs = int(batch.batch.max()) + 1
        cell = getattr(batch, "cell", None)
        cells = (
            torch.eye(3, dtype=born_charges.dtype, device=born_charges.device)
            .expand(graphs, 3, 3)
            if cell is None else cell.reshape(-1, 3, 3)
        )
        displacement = self._coupling_voigt(displacement_response)
        per_atom = torch.einsum("nai,nav->niv", born_charges, displacement)
        contracted = scatter(
            per_atom, batch.batch, dim=0, dim_size=graphs, reduce="sum"
        )
        volume = torch.linalg.det(cells).abs().clamp_min(
            torch.finfo(cells.dtype).eps
        )
        return piezo_voigt_to_cartesian(
            self.PIEZO_C_PER_M2 * contracted / volume[:, None, None]
        )

    def responses(
        self,
        electronic_piezo: torch.Tensor,
        born_charges: torch.Tensor,
        internal_strain: torch.Tensor,
        force_constants_flat: torch.Tensor,
        strain_curvature: torch.Tensor,
        batch,
        elastic_background: torch.Tensor,
        dielectric_background: torch.Tensor,
        solve_policy: str | None = None,
        regularization: float | None = None,
        return_optical_operator: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ptr = _graph_ptr(batch.batch)
        cell = getattr(batch, "cell", None)
        cells = (
            torch.eye(3, dtype=born_charges.dtype, device=born_charges.device)
            .expand(ptr.numel() - 1, 3, 3)
            if cell is None else cell.reshape(-1, 3, 3)
        )
        ionic_piezo, ionic_dielectric, shared_clamped, elastic_softening, inverse_flat = [], [], [], [], []
        force_offset = 0
        for graph_index in range(ptr.numel() - 1):
            start, stop = int(ptr[graph_index]), int(ptr[graph_index + 1])
            atoms = stop - start
            block_values = 9 * atoms * atoms
            blocks = force_constants_flat[force_offset : force_offset + block_values].reshape(atoms, atoms, 3, 3)
            force_offset += block_values
            coupling = self._coupling_voigt(internal_strain[start:stop]).reshape(3 * atoms, 6)
            # VASP BEC rows are atomic force/displacement directions and
            # columns are electric-field/polarization directions.
            charge = born_charges[start:stop].reshape(3 * atoms, 3)
            relaxed = self.apply_optical_operator(
                blocks, torch.cat((charge, coupling), dim=-1), solve_policy, regularization
            )
            inverse_charge, inverse_coupling = relaxed[:, :3], relaxed[:, 3:]
            if return_optical_operator:
                operator = self.optical_operator(blocks, solve_policy, regularization)
                inverse_flat.append(self._blocks_from_matrix(operator, atoms).reshape(-1))
            volume = torch.linalg.det(cells[graph_index]).abs().clamp_min(torch.finfo(cells.dtype).eps)
            ionic_piezo.append(self.PIEZO_C_PER_M2 * (charge.transpose(0, 1) @ inverse_coupling) / volume)
            ionic_dielectric.append(
                self.DIELECTRIC_RELATIVE * (charge.transpose(0, 1) @ inverse_charge) / volume
            )
            shared_clamped.append(
                self.EV_PER_A3_TO_GPA * strain_curvature[graph_index] / volume
            )
            elastic_softening.append(
                self.EV_PER_A3_TO_GPA * (coupling.transpose(0, 1) @ inverse_coupling) / volume
            )
        ionic_piezo_cart = piezo_voigt_to_cartesian(torch.stack(ionic_piezo))
        ionic_dielectric_tensor = torch.stack(ionic_dielectric)
        elastic_softening_tensor = torch.stack(elastic_softening)
        dielectric = dielectric_background + ionic_dielectric_tensor
        shared_clamped_tensor = torch.stack(shared_clamped)
        # Case A in the energy decomposition: the separately predicted
        # background is K_direct, while S^T K S is included exactly once here.
        elastic = elastic_background + shared_clamped_tensor - elastic_softening_tensor
        return (
            electronic_piezo + ionic_piezo_cart,
            dielectric,
            elastic,
            torch.cat(inverse_flat) if inverse_flat else force_constants_flat.new_empty(0),
            shared_clamped_tensor,
            ionic_dielectric_tensor,
            elastic_softening_tensor,
        )

    def forward(
        self,
        piezo_c_per_m2: torch.Tensor,
        elastic_gpa: torch.Tensor,
        dielectric_relative: torch.Tensor,
        field: torch.Tensor,
        eta6: torch.Tensor,
    ) -> torch.Tensor:
        """Physical response energy density in eV/Angstrom^3.

        ``field`` is in V/Angstrom.  SI-valued response coefficients are first
        converted back to one coherent eV/Angstrom^3 convention before their
        quadratic form is assembled.
        """
        if field.shape[-1] != 3 or eta6.shape[-1] != 6:
            raise ValueError("Expected field [...,3] and eta6 [...,6]")
        piezo_internal = piezo_c_per_m2 / self.PIEZO_C_PER_M2
        elastic_internal = elastic_gpa / self.EV_PER_A3_TO_GPA
        dielectric_internal = dielectric_relative / self.DIELECTRIC_RELATIVE
        mixed_energy = -torch.einsum(
            "bi,bijk,bjk->b", field, piezo_internal, voigt_to_symmetric_matrix(eta6)
        )
        elastic_energy = 0.5 * torch.einsum("bI,bIJ,bJ->b", eta6, elastic_internal, eta6)
        dielectric_energy = -0.5 * torch.einsum(
            "bi,bij,bj->b", field, dielectric_internal, field
        )
        return mixed_energy + elastic_energy + dielectric_energy


class PiezoJet(nn.Module):
    def __init__(
        self,
        global_context_dim: int = 128,
        spectral_channels: int = 16,
        spectral_shells: int = 8,
        polar_fluctuation_shells: int = 8,
        reciprocal_cutoff: float = 7.0,
        optical_regularization: float = 1e-3,
        optical_stability_cutoff: float = 1e-4,
        optical_solve_policy: str = "regularized",
        factor_architecture: str = "independent_quadratic_response",
        background_architecture: str = "isotropic",
        displacement_attention_dim: int = 64,
        displacement_cross_rank: int = 24,
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = CartesianLocalEnvironmentEncoder(**encoder_kwargs)
        self.electronic_head = CartesianPiezoTensorHead(
            self.encoder.scalar_dim, self.encoder.channels
        )
        # Total-only labels have a branch-allocation gauge and therefore use
        # an independent macro tower. Their gradients cannot enter the
        # physical encoder or any Z/Phi/Lambda/U_eta decoder.
        self.macro_encoder = CartesianLocalEnvironmentEncoder(**encoder_kwargs)
        self.macro_total_head = CartesianPiezoTensorHead(
            self.macro_encoder.scalar_dim, self.macro_encoder.channels
        )
        self.macro_dielectric_head = CartesianMacroDielectricHead(
            self.macro_encoder.scalar_dim, self.macro_encoder.channels
        )
        self.macro_elastic_head = CartesianMacroElasticHead(
            self.macro_encoder.scalar_dim, self.macro_encoder.channels
        )
        if background_architecture != "isotropic":
            raise ValueError("PiezoJet supports only the maintained isotropic background")
        self.ionic_parameterization = "isolated_global_octupole_displacement"
        # U is a global Green-response field rather than a local DFPT factor.
        # Its representation is therefore isolated from Z/Phi/Lambda updates;
        # all towers start from the same inductive structural initialization.
        self.displacement_encoder = CartesianLocalEnvironmentEncoder(
            **encoder_kwargs
        )
        self.born_head = CartesianBornChargeHead(self.encoder.scalar_dim, self.encoder.channels)
        self.local_polar_mode = CartesianPolarReadout(self.encoder.scalar_dim, self.encoder.channels)
        self.global_context = CrystalGlobalContext(
            global_context_dim, spectral_channels, spectral_shells, polar_fluctuation_shells, reciprocal_cutoff
        )
        self.displacement_local_polar = CartesianPolarReadout(
            self.displacement_encoder.scalar_dim,
            self.displacement_encoder.channels,
        )
        self.displacement_global_context = CrystalGlobalContext(
            global_context_dim,
            spectral_channels,
            spectral_shells,
            polar_fluctuation_shells,
            reciprocal_cutoff,
        )
        displacement_kwargs = dict(
            scalar_dim=self.displacement_encoder.scalar_dim,
            channels=self.displacement_encoder.channels,
            context_dim=global_context_dim,
            attention_dim=displacement_attention_dim,
            cross_rank=displacement_cross_rank,
            radial_basis=int(encoder_kwargs.get("radial_basis", 12)),
            radial_hidden=int(encoder_kwargs.get("radial_hidden", 64)),
            cutoff=float(encoder_kwargs.get("cutoff", 5.0)),
        )
        self.displacement_response_head = OctupoleGlobalDisplacementResponseHead(
            **displacement_kwargs
        )
        # V is an auxiliary training coordinate for the first-order real
        # block form of (Phi + i delta I)^-1.  It is never used at inference.
        self.displacement_auxiliary_head = OctupoleGlobalDisplacementResponseHead(
            **displacement_kwargs
        )
        if factor_architecture != "independent_quadratic_response":
            raise ValueError(
                "PiezoJet supports only factor_architecture="
                "independent_quadratic_response"
            )
        self.factor_architecture = factor_architecture
        self.response_factors = IndependentQuadraticResponseHead(
            self.encoder.scalar_dim,
            self.encoder.channels,
            int(encoder_kwargs.get("radial_basis", 12)),
            float(encoder_kwargs.get("cutoff", 5.0)),
            global_context_dim,
        )
        self.background_architecture = background_architecture
        self.background = LinearResponseBackground(global_context_dim)
        self.response = AtomCoordinateResponsePotential(
            optical_regularization,
            optical_stability_cutoff,
            optical_solve_policy,
        )

    def encode(self, batch, atomic_numbers: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(batch, atomic_numbers)

    def _factor_features(self, batch):
        features = self.encode(batch)
        born_charges = self.born_head(features)
        born_mean = scatter(
            born_charges, batch.batch, dim=0, dim_size=int(batch.batch.max()) + 1, reduce="mean"
        )
        born_charges = born_charges - born_mean[batch.batch]
        local_polar = self.local_polar_mode(features)
        context, spectral_operator = self.global_context(
            batch, batch.batch, local_polar, return_operator=True
        )
        force_constants_flat, internal_strain, strain_curvature = self.response_factors(
            features, local_polar, context, batch
        )
        factors = AtomCoordinateFactors(
            born_charges, force_constants_flat, internal_strain, strain_curvature
        )
        return features, local_polar, factors, context, spectral_operator

    def predict_factors(self, batch) -> AtomCoordinateFactors:
        """Predict DFPT factors without evaluating their stiff inverse product."""
        return self._factor_features(batch)[2]

    def predict_macro_responses(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict total-only responses on a tower isolated from physical factors."""
        macro_features = self.macro_encoder(batch)
        macro_total = self.macro_total_head(macro_features, batch.batch)
        macro_total = 0.5 * (macro_total + macro_total.transpose(-1, -2))
        return (
            macro_total,
            self.macro_dielectric_head(macro_features, batch.batch),
            self.macro_elastic_head(macro_features, batch.batch),
        )

    def predict_macro_total(self, batch) -> torch.Tensor:
        """Backward-compatible total-piezo-only entry point."""
        return self.predict_macro_responses(batch)[0]

    def predict_displacement_response(self, batch) -> torch.Tensor:
        """Predict the independent translation-free regularized response field.

        This intentionally bypasses the BEC, Hessian, Lambda, electronic, and
        macro branches.  It is used only in the teacher-forced U curriculum,
        where a nonzero DFPT target must establish a well-conditioned learning
        signal before the bilinear ``Z*^T U`` loss and first-order block
        consistency are enabled.
        """
        features, context, spectral_operator = self._displacement_features(batch)
        return self.displacement_response_head(
            features, batch, context, spectral_operator
        )

    def _displacement_features(self, batch):
        """Encode the isolated global response tower once."""
        features = self.displacement_encoder(batch)
        local_polar = self.displacement_local_polar(features)
        context, spectral_operator = self.displacement_global_context(
            batch, batch.batch, local_polar, return_operator=True
        )
        return features, context, spectral_operator

    def predict_displacement_block_response(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the physical U field and training-only first-order V field."""
        features, context, spectral_operator = self._displacement_features(batch)
        return (
            self.displacement_response_head(
                features, batch, context, spectral_operator
            ),
            self.displacement_auxiliary_head(
                features, batch, context, spectral_operator
            ),
        )

    def predict_components(
        self,
        batch,
        return_optical_operator: bool = False,
        *,
        compute_macro_response: bool = True,
        compute_factorized_response: bool = True,
    ) -> AtomCoordinatePrediction:
        """Predict response components.

        Full optical Green matrices are excluded from the regular path.  They
        can be requested only for a small, explicitly diagnostic calculation.
        Multistream training may omit macro or propagated-factor outputs when
        no active objective consumes them; the maintained physical
        ``electronic + Z*^T U`` path is unchanged.
        """
        if return_optical_operator and not compute_factorized_response:
            raise ValueError(
                "A diagnostic optical operator requires factorized response propagation"
            )
        features, local_polar, factors, context, spectral_operator = self._factor_features(batch)
        graphs = context.shape[0]
        electronic_direct = self.electronic_head(features, batch.batch)
        electronic_piezo = 0.5 * (
            electronic_direct + spectral_operator
            + (electronic_direct + spectral_operator).transpose(-1, -2)
        )
        if compute_macro_response:
            macro_total, macro_dielectric, macro_elastic = self.predict_macro_responses(batch)
        else:
            macro_total = electronic_piezo.new_zeros(graphs, 3, 3, 3)
            macro_dielectric = electronic_piezo.new_zeros(graphs, 3, 3)
            macro_elastic = electronic_piezo.new_zeros(graphs, 6, 6)
        internal_strain = factors.internal_strain
        displacement_response = self.predict_displacement_response(batch)
        ionic_piezo = self.response.ionic_piezo_from_displacement_response(
            factors.born_charges, displacement_response, batch
        )
        if compute_factorized_response:
            elastic_background, dielectric_background = self.background(context)
            (
                factorized_tensor, dielectric, elastic, optical_operator_flat,
                shared_clamped_elastic, ionic_dielectric, elastic_softening,
            ) = self.response.responses(
                electronic_piezo, factors.born_charges, internal_strain,
                factors.force_constants_flat, factors.strain_curvature, batch,
                elastic_background, dielectric_background,
                return_optical_operator=return_optical_operator,
            )
        else:
            factorized_tensor = electronic_piezo
            optical_operator_flat = factors.force_constants_flat.new_empty(0)
            elastic_background = electronic_piezo.new_zeros(graphs, 6, 6)
            dielectric_background = electronic_piezo.new_zeros(graphs, 3, 3)
            shared_clamped_elastic = electronic_piezo.new_zeros(graphs, 6, 6)
            ionic_dielectric = electronic_piezo.new_zeros(graphs, 3, 3)
            elastic_softening = electronic_piezo.new_zeros(graphs, 6, 6)
            dielectric = dielectric_background
            elastic = elastic_background
        factorized_ionic_piezo = factorized_tensor - electronic_piezo
        physical_tensor = electronic_piezo + ionic_piezo
        return AtomCoordinatePrediction(
            # The direct and reciprocal terms are strain-symmetric by
            # construction.  Retain this final projection because the
            # regularized optical solve and finite-precision subtraction above
            # can introduce roundoff-level antisymmetry in the assembled
            # response; this is a numerical invariant enforcement, not a
            # second learned symmetrization.
            macro_total,
            macro_dielectric, macro_elastic,
            0.5 * (physical_tensor + physical_tensor.transpose(-1, -2)),
            electronic_piezo, ionic_piezo,
            factorized_ionic_piezo, displacement_response,
            factors.born_charges, factors.force_constants_flat, internal_strain, optical_operator_flat,
            shared_clamped_elastic, elastic_background, dielectric_background,
            ionic_dielectric, elastic_softening, dielectric, elastic,
        )

    def forward(self, batch) -> torch.Tensor:
        return self.predict_components(batch).tensor

    def potential(self, batch, field: torch.Tensor, eta6: torch.Tensor) -> torch.Tensor:
        components = self.predict_components(batch)
        return self.response(
            components.physical_tensor, components.elastic, components.dielectric, field, eta6,
        )


def model_from_config(config: Mapping[str, object]) -> PiezoJet:
    """Construct the single production PiezoJet architecture from a run config."""
    factor_architecture = str(
        config.get("factor_architecture", "independent_quadratic_response")
    )
    if factor_architecture != "independent_quadratic_response":
        raise ValueError(
            "Production PiezoJet requires "
            "factor_architecture=independent_quadratic_response; "
            "legacy architecture switches are not accepted by model_from_config"
        )
    solve_policy = str(config.get("optical_solve_policy", "regularized"))
    if solve_policy != "regularized":
        raise ValueError(
            "Production PiezoJet requires optical_solve_policy=regularized; "
            "exact propagation is available only through explicit DFPT diagnostics"
        )
    background_architecture = str(config.get("background_architecture", "isotropic"))
    if background_architecture != "isotropic":
        raise ValueError(
            "Production PiezoJet requires background_architecture=isotropic; "
            "the unselected anisotropic candidate is not a maintained fallback"
        )
    return PiezoJet(
        embedding_dim=int(config["embedding_dim"]), cutoff=float(config["cutoff"]),
        num_blocks=int(config["num_blocks"]), radial_basis=int(config["radial_basis"]), radial_hidden=int(config["radial_hidden"]),
        cartesian_channels=int(config.get("cartesian_channels", 48)),
        global_context_dim=int(config.get("global_context_dim", 128)), spectral_channels=int(config.get("spectral_channels", 16)),
        spectral_shells=int(config.get("spectral_shells", 8)), polar_fluctuation_shells=int(config.get("polar_fluctuation_shells", 8)),
        reciprocal_cutoff=float(config.get("reciprocal_cutoff", 7.0)),
        optical_regularization=float(config.get("optical_regularization", 1e-3)),
        optical_stability_cutoff=float(config.get("optical_stability_cutoff", 1e-4)),
        optical_solve_policy=solve_policy,
        factor_architecture=factor_architecture,
        background_architecture=background_architecture,
        displacement_attention_dim=int(config.get("displacement_attention_dim", 64)),
        displacement_cross_rank=int(config.get("displacement_cross_rank", 24)),
    )
