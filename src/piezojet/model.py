"""O(3)-equivariant periodic encoders and atom-coordinate response operators."""

from __future__ import annotations

import math
from typing import Mapping, NamedTuple

import torch
from e3nn import o3
from e3nn.nn import Gate
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter, to_dense_batch

from .tensor_ops import (
    BEC_IRREP_SLICES,
    BEC_TYPE,
    PIEZO_TYPE,
    PIEZO_IRREP_SLICES,
    born_from_irreps,
    piezo_from_irreps,
    piezo_voigt_to_cartesian,
    voigt_to_symmetric_matrix,
)
from .projectors import translation_projector
from .elastic_dielectric_ops import (
    DIELECTRIC_TENSOR,
    dielectric_from_irreps,
    elastic_cartesian_to_voigt,
)


def _radial_basis(distance: torch.Tensor, centers: torch.Tensor, cutoff: float) -> torch.Tensor:
    centers = centers.to(dtype=distance.dtype)
    width = cutoff / max(centers.numel() - 1, 1)
    # The cubic compact-support envelope and its first two derivatives vanish
    # at the cutoff.  This C2 contract is required by response models whose
    # parameter gradients differentiate a geometry Jacobian; the former
    # quadratic envelope was only C1 at the neighbor boundary.
    envelope = (1 - distance / cutoff).clamp_min(0).pow(3)
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


_E3NN_HIDDEN_LAYOUT = (
    (64, "0e"),
    (16, "0o"),
    (24, "1e"),
    (24, "1o"),
    (12, "2e"),
    (12, "2o"),
    (6, "3e"),
    (6, "3o"),
)


def _scaled_hidden_irreps(width_multiplier: float) -> o3.Irreps:
    """Return the deterministic width-scaled electrostatic hidden layout."""
    if not math.isfinite(width_multiplier) or width_multiplier <= 0.0:
        raise ValueError("encoder width multiplier must be finite and positive")
    terms = [
        (max(1, math.floor(base * width_multiplier + 0.5)), irrep)
        for base, irrep in _E3NN_HIDDEN_LAYOUT
    ]
    return o3.Irreps(terms)


def _gate_for(irreps_out: o3.Irreps) -> Gate:
    """Create a scalar-plus-gated non-scalar e3nn activation."""
    scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
    gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
    if not scalars or not gated or irreps_out != scalars + gated:
        raise ValueError("PiezoJet hidden irreps must list scalars before tensors")
    gates = o3.Irreps(f"{gated.num_irreps}x0e")
    scalar_activations = [F.silu if ir.p == 1 else torch.tanh for _, ir in scalars]
    gate = Gate(
        scalars,
        scalar_activations,
        gates,
        [torch.sigmoid],
        gated,
    )
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
        width_multiplier: float = 1.0,
    ):
        super().__init__()
        self.cutoff, self.radial_basis = cutoff, radial_basis
        self.width_multiplier = float(width_multiplier)
        self.register_buffer("radial_centers", torch.linspace(0, cutoff, radial_basis), persistent=False)
        self.embedding = nn.Embedding(119, embedding_dim)
        self.input_irreps = o3.Irreps(f"{embedding_dim}x0e")
        self.hidden_irreps = _scaled_hidden_irreps(self.width_multiplier)
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

    def _node_bases(self, features: CartesianNodeFeatures) -> torch.Tensor:
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
        return torch.stack(
            (vector_identity, identity_vector, vector_quadrupole, quadrupole_vector),
            dim=2,
        )

    def output_basis(
        self,
        features: CartesianNodeFeatures,
        batch_index: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Unrestricted geometric readout span for each material.

        This is an oracle upper bound: every atom/channel/family coefficient is
        allowed to vary independently.  It therefore diagnoses whether the
        actual Cartesian tensor bases can represent a target at all, without
        conflating that question with the coefficient MLP or optimizer.
        """
        bases = self._node_bases(features)
        graphs = int(batch_index.max()) + 1
        result: list[torch.Tensor] = []
        for graph_index in range(graphs):
            mask = batch_index == graph_index
            count = int(mask.sum())
            if count == 0:
                raise ValueError("Piezo output basis received an empty graph")
            result.append(bases[mask].reshape(-1, 3, 3, 3) / float(count))
        return result

    def forward(self, features: CartesianNodeFeatures, batch_index: torch.Tensor) -> torch.Tensor:
        bases = self._node_bases(features)
        vector = features.vector
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


class _CrystalGeometryCache(NamedTuple):
    """Non-learned reciprocal/edge geometry reused for a fixed PyG batch."""

    signature: tuple[object, ...]
    counts: torch.Tensor
    lattice: torch.Tensor
    cosine: torch.Tensor
    sine: torch.Tensor
    radial: torch.Tensor
    directions: torch.Tensor
    reciprocal_mask: torch.Tensor
    edge_source: torch.Tensor
    edge_target: torch.Tensor
    edge_graph: torch.Tensor
    local_kernel: torch.Tensor
    local_kernel_sum: torch.Tensor


def _batch_graph_count(batch, batch_index: torch.Tensor) -> int:
    """Read the PyG batch size without synchronizing a CUDA index tensor."""
    try:
        graphs = batch.num_graphs
    except AttributeError:
        graphs = None
    if graphs is not None:
        return int(graphs)
    ptr = getattr(batch, "ptr", None)
    if isinstance(ptr, torch.Tensor):
        return int(ptr.shape[0] - 1)
    cell = getattr(batch, "cell", None)
    if isinstance(cell, torch.Tensor) and cell.ndim == 3:
        return int(cell.shape[0])
    return int(batch_index.max()) + 1


class CrystalGlobalContext(nn.Module):
    """Invariant context plus a cell-basis-invariant tensorial response operator.

    A reciprocal shell is enumerated by its *physical* scaled length
    ``|2 pi h L^{-T}| V^(1/3)`` rather than by a fixed cube of integer labels.
    Consequently an integral change of lattice basis only re-indexes the same
    shell.  In addition to scalar power spectra, the module constructs the
    origin-invariant polar cross-spectrum ``Re[P_h S_h^*]`` and contracts it
    with ``g_hat ⊗ g_hat`` to yield a polar rank-three response operator.
    """

    _geometry_cache: _CrystalGeometryCache | None

    def __init__(
        self,
        context_dim: int = 128,
        spectral_channels: int = 16,
        spectral_shells: int = 8,
        polar_fluctuation_shells: int = 8,
        reciprocal_cutoff: float = 7.0,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
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
        # This is deliberately a one-batch, non-persistent cache.  It contains
        # only fixed geometry (no learned embeddings and no autograd graph), so
        # repeated optimization steps on the same PyG batch avoid rebuilding
        # reciprocal shells, phases, and edge kernels.  The tensor signatures
        # below invalidate it after device moves or in-place geometry edits.
        self._geometry_cache = None

    @staticmethod
    def _as_cells(cell: torch.Tensor | None, graphs: int) -> torch.Tensor | None:
        if cell is None:
            return None
        if cell.ndim == 2:
            cell = cell.unsqueeze(0)
        if cell.shape != (graphs, 3, 3):
            raise ValueError(f"Expected graph cells [{graphs},3,3], got {tuple(cell.shape)}")
        return cell

    def _physical_shell_from_basis(
        self,
        reciprocal_basis: torch.Tensor,
        limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter one precomputed reciprocal basis by the physical cutoff."""
        dtype, device = reciprocal_basis.dtype, reciprocal_basis.device
        values = torch.arange(-limit, limit + 1, dtype=dtype, device=device)
        integer_indices = torch.stack(torch.meshgrid(values, values, values, indexing="ij"), dim=-1).reshape(-1, 3)
        scaled_reciprocal = integer_indices @ reciprocal_basis
        scaled_norm = torch.linalg.vector_norm(scaled_reciprocal, dim=-1)
        keep = (scaled_norm > torch.finfo(dtype).eps) & (scaled_norm <= self.reciprocal_cutoff)
        integer_indices, scaled_reciprocal, scaled_norm = integer_indices[keep], scaled_reciprocal[keep], scaled_norm[keep]
        return integer_indices, scaled_reciprocal / scaled_norm[:, None], scaled_norm

    @staticmethod
    def _tensor_signature(value: torch.Tensor | None) -> tuple[object, ...]:
        if value is None:
            return (None,)
        return (
            value.data_ptr(),
            tuple(value.shape),
            str(value.device),
            value.dtype,
            value._version,
        )

    def _geometry_signature(self, batch, batch_index: torch.Tensor) -> tuple[object, ...]:
        return (
            id(batch),
            self.reciprocal_cutoff,
            self.spectral_shells,
            self.polar_fluctuation_shells,
            self._tensor_signature(batch_index),
            self._tensor_signature(batch.pos),
            self._tensor_signature(getattr(batch, "cell", None)),
            self._tensor_signature(getattr(batch, "frac", None)),
            self._tensor_signature(batch.edge_index),
            self._tensor_signature(batch.edge_shift),
        )

    def _fixed_geometry(self, batch, batch_index: torch.Tensor) -> _CrystalGeometryCache:
        """Build padded reciprocal geometry once, then reuse it exactly."""
        signature = self._geometry_signature(batch, batch_index)
        differentiable_geometry = any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for value in (
                batch.pos,
                getattr(batch, "cell", None),
                getattr(batch, "frac", None),
                batch.edge_shift,
            )
        )
        if (
            not differentiable_geometry
            and self._geometry_cache is not None
            and self._geometry_cache.signature == signature
        ):
            return self._geometry_cache

        graphs = _batch_graph_count(batch, batch_index)
        dtype, device = batch.pos.dtype, batch.pos.device
        node_counts = torch.bincount(batch_index, minlength=graphs)
        counts = node_counts.to(dtype)
        cell = self._as_cells(getattr(batch, "cell", None), graphs)
        frac = getattr(batch, "frac", None)
        lattice = torch.zeros(graphs, 2, dtype=dtype, device=device)
        shells: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        max_shell = 0
        reciprocal_basis = None
        limits = None
        if cell is not None:
            volume = torch.linalg.det(cell).abs().clamp_min(torch.finfo(dtype).eps)
            lattice = torch.stack((volume.log(), (counts / volume).log()), dim=-1)
            if frac is not None:
                reciprocal_basis = (
                    2.0
                    * torch.pi
                    * torch.linalg.inv(cell).transpose(-1, -2)
                    * volume.pow(1.0 / 3.0)[:, None, None]
                )
                smallest = torch.linalg.svdvals(reciprocal_basis).amin(dim=-1)
                smallest = smallest.clamp_min(torch.finfo(dtype).eps)
                limits = torch.ceil(self.reciprocal_cutoff / smallest)
                limits = limits.clamp_min(1).to(dtype=torch.long)

        # Transfer all ragged integer metadata once.  Per-graph ``int(tensor)``
        # conversions serialize CUDA execution and dominated small-batch runs.
        metadata = (
            node_counts[:, None]
            if limits is None
            else torch.stack((node_counts, limits), dim=-1)
        )
        metadata_rows = metadata.detach().cpu().tolist()
        node_count_values = [int(row[0]) for row in metadata_rows]
        if reciprocal_basis is not None:
            for graph_index, row in enumerate(metadata_rows):
                shell = self._physical_shell_from_basis(
                    reciprocal_basis[graph_index], int(row[1])
                )
                shells.append(shell)
                max_shell = max(max_shell, int(shell[0].shape[0]))

        max_nodes = max(node_count_values)
        cosine = torch.zeros(graphs, max_nodes, max_shell, dtype=dtype, device=device)
        sine = torch.zeros_like(cosine)
        radial = torch.zeros(
            graphs, max_shell, self.spectral_shells, dtype=dtype, device=device
        )
        directions = torch.zeros(graphs, max_shell, 3, dtype=dtype, device=device)
        reciprocal_mask = torch.zeros(graphs, max_shell, dtype=torch.bool, device=device)
        if shells:
            for graph_index, (indices, shell_directions, scaled_norm) in enumerate(shells):
                shell_size = int(indices.shape[0])
                if shell_size == 0:
                    continue
                node_mask = batch_index == graph_index
                node_count = node_count_values[graph_index]
                phase = 2.0 * torch.pi * (frac[node_mask] @ indices.transpose(0, 1))
                cosine[graph_index, :node_count, :shell_size] = torch.cos(phase)
                sine[graph_index, :node_count, :shell_size] = torch.sin(phase)
                radial[graph_index, :shell_size] = torch.exp(
                    -((scaled_norm.unsqueeze(-1) - self.spectral_centers.to(dtype)) / 0.75).square()
                )
                directions[graph_index, :shell_size] = shell_directions
                reciprocal_mask[graph_index, :shell_size] = True

        edge_source, edge_target = batch.edge_index
        edge_graph = batch_index[edge_target]
        edge_vectors = batch.pos[edge_source] - batch.pos[edge_target] + batch.edge_shift
        edge_distance = torch.linalg.vector_norm(edge_vectors, dim=-1)
        local_kernel = torch.exp(
            -((edge_distance.unsqueeze(-1) - self.fluctuation_centers.to(dtype)) / 0.6).square()
        )
        local_kernel_sum = scatter(
            local_kernel, edge_graph, dim=0, dim_size=graphs, reduce="sum"
        )
        geometry = _CrystalGeometryCache(
            signature, counts, lattice, cosine, sine, radial, directions,
            reciprocal_mask, edge_source, edge_target, edge_graph,
            local_kernel, local_kernel_sum,
        )
        # Holding a response Jacobian graph on the module would retain the
        # previous optimizer step and double peak memory on the next call.
        # Fixed reference batches keep the fast cache; differentiable
        # perturbed geometry is deliberately ephemeral.
        if not differentiable_geometry:
            self._geometry_cache = geometry
        return geometry

    def forward(
        self,
        batch,
        batch_index: torch.Tensor,
        polar_vectors: torch.Tensor | None = None,
        *,
        return_operator: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        graphs = _batch_graph_count(batch, batch_index)
        dtype, device = batch.pos.dtype, batch.pos.device
        geometry = self._fixed_geometry(batch, batch_index)
        counts = geometry.counts
        composition = scatter(F.one_hot(batch.z, num_classes=119).to(dtype), batch_index, dim=0, dim_size=graphs, reduce="sum")
        composition = composition / counts.unsqueeze(-1).clamp_min(1.0)
        chemical_spectrum = torch.zeros(graphs, self.spectral_shells * self.spectral_channels, dtype=dtype, device=device)
        polar_spectrum = torch.zeros(graphs, self.spectral_shells, dtype=dtype, device=device)
        fluctuation_spectrum = torch.zeros(graphs, self.polar_fluctuation_shells, dtype=dtype, device=device)
        tensorial_operator = torch.zeros(graphs, 3, 3, 3, dtype=dtype, device=device)
        if geometry.radial.shape[1] > 0:
            species = self.species_features(batch.z).to(dtype)
            species_dense, _ = to_dense_batch(species, batch_index, batch_size=graphs)
            cosine_t = geometry.cosine.transpose(1, 2)
            sine_t = geometry.sine.transpose(1, 2)
            chemical_real = torch.bmm(cosine_t, species_dense)
            chemical_imag = torch.bmm(sine_t, species_dense)
            chemical_power = (chemical_real.square() + chemical_imag.square()) / counts.square().clamp_min(1.0)[:, None, None]
            radial_sum = geometry.radial.sum(dim=1).clamp_min(1e-8)
            chemical_spectrum = (
                torch.einsum("ghs,ghc->gsc", geometry.radial, chemical_power)
                / radial_sum.unsqueeze(-1)
            ).reshape(graphs, -1)
            if polar_vectors is not None:
                polar_dense, _ = to_dense_batch(polar_vectors, batch_index, batch_size=graphs)
                inverse_counts = counts.clamp_min(1.0).reciprocal()[:, None, None]
                polar_real = torch.bmm(cosine_t, polar_dense) * inverse_counts
                polar_imag = torch.bmm(sine_t, polar_dense) * inverse_counts
                polar_power = polar_real.square().sum(dim=-1) + polar_imag.square().sum(dim=-1)
                polar_spectrum = (
                    torch.einsum("ghs,gh->gs", geometry.radial, polar_power)
                    / radial_sum
                )
                references = self.phase_reference(batch.z).to(dtype)
                reference_dense, _ = to_dense_batch(references, batch_index, batch_size=graphs)
                reference_real = torch.bmm(cosine_t, reference_dense).squeeze(-1) * inverse_counts.squeeze(-1)
                reference_imag = torch.bmm(sine_t, reference_dense).squeeze(-1) * inverse_counts.squeeze(-1)
                cross_polar = polar_real * reference_real.unsqueeze(-1) + polar_imag * reference_imag.unsqueeze(-1)
                shell_weight = self.operator_kernel(geometry.radial).squeeze(-1)
                tensorial_operator = torch.einsum(
                    "gh,ghi,ghj,ghk->gijk",
                    shell_weight, cross_polar, geometry.directions, geometry.directions,
                ) / geometry.reciprocal_mask.sum(dim=1).clamp_min(1)[:, None, None, None]
                uniform = scatter(polar_vectors, batch_index, dim=0, dim_size=graphs, reduce="mean")
                uniform_shape = torch.einsum("gi,gj,gk->gijk", uniform, uniform, uniform)
                tensorial_operator = tensorial_operator + (
                    self.uniform_operator_weight * uniform_shape
                    / uniform.square().sum(dim=-1).clamp_min(1e-8)[:, None, None, None]
                )
        if polar_vectors is not None and geometry.local_kernel.numel() > 0:
            centered = polar_vectors - scatter(
                polar_vectors, batch_index, dim=0, dim_size=graphs, reduce="mean"
            )[batch_index]
            correlation = (
                centered[geometry.edge_source] * centered[geometry.edge_target]
            ).sum(dim=-1, keepdim=True)
            numerator = scatter(
                correlation * geometry.local_kernel, geometry.edge_graph,
                dim=0, dim_size=graphs, reduce="sum",
            )
            fluctuation_spectrum = numerator / geometry.local_kernel_sum.clamp_min(1e-8)
        context = self.network(torch.cat((composition, geometry.lattice, chemical_spectrum, polar_spectrum, fluctuation_spectrum), dim=-1))
        return (context, tensorial_operator) if return_operator else context


class ElectromechanicalJetPrediction(NamedTuple):
    """Identifiable coefficients of a first-order electromechanical response jet.

    These coefficients are response derivatives, not a claim that the
    electrostatic branch is generated by the mechanical factor energy.
    """

    born_charges: torch.Tensor
    electronic_piezo: torch.Tensor
    electronic_dielectric: torch.Tensor | None


class _PerturbedCrystalBatch(NamedTuple):
    """Minimal immutable graph view for differential polarization."""

    z: torch.Tensor
    pos: torch.Tensor
    frac: torch.Tensor
    cell: torch.Tensor
    edge_index: torch.Tensor
    edge_shift: torch.Tensor
    batch: torch.Tensor
    num_nodes: int


class EquivariantGlobalAttention(nn.Module):
    """Invariant attention weights acting on complete equivariant node values."""

    def __init__(self, irreps: o3.Irreps, attention_dim: int = 64):
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        scalar_irreps = o3.Irreps(f"{attention_dim}x0e")
        self.query = o3.Linear(irreps, scalar_irreps)
        self.key = o3.Linear(irreps, scalar_irreps)
        self.value = o3.Linear(irreps, irreps)
        self.scale = attention_dim**-0.5
        self.residual_gate = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        node_features: torch.Tensor,
        batch_index: torch.Tensor,
        graphs: int | None = None,
    ) -> torch.Tensor:
        graphs = int(graphs) if graphs is not None else int(batch_index.max()) + 1
        queries, mask = to_dense_batch(
            self.query(node_features), batch_index, batch_size=graphs
        )
        keys, key_mask = to_dense_batch(
            self.key(node_features), batch_index, batch_size=graphs
        )
        values, value_mask = to_dense_batch(
            self.value(node_features), batch_index, batch_size=graphs
        )
        if not torch.equal(mask, key_mask) or not torch.equal(mask, value_mask):
            raise RuntimeError("Electrostatic attention batching mismatch")
        scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale
        scores = scores.masked_fill(~mask[:, None, :], -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.bmm(attention, values)[mask]
        return node_features + torch.tanh(self.residual_gate) * attended


class EquivariantResponseAdapter(nn.Module):
    """Small task-specific, context-gated equivariant residual adapter.

    The mixing map is O(3)-equivariant and the gate depends only on the
    invariant crystal context.  Thus the adapter can specialize a response
    task without choosing a frame or turning the shared first-order response
    jet into a nonlinear polarization model.  Its residual scale starts at
    zero, so loading an A1 encoder has a well-defined shared-trunk limit.
    """

    def __init__(self, irreps: o3.Irreps, context_dim: int):
        super().__init__()
        self.irreps = irreps
        self.mix = o3.Linear(irreps, irreps)
        self.context_gates = nn.Sequential(
            nn.Linear(context_dim, context_dim), nn.SiLU(),
            nn.Linear(context_dim, len(irreps)),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        node_features: torch.Tensor,
        context: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        gates = self.context_gates(context)[batch_index]
        mixed = self.mix(node_features)
        pieces = [
            mixed[..., block] * (1.0 + torch.tanh(gates[..., index : index + 1]))
            for index, block in enumerate(self.irreps.slices())
        ]
        adapted = torch.cat(pieces, dim=-1)
        return node_features + torch.tanh(self.residual_scale) * adapted


class IrrepRMSNorm(nn.Module):
    """RMS-normalize each irrep multiplicity without mixing tensor components."""

    def __init__(self, irreps: o3.Irreps, epsilon: float = 1e-8):
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError("Irrep RMS epsilon must be positive")
        self.irreps = o3.Irreps(irreps)
        self.epsilon = float(epsilon)
        self.gain = nn.Parameter(torch.ones(self.irreps.num_irreps))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.irreps.dim:
            raise ValueError("Irrep RMS input dimension mismatch")
        pieces: list[torch.Tensor] = []
        offset = 0
        gain_offset = 0
        for (multiplicity, irrep), block in zip(
            self.irreps, self.irreps.slices(), strict=True
        ):
            values = features[..., block].reshape(
                *features.shape[:-1], multiplicity, irrep.dim
            )
            # Normalize one complete multiplicity block.  In particular, do
            # not normalize each l=0 channel independently: that would reduce
            # every nonzero scalar to its sign and erase response amplitude.
            # Summing squared tensor components and channels is O(3)-invariant
            # while preserving their relative magnitudes.
            rms = values.square().mean(dim=(-2, -1), keepdim=True).add(
                self.epsilon
            ).sqrt()
            gain = self.gain[
                gain_offset : gain_offset + multiplicity
            ].reshape(*([1] * (values.ndim - 2)), multiplicity, 1)
            pieces.append((values / rms * gain).reshape(*features.shape[:-1], -1))
            offset += multiplicity * irrep.dim
            gain_offset += multiplicity
        if offset != features.shape[-1] or gain_offset != self.irreps.num_irreps:
            raise RuntimeError("Irrep RMS layout accounting mismatch")
        return torch.cat(pieces, dim=-1)


class TrainableIrrepAdapter(nn.Module):
    """Nonzero, channel-gated equivariant residual adapter.

    The residual amplitude and the crystal-context gate are invariant scalars
    defined separately for every irrep multiplicity.  Unlike the retained
    A1.5 zero-gate control, both the equivariant mixing weights and the context
    route receive gradients on the first optimization step.
    """

    def __init__(
        self,
        irreps: o3.Irreps,
        context_dim: int,
        initial_scale: float = 0.075,
        mixing_std: float = 1e-3,
    ):
        super().__init__()
        if initial_scale <= 0.0 or not math.isfinite(initial_scale):
            raise ValueError("Adapter initial scale must be finite and positive")
        if mixing_std <= 0.0 or not math.isfinite(mixing_std):
            raise ValueError("Adapter mixing std must be finite and positive")
        self.irreps = o3.Irreps(irreps)
        self.normalization = IrrepRMSNorm(self.irreps)
        self.mix = o3.Linear(self.irreps, self.irreps)
        self.context_gates = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, self.irreps.num_irreps),
        )
        inverse_softplus = math.log(math.expm1(initial_scale))
        self.scale_logits = nn.Parameter(
            torch.full((self.irreps.num_irreps,), inverse_softplus)
        )
        nn.init.normal_(self.mix.weight, mean=0.0, std=mixing_std)
        final_context = self.context_gates[-1]
        nn.init.normal_(final_context.weight, mean=0.0, std=mixing_std)
        nn.init.zeros_(final_context.bias)

    def _expand_multiplicity_scalars(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.irreps.num_irreps:
            raise ValueError("Adapter multiplicity-gate dimension mismatch")
        pieces: list[torch.Tensor] = []
        offset = 0
        for multiplicity, irrep in self.irreps:
            channel = values[..., offset : offset + multiplicity]
            pieces.append(
                channel.unsqueeze(-1)
                .expand(*channel.shape, irrep.dim)
                .reshape(*channel.shape[:-1], multiplicity * irrep.dim)
            )
            offset += multiplicity
        if offset != self.irreps.num_irreps:
            raise RuntimeError("Adapter gate layout accounting mismatch")
        return torch.cat(pieces, dim=-1)

    def forward(
        self,
        node_features: torch.Tensor,
        context: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.normalization(node_features)
        mixed = self.mix(normalized)
        contextual = 1.0 + torch.tanh(self.context_gates(context)[batch_index])
        amplitude = F.softplus(self.scale_logits) * contextual
        expanded_amplitude = self._expand_multiplicity_scalars(amplitude)
        return node_features + expanded_amplitude * mixed


class ElectromechanicalJetHead(nn.Module):
    r"""Exact first-order electromechanical jet around a reference crystal.

    The model never predicts an absolute Berry-phase polarization.  Instead it
        represents the complete identifiable displacement--strain part of the
        local polarization-response jet

    ``Delta P = c_e/Omega sum_k Z_k^T u_k + e_el : eta + chi_el E``.

    Consequently the BEC and electronic piezo tensors are, by construction,
    the displacement and strain Jacobians of one vector-valued map.  A PBC
    steerable encoder supplies explicit l=1,2,3 channels; invariant global
    attention and reciprocal polar--chemical context act without selecting a
    crystal frame.  This is one declared candidate, not a conditional fallback.
    """

    PIEZO_C_PER_M2 = 16.02176634
    VACUUM_PERMITTIVITY_C_PER_VM = 8.8541878128e-12

    def __init__(
        self,
        embedding_dim: int = 32,
        cutoff: float = 5.0,
        lmax: int = 3,
        num_blocks: int = 3,
        radial_basis: int = 12,
        radial_hidden: int = 64,
        global_context_dim: int = 128,
        spectral_channels: int = 16,
        spectral_shells: int = 8,
        polar_fluctuation_shells: int = 8,
        reciprocal_cutoff: float = 7.0,
        attention_dim: int = 64,
        encoder_width_multiplier: float = 1.0,
    ):
        super().__init__()
        if lmax < 3:
            raise ValueError("ElectromechanicalJetHead requires explicit lmax >= 3")
        self.encoder = PeriodicCrystalEncoder(
            embedding_dim=embedding_dim,
            cutoff=cutoff,
            lmax=lmax,
            num_blocks=num_blocks,
            radial_basis=radial_basis,
            radial_hidden=radial_hidden,
            width_multiplier=encoder_width_multiplier,
        )
        irreps = self.encoder.hidden_irreps
        self.global_attention = EquivariantGlobalAttention(irreps, attention_dim)
        self.local_polar = o3.Linear(irreps, o3.Irreps("1x1o"))
        self.global_context = CrystalGlobalContext(
            global_context_dim,
            spectral_channels,
            spectral_shells,
            polar_fluctuation_shells,
            reciprocal_cutoff,
        )
        self.electronic_irreps = o3.Linear(irreps, PIEZO_TYPE)
        self.born_irreps = o3.Linear(irreps, BEC_TYPE)
        self.dielectric_irreps = o3.Linear(irreps, DIELECTRIC_TENSOR)
        self.electronic_context_gates = nn.Sequential(
            nn.Linear(global_context_dim, global_context_dim), nn.SiLU(),
            nn.Linear(global_context_dim, len(PIEZO_IRREP_SLICES)),
        )
        self.born_context_gates = nn.Sequential(
            nn.Linear(global_context_dim, global_context_dim), nn.SiLU(),
            nn.Linear(global_context_dim, len(BEC_IRREP_SLICES)),
        )

    @staticmethod
    def _gated_blocks(
        coordinates: torch.Tensor,
        gates: torch.Tensor,
        blocks: Mapping[str, slice],
    ) -> torch.Tensor:
        if gates.shape[-1] != len(blocks):
            raise ValueError("Context gate count does not match irrep blocks")
        return torch.cat(
            [
                coordinates[..., block] * (1.0 + torch.tanh(gates[..., index : index + 1]))
                for index, block in enumerate(blocks.values())
            ],
            dim=-1,
        )

    def encode_response_features(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = self.encoder(batch)
        graphs = _batch_graph_count(batch, batch.batch)
        node_features = self.global_attention(node_features, batch.batch, graphs)
        local_polar = self.local_polar(node_features)
        context = self.global_context(batch, batch.batch, local_polar)
        graphs = context.shape[0]
        graph_features = scatter(
            node_features, batch.batch, dim=0, dim_size=graphs, reduce="mean"
        )
        return node_features, graph_features, context

    def decode_born(
        self,
        node_features: torch.Tensor,
        context: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        graphs = context.shape[0]
        born_coordinates = self._gated_blocks(
            self.born_irreps(node_features),
            self.born_context_gates(context)[batch_index],
            BEC_IRREP_SLICES,
        )
        born = born_from_irreps(born_coordinates)
        # Acoustic charge neutrality is an exact response-Jacobian invariant.
        born = born - scatter(
            born, batch_index, dim=0, dim_size=graphs, reduce="mean"
        )[batch_index]
        return born

    def decode_electronic_piezo(
        self,
        graph_features: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        electronic_coordinates = self._gated_blocks(
            self.electronic_irreps(graph_features),
            self.electronic_context_gates(context),
            PIEZO_IRREP_SLICES,
        )
        electronic = piezo_from_irreps(electronic_coordinates)
        return 0.5 * (electronic + electronic.transpose(-1, -2))

    def decode_dielectric(self, graph_features: torch.Tensor) -> torch.Tensor:
        dielectric_root = dielectric_from_irreps(
            self.dielectric_irreps(graph_features)
        )
        identity = torch.eye(
            3, dtype=dielectric_root.dtype, device=dielectric_root.device
        )
        return (
            identity
            + dielectric_root @ dielectric_root.transpose(-1, -2)
        )

    def decode_electronic(
        self,
        graph_features: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.decode_electronic_piezo(graph_features, context),
            self.decode_dielectric(graph_features),
        )

    def coefficients(self, batch) -> ElectromechanicalJetPrediction:
        node_features, graph_features, context = self.encode_response_features(batch)
        born = self.decode_born(node_features, context, batch.batch)
        electronic, electronic_dielectric = self.decode_electronic(
            graph_features, context
        )
        return ElectromechanicalJetPrediction(
            born, electronic, electronic_dielectric
        )

    @classmethod
    def polarization_increment_from_coefficients(
        cls,
        prediction: ElectromechanicalJetPrediction,
        displacement: torch.Tensor,
        strain: torch.Tensor,
        batch,
        electric_field_v_per_m: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if displacement.shape != (batch.num_nodes, 3):
            raise ValueError("Displacement probe must have shape [num_nodes,3]")
        graphs = prediction.electronic_piezo.shape[0]
        if strain.shape != (graphs, 3, 3):
            raise ValueError("Strain probe must have shape [graphs,3,3]")
        if not torch.allclose(strain, strain.transpose(-1, -2), atol=1e-7, rtol=1e-7):
            raise ValueError("Electromechanical-jet strain probe must be symmetric")
        if electric_field_v_per_m is None:
            electric_field_v_per_m = strain.new_zeros((graphs, 3))
        if electric_field_v_per_m.shape != (graphs, 3):
            raise ValueError("Electric-field probe must have shape [graphs,3]")
        cells = batch.cell.reshape(graphs, 3, 3)
        volume = torch.linalg.det(cells).abs().clamp_min(
            torch.finfo(cells.dtype).eps
        )
        ionic = scatter(
            torch.einsum("nai,na->ni", prediction.born_charges, displacement),
            batch.batch,
            dim=0,
            dim_size=graphs,
            reduce="sum",
        )
        ionic = cls.PIEZO_C_PER_M2 * ionic / volume[:, None]
        electronic = torch.einsum(
            "gijk,gjk->gi", prediction.electronic_piezo, strain
        )
        identity = torch.eye(
            3,
            dtype=prediction.electronic_dielectric.dtype,
            device=prediction.electronic_dielectric.device,
        )
        field = cls.VACUUM_PERMITTIVITY_C_PER_VM * torch.einsum(
            "gij,gj->gi",
            prediction.electronic_dielectric - identity,
            electric_field_v_per_m,
        )
        return ionic + electronic + field

    def forward(
        self,
        batch,
        displacement: torch.Tensor,
        strain: torch.Tensor,
        electric_field_v_per_m: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.polarization_increment_from_coefficients(
            self.coefficients(batch), displacement, strain, batch,
            electric_field_v_per_m,
        )


class SoftSharedElectromechanicalJetHead(ElectromechanicalJetHead):
    """A1.5: one shared electrostatic trunk with response-specific adapters.

    BECs and clamped-ion piezoelectricity are still the coefficients of the
    same *linear* response map.  Only their hidden representations receive
    separate equivariant residual adapters, avoiding A1's unnecessary
    hard-sharing hypothesis while retaining the same tensor constraints and
    data interface.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.born_adapter = EquivariantResponseAdapter(
            self.encoder.hidden_irreps, self.global_context.context_dim
        )
        self.electronic_adapter = EquivariantResponseAdapter(
            self.encoder.hidden_irreps, self.global_context.context_dim
        )
        self.dielectric_adapter = EquivariantResponseAdapter(
            self.encoder.hidden_irreps, self.global_context.context_dim
        )

    def coefficients(self, batch) -> ElectromechanicalJetPrediction:
        node_features, _, context = self.encode_response_features(batch)
        born_features = self.born_adapter(node_features, context, batch.batch)
        electronic_features = self.electronic_adapter(
            node_features, context, batch.batch
        )
        dielectric_features = self.dielectric_adapter(
            node_features, context, batch.batch
        )
        graphs = context.shape[0]
        electronic_graph_features = scatter(
            electronic_features, batch.batch, dim=0, dim_size=graphs, reduce="mean"
        )
        dielectric_graph_features = scatter(
            dielectric_features, batch.batch, dim=0, dim_size=graphs, reduce="mean"
        )
        born = self.decode_born(born_features, context, batch.batch)
        electronic = self.decode_electronic_piezo(
            electronic_graph_features, context
        )
        dielectric = self.decode_dielectric(dielectric_graph_features)
        return ElectromechanicalJetPrediction(born, electronic, dielectric)


class HierarchicalElectromechanicalJetHead(ElectromechanicalJetHead):
    """A1.6: hierarchical sharing for the first-order electrostatic jet.

    A common periodic chemistry/geometry encoder feeds a charge--screening
    response trunk shared by BEC and dielectric prediction, and an independent
    polar--strain response trunk for clamped-ion piezoelectricity.  Each task
    then receives its own nonzero per-irrep adapter.  The split changes neural
    parameter sharing only; all three tensors remain coefficients of the same
    declared first-order polarization increment.

    The trunk names intentionally describe response statistics rather than
    deleting odd-parity hidden covariants: even outputs may depend on even
    combinations of odd covariants, so both trunks retain the complete O(3)
    hidden representation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        irreps = self.encoder.hidden_irreps
        context_dim = self.global_context.context_dim
        self.charge_screening_trunk = TrainableIrrepAdapter(irreps, context_dim)
        self.polar_strain_trunk = TrainableIrrepAdapter(irreps, context_dim)
        self.born_adapter = TrainableIrrepAdapter(irreps, context_dim)
        self.dielectric_adapter = TrainableIrrepAdapter(irreps, context_dim)
        self.electronic_adapter = TrainableIrrepAdapter(irreps, context_dim)

    def coefficients(self, batch) -> ElectromechanicalJetPrediction:
        node_features, _, context = self.encode_response_features(batch)
        charge_screening = self.charge_screening_trunk(
            node_features, context, batch.batch
        )
        polar_strain = self.polar_strain_trunk(
            node_features, context, batch.batch
        )
        born_features = self.born_adapter(
            charge_screening, context, batch.batch
        )
        dielectric_features = self.dielectric_adapter(
            charge_screening, context, batch.batch
        )
        electronic_features = self.electronic_adapter(
            polar_strain, context, batch.batch
        )
        graphs = context.shape[0]
        electronic_graph_features = scatter(
            electronic_features,
            batch.batch,
            dim=0,
            dim_size=graphs,
            reduce="mean",
        )
        dielectric_graph_features = scatter(
            dielectric_features,
            batch.batch,
            dim=0,
            dim_size=graphs,
            reduce="mean",
        )
        born = self.decode_born(born_features, context, batch.batch)
        electronic = self.decode_electronic_piezo(
            electronic_graph_features, context
        )
        dielectric = self.decode_dielectric(dielectric_graph_features)
        return ElectromechanicalJetPrediction(born, electronic, dielectric)


class IndependentElectrostaticHeads(nn.Module):
    """A0 control with statistically independent BEC and piezo generators."""

    def __init__(self, **kwargs):
        super().__init__()
        self.born_generator = ElectromechanicalJetHead(**kwargs)
        self.piezo_generator = ElectromechanicalJetHead(**kwargs)
        self.dielectric_generator = ElectromechanicalJetHead(**kwargs)
        # A0 changes parameter sharing, not the initial response function.
        # Clone values before pruning so every independent task starts from
        # exactly the same random trunk/decoder state as A1 under the same
        # seed, while retaining distinct Parameter objects and optimizer state.
        initial_state = self.born_generator.state_dict()
        self.piezo_generator.load_state_dict(initial_state, strict=True)
        self.dielectric_generator.load_state_dict(initial_state, strict=True)
        # A0 has three independent backbones but no dead task heads.  Removing
        # the unused decoders keeps its parameter count and optimizer state
        # faithful to the actual control rather than silently carrying half of
        # two A1 models.
        del self.born_generator.electronic_irreps
        del self.born_generator.dielectric_irreps
        del self.born_generator.electronic_context_gates
        del self.piezo_generator.born_irreps
        del self.piezo_generator.born_context_gates
        del self.piezo_generator.dielectric_irreps
        del self.dielectric_generator.born_irreps
        del self.dielectric_generator.born_context_gates
        del self.dielectric_generator.electronic_irreps
        del self.dielectric_generator.electronic_context_gates

    def born_charges(self, batch) -> torch.Tensor:
        born_features, _, born_context = (
            self.born_generator.encode_response_features(batch)
        )
        return self.born_generator.decode_born(
            born_features, born_context, batch.batch
        )

    def electronic_response(
        self, batch
    ) -> torch.Tensor:
        _, piezo_features, piezo_context = (
            self.piezo_generator.encode_response_features(batch)
        )
        return self.piezo_generator.decode_electronic_piezo(
            piezo_features, piezo_context
        )

    def dielectric_response(self, batch) -> torch.Tensor:
        _, dielectric_features, _ = (
            self.dielectric_generator.encode_response_features(batch)
        )
        return self.dielectric_generator.decode_dielectric(dielectric_features)

    def coefficients(self, batch) -> ElectromechanicalJetPrediction:
        born = self.born_charges(batch)
        electronic = self.electronic_response(batch)
        dielectric = self.dielectric_response(batch)
        return ElectromechanicalJetPrediction(
            born, electronic, dielectric,
        )


class PolarizationStateNetwork(nn.Module):
    r"""Equivariant polarization state used only through relative changes.

    No absolute or Berry-branch polarization label is assigned to this output.
    Explicit ``l<=3`` node features, invariant global attention, and reciprocal
    scalar context provide the nonlinear geometry dependence whose response
    Jacobian is supervised.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        cutoff: float = 5.0,
        lmax: int = 3,
        num_blocks: int = 3,
        radial_basis: int = 12,
        radial_hidden: int = 64,
        global_context_dim: int = 128,
        spectral_channels: int = 16,
        spectral_shells: int = 8,
        polar_fluctuation_shells: int = 8,
        reciprocal_cutoff: float = 7.0,
        attention_dim: int = 64,
    ):
        super().__init__()
        if lmax < 3:
            raise ValueError("PolarizationStateNetwork requires explicit lmax >= 3")
        self.encoder = PeriodicCrystalEncoder(
            embedding_dim=embedding_dim,
            cutoff=cutoff,
            lmax=lmax,
            num_blocks=num_blocks,
            radial_basis=radial_basis,
            radial_hidden=radial_hidden,
        )
        irreps = self.encoder.hidden_irreps
        self.global_attention = EquivariantGlobalAttention(irreps, attention_dim)
        self.local_polar = o3.Linear(irreps, o3.Irreps("1x1o"))
        self.global_context = CrystalGlobalContext(
            global_context_dim,
            spectral_channels,
            spectral_shells,
            polar_fluctuation_shells,
            reciprocal_cutoff,
        )
        self.polarization_head = o3.Linear(irreps, o3.Irreps("1x1o"))
        self.context_gate = nn.Sequential(
            nn.Linear(global_context_dim, global_context_dim),
            nn.SiLU(),
            nn.Linear(global_context_dim, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        node_features = self.encoder(batch)
        graphs = _batch_graph_count(batch, batch.batch)
        node_features = self.global_attention(node_features, batch.batch, graphs)
        local_polar = self.local_polar(node_features)
        context = self.global_context(batch, batch.batch, local_polar)
        graphs = context.shape[0]
        graph_features = scatter(
            node_features, batch.batch, dim=0, dim_size=graphs, reduce="mean"
        )
        vector = self.polarization_head(graph_features)
        return vector * (1.0 + torch.tanh(self.context_gate(context)))


class NonlinearDifferentialPolarizationTower(nn.Module):
    r"""Literal nonlinear differential-polarization response generator.

    ``Delta P_theta(x;u,eta) = P_theta(T_eta(x+u_o)) - P_theta(x)`` is
    evaluated on perturbed geometry, and both Born charges and electronic
    piezo coefficients are differentiated from that same map at zero.  This is
    deliberately separate from the retained shared linear-coefficient control.
    """

    PIEZO_C_PER_M2 = ElectromechanicalJetHead.PIEZO_C_PER_M2

    def __init__(self, *, polarization_variable: str, **kwargs):
        super().__init__()
        if polarization_variable not in {"cartesian", "reduced"}:
            raise ValueError(
                "polarization_variable must be exactly 'cartesian' or 'reduced'"
            )
        self.polarization_variable = polarization_variable
        self.state = PolarizationStateNetwork(**kwargs)

    def _polarization_variable(
        self,
        cartesian: torch.Tensor,
        eta6: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw P or reduced P0=det(F) F^-1 P for each graph."""
        if self.polarization_variable == "cartesian":
            return cartesian
        strain = voigt_to_symmetric_matrix(eta6)
        identity = torch.eye(3, dtype=eta6.dtype, device=eta6.device)
        deformation = identity.unsqueeze(0) + strain
        inverse_action = torch.linalg.solve(
            deformation, cartesian.unsqueeze(-1)
        ).squeeze(-1)
        return torch.linalg.det(deformation).unsqueeze(-1) * inverse_action

    @staticmethod
    def _cells(batch, graphs: int) -> torch.Tensor:
        cells = batch.cell
        if cells.ndim == 2:
            cells = cells.unsqueeze(0)
        if cells.shape != (graphs, 3, 3):
            raise ValueError(
                f"Expected cells [{graphs},3,3], got {tuple(cells.shape)}"
            )
        return cells

    @classmethod
    def _perturbed_batch(
        cls,
        batch,
        displacement: torch.Tensor,
        eta6: torch.Tensor,
    ) -> _PerturbedCrystalBatch:
        if displacement.shape != (batch.num_nodes, 3):
            raise ValueError("Displacement must have shape [num_nodes,3]")
        graphs = int(batch.batch.max()) + 1
        if eta6.shape != (graphs, 6):
            raise ValueError(f"Engineering strain must have shape [{graphs},6]")
        cells = cls._cells(batch, graphs)
        centered = displacement - scatter(
            displacement, batch.batch, dim=0, dim_size=graphs, reduce="mean"
        )[batch.batch]
        strain = voigt_to_symmetric_matrix(eta6)
        identity = torch.eye(3, dtype=batch.pos.dtype, device=batch.pos.device)
        deformation = identity.unsqueeze(0) + strain
        displaced = batch.pos + centered
        pos = torch.einsum("nj,nij->ni", displaced, deformation[batch.batch])
        cell = torch.einsum("gaj,gij->gai", cells, deformation)
        edge_graph = batch.batch[batch.edge_index[1]]
        edge_shift = torch.einsum(
            "ej,eij->ei", batch.edge_shift, deformation[edge_graph]
        )
        # Use the undeformed reference lattice for internal fractional
        # coordinates; a homogeneous strain then cancels exactly here.
        inverse_reference = torch.linalg.inv(cells)
        frac = torch.einsum(
            "nj,njk->nk", displaced, inverse_reference[batch.batch]
        )
        return _PerturbedCrystalBatch(
            batch.z, pos, frac, cell, batch.edge_index, edge_shift,
            batch.batch, int(batch.num_nodes),
        )

    def polarization_increment(
        self,
        batch,
        displacement: torch.Tensor,
        eta6: torch.Tensor,
    ) -> torch.Tensor:
        graphs = int(batch.batch.max()) + 1
        zeros_u = torch.zeros_like(batch.pos)
        zeros_eta = torch.zeros(
            graphs, 6, dtype=batch.pos.dtype, device=batch.pos.device
        )
        # Both arguments use the same canonical proxy, making Delta P(0,0)
        # bitwise zero without giving P_theta(x) an absolute physical branch.
        reference = self._perturbed_batch(batch, zeros_u, zeros_eta)
        perturbed = self._perturbed_batch(batch, displacement, eta6)
        perturbed_state = self._polarization_variable(self.state(perturbed), eta6)
        reference_state = self._polarization_variable(self.state(reference), zeros_eta)
        return perturbed_state - reference_state

    def coefficients(
        self,
        batch,
        *,
        create_graph: bool | None = None,
    ) -> ElectromechanicalJetPrediction:
        """Use exactly three reverse VJPs, one per polarization component.

        e3nn's scripted tensor products contain a detach view without a vmap
        batching rule, so ``is_grads_batched`` is not a valid execution path.
        The output dimension is physically fixed at three; this loop is
        constant-size and is the sole maintained Jacobian implementation.
        """
        create_graph = self.training if create_graph is None else bool(create_graph)
        graphs = int(batch.batch.max()) + 1
        # no_grad can be locally overridden; inference_mode must not wrap this
        # response-Jacobian path.
        with torch.enable_grad():
            displacement = torch.zeros_like(batch.pos, requires_grad=True)
            eta6 = torch.zeros(
                graphs, 6, dtype=batch.pos.dtype, device=batch.pos.device,
                requires_grad=True,
            )
            perturbed = self._perturbed_batch(batch, displacement, eta6)
            # d[P(T(x)) - P(x)]/d(u,eta) = dP(T(x))/d(u,eta): the reference
            # state is constant with respect to both perturbations.  Eliding
            # that second network evaluation is an exact derivative identity,
            # and halves the second-order training graph without changing the
            # literal public increment above.
            increment = self._polarization_variable(self.state(perturbed), eta6)
            displacement_rows = []
            strain_rows = []
            for component in range(3):
                grad_u, grad_eta = torch.autograd.grad(
                    increment[:, component].sum(),
                    (displacement, eta6),
                    create_graph=create_graph,
                    retain_graph=True,
                )
                displacement_rows.append(grad_u)
                strain_rows.append(grad_eta)
            displacement_jacobian = torch.stack(displacement_rows, dim=-1)
            electronic_voigt = torch.stack(strain_rows, dim=1)
            cells = self._cells(batch, graphs)
            volume = torch.linalg.det(cells).abs().clamp_min(
                torch.finfo(cells.dtype).eps
            )
            born = (
                displacement_jacobian
                * volume[batch.batch, None, None]
                / self.PIEZO_C_PER_M2
            )
            electronic = piezo_voigt_to_cartesian(electronic_voigt)
        return ElectromechanicalJetPrediction(born, electronic, None)

    def forward(
        self,
        batch,
        displacement: torch.Tensor,
        eta6: torch.Tensor,
    ) -> torch.Tensor:
        return self.polarization_increment(batch, displacement, eta6)


class MacroscopicPiezoelectricCoupling(nn.Module):
    """Evaluate only the observable scalar ``-E_i e_ijk eta_jk``.

    This bookkeeping contraction is not a microscopic potential and is not
    used by the production PiezoJet forward path.
    """

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

    def electronic_output_basis(self, batch) -> list[torch.Tensor]:
        """Return the current electronic geometric readout span per graph.

        The Cartesian head bases and the reciprocal polar--chemical rank-three
        operator are included.  Their coefficients are unrestricted, making
        this a read-only model-class upper bound rather than a prediction or a
        trainable fallback.
        """
        features, _, _, _, spectral_operator = self._factor_features(batch)
        local_bases = self.electronic_head.output_basis(features, batch.batch)
        return [
            torch.cat((basis, spectral_operator[index : index + 1]), dim=0)
            for index, basis in enumerate(local_bases)
        ]

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
