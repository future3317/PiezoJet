"""Small O(3)-equivariant periodic encoder and response potential."""

from __future__ import annotations

from typing import Mapping, NamedTuple

import math

import torch
from e3nn import o3
from e3nn.nn import Gate
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter

from .tensor_ops import PIEZO_TYPE, piezo_from_irreps, piezo_voigt_to_cartesian, voigt_to_symmetric_matrix


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


def _translation_projector(atoms: int, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cartesian optical projector and its three normalized translations."""
    size = 3 * atoms
    translation = reference.new_zeros(size, 3)
    axes = torch.arange(3, device=reference.device)
    translation[axes.repeat(atoms) + 3 * torch.arange(atoms, device=reference.device).repeat_interleave(3), axes.repeat(atoms)] = atoms ** -0.5
    projector = torch.eye(size, dtype=reference.dtype, device=reference.device) - translation @ translation.transpose(0, 1)
    return projector, translation


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


class CartesianForceConstantHead(nn.Module):
    """Variable-size atom-coordinate Hessian with exact translation nullspace.

    Equivariant edge blocks first form an unrestricted symmetric operator
    ``H``.  For each graph we return

    ``Phi = P (H + H^T) P / 2``,

    where ``P`` removes the three uniform translations. This gives block
    transpose symmetry and the acoustic sum rule while retaining genuine
    negative/unstable DFPT modes. Stability is handled by the signed damped
    pseudoinverse in ``AtomCoordinateResponsePotential``. Per-graph matrices
    are concatenated in flattened form to avoid cross-graph padding.
    """

    def __init__(self, scalar_dim: int, radial_basis: int, cutoff: float):
        super().__init__()
        self.cutoff = float(cutoff)
        self.register_buffer("radial_centers", torch.linspace(0, cutoff, radial_basis), persistent=False)
        hidden = max(32, scalar_dim)
        self.edge_coefficients = nn.Sequential(
            nn.Linear(2 * scalar_dim + radial_basis, hidden), nn.SiLU(), nn.Linear(hidden, 4)
        )
        self.register_buffer("identity", torch.eye(3), persistent=False)

    def forward(
        self,
        features: CartesianNodeFeatures,
        local_polar: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        source, target = batch.edge_index
        vectors = batch.pos[source] - batch.pos[target] + batch.edge_shift
        distance = torch.linalg.vector_norm(vectors, dim=-1)
        direction = vectors / distance.clamp_min(torch.finfo(vectors.dtype).eps).unsqueeze(-1)
        radial = _radial_basis(distance, self.radial_centers, self.cutoff)
        symmetric_scalar = features.scalar[source] + features.scalar[target]
        contrast = (features.scalar[source] - features.scalar[target]).abs()
        coefficients = self.edge_coefficients(torch.cat((symmetric_scalar, contrast, radial), dim=-1))
        identity = self.identity.to(dtype=vectors.dtype)
        rr = direction.unsqueeze(-1) * direction.unsqueeze(-2)
        polar_r = local_polar[source].unsqueeze(-1) * direction.unsqueeze(-2)
        r_polar = direction.unsqueeze(-1) * local_polar[target].unsqueeze(-2)
        edge_blocks = (
            coefficients[:, 0, None, None] * identity
            + coefficients[:, 1, None, None] * rr
            + coefficients[:, 2, None, None] * polar_r
            + coefficients[:, 3, None, None] * r_polar
        )
        ptr = _graph_ptr(batch.batch)
        flattened = []
        for graph_index in range(ptr.numel() - 1):
            start, stop = int(ptr[graph_index]), int(ptr[graph_index + 1])
            atoms = stop - start
            edge_mask = batch.batch[target] == graph_index
            local_source, local_target = source[edge_mask] - start, target[edge_mask] - start
            blocks = edge_blocks[edge_mask]
            raw_blocks = blocks.new_zeros(atoms, atoms, 3, 3)
            raw_blocks.index_put_((local_source, local_target), blocks, accumulate=True)
            raw = raw_blocks.permute(0, 2, 1, 3).reshape(3 * atoms, 3 * atoms)
            raw = 0.5 * (raw + raw.transpose(0, 1))
            projector, _ = _translation_projector(atoms, raw)
            force_constants = projector @ raw @ projector
            blocks_out = force_constants.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)
            flattened.append(blocks_out.reshape(-1))
        return torch.cat(flattened)


class StrainAwareQuadraticEnergyHead(nn.Module):
    """Joint Hessian and strain-force factors from one quadratic bond energy.

    For every directed periodic bond, the model predicts a symmetric signed
    stiffness ``K_e``.  The shared local energy is

    ``1/2 (B_e u + S_e eta)^T K_e (B_e u + S_e eta)``.

    Consequently ``Phi = sum B_e^T K_e B_e`` and
    ``Lambda = -sum B_e^T K_e S_e`` are integrable derivatives of the same
    strain-aware scalar.  Signed stiffnesses retain unstable optical modes.
    Reciprocal/global context conditions ``K_e`` so collective information is
    part of the ionic path instead of only an electronic side branch.
    """

    def __init__(
        self,
        scalar_dim: int,
        radial_basis: int,
        cutoff: float,
        context_dim: int,
        learned_strain_map: bool = False,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.learned_strain_map = bool(learned_strain_map)
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
        if self.learned_strain_map:
            self.edge_strain_map = nn.Sequential(
                nn.Linear(2 * scalar_dim + radial_basis + context_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 7),
            )
            # Start from the physical affine bond strain and let direct Lambda
            # plus ionic-response supervision learn only the required residual.
            nn.init.zeros_(self.edge_strain_map[-1].weight)
            nn.init.zeros_(self.edge_strain_map[-1].bias)

    @staticmethod
    def _internal_tensor(coupling: torch.Tensor) -> torch.Tensor:
        """Map canonical engineering-Voigt coupling to symmetric Cartesian form."""
        atoms = coupling.shape[0]
        output = coupling.new_zeros(atoms, 3, 3, 3)
        output[..., 0, 0] = coupling[..., 0]
        output[..., 1, 1] = coupling[..., 1]
        output[..., 2, 2] = coupling[..., 2]
        output[..., 1, 2] = output[..., 2, 1] = coupling[..., 3]
        output[..., 0, 2] = output[..., 2, 0] = coupling[..., 4]
        output[..., 0, 1] = output[..., 1, 0] = coupling[..., 5]
        return output

    def forward(
        self,
        features: CartesianNodeFeatures,
        local_polar: torch.Tensor,
        context: torch.Tensor,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        if self.learned_strain_map:
            strain_coefficients = self.edge_strain_map(edge_context)
            r, p = direction, polar
            r_identity = torch.einsum("ei,jk->eijk", r, identity)
            identity_r = 0.5 * (
                torch.einsum("ij,ek->eijk", identity, r)
                + torch.einsum("ik,ej->eijk", identity, r)
            )
            rrr = torch.einsum("ei,ej,ek->eijk", r, r, r)
            p_identity = torch.einsum("ei,jk->eijk", p, identity)
            identity_p = 0.5 * (
                torch.einsum("ij,ek->eijk", identity, p)
                + torch.einsum("ik,ej->eijk", identity, p)
            )
            p_rr = torch.einsum("ei,ej,ek->eijk", p, r, r)
            r_pr = 0.5 * (
                torch.einsum("ei,ej,ek->eijk", r, p, r)
                + torch.einsum("ei,ek,ej->eijk", r, p, r)
            )
            bases = torch.stack(
                (r_identity, identity_r, rrr, p_identity, identity_p, p_rr, r_pr),
                dim=1,
            )
            correction = torch.einsum("ec,ecijk->eijk", strain_coefficients, bases)
            correction = correction * distance[:, None, None, None]
            correction_voigt = torch.stack(
                (
                    correction[..., 0, 0], correction[..., 1, 1], correction[..., 2, 2],
                    correction[..., 1, 2], correction[..., 0, 2], correction[..., 0, 1],
                ),
                dim=-1,
            )
            strain_map = strain_map + correction_voigt
        stiffness_strain = stiffness @ strain_map

        ptr = _graph_ptr(batch.batch)
        force_flat, internal = [], []
        for graph_index in range(ptr.numel() - 1):
            start, stop = int(ptr[graph_index]), int(ptr[graph_index + 1])
            atoms = stop - start
            edge_mask = edge_graph == graph_index
            local_source = source[edge_mask] - start
            local_target = target[edge_mask] - start
            local_stiffness = stiffness[edge_mask]
            local_strain = stiffness_strain[edge_mask]

            blocks = local_stiffness.new_zeros(atoms, atoms, 3, 3)
            blocks.index_put_((local_source, local_source), local_stiffness, accumulate=True)
            blocks.index_put_((local_target, local_target), local_stiffness, accumulate=True)
            blocks.index_put_((local_source, local_target), -local_stiffness, accumulate=True)
            blocks.index_put_((local_target, local_source), -local_stiffness, accumulate=True)
            blocks = 0.5 * (blocks + blocks.permute(1, 0, 3, 2))

            coupling = local_stiffness.new_zeros(atoms, 3, 6)
            # Lambda = -B^T K S for the declared quadratic energy.
            coupling.index_put_((local_source,), -local_strain, accumulate=True)
            coupling.index_put_((local_target,), local_strain, accumulate=True)
            force_flat.append(blocks.reshape(-1))
            internal.append(self._internal_tensor(coupling))
        return torch.cat(force_flat), torch.cat(internal)


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
        dielectric = susceptibility[:, None, None] * self.identity.to(dtype=context.dtype)
        return elastic, dielectric


class AtomCoordinatePrediction(NamedTuple):
    """Responses propagated through the physical ``3N-3`` optical space."""

    tensor: torch.Tensor
    electronic_piezo: torch.Tensor
    ionic_piezo: torch.Tensor
    born_charges: torch.Tensor
    force_constants_flat: torch.Tensor
    internal_strain: torch.Tensor
    optical_operator_flat: torch.Tensor
    elastic_background: torch.Tensor
    dielectric_background: torch.Tensor
    dielectric: torch.Tensor
    elastic: torch.Tensor


class AtomCoordinateFactors(NamedTuple):
    """Directly supervised atom-coordinate factors before the response solve."""

    born_charges: torch.Tensor
    force_constants_flat: torch.Tensor
    internal_strain: torch.Tensor


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
    TRANSLATION_PENALTY = 1.0

    def __init__(
        self,
        optical_regularization: float = 1e-3,
        optical_stability_cutoff: float = 1e-4,
        optical_solve_policy: str = "auto",
    ):
        super().__init__()
        if optical_regularization <= 0:
            raise ValueError("optical_regularization must be positive")
        if optical_stability_cutoff <= 0:
            raise ValueError("optical_stability_cutoff must be positive")
        if optical_solve_policy not in {"auto", "exact", "regularized"}:
            raise ValueError("optical_solve_policy must be auto, exact, or regularized")
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

    @staticmethod
    def _finalize_operator(matrix: torch.Tensor, atoms: int, dtype: torch.dtype) -> torch.Tensor:
        """Symmetrize and re-project after a possible float64-to-float32 cast."""
        matrix = (0.5 * (matrix + matrix.transpose(0, 1))).to(dtype)
        projector, _ = _translation_projector(atoms, matrix)
        matrix = projector @ matrix @ projector
        return 0.5 * (matrix + matrix.transpose(0, 1))

    @staticmethod
    def _optical_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
        """Return all non-translational eigenvalues of an ASR-projected matrix."""
        values = torch.linalg.eigvalsh(matrix)
        if values.numel() <= 3:
            return values.new_empty(0)
        keep = torch.argsort(values.abs())[3:]
        return values[keep]

    def exact_optical_inverse(self, force_constants: torch.Tensor) -> torch.Tensor:
        """Exact inverse on a nonsingular optical subspace.

        The translation block is lifted only to make the full-coordinate solve
        nonsingular; the outer projectors remove it from the returned operator.
        This is the stationary response of the original quadratic form.  It is
        also well-defined for an indefinite saddle when no optical eigenvalue
        is zero, although ``auto`` selects it only for stable structures.
        """
        atoms = force_constants.shape[0]
        matrix = self._matrix_from_blocks(force_constants)
        output_dtype = matrix.dtype
        if matrix.dtype in (torch.float16, torch.bfloat16, torch.float32):
            matrix = matrix.to(torch.float64)
        projector, translation = _translation_projector(atoms, matrix)
        translation_projector = translation @ translation.transpose(0, 1)
        optical = self._optical_eigenvalues(matrix)
        if optical.numel() and bool((optical.abs().min() <= self.optical_stability_cutoff).item()):
            raise RuntimeError(
                "Exact optical inverse requested with a mode at or below the stability cutoff"
            )
        augmented = matrix + self.TRANSLATION_PENALTY * translation_projector
        inverse = projector @ torch.linalg.solve(augmented, projector) @ projector
        return self._finalize_operator(inverse, atoms, output_dtype)

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
        delta = self.optical_regularization if regularization is None else float(regularization)
        if delta <= 0:
            raise ValueError("regularization must be positive")
        atoms = force_constants.shape[0]
        matrix = self._matrix_from_blocks(force_constants)
        output_dtype = matrix.dtype
        # delta^2 can be 1e-6 in native eV/Angstrom^2 units. Perform the
        # small per-crystal solve in float64 so rotations do not get amplified
        # by float32 conditioning near a soft crossing.
        if matrix.dtype in (torch.float16, torch.bfloat16, torch.float32):
            matrix = matrix.to(torch.float64)
        projector, translation = _translation_projector(atoms, matrix)
        translation_projector = translation @ translation.transpose(0, 1)
        # Signed Tikhonov/Moore--Penrose filter lambda/(lambda^2+delta^2):
        # negative and unstable modes retain their sign, exact translations
        # remain zero, and gradients stay finite at soft-mode crossings.
        normal = (
            matrix @ matrix
            + delta ** 2 * projector
            + self.TRANSLATION_PENALTY * translation_projector
        )
        inverse = projector @ torch.linalg.solve(normal, matrix) @ projector
        return self._finalize_operator(inverse, atoms, output_dtype)

    def optical_operator(
        self,
        force_constants: torch.Tensor,
        solve_policy: str | None = None,
        regularization: float | None = None,
    ) -> torch.Tensor:
        """Choose the declared exact or regularized optical response policy."""
        policy = self.optical_solve_policy if solve_policy is None else solve_policy
        if policy not in {"auto", "exact", "regularized"}:
            raise ValueError("solve_policy must be auto, exact, or regularized")
        if policy == "exact":
            return self.exact_optical_inverse(force_constants)
        if policy == "regularized":
            return self.signed_regularized_optical_green(force_constants, regularization)

        matrix = self._matrix_from_blocks(force_constants)
        spectral_matrix = matrix.to(torch.float64) if matrix.dtype in (torch.float16, torch.bfloat16, torch.float32) else matrix
        optical = self._optical_eigenvalues(spectral_matrix)
        stable = optical.numel() == 0 or bool((optical.min() > self.optical_stability_cutoff).item())
        if stable:
            return self.exact_optical_inverse(force_constants)
        return self.signed_regularized_optical_green(force_constants, regularization)

    def responses(
        self,
        electronic_piezo: torch.Tensor,
        born_charges: torch.Tensor,
        internal_strain: torch.Tensor,
        force_constants_flat: torch.Tensor,
        batch,
        elastic_background: torch.Tensor,
        dielectric_background: torch.Tensor,
        solve_policy: str | None = None,
        regularization: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ptr = _graph_ptr(batch.batch)
        cell = getattr(batch, "cell", None)
        cells = (
            torch.eye(3, dtype=born_charges.dtype, device=born_charges.device)
            .expand(ptr.numel() - 1, 3, 3)
            if cell is None else cell.reshape(-1, 3, 3)
        )
        ionic_piezo, ionic_dielectric, elastic_softening, inverse_flat = [], [], [], []
        force_offset = 0
        for graph_index in range(ptr.numel() - 1):
            start, stop = int(ptr[graph_index]), int(ptr[graph_index + 1])
            atoms = stop - start
            block_values = 9 * atoms * atoms
            blocks = force_constants_flat[force_offset : force_offset + block_values].reshape(atoms, atoms, 3, 3)
            force_offset += block_values
            operator = self.optical_operator(blocks, solve_policy, regularization)
            inverse_flat.append(self._blocks_from_matrix(operator, atoms).reshape(-1))
            coupling = self._coupling_voigt(internal_strain[start:stop]).reshape(3 * atoms, 6)
            # VASP BEC rows are atomic force/displacement directions and
            # columns are electric-field/polarization directions.
            charge = born_charges[start:stop].reshape(3 * atoms, 3)
            inverse_coupling = operator @ coupling
            volume = torch.linalg.det(cells[graph_index]).abs().clamp_min(torch.finfo(cells.dtype).eps)
            ionic_piezo.append(self.PIEZO_C_PER_M2 * (charge.transpose(0, 1) @ inverse_coupling) / volume)
            ionic_dielectric.append(
                self.DIELECTRIC_RELATIVE * (charge.transpose(0, 1) @ operator @ charge) / volume
            )
            elastic_softening.append(
                self.EV_PER_A3_TO_GPA * (coupling.transpose(0, 1) @ inverse_coupling) / volume
            )
        ionic_piezo_cart = piezo_voigt_to_cartesian(torch.stack(ionic_piezo))
        dielectric = dielectric_background + torch.stack(ionic_dielectric)
        elastic = elastic_background - torch.stack(elastic_softening)
        return electronic_piezo + ionic_piezo_cart, dielectric, elastic, torch.cat(inverse_flat)

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
        optical_solve_policy: str = "auto",
        factor_architecture: str = "energy_learned_strain",
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = CartesianLocalEnvironmentEncoder(**encoder_kwargs)
        self.head = CartesianPiezoTensorHead(self.encoder.input[-2].out_features, self.encoder.channels)
        self.born_head = CartesianBornChargeHead(self.encoder.input[-2].out_features, self.encoder.channels)
        self.local_polar_mode = CartesianPolarReadout(self.encoder.input[-2].out_features, self.encoder.channels)
        self.global_context = CrystalGlobalContext(
            global_context_dim, spectral_channels, spectral_shells, polar_fluctuation_shells, reciprocal_cutoff
        )
        if factor_architecture not in {"legacy", "energy", "energy_learned_strain"}:
            raise ValueError("factor_architecture must be legacy, energy, or energy_learned_strain")
        self.factor_architecture = factor_architecture
        if factor_architecture == "legacy":
            self.force_constants = CartesianForceConstantHead(
                self.encoder.input[-2].out_features,
                int(encoder_kwargs.get("radial_basis", 12)),
                float(encoder_kwargs.get("cutoff", 5.0)),
            )
            self.internal_strain = CartesianInternalStrainHead(
                self.encoder.input[-2].out_features, self.encoder.channels
            )
        else:
            self.energy_factors = StrainAwareQuadraticEnergyHead(
                self.encoder.input[-2].out_features,
                int(encoder_kwargs.get("radial_basis", 12)),
                float(encoder_kwargs.get("cutoff", 5.0)),
                global_context_dim,
                learned_strain_map=factor_architecture == "energy_learned_strain",
            )
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
        if self.factor_architecture == "legacy":
            force_constants_flat = self.force_constants(features, local_polar, batch)
            internal_strain = self.internal_strain(features, batch.batch)
        else:
            force_constants_flat, internal_strain = self.energy_factors(
                features, local_polar, context, batch
            )
        factors = AtomCoordinateFactors(born_charges, force_constants_flat, internal_strain)
        return features, local_polar, factors, context, spectral_operator

    def predict_factors(self, batch) -> AtomCoordinateFactors:
        """Predict DFPT factors without evaluating their stiff inverse product."""
        return self._factor_features(batch)[2]

    def predict_components(self, batch) -> AtomCoordinatePrediction:
        features, local_polar, factors, context, spectral_operator = self._factor_features(batch)
        direct = self.head(features, batch.batch)
        electronic_piezo = 0.5 * (direct + spectral_operator + (direct + spectral_operator).transpose(-1, -2))
        elastic_background, dielectric_background = self.background(context)
        tensor, dielectric, elastic, optical_operator_flat = self.response.responses(
            electronic_piezo, factors.born_charges, factors.internal_strain,
            factors.force_constants_flat, batch,
            elastic_background, dielectric_background
        )
        ionic_piezo = tensor - electronic_piezo
        return AtomCoordinatePrediction(
            0.5 * (tensor + tensor.transpose(-1, -2)), electronic_piezo, ionic_piezo,
            factors.born_charges, factors.force_constants_flat, factors.internal_strain, optical_operator_flat,
            elastic_background, dielectric_background, dielectric, elastic,
        )

    def forward(self, batch) -> torch.Tensor:
        return self.predict_components(batch).tensor

    def potential(self, batch, field: torch.Tensor, eta6: torch.Tensor) -> torch.Tensor:
        components = self.predict_components(batch)
        return self.response(
            components.tensor, components.elastic, components.dielectric, field, eta6,
        )


def model_from_config(config: Mapping[str, object]) -> PiezoJet:
    """Construct the single production PiezoJet architecture from a run config."""
    return PiezoJet(
        embedding_dim=int(config["embedding_dim"]), cutoff=float(config["cutoff"]),
        num_blocks=int(config["num_blocks"]), radial_basis=int(config["radial_basis"]), radial_hidden=int(config["radial_hidden"]),
        cartesian_channels=int(config.get("cartesian_channels", 48)),
        global_context_dim=int(config.get("global_context_dim", 128)), spectral_channels=int(config.get("spectral_channels", 16)),
        spectral_shells=int(config.get("spectral_shells", 8)), polar_fluctuation_shells=int(config.get("polar_fluctuation_shells", 8)),
        reciprocal_cutoff=float(config.get("reciprocal_cutoff", 7.0)),
        optical_regularization=float(config.get("optical_regularization", 1e-3)),
        optical_stability_cutoff=float(config.get("optical_stability_cutoff", 1e-4)),
        optical_solve_policy=str(config.get("optical_solve_policy", "auto")),
        factor_architecture=str(config.get("factor_architecture", "legacy")),
    )
