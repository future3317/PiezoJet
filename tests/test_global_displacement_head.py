import torch
from types import SimpleNamespace

from piezojet.model import (
    CartesianNodeFeatures,
    GlobalDisplacementResponseHead,
    OctupoleGlobalDisplacementResponseHead,
)


def _features(nodes: int = 7, channels: int = 5, scalar_dim: int = 8):
    scalar = torch.randn(nodes, scalar_dim)
    vector = torch.randn(nodes, channels, 3)
    raw = torch.randn(nodes, channels, 3, 3)
    symmetric = 0.5 * (raw + raw.transpose(-1, -2))
    trace = torch.diagonal(symmetric, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3)
    quadrupole = symmetric - trace[..., None, None] * identity / 3.0
    return CartesianNodeFeatures(scalar, vector, quadrupole)


def _complete_graph(pos: torch.Tensor) -> SimpleNamespace:
    nodes = pos.shape[0]
    source, target = torch.where(~torch.eye(nodes, dtype=torch.bool))
    return SimpleNamespace(
        pos=pos,
        edge_index=torch.stack((source, target)),
        edge_shift=torch.zeros(source.shape[0], 3),
        batch=torch.zeros(nodes, dtype=torch.long),
    )


def _loop_nonlocal_reference(head, features, batch_index):
    """Pre-vectorization implementation retained as a numerical oracle."""
    vector_values = torch.einsum(
        "cd,ndj->ncj", head.vector_value_mix, features.vector
    )
    quadrupole_values = torch.einsum(
        "cd,ndjk->ncjk", head.quadrupole_value_mix, features.quadrupole
    )
    queries, keys = head.query(features.scalar), head.key(features.scalar)
    vector_global = torch.zeros_like(features.vector)
    quadrupole_global = torch.zeros_like(features.quadrupole)
    scale = head.attention_dim**-0.5
    for graph_index in range(int(batch_index.max()) + 1):
        node_indices = torch.nonzero(
            batch_index == graph_index, as_tuple=False
        ).squeeze(-1)
        attention = torch.softmax(
            queries[node_indices] @ keys[node_indices].transpose(0, 1) * scale,
            dim=-1,
        )
        vector_global[node_indices] = torch.einsum(
            "ij,jck->ick", attention, vector_values[node_indices]
        )
        quadrupole_global[node_indices] = torch.einsum(
            "ij,jckl->ickl", attention, quadrupole_values[node_indices]
        )
    return (
        features.vector
        + torch.tanh(head.vector_nonlocal_gate) * vector_global,
        features.quadrupole
        + torch.tanh(head.quadrupole_nonlocal_gate) * quadrupole_global,
    )


def test_vectorized_nonlocal_attention_matches_loop_output_and_gradients():
    torch.manual_seed(90)
    features = _features(nodes=12, channels=4, scalar_dim=6)
    features = CartesianNodeFeatures(
        *(value.requires_grad_() for value in features)
    )
    batch = torch.tensor([0] * 3 + [1] * 5 + [2] * 4)
    head = GlobalDisplacementResponseHead(
        6, 4, 9, attention_dim=5, cross_rank=3
    )

    expected = _loop_nonlocal_reference(head, features, batch)
    observed = head._nonlocal_features(features, batch)
    for actual, reference in zip(observed, expected):
        assert torch.allclose(actual, reference, atol=2e-6, rtol=2e-6)

    parameters = tuple(head.parameters())
    reference_loss = expected[0].square().mean() + expected[1].square().mean()
    observed_loss = observed[0].square().mean() + observed[1].square().mean()
    reference_gradients = torch.autograd.grad(
        reference_loss, (*features, *parameters), retain_graph=True,
        allow_unused=True,
    )
    observed_gradients = torch.autograd.grad(
        observed_loss, (*features, *parameters), allow_unused=True
    )
    for actual, reference in zip(observed_gradients, reference_gradients):
        if reference is None:
            assert actual is None
        else:
            assert actual is not None
            assert torch.allclose(actual, reference, atol=3e-6, rtol=3e-5)


def test_global_displacement_head_is_translation_free_and_permutation_equivariant():
    torch.manual_seed(91)
    features = _features()
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    context = torch.randn(2, 11)
    spectral = torch.randn(2, 3, 3, 3)
    spectral = 0.5 * (spectral + spectral.transpose(-1, -2))
    head = GlobalDisplacementResponseHead(8, 5, 11, attention_dim=7, cross_rank=6)
    reference = head(features, batch, context, spectral)
    assert torch.allclose(reference[batch == 0].sum(0), torch.zeros(3, 3, 3), atol=2e-5)
    assert torch.allclose(reference[batch == 1].sum(0), torch.zeros(3, 3, 3), atol=2e-5)

    permutation = torch.tensor([2, 0, 3, 1, 6, 4, 5])
    permuted_features = CartesianNodeFeatures(
        features.scalar[permutation],
        features.vector[permutation],
        features.quadrupole[permutation],
    )
    observed = head(permuted_features, batch[permutation], context, spectral)
    assert torch.allclose(observed, reference[permutation], atol=2e-5, rtol=2e-5)


def test_global_displacement_head_is_o3_equivariant():
    torch.manual_seed(92)
    features = _features(nodes=5)
    batch = torch.zeros(5, dtype=torch.long)
    context = torch.randn(1, 11)
    spectral = torch.randn(1, 3, 3, 3)
    spectral = 0.5 * (spectral + spectral.transpose(-1, -2))
    head = GlobalDisplacementResponseHead(8, 5, 11, attention_dim=7, cross_rank=6)
    reference = head(features, batch, context, spectral)

    raw = torch.randn(3, 3)
    rotation, _ = torch.linalg.qr(raw)
    rotation[:, 0] *= torch.det(rotation).sign()
    rotated_features = CartesianNodeFeatures(
        features.scalar,
        torch.einsum("ia,nca->nci", rotation, features.vector),
        torch.einsum("ia,ncab,jb->ncij", rotation, features.quadrupole, rotation),
    )
    rotated_spectral = torch.einsum(
        "ia,jb,kc,nabc->nijk", rotation, rotation, rotation, spectral
    )
    observed = head(rotated_features, batch, context, rotated_spectral)
    expected = torch.einsum(
        "ia,jb,kc,nabc->nijk", rotation, rotation, rotation, reference
    )
    assert torch.allclose(observed, expected, atol=4e-5, rtol=4e-5)


def test_octupole_global_head_is_translation_free_permutation_and_o3_equivariant():
    torch.manual_seed(93)
    nodes = 6
    features = _features(nodes=nodes)
    pos = 0.6 * torch.randn(nodes, 3)
    graph = _complete_graph(pos)
    context = torch.randn(1, 11)
    spectral = torch.randn(1, 3, 3, 3)
    spectral = 0.5 * (spectral + spectral.transpose(-1, -2))
    head = OctupoleGlobalDisplacementResponseHead(
        8, 5, 11, attention_dim=7, cross_rank=6,
        radial_basis=5, radial_hidden=13, cutoff=5.0,
    )
    reference = head(features, graph, context, spectral)
    assert torch.allclose(reference.sum(0), torch.zeros(3, 3, 3), atol=3e-5)

    permutation = torch.tensor([2, 5, 0, 4, 1, 3])
    permuted_features = CartesianNodeFeatures(
        features.scalar[permutation],
        features.vector[permutation],
        features.quadrupole[permutation],
    )
    observed = head(
        permuted_features, _complete_graph(pos[permutation]), context, spectral
    )
    assert torch.allclose(observed, reference[permutation], atol=5e-5, rtol=5e-5)

    raw = torch.randn(3, 3)
    rotation, _ = torch.linalg.qr(raw)
    rotation[:, 0] *= torch.det(rotation).sign()
    rotated_features = CartesianNodeFeatures(
        features.scalar,
        torch.einsum("ia,nca->nci", rotation, features.vector),
        torch.einsum("ia,ncab,jb->ncij", rotation, features.quadrupole, rotation),
    )
    rotated_spectral = torch.einsum(
        "ia,jb,kc,nabc->nijk", rotation, rotation, rotation, spectral
    )
    rotated_pos = torch.einsum("ia,na->ni", rotation, pos)
    observed = head(
        rotated_features, _complete_graph(rotated_pos), context, rotated_spectral
    )
    expected = torch.einsum(
        "ia,jb,kc,nabc->nijk", rotation, rotation, rotation, reference
    )
    assert torch.allclose(observed, expected, atol=8e-5, rtol=8e-5)
