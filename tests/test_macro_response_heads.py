import torch
from torch_geometric.data import Batch

from piezojet.data import load_gmtnet_records, record_to_graph
from piezojet.elastic_dielectric_ops import (
    elastic_voigt_to_cartesian,
    rotate_elastic,
)
from piezojet.model import (
    CartesianMacroDielectricHead,
    CartesianMacroElasticHead,
    CartesianNodeFeatures,
    PiezoJet,
)
from tests.data_paths import gmtnet_root


def _rotation(dtype):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))
    return q


def test_macro_dielectric_and_elastic_heads_are_spd_and_o3_covariant():
    torch.manual_seed(1701)
    dtype = torch.float64
    nodes, scalar_dim, channels = 5, 7, 6
    scalar = torch.randn(nodes, scalar_dim, dtype=dtype)
    vector = torch.randn(nodes, channels, 3, dtype=dtype)
    quadrupole = torch.randn(nodes, channels, 3, 3, dtype=dtype)
    quadrupole = 0.5 * (quadrupole + quadrupole.transpose(-1, -2))
    quadrupole = quadrupole - torch.diagonal(quadrupole, dim1=-2, dim2=-1).sum(-1)[..., None, None] * torch.eye(3, dtype=dtype) / 3
    features = CartesianNodeFeatures(scalar, vector, quadrupole)
    batch = torch.tensor([0, 0, 0, 1, 1])
    dielectric_head = CartesianMacroDielectricHead(scalar_dim, channels).to(dtype)
    elastic_head = CartesianMacroElasticHead(scalar_dim, channels, modes=8).to(dtype)
    dielectric = dielectric_head(features, batch)
    elastic = elastic_voigt_to_cartesian(elastic_head(features, batch))
    assert torch.linalg.eigvalsh(dielectric).min() >= 1.0 - 1e-12
    strain = torch.randn(2, 3, 3, dtype=dtype)
    strain = 0.5 * (strain + strain.transpose(-1, -2))
    energy = torch.einsum("bij,bijkl,bkl->b", strain, elastic, strain)
    assert torch.all(energy >= -1e-10)

    rotation = _rotation(dtype)
    rotated = CartesianNodeFeatures(
        scalar,
        torch.einsum("ia,nca->nci", rotation, vector),
        torch.einsum("ia,ncab,jb->ncij", rotation, quadrupole, rotation),
    )
    rotated_dielectric = dielectric_head(rotated, batch)
    rotated_elastic = elastic_voigt_to_cartesian(elastic_head(rotated, batch))
    expected_dielectric = torch.einsum("ia,bac,jc->bij", rotation, dielectric, rotation)
    expected_elastic = rotate_elastic(elastic, rotation)
    assert torch.allclose(rotated_dielectric, expected_dielectric, atol=1e-9, rtol=1e-9)
    assert torch.allclose(rotated_elastic, expected_elastic, atol=1e-9, rtol=1e-9)


def test_macro_response_gradients_do_not_enter_physical_tower():
    record = load_gmtnet_records(gmtnet_root())[0]
    batch = Batch.from_data_list([record_to_graph(record, cutoff=5.0, max_neighbors=16)])
    model = PiezoJet(cutoff=5.0, num_blocks=1, radial_basis=6, radial_hidden=16, cartesian_channels=8)
    piezo, dielectric, elastic = model.predict_macro_responses(batch)
    (piezo.square().mean() + dielectric.square().mean() + elastic.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in model.macro_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.macro_dielectric_head.parameters())
    assert any(parameter.grad is not None for parameter in model.macro_elastic_head.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.response_factors.parameters())
    assert all(parameter.grad is None for parameter in model.born_head.parameters())
