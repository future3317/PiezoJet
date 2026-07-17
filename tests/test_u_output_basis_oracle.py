from types import SimpleNamespace

import torch

from piezojet.u_capacity_adjudication import UCapacityModel
from piezojet.u_output_basis_oracle import _head_bases


def _config() -> dict:
    return {
        "embedding_dim": 8,
        "cutoff": 4.0,
        "num_blocks": 1,
        "radial_basis": 4,
        "radial_hidden": 8,
        "cartesian_channels": 3,
        "global_context_dim": 12,
        "spectral_channels": 3,
        "spectral_shells": 2,
        "polar_fluctuation_shells": 2,
        "reciprocal_cutoff": 4.0,
        "displacement_attention_dim": 4,
        "displacement_cross_rank": 2,
    }


def _batch() -> SimpleNamespace:
    return SimpleNamespace(
        num_nodes=2,
        num_graphs=1,
        z=torch.tensor([8, 8]),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_shift=torch.zeros(2, 3),
        batch=torch.zeros(2, dtype=torch.long),
        cell=torch.eye(3).unsqueeze(0) * 4.0,
    )


def test_global_l3_oracle_includes_octupole_basis_family():
    config = _config()
    model = UCapacityModel(config, architecture="global_l3", first_order_auxiliary=False)
    bases, spectral = _head_bases(model, _batch())

    cross_rank = config["displacement_cross_rank"]
    assert bases.shape == (2, 5 * cross_rank, 3, 3, 3)
    assert spectral is not None and spectral.shape == (2, 3, 3, 3)
    # A bond on x has a nonzero STF xxx octupole component, so the newly
    # included fifth family cannot be an all-zero placeholder.
    assert torch.count_nonzero(bases[:, 4 * cross_rank :]).item() > 0
