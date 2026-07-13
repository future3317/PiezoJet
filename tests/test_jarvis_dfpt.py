import torch
from torch_geometric.loader import DataLoader

from piezojet.data import PiezoDataset, load_gmtnet_records
from piezojet.jarvis_dfpt import DFPT_CACHE_SCHEMA, JarvisDFPTCache
from piezojet.model import PiezoJet
from piezojet.train import (
    born_loss,
    force_constant_loss,
    internal_strain_loss,
    ionic_piezo_loss,
    freeze_factor_stack,
)


def _payload(record):
    atoms = len(record["atoms"]["elements"])
    modes = 3 * atoms
    translation = torch.zeros(modes, 3)
    for axis in range(3):
        translation[axis::3, axis] = atoms ** -0.5
    projector = torch.eye(modes) - translation @ translation.T
    force_constants = projector.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)
    return {
        "schema": DFPT_CACHE_SCHEMA,
        "jid": str(record["JARVIS_ID"]),
        "source_archive": "unit-test.zip",
        "born_charges": torch.ones(atoms, 3, 3),
        "dynamical_eigenvalues": torch.arange(modes, dtype=torch.float32),
        "dynamical_eigenvectors": torch.zeros(modes, atoms, 3),
        "masses": torch.ones(atoms),
        "force_constants": force_constants,
        "dynamical_matrix": torch.zeros(atoms, atoms, 3, 3),
        "ionic_piezo_source": torch.zeros(3, 6),
        "total_piezo_source": torch.zeros(3, 6),
        "internal_strain_tensors": torch.ones(1, 3, 3),
        "internal_strain_ions": torch.zeros(1, dtype=torch.long),
        "internal_strain_directions": torch.zeros(1, dtype=torch.long),
        "epsilon": {},
    }


def test_dfpt_cache_attaches_node_bec_and_ragged_mode_metadata(tmp_path):
    records = load_gmtnet_records("data/raw/gmtnet")[:2]
    cache = JarvisDFPTCache(tmp_path / "dfpt")
    for record in records:
        cache.save(_payload(record))
    ids = [str(record["JARVIS_ID"]) for record in records]
    dataset = PiezoDataset(records, ids, 5.0, 32, processed_dir=tmp_path / "graphs", dfpt_dir=tmp_path / "dfpt")
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch.born_mask.all()
    assert batch.y_born.shape == (sum(len(record["atoms"]["elements"]) for record in records), 3, 3)
    for graph_index in range(len(records)):
        assert torch.allclose(batch.y_born[batch.batch == graph_index].sum(dim=0), torch.zeros(3, 3))
    assert batch.born_raw_asr_max_abs_e.shape == (len(records),)
    assert batch.born_raw_asr_rel.shape == (len(records),)
    assert batch.born_projection_rel.shape == (len(records),)
    assert batch.dfpt_mode_count.tolist() == [3 * len(record["atoms"]["elements"]) for record in records]
    assert batch.dfpt_dynamical_eigenvectors_flat.numel() == sum((3 * len(record["atoms"]["elements"])) * len(record["atoms"]["elements"]) * 3 for record in records)
    assert batch.force_constant_mask.all()
    assert batch.dfpt_internal_strain_count.tolist() == [1, 1]


def test_born_loss_respects_missing_label_mask():
    prediction = torch.randn(3, 3, 3, requires_grad=True)
    target = torch.randn(3, 3, 3)
    loss = born_loss(prediction, target, torch.tensor([False, True, False]))
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad[0].abs().sum() == 0
    assert prediction.grad[1].abs().sum() > 0


def test_variable_atom_count_dfpt_path_has_no_mode_padding_and_finite_gradients(tmp_path):
    records = load_gmtnet_records("data/raw/gmtnet")
    first = records[0]
    second = next(record for record in records[1:] if len(record["atoms"]["elements"]) != len(first["atoms"]["elements"]))
    selected = [first, second]
    cache = JarvisDFPTCache(tmp_path / "dfpt")
    for record in selected:
        cache.save(_payload(record))
    ids = [str(record["JARVIS_ID"]) for record in selected]
    dataset = PiezoDataset(selected, ids, 5.0, 32, processed_dir=tmp_path / "graphs", dfpt_dir=tmp_path / "dfpt")
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    model = PiezoJet(cutoff=5.0, num_blocks=1)
    components = model.predict_components(batch)
    atom_counts = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
    assert components.force_constants_flat.numel() == sum(9 * atoms * atoms for atoms in atom_counts)
    loss = born_loss(components.born_charges, batch.y_born, batch.born_mask, batch.batch)
    loss = loss + force_constant_loss(
        components.force_constants_flat, batch.dfpt_force_constants_flat, batch.ptr, batch.force_constant_mask
    )
    loss = loss + internal_strain_loss(
        components.internal_strain, batch.dfpt_internal_strain_flat,
        batch.dfpt_internal_strain_ions, batch.dfpt_internal_strain_directions,
        batch.dfpt_internal_strain_count, batch.ptr,
    )
    loss = loss + ionic_piezo_loss(
        components.ionic_piezo, batch.y_ionic_piezo, batch.ionic_piezo_mask
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_direct_factor_path_matches_full_response_components(tmp_path):
    records = load_gmtnet_records("data/raw/gmtnet")[:2]
    cache = JarvisDFPTCache(tmp_path / "dfpt")
    for record in records:
        cache.save(_payload(record))
    ids = [str(record["JARVIS_ID"]) for record in records]
    dataset = PiezoDataset(records, ids, 5.0, 32, processed_dir=tmp_path / "graphs", dfpt_dir=cache.directory)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    model = PiezoJet(cutoff=5.0, num_blocks=1)
    model.eval()
    with torch.no_grad():
        factors = model.predict_factors(batch)
        components = model.predict_components(batch)
    assert torch.allclose(factors.born_charges, components.born_charges)
    assert torch.allclose(factors.force_constants_flat, components.force_constants_flat)
    assert torch.allclose(factors.internal_strain, components.internal_strain)


def test_freeze_factor_stack_leaves_response_heads_trainable():
    model = PiezoJet(cutoff=5.0, num_blocks=1)
    frozen = freeze_factor_stack(model)
    assert set(frozen) == {
        "encoder", "born_head", "local_polar_mode", "global_context", "energy_factors"
    }
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.born_head.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert not any(parameter.requires_grad for parameter in model.global_context.parameters())
