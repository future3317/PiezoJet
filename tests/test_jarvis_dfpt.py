import torch
from types import SimpleNamespace
from torch_geometric.loader import DataLoader

from piezojet.data import PiezoDataset, load_gmtnet_records
from piezojet.jarvis_dfpt import DFPT_CACHE_SCHEMA, JarvisDFPTCache
import piezojet.jarvis_dfpt as jarvis_dfpt_module
from piezojet.evaluate_dfpt import ionic_piezo_from_factors
from piezojet.model import AtomCoordinateResponsePotential, PiezoJet
from piezojet.train import (
    born_loss,
    dielectric_loss,
    full_internal_strain_loss,
    force_constant_loss,
    internal_strain_loss,
    ionic_piezo_loss,
    macroscopic_piezo_loss,
    response_active_internal_strain_loss,
)
from tests.data_paths import gmtnet_root


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
        "epsilon": {"epsilon_ion": torch.eye(3)},
        "provenance": {
            "schema": 1,
            "status": "synthetic_unit_test",
            "source_archive": {"name": "unit-test.zip"},
            "parser": {"parser_schema": "synthetic"},
        },
    }


def test_outcar_numeric_overflow_is_quarantined_without_discarding_other_dfpt(monkeypatch, tmp_path):
    monkeypatch.setattr(
        jarvis_dfpt_module,
        "Outcar",
        lambda _: SimpleNamespace(
            piezoelectric_tensor=(torch.zeros(3, 6).numpy(), torch.zeros(3, 6).numpy())
        ),
    )
    text = b"""INTERNAL STRAIN TENSOR  FOR ION    1  DIRECTION 1   (eV/Angst):
 -----------------------------------------------------------------------------
   ********* 50.02702  0.00000
    49.66175 62.91693  0.00000
     0.00000  0.00000*********
"""
    _, _, blocks, ions, directions, halfwidth, audit = JarvisDFPTCache._outcar_dfpt_labels(
        text, tmp_path, "JVASP-overflow"
    )
    assert blocks.shape == (0, 3, 3)
    assert halfwidth.shape == (0, 3, 3)
    assert ions.numel() == directions.numel() == 0
    assert audit["valid_blocks"] == 0
    assert audit["malformed_blocks"][0]["reason"] == "numeric_overflow"
    assert not audit["complete_observed_block_parse"]


def test_dfpt_cache_attaches_node_bec_and_ragged_mode_metadata(tmp_path):
    records = load_gmtnet_records(gmtnet_root())[:2]
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
    assert batch.dfpt_branch_mask.all()
    assert batch.dfpt_ionic_dielectric_mask.all()
    assert torch.allclose(batch.y_dfpt_ionic_dielectric, torch.eye(3).expand(len(records), 3, 3))
    assert torch.allclose(batch.y_electronic_piezo, batch.y_dfpt_total_piezo - batch.y_ionic_piezo)


def test_electrostatic_profile_keeps_identical_required_labels_without_ragged_arrays(tmp_path):
    record = load_gmtnet_records(gmtnet_root())[0]
    cache = JarvisDFPTCache(tmp_path / "dfpt")
    cache.save(_payload(record))
    material_id = str(record["JARVIS_ID"])
    common = {
        "processed_dir": tmp_path / "graphs",
        "dfpt_dir": tmp_path / "dfpt",
    }
    full = PiezoDataset([record], [material_id], 5.0, 32, **common)[0]
    electrostatic = PiezoDataset(
        [record], [material_id], 5.0, 32,
        dfpt_profile="electrostatic", **common,
    )[0]
    assert torch.allclose(electrostatic.y_born, full.y_born)
    assert torch.allclose(
        electrostatic.y_electronic_piezo, full.y_electronic_piezo
    )
    assert torch.allclose(
        electrostatic.y_dfpt_electronic_dielectric,
        full.y_dfpt_electronic_dielectric,
    )
    assert not hasattr(electrostatic, "dfpt_force_constants_flat")
    assert not hasattr(electrostatic, "dfpt_dynamical_eigenvectors_flat")
    assert not hasattr(electrostatic, "dfpt_internal_strain_flat")


def test_projected_outcar_branch_labels_close_in_one_target_space(tmp_path):
    record = load_gmtnet_records(gmtnet_root())[5]
    payload = _payload(record)
    payload["ionic_piezo_source"] = torch.tensor(
        [[0.11, -0.23, 0.37, 0.19, -0.41, 0.29],
         [-0.07, 0.31, 0.13, -0.17, 0.43, -0.09],
         [0.21, 0.05, -0.27, 0.39, 0.15, -0.33]],
        dtype=torch.float32,
    )
    payload["total_piezo_source"] = payload["ionic_piezo_source"] + 0.17
    cache = JarvisDFPTCache(tmp_path / "dfpt")
    cache.save(payload)
    graph = PiezoDataset(
        [record], [str(record["JARVIS_ID"])], 5.0, 32,
        processed_dir=tmp_path / "graphs", dfpt_dir=tmp_path / "dfpt",
    )[0]
    reconstructed = graph.y_electronic_piezo + graph.y_ionic_piezo
    assert torch.allclose(reconstructed, graph.y_dfpt_total_piezo, atol=2e-7, rtol=2e-7)


def test_auxiliary_tensor_losses_are_invariant_under_common_rotation():
    torch.manual_seed(703)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    mask = torch.tensor([True, True])

    dielectric_target = torch.randn(2, 3, 3, dtype=torch.float64)
    dielectric_target = dielectric_target @ dielectric_target.transpose(-1, -2)
    dielectric_prediction = dielectric_target + 0.2 * torch.randn_like(dielectric_target)
    def rotate_rank2(value):
        return torch.einsum("ia,...ab,jb->...ij", rotation, value, rotation)
    original_dielectric = dielectric_loss(dielectric_prediction, dielectric_target, mask)
    rotated_dielectric = dielectric_loss(
        rotate_rank2(dielectric_prediction), rotate_rank2(dielectric_target), mask
    )
    assert torch.allclose(original_dielectric, rotated_dielectric, atol=1e-12, rtol=1e-11)

    born_target = torch.randn(2, 3, 3, dtype=torch.float64)
    born_prediction = born_target + 0.2 * torch.randn_like(born_target)
    original_born = born_loss(born_prediction, born_target, mask)
    rotated_born = born_loss(rotate_rank2(born_prediction), rotate_rank2(born_target), mask)
    assert torch.allclose(original_born, rotated_born, atol=1e-12, rtol=1e-11)

    piezo_target = torch.randn(2, 3, 3, 3, dtype=torch.float64)
    piezo_target = 0.5 * (piezo_target + piezo_target.transpose(-1, -2))
    piezo_prediction = piezo_target + 0.2 * torch.randn_like(piezo_target)
    def rotate_rank3(value):
        return torch.einsum(
            "ia,jb,kc,...abc->...ijk", rotation, rotation, rotation, value
        )
    original_piezo = macroscopic_piezo_loss(piezo_prediction, piezo_target, mask)
    rotated_piezo = macroscopic_piezo_loss(
        rotate_rank3(piezo_prediction), rotate_rank3(piezo_target), mask
    )
    assert torch.allclose(original_piezo, rotated_piezo, atol=1e-12, rtol=1e-11)


def test_dfpt_branch_losses_are_masked_when_projected_total_conflicts_with_outcar(tmp_path):
    record = load_gmtnet_records(gmtnet_root())[5]
    payload = _payload(record)
    # This deliberately disagrees with the GMTNet total by much more than both
    # registered tolerances; it must not receive simultaneous total/branch
    # supervision.
    payload["total_piezo_source"] = 10.0 * torch.as_tensor(
        record["piezoelectric_C_m2"], dtype=torch.float32
    )
    cache = JarvisDFPTCache(tmp_path / "dfpt")
    cache.save(payload)
    dataset = PiezoDataset(
        [record], [str(record["JARVIS_ID"])], 5.0, 32,
        processed_dir=tmp_path / "graphs", dfpt_dir=tmp_path / "dfpt",
    )
    graph = dataset[0]
    assert not bool(graph.dfpt_total_consistency_mask)
    assert not bool(graph.ionic_piezo_mask)
    assert not bool(graph.dfpt_branch_mask)
    assert graph.dfpt_total_consistency_abs_c_per_m2.item() > 0.05
    assert graph.dfpt_total_consistency_rel.item() > 0.05


def test_born_loss_respects_missing_label_mask():
    prediction = torch.randn(3, 3, 3, requires_grad=True)
    target = torch.randn(3, 3, 3)
    loss = born_loss(prediction, target, torch.tensor([False, True, False]))
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad[0].abs().sum() == 0
    assert prediction.grad[1].abs().sum() > 0


def test_variable_atom_count_dfpt_path_has_no_mode_padding_and_finite_gradients(tmp_path):
    records = load_gmtnet_records(gmtnet_root())
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
    records = load_gmtnet_records(gmtnet_root())[:2]
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


def test_response_active_lambda_loss_is_zero_for_the_true_observable_response():
    response = AtomCoordinateResponsePotential(optical_regularization=0.1)
    atoms = 2
    relative = torch.zeros(3, 3 * atoms)
    for axis in range(3):
        relative[axis, axis] = 2.0 ** -0.5
        relative[axis, axis + 3] = -(2.0 ** -0.5)
    matrix = torch.einsum("a,ai,aj->ij", torch.tensor([2.0, 3.0, 4.0]), relative, relative)
    blocks = matrix.reshape(atoms, 3, atoms, 3).permute(0, 2, 1, 3)
    born = torch.randn(atoms, 3, 3)
    born = born - born.mean(dim=0, keepdim=True)
    true_lambda = torch.randn(atoms, 3, 3, 3)
    true_lambda = 0.5 * (true_lambda + true_lambda.transpose(-1, -2))
    true_lambda = true_lambda - true_lambda.mean(dim=0, keepdim=True)
    cell = torch.diag(torch.tensor([2.0, 2.0, 2.5])).unsqueeze(0)
    target = ionic_piezo_from_factors(
        response, born, blocks, true_lambda, torch.linalg.det(cell[0]), "regularized"
    ).unsqueeze(0)
    prediction = true_lambda.clone().requires_grad_()
    exact_loss = response_active_internal_strain_loss(
        prediction,
        born,
        blocks.reshape(-1),
        target,
        torch.tensor([True]),
        torch.tensor([0, atoms]),
        cell,
        response,
    )
    perturbed_loss = response_active_internal_strain_loss(
        prediction + 0.2 * torch.randn_like(prediction),
        born,
        blocks.reshape(-1),
        target,
        torch.tensor([True]),
        torch.tensor([0, atoms]),
        cell,
        response,
    )
    assert exact_loss < 1e-10
    assert perturbed_loss > exact_loss
    perturbed_loss.backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0


def test_independent_displacement_response_has_no_pinv_chart_or_ghost_gradient(tmp_path):
    record = load_gmtnet_records(gmtnet_root())[0]
    cache = JarvisDFPTCache(tmp_path / "dfpt")
    cache.save(_payload(record))
    dataset = PiezoDataset(
        [record], [str(record["JARVIS_ID"])], 5.0, 32,
        processed_dir=tmp_path / "graphs", dfpt_dir=cache.directory,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model = PiezoJet(cutoff=5.0, num_blocks=1)
    components = model.predict_components(batch)
    assert components.ionic_piezo.shape == (1, 3, 3, 3)
    assert components.factorized_ionic_piezo.shape == (1, 3, 3, 3)
    assert components.displacement_response.shape == (batch.num_nodes, 3, 3, 3)
    assert components.internal_strain.sum(dim=0).abs().max() < 1e-5
    assert components.displacement_response.sum(dim=0).abs().max() < 1e-5
    assert not torch.allclose(components.ionic_piezo, components.factorized_ionic_piezo)

    components.ionic_piezo.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.displacement_response_head.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.born_head.parameters())
    assert all(parameter.grad is None for parameter in model.response_factors.parameters())

    model.zero_grad(set_to_none=True)
    teacher_forced = model.predict_displacement_response(batch)
    assert torch.allclose(teacher_forced, components.displacement_response)
    teacher_forced.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.displacement_response_head.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.displacement_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.born_head.parameters())
    assert all(parameter.grad is None for parameter in model.response_factors.parameters())

    model.zero_grad(set_to_none=True)
    components = model.predict_components(batch)
    components.factorized_ionic_piezo.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.born_head.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.response_factors.parameters()
    )


def test_full_lambda_loss_respects_strict_graph_mask():
    prediction = torch.zeros(3, 3, 3, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    batch = torch.tensor([0, 0, 1])
    graph_mask = torch.tensor([True, False])
    target[2] = 10.0
    masked = full_internal_strain_loss(prediction, target, graph_mask, batch)
    assert masked == 0.0
    target[0] = 1.0
    supervised = full_internal_strain_loss(prediction, target, graph_mask, batch)
    assert supervised > 0.0
    supervised.backward()
    assert prediction.grad[0].abs().sum() > 0
    assert prediction.grad[2].abs().sum() == 0
