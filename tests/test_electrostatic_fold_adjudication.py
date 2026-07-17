import json

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from piezojet.audit_dft3d_release import _fractional
from piezojet.build_electrostatic_development_folds import (
    electrostatic_availability,
    electrostatic_fold_train_ids,
)
from piezojet.data import (
    deterministic_subset,
    load_gmtnet_records,
    record_to_graph,
)
from piezojet.electrostatic_fold_adjudication import (
    ARCHITECTURES,
    backward_training_objective,
    response_active_diagnostic_indices,
)
from piezojet.electronic_capacity import (
    born_material_balanced_loss,
    irrep_balanced_capacity_loss,
)
from piezojet.model import IndependentElectrostaticHeads
from piezojet.pretrain_e3nn import electrostatic_pretraining_ids
from piezojet.prepare_electrostatic_adjudication import build_plan


def test_official_cartesian_structure_coordinates_convert_to_fractional():
    atoms = {
        "lattice_mat": [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        "coords": [[1.0, 1.5, 2.0]],
        "cartesian": True,
    }
    assert np.allclose(_fractional(atoms), [[0.5, 0.5, 0.5]])


def test_fold_limit_selection_is_deterministic_without_reordering_source():
    values = [f"id-{index}" for index in range(20)]
    selected = deterministic_subset(values, 7, 123)
    assert selected == deterministic_subset(values, 7, 123)
    assert [values.index(value) for value in selected] == sorted(
        values.index(value) for value in selected
    )


def test_independent_control_has_no_shared_task_parameters():
    model = IndependentElectrostaticHeads(
        embedding_dim=4,
        cutoff=3.0,
        lmax=3,
        num_blocks=1,
        radial_basis=3,
        radial_hidden=8,
        global_context_dim=8,
        spectral_channels=2,
        spectral_shells=2,
        polar_fluctuation_shells=2,
        reciprocal_cutoff=3.0,
        attention_dim=4,
    )
    born_ids = {id(parameter) for parameter in model.born_generator.parameters()}
    piezo_ids = {id(parameter) for parameter in model.piezo_generator.parameters()}
    assert not born_ids & piezo_ids
    assert not hasattr(model.born_generator, "electronic_irreps")
    assert not hasattr(model.piezo_generator, "born_irreps")


def test_independent_control_runs_only_its_declared_decoders():
    graph = record_to_graph(load_gmtnet_records("data/raw/gmtnet")[10], 5.0, 12)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = IndependentElectrostaticHeads(
        embedding_dim=4,
        cutoff=5.0,
        lmax=3,
        num_blocks=1,
        radial_basis=3,
        radial_hidden=8,
        global_context_dim=8,
        spectral_channels=2,
        spectral_shells=2,
        polar_fluctuation_shells=2,
        reciprocal_cutoff=3.0,
        attention_dim=4,
    ).eval()
    prediction = model.coefficients(graph)
    assert prediction.born_charges.shape == (graph.num_nodes, 3, 3)
    assert prediction.electronic_piezo.shape == (1, 3, 3, 3)
    assert torch.linalg.eigvalsh(prediction.electronic_dielectric).min() > 0.999


def test_independent_control_sequential_backward_matches_joint_objective():
    graph = record_to_graph(load_gmtnet_records("data/raw/gmtnet")[10], 5.0, 12)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    graph.y_born = torch.linspace(
        -0.2, 0.3, graph.num_nodes * 9, dtype=torch.float32
    ).reshape(graph.num_nodes, 3, 3)
    graph.y_electronic_piezo = torch.linspace(
        -0.4, 0.5, 27, dtype=torch.float32
    ).reshape(1, 3, 3, 3)
    kwargs = {
        "embedding_dim": 4,
        "cutoff": 5.0,
        "lmax": 3,
        "num_blocks": 1,
        "radial_basis": 3,
        "radial_hidden": 8,
        "global_context_dim": 8,
        "spectral_channels": 2,
        "spectral_shells": 2,
        "polar_fluctuation_shells": 2,
        "reciprocal_cutoff": 3.0,
        "attention_dim": 4,
    }
    model = IndependentElectrostaticHeads(**kwargs)
    reference = IndependentElectrostaticHeads(**kwargs)
    reference.load_state_dict(model.state_dict())

    prediction = reference.coefficients(graph)
    reference_electronic = irrep_balanced_capacity_loss(
        prediction.electronic_piezo, graph.y_electronic_piezo
    )
    reference_born = born_material_balanced_loss(
        prediction.born_charges, graph.y_born, graph.batch
    )
    (reference_electronic + reference_born).backward()

    sequential_electronic, sequential_born = backward_training_objective(
        model, graph, "a0_independent_irreps"
    )
    assert torch.allclose(sequential_electronic, reference_electronic)
    assert torch.allclose(sequential_born, reference_born)
    for (name, parameter), (reference_name, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == reference_name
        if parameter.grad is None or reference_parameter.grad is None:
            assert parameter.grad is None and reference_parameter.grad is None, name
            continue
        assert torch.allclose(
            parameter.grad, reference_parameter.grad, rtol=2e-5, atol=2e-7
        ), name


def test_gradient_audit_uses_active_norm_strata_not_dataset_prefix():
    class Dataset:
        def __init__(self):
            self.graphs = []
            for index, norm in enumerate([0.0, 0.1, 0.3, 0.6, 1.2]):
                target = torch.zeros(1, 3, 3, 3)
                target.reshape(-1)[0] = norm
                self.graphs.append(Data(
                    y_electronic_piezo=target,
                    material_id=f"JVASP-{index}",
                ))

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, index):
            return self.graphs[index]

    indices, audit = response_active_diagnostic_indices(
        Dataset(), 2, component_floor=0.05
    )
    assert indices == [2, 4]
    assert [row["material_id"] for row in audit] == ["JVASP-2", "JVASP-4"]


def test_electrostatic_coverage_requires_finite_shape_valid_arrays():
    payload = {
        "born_charges": torch.zeros(2, 3, 3),
        "total_piezo_source": torch.zeros(3, 6),
        "ionic_piezo_source": torch.zeros(3, 6),
        "force_constants": torch.zeros(2, 2, 3, 3),
        "epsilon": {"epsilon": torch.eye(3)},
    }
    assert all(electrostatic_availability(payload).values())
    payload["force_constants"] = torch.full((2, 2, 3, 3), torch.nan)
    availability = electrostatic_availability(payload)
    assert not availability["force_constants"]
    assert availability["born_charges"]


def test_nonredundant_fold_manifest_derives_formula_safe_train_panel():
    payload = {
        "material_ids": ["a", "b", "c"],
        "folds": [{"fold": 0, "development": ["b"]}],
    }
    assert electrostatic_fold_train_ids(payload, 0) == ["a", "c"]
    payload["folds"][0]["development"] = ["outside"]
    with pytest.raises(ValueError, match="population subset"):
        electrostatic_fold_train_ids(payload, 0)


def test_electrostatic_pretraining_accepts_nonredundant_schema_two_manifest():
    payload = {
        "material_ids": ["a", "b", "c"],
        "folds": [{"fold": 0, "development": ["b"]}],
    }
    assert electrostatic_pretraining_ids(
        payload, 0, {"a", "b", "c"}, train_limit=0, seed=42
    ) == ["a", "c"]
    with pytest.raises(ValueError, match="Unknown electrostatic train IDs"):
        electrostatic_pretraining_ids(
            payload, 0, {"a", "b"}, train_limit=0, seed=42
        )


def test_adjudication_plan_is_nonexecuting_fresh_and_frozen_panel_safe(tmp_path):
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({
        "frozen_validation_test_labels_read": False,
        "folds": [{"fold": 0, "development": ["b"]}],
    }), encoding="utf-8")
    plan = build_plan(
        folds_path=folds,
        config_path=tmp_path / "config.yaml",
        cohort_root=tmp_path / "cohort",
        fold_index=0,
        seed=42,
        train_limit=100,
        development_limit=100,
        pretrain_epochs=20,
        updates=100,
        batch_size=4,
        eval_interval=25,
    )
    assert plan["status"] == "planned_not_executed"
    assert plan["data_boundary"]["frozen_validation_test_labels_read"] is False
    assert len(plan["steps"]) == 6
    assert all(
        step.get("architecture") in ARCHITECTURES
        for step in plan["steps"][1:5]
    )
