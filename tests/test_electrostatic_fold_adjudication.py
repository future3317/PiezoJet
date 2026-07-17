import json

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

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
    load_structure_pretraining,
    make_model,
    response_active_diagnostic_indices,
)
from piezojet.checkpoint_provenance import validate_checkpoint_provenance
from piezojet.electronic_capacity import (
    born_material_balanced_loss,
    dielectric_capacity_metrics,
    irrep_balanced_capacity_loss,
)
from piezojet.model import (
    IndependentElectrostaticHeads,
    SoftSharedElectromechanicalJetHead,
)
from piezojet.pretrain_e3nn import electrostatic_pretraining_ids
from piezojet.pretraining_protocol import provenance
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


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_microbatch_gradient_matches_one_logical_material_mean(architecture):
    records = load_gmtnet_records("data/raw/gmtnet")
    graphs = [
        record_to_graph(records[index], 5.0, 12)
        for index in (10, 11)
    ]
    for offset, graph in enumerate(graphs):
        graph.y_born = torch.linspace(
            -0.2 + offset, 0.3 + offset, graph.num_nodes * 9
        ).reshape(graph.num_nodes, 3, 3)
        graph.y_electronic_piezo = torch.linspace(
            -0.4 + offset, 0.5 + offset, 27
        ).reshape(1, 3, 3, 3)
    kwargs = {
        "embedding_dim": 4, "cutoff": 5.0, "lmax": 3, "num_blocks": 1,
        "radial_basis": 3, "radial_hidden": 8, "global_context_dim": 8,
        "spectral_channels": 2, "spectral_shells": 2,
        "polar_fluctuation_shells": 2, "reciprocal_cutoff": 3.0,
        "global_attention_dim": 4,
    }
    microbatched = make_model(architecture, kwargs)
    logical = make_model(architecture, kwargs)
    logical.load_state_dict(microbatched.state_dict())
    backward_training_objective(
        logical, Batch.from_data_list(graphs), architecture
    )
    for graph in graphs:
        backward_training_objective(
            microbatched, Batch.from_data_list([graph]),
            architecture, gradient_scale=0.5,
        )
    for (name, parameter), (reference_name, reference_parameter) in zip(
        microbatched.named_parameters(), logical.named_parameters(), strict=True
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


def test_fold_pretraining_checkpoint_strictly_initializes_every_stage_a_encoder(tmp_path):
    """Every maintained candidate rejects encoder-layout drift."""
    config = {
        "embedding_dim": 4,
        "cutoff": 3.0,
        "lmax": 3,
        "num_blocks": 1,
        "radial_basis": 3,
        "radial_hidden": 8,
        "global_context_dim": 8,
        "spectral_channels": 2,
        "spectral_shells": 2,
        "polar_fluctuation_shells": 2,
        "reciprocal_cutoff": 3.0,
        "global_attention_dim": 4,
    }
    from piezojet.baselines import e3nn_direct_baseline_from_config

    source = e3nn_direct_baseline_from_config(config)
    checkpoint = tmp_path / "encoder.pt"
    split_source = tmp_path / "folds.json"
    split_source.write_text("{}", encoding="utf-8")
    torch.save({
        "architecture": "e3nn_periodic_v1",
        "encoder": source.encoder.state_dict(),
        "pretraining_provenance": provenance(["train-id"], split_source, "train"),
        "epoch": 1,
        "loss": 0.5,
    }, checkpoint)
    for architecture in ARCHITECTURES:
        model = make_model(architecture, config)
        loaded_provenance = load_structure_pretraining(
            model, architecture, checkpoint, torch.device("cpu"),
            ["train-id"], ["development-id"],
        )
        assert loaded_provenance["encoder_copies_initialized"] == (
            2 if architecture == "a0_independent_irreps" else 1
        )
        encoders = (
            [model.born_generator.encoder, model.piezo_generator.encoder]
            if architecture == "a0_independent_irreps"
            else [model.encoder]
        )
        for encoder in encoders:
            for name, value in source.encoder.state_dict().items():
                assert torch.equal(encoder.state_dict()[name], value), name


def test_fold_pretraining_checkpoint_rejects_development_structure_provenance(tmp_path):
    config = {
        "embedding_dim": 4, "cutoff": 3.0, "lmax": 3, "num_blocks": 1,
        "radial_basis": 3, "radial_hidden": 8, "global_context_dim": 8,
        "spectral_channels": 2, "spectral_shells": 2,
        "polar_fluctuation_shells": 2, "reciprocal_cutoff": 3.0,
        "global_attention_dim": 4,
    }
    from piezojet.baselines import e3nn_direct_baseline_from_config

    source = e3nn_direct_baseline_from_config(config)
    checkpoint = tmp_path / "leaked_encoder.pt"
    split_source = tmp_path / "folds.json"
    split_source.write_text("{}", encoding="utf-8")
    torch.save({
        "architecture": "e3nn_periodic_v1",
        "encoder": source.encoder.state_dict(),
        "pretraining_provenance": provenance(
            ["development-id"], split_source, "train"
        ),
    }, checkpoint)
    with pytest.raises(ValueError, match="held-out IDs"):
        load_structure_pretraining(
            make_model("a1_electromechanical_jet", config),
            "a1_electromechanical_jet", checkpoint, torch.device("cpu"),
            ["train-id"], ["development-id"],
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
    assert len(plan["steps"]) == 5
    assert all(
        step.get("architecture") in ARCHITECTURES
        for step in plan["steps"][1:4]
    )


def test_stage_a_checkpoint_provenance_binds_fold_ids_and_source(tmp_path):
    source = tmp_path / "folds.json"
    source.write_text("{}", encoding="utf-8")
    config = {
        "seed": 42,
        "data_commit": "a" * 40,
        "fold_identity": "electrostatic-development-fold-0",
    }
    from piezojet.checkpoint_provenance import build_checkpoint_provenance

    expected = build_checkpoint_provenance(
        {"train": ["train-a"], "val": ["dev-a"], "test": []},
        source,
        config,
        split_kind="electrostatic_development_fold_0",
    )
    assert validate_checkpoint_provenance(
        {"checkpoint_provenance": expected}, expected
    ) == expected
    changed = build_checkpoint_provenance(
        {"train": ["train-b"], "val": ["dev-a"], "test": []},
        source,
        config,
        split_kind="electrostatic_development_fold_0",
    )
    with pytest.raises(ValueError, match="split_id_sha256|all_ids_sha256"):
        validate_checkpoint_provenance(
            {"checkpoint_provenance": expected}, changed
        )


def test_soft_shared_jet_preserves_tensor_shapes_and_has_task_adapters():
    graph = record_to_graph(load_gmtnet_records("data/raw/gmtnet")[10], 5.0, 12)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = SoftSharedElectromechanicalJetHead(
        embedding_dim=4, cutoff=5.0, lmax=3, num_blocks=1, radial_basis=3,
        radial_hidden=8, global_context_dim=8, spectral_channels=2,
        spectral_shells=2, polar_fluctuation_shells=2, reciprocal_cutoff=3.0,
        attention_dim=4,
    ).eval()
    prediction = model.coefficients(graph)
    assert prediction.born_charges.shape == (graph.num_nodes, 3, 3)
    assert prediction.electronic_piezo.shape == (1, 3, 3, 3)
    assert model.born_adapter is not model.electronic_adapter
    assert float(model.born_adapter.residual_scale.detach()) == 0.0


def test_dielectric_audit_is_availability_masked_without_zero_fill_bias():
    target = torch.stack((torch.eye(3), 2.0 * torch.eye(3)))
    prediction = target.clone()
    prediction[1] = 0.0
    metrics = dielectric_capacity_metrics(
        prediction, target, torch.tensor([True, False])
    )
    assert metrics["available_materials"] == 1
    assert metrics["mean_stabilized_relative_frobenius_error"] == 0.0
