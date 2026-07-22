import json
from hashlib import sha256

import numpy as np
import pytest
import torch
from e3nn import o3
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
    _evaluate,
    backward_training_objective,
    load_electronic_response_pretraining,
    load_structure_pretraining,
    load_bec_response_pretraining,
    make_model,
    response_active_diagnostic_indices,
)
from piezojet.electrostatic_a0_fold_adjudication import (
    TASKS,
    _evaluate_task,
    _restore_a0_progress,
)
from tests.data_paths import gmtnet_root
from piezojet.electrostatic_protocol import (
    A0_ARCHITECTURES,
    STABILIZED_SELECTION_VERSION,
    compact_training_curve_row,
    development_early_stopping,
    development_selection,
    matched_material_schedule,
)
from piezojet.checkpoint_provenance import validate_checkpoint_provenance
from piezojet.electronic_capacity import (
    born_material_balanced_loss,
    dielectric_capacity_metrics,
    dielectric_material_balanced_loss,
    irrep_balanced_capacity_loss,
)
from piezojet.model import (
    HierarchicalElectromechanicalJetHead,
    IndependentElectrostaticHeads,
    SoftSharedElectromechanicalJetHead,
    TrainableIrrepAdapter,
)
from piezojet.pretrain_e3nn import (
    electrostatic_pretraining_ids,
    logical_pretraining_batch_sizes,
)
from piezojet.pretrain_bec_e3nn import validate_resume_payload as validate_bec_resume_payload
from piezojet.pretraining_protocol import (
    BEC_RESPONSE_PRETRAINING_ARCHITECTURE,
    BEC_RESPONSE_PRETRAINING_OBJECTIVE,
    ELECTRONIC_RESPONSE_PRETRAINING_ARCHITECTURE,
    ELECTRONIC_RESPONSE_PRETRAINING_OBJECTIVE,
)
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
    dielectric_ids = {
        id(parameter) for parameter in model.dielectric_generator.parameters()
    }
    assert not born_ids & piezo_ids
    assert not born_ids & dielectric_ids
    assert not piezo_ids & dielectric_ids
    assert not hasattr(model.born_generator, "electronic_irreps")
    assert not hasattr(model.piezo_generator, "born_irreps")


def test_parameter_matched_a0_is_close_to_shared_candidate_capacity():
    config = {
        "embedding_dim": 32,
        "cutoff": 5.0,
        "lmax": 3,
        "num_blocks": 3,
        "radial_basis": 12,
        "radial_hidden": 64,
        "global_context_dim": 128,
        "spectral_channels": 16,
        "spectral_shells": 8,
        "polar_fluctuation_shells": 8,
        "reciprocal_cutoff": 7.0,
        "global_attention_dim": 64,
        "a0_parameter_matched_width_multiplier": 0.56,
    }
    independent = make_model("a0_parameter_matched_irreps", config)
    shared = make_model("a1_electromechanical_jet", config)
    hierarchical = make_model("a16_hierarchical_electromechanical_jet", config)
    counts = {
        "a0_pm": sum(parameter.numel() for parameter in independent.parameters()),
        "a1": sum(parameter.numel() for parameter in shared.parameters()),
        "a16": sum(parameter.numel() for parameter in hierarchical.parameters()),
    }
    assert counts == {"a0_pm": 6_358_299, "a1": 6_454_490, "a16": 6_673_790}
    assert abs(counts["a0_pm"] / counts["a1"] - 1.0) < 0.02
    assert abs(counts["a0_pm"] / counts["a16"] - 1.0) < 0.05


def test_trainable_irrep_adapter_is_equivariant_and_live_on_first_backward():
    irreps = o3.Irreps(
        "2x0e + 1x0o + 2x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o"
    )
    torch.manual_seed(7)
    adapter = TrainableIrrepAdapter(irreps, context_dim=5)
    features = torch.randn(4, irreps.dim, requires_grad=True)
    context = torch.randn(2, 5)
    batch_index = torch.tensor([0, 0, 1, 1])
    rotation = o3.rand_matrix()
    representation = irreps.D_from_matrix(rotation)
    rotated = adapter(features @ representation.T, context, batch_index)
    expected = adapter(features, context, batch_index) @ representation.T
    assert torch.allclose(rotated, expected, rtol=2e-4, atol=2e-5)

    adapter(features, context, batch_index).square().mean().backward()
    assert adapter.scale_logits.grad is not None
    assert torch.linalg.vector_norm(adapter.scale_logits.grad) > 0
    assert adapter.mix.weight.grad is not None
    assert torch.linalg.vector_norm(adapter.mix.weight.grad) > 0
    assert adapter.context_gates[-1].weight.grad is not None
    assert torch.linalg.vector_norm(adapter.context_gates[-1].weight.grad) > 0
    assert torch.allclose(
        torch.nn.functional.softplus(adapter.scale_logits.detach()),
        torch.full_like(adapter.scale_logits, 0.075),
    )


def test_a0_matched_schedule_reuses_identical_complete_microbatch_passes():
    schedule = matched_material_schedule(5, 3, 4, 2, 17)
    assert schedule == matched_material_schedule(5, 3, 4, 2, 17)
    assert len(schedule) == 12
    for start in range(0, len(schedule), 4):
        assert len(set(schedule[start : start + 4])) == 4


def test_a0_resume_restores_all_towers_and_optimizers_at_common_boundary():
    class TinyA0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.piezo_generator = torch.nn.Linear(2, 2)
            self.born_generator = torch.nn.Linear(2, 2)
            self.dielectric_generator = torch.nn.Linear(2, 2)

    source = TinyA0()
    source_optimizers = {
        task: torch.optim.AdamW(getattr(
            source,
            {"electronic": "piezo_generator", "born": "born_generator", "dielectric": "dielectric_generator"}[task],
        ).parameters())
        for task in TASKS
    }
    for task, optimizer in source_optimizers.items():
        tower = getattr(
            source,
            {"electronic": "piezo_generator", "born": "born_generator", "dielectric": "dielectric_generator"}[task],
        )
        optimizer.zero_grad(set_to_none=True)
        tower(torch.ones(1, 2)).sum().backward()
        optimizer.step()

    contract = {"updates": 10, "eval_interval": 5, "seed": 42}
    provenance = {"all_ids_sha256": "a" * 64}
    payload = {
        "status": "running",
        "completed_update": 5,
        "training_contract": contract,
        "checkpoint_provenance": provenance,
        "model": {
            "electronic": source.piezo_generator.state_dict(),
            "born": source.born_generator.state_dict(),
            "dielectric": source.dielectric_generator.state_dict(),
        },
        "optimizer": {
            task: optimizer.state_dict() for task, optimizer in source_optimizers.items()
        },
        "history": [{"update": 5}],
        "best_score": 1.25,
        "best_update": 5,
        "best_model": {task: {} for task in TASKS},
        "best_metrics": {task: {} for task in TASKS},
        "initial_gradients": {task: {"loss": 1.0} for task in TASKS},
    }
    restored_model = TinyA0()
    restored_optimizers = {
        task: torch.optim.AdamW(getattr(
            restored_model,
            {"electronic": "piezo_generator", "born": "born_generator", "dielectric": "dielectric_generator"}[task],
        ).parameters())
        for task in TASKS
    }
    restored = _restore_a0_progress(
        restored_model, restored_optimizers, payload, contract, provenance
    )
    assert restored["start_block"] == 5
    assert restored["history"] == [{"update": 5}]
    for task in TASKS:
        source_tower = getattr(
            source,
            {"electronic": "piezo_generator", "born": "born_generator", "dielectric": "dielectric_generator"}[task],
        )
        restored_tower = getattr(
            restored_model,
            {"electronic": "piezo_generator", "born": "born_generator", "dielectric": "dielectric_generator"}[task],
        )
        for expected, actual in zip(source_tower.parameters(), restored_tower.parameters(), strict=True):
            assert torch.equal(expected, actual)
        assert restored_optimizers[task].state

    with pytest.raises(ValueError, match="training contract"):
        _restore_a0_progress(
            restored_model, restored_optimizers, payload,
            {**contract, "seed": 7}, provenance,
        )
    with pytest.raises(ValueError, match="block boundary"):
        _restore_a0_progress(
            restored_model, restored_optimizers,
            {**payload, "completed_update": 3}, contract, provenance,
        )


def test_independent_control_runs_only_its_declared_decoders():
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[10], 5.0, 12)
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


def test_a0_starts_from_the_same_response_function_as_a1_without_sharing_parameters():
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[10], 5.0, 12)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
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
    torch.manual_seed(23)
    independent = IndependentElectrostaticHeads(**kwargs).eval()
    torch.manual_seed(23)
    shared = make_model("a1_electromechanical_jet", {
        **kwargs, "global_attention_dim": kwargs["attention_dim"],
    }).eval()
    independent_prediction = independent.coefficients(graph)
    shared_prediction = shared.coefficients(graph)
    assert torch.equal(
        independent_prediction.born_charges, shared_prediction.born_charges
    )
    assert torch.equal(
        independent_prediction.electronic_piezo,
        shared_prediction.electronic_piezo,
    )
    assert torch.equal(
        independent_prediction.electronic_dielectric,
        shared_prediction.electronic_dielectric,
    )


def test_independent_control_sequential_backward_matches_joint_objective():
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[10], 5.0, 12)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    graph.y_born = torch.linspace(
        -0.2, 0.3, graph.num_nodes * 9, dtype=torch.float32
    ).reshape(graph.num_nodes, 3, 3)
    graph.y_electronic_piezo = torch.linspace(
        -0.4, 0.5, 27, dtype=torch.float32
    ).reshape(1, 3, 3, 3)
    graph.y_dfpt_electronic_dielectric = (2.0 * torch.eye(3)).unsqueeze(0)
    graph.dfpt_electronic_dielectric_mask = torch.tensor(True)
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
    reference_dielectric = dielectric_material_balanced_loss(
        prediction.electronic_dielectric,
        graph.y_dfpt_electronic_dielectric,
        graph.dfpt_electronic_dielectric_mask,
    )
    (reference_electronic + reference_born + reference_dielectric).backward()

    sequential_electronic, sequential_born, sequential_dielectric = backward_training_objective(
        model, graph, "a0_independent_irreps"
    )
    assert torch.allclose(sequential_electronic, reference_electronic)
    assert torch.allclose(sequential_born, reference_born)
    assert torch.allclose(sequential_dielectric, reference_dielectric)
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


def test_evaluation_preserves_version_counters_required_by_geometry_cache():
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[10], 5.0, 12)
    graph.y_born = torch.zeros(graph.num_nodes, 3, 3)
    graph.y_electronic_piezo = torch.zeros(1, 3, 3, 3)
    graph.y_dfpt_electronic_dielectric = (2.0 * torch.eye(3)).unsqueeze(0)
    graph.dfpt_electronic_dielectric_mask = torch.tensor(True)
    batch = Batch.from_data_list([graph])
    config = {
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
        "global_attention_dim": 4,
    }
    shared = make_model("a1_electromechanical_jet", config)
    metrics = _evaluate(shared, [batch], "a1_electromechanical_jet")
    assert set(metrics) == {"electronic", "born", "dielectric_audit"}

    independent = make_model("a0_independent_irreps", config)
    electronic = _evaluate_task(
        independent.piezo_generator, [batch], "electronic"
    )
    assert electronic["materials"] == 1


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_microbatch_gradient_matches_one_logical_material_mean(architecture):
    records = load_gmtnet_records(gmtnet_root())
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
        graph.y_dfpt_electronic_dielectric = (
            (2.0 + offset) * torch.eye(3)
        ).unsqueeze(0)
        graph.dfpt_electronic_dielectric_mask = torch.tensor(True)
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


def test_pretraining_logical_batches_preserve_complete_exposure_epoch():
    sizes = logical_pretraining_batch_sizes(3951, 4, 32)
    assert sizes == [32] * 123 + [15]
    assert sum(sizes) == 3951
    with pytest.raises(ValueError, match="divisible"):
        logical_pretraining_batch_sizes(3951, 4, 30)


def test_stabilized_selection_guardrails_cannot_be_hidden_by_scalar_score():
    metrics = {
        "electronic": {
            "mean_stabilized_relative_frobenius_error": 0.4,
            "mean_active_cosine": 0.2,
            "mean_active_amplitude_ratio": 0.1,
        },
        "born": {
            "mean_stabilized_relative_frobenius_error": 0.3,
            "mean_nonzero_cosine": 0.1,
            "mean_exact_zero_prediction_norm_e": 0.02,
        },
        "dielectric": {"mean_stabilized_relative_frobenius_error": 0.5},
    }
    selection = development_selection(metrics)
    assert selection["version"] == STABILIZED_SELECTION_VERSION
    assert selection["raw_score"] == pytest.approx(1.2)
    assert selection["eligible"] is True
    assert selection["exact_zero_bec_absolute_leakage_e"] == pytest.approx(0.02)

    collapsed = development_selection({
        **metrics,
        "electronic": {
            **metrics["electronic"],
            "mean_active_amplitude_ratio": 0.01,
        },
    })
    assert collapsed["raw_score"] == pytest.approx(selection["raw_score"])
    assert collapsed["eligible"] is False
    assert "electronic_active_amplitude_collapse" in collapsed["guardrail_failures"]


def test_early_stopping_waits_for_an_eligible_checkpoint_then_counts_evaluations():
    collapsed = development_early_stopping(
        score=1.0,
        eligible=False,
        best_score=float("inf"),
        non_improving_evaluations=0,
        patience_evaluations=2,
        minimum_improvement=0.0,
    )
    assert collapsed["improved"] is False
    assert collapsed["non_improving_evaluations"] == 0
    assert collapsed["should_stop"] is False

    first_eligible = development_early_stopping(
        score=2.0,
        eligible=True,
        best_score=float("inf"),
        non_improving_evaluations=0,
        patience_evaluations=2,
        minimum_improvement=0.0,
    )
    assert first_eligible["improved"] is True
    assert first_eligible["non_improving_evaluations"] == 0

    first_miss = development_early_stopping(
        score=2.1,
        eligible=True,
        best_score=2.0,
        non_improving_evaluations=0,
        patience_evaluations=2,
        minimum_improvement=0.0,
    )
    assert first_miss["non_improving_evaluations"] == 1
    assert first_miss["should_stop"] is False
    second_miss = development_early_stopping(
        score=2.2,
        eligible=True,
        best_score=2.0,
        non_improving_evaluations=1,
        patience_evaluations=2,
        minimum_improvement=0.0,
    )
    assert second_miss["non_improving_evaluations"] == 2
    assert second_miss["should_stop"] is True


def test_early_stopping_minimum_improvement_and_disable_switch():
    below_delta = development_early_stopping(
        score=0.95,
        eligible=True,
        best_score=1.0,
        non_improving_evaluations=0,
        patience_evaluations=2,
        minimum_improvement=0.1,
    )
    assert below_delta["improved"] is False
    disabled = development_early_stopping(
        score=2.0,
        eligible=False,
        best_score=1.0,
        non_improving_evaluations=99,
        patience_evaluations=0,
        minimum_improvement=0.0,
    )
    assert disabled["should_stop"] is False


def test_compact_training_curve_row_drops_per_material_payloads():
    row = {
        "update": 200,
        "train_loss": 0.5,
        "development_selection_score": 2.0,
        "development_selection": {
            "eligible": True,
            "components": {"electronic_stabilized_relative": 0.7},
            "guardrails": {"electronic_active_cosine": 0.2},
            "guardrail_failures": [],
        },
        "train_selection": {"raw_score": 1.2},
        "generalization_score_gap": 0.8,
        "development_metrics": {"electronic": {"per_material": [1, 2, 3]}},
        "early_stopping": {"should_stop": False},
    }
    compact = compact_training_curve_row(row)
    assert compact is not None
    assert compact["update"] == 200
    assert compact["train_selection_score"] == pytest.approx(1.2)
    assert compact["generalization_score_gap"] == pytest.approx(0.8)
    assert "development_metrics" not in compact


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
    for architecture in (
        architecture
        for architecture in ARCHITECTURES
        if architecture != "a0_parameter_matched_irreps"
    ):
        model = make_model(architecture, config)
        loaded_provenance = load_structure_pretraining(
            model, architecture, checkpoint, torch.device("cpu"),
            ["train-id"], ["development-id"],
        )
        assert loaded_provenance["encoder_copies_initialized"] == (
            3 if architecture in A0_ARCHITECTURES else 1
        )
        encoders = (
            [
                model.born_generator.encoder,
                model.piezo_generator.encoder,
                model.dielectric_generator.encoder,
            ]
            if architecture in A0_ARCHITECTURES
            else [model.encoder]
        )
        for encoder in encoders:
            for name, value in source.encoder.state_dict().items():
                assert torch.equal(encoder.state_dict()[name], value), name


def test_full_width_pretraining_cannot_initialize_parameter_matched_a0(tmp_path):
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
        "a0_parameter_matched_width_multiplier": 0.56,
        "code_commit": "b" * 40,
    }
    from piezojet.baselines import e3nn_direct_baseline_from_config

    source = e3nn_direct_baseline_from_config(config)
    split_source = tmp_path / "folds.json"
    split_source.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "full_width_encoder.pt"
    torch.save({
        "architecture": "e3nn_periodic_v1",
        "encoder": source.encoder.state_dict(),
        "pretraining_provenance": provenance(["train-id"], split_source, "train"),
        "code_commit": "a" * 40,
        "config": dict(config),
        "pretraining_contract": {
            "objective": "masked_species_plus_translation_free_coordinate_denoising",
            "response_label_count": 0,
        },
    }, checkpoint)
    with pytest.raises(ValueError, match="encoder configuration differs"):
        load_structure_pretraining(
            make_model("a0_parameter_matched_irreps", config),
            "a0_parameter_matched_irreps",
            checkpoint,
            torch.device("cpu"),
            ["train-id"],
            ["development-id"],
            config,
        )


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


def test_fold_pretraining_checkpoint_records_audited_cross_commit_reuse(tmp_path):
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
        "code_commit": "b" * 40,
    }
    from piezojet.baselines import e3nn_direct_baseline_from_config

    source = e3nn_direct_baseline_from_config(config)
    split_source = tmp_path / "folds.json"
    split_source.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "encoder.pt"
    source_config = dict(config)
    source_config["code_commit"] = "a" * 40
    torch.save({
        "architecture": "e3nn_periodic_v1",
        "encoder": source.encoder.state_dict(),
        "pretraining_provenance": provenance(["train-id"], split_source, "train"),
        "code_commit": "a" * 40,
        "config": source_config,
        "pretraining_contract": {
            "objective": "masked_species_plus_translation_free_coordinate_denoising",
            "response_label_count": 0,
        },
    }, checkpoint)
    audit = load_structure_pretraining(
        make_model("a1_electromechanical_jet", config),
        "a1_electromechanical_jet",
        checkpoint,
        torch.device("cpu"),
        ["train-id"],
        ["development-id"],
        config,
    )
    assert audit["pretraining_source_code_commit"] == "a" * 40
    assert audit["downstream_code_commit"] == "b" * 40
    assert audit["cross_commit_reuse"] is True


def test_fold_pretraining_checkpoint_rejects_encoder_config_drift(tmp_path):
    config = {
        "embedding_dim": 4, "cutoff": 3.0, "lmax": 3, "num_blocks": 1,
        "radial_basis": 3, "radial_hidden": 8, "global_context_dim": 8,
        "spectral_channels": 2, "spectral_shells": 2,
        "polar_fluctuation_shells": 2, "reciprocal_cutoff": 3.0,
        "global_attention_dim": 4, "code_commit": "b" * 40,
    }
    from piezojet.baselines import e3nn_direct_baseline_from_config

    source = e3nn_direct_baseline_from_config(config)
    split_source = tmp_path / "folds.json"
    split_source.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "encoder.pt"
    saved_config = dict(config)
    saved_config["cutoff"] = 4.0
    torch.save({
        "architecture": "e3nn_periodic_v1",
        "encoder": source.encoder.state_dict(),
        "pretraining_provenance": provenance(["train-id"], split_source, "train"),
        "code_commit": "a" * 40,
        "config": saved_config,
        "pretraining_contract": {
            "objective": "masked_species_plus_translation_free_coordinate_denoising",
            "response_label_count": 0,
        },
    }, checkpoint)
    with pytest.raises(ValueError, match="encoder configuration differs"):
        load_structure_pretraining(
            make_model("a1_electromechanical_jet", config),
            "a1_electromechanical_jet",
            checkpoint,
            torch.device("cpu"),
            ["train-id"],
            ["development-id"],
            config,
        )


def test_bec_response_pretraining_strictly_initializes_only_a0pm_born_tower(tmp_path):
    config = {
        "embedding_dim": 4, "cutoff": 3.0, "max_neighbors": 8, "lmax": 3,
        "num_blocks": 1, "radial_basis": 3, "radial_hidden": 8,
        "global_context_dim": 8, "spectral_channels": 2, "spectral_shells": 2,
        "polar_fluctuation_shells": 2, "reciprocal_cutoff": 3.0,
        "global_attention_dim": 4, "a0_parameter_matched_width_multiplier": 0.56,
        "code_commit": "b" * 40, "fold_identity": "electrostatic-development-fold-0",
    }
    split_source = tmp_path / "folds.json"
    split_source.write_text("{}", encoding="utf-8")
    source = make_model("a0_parameter_matched_irreps", config)
    entry = provenance(["train-id"], split_source, "train", config)
    entry.update({
        "development_material_id_sha256": sha256(b"development-id").hexdigest(),
        "development_formula_overlap_count": 0,
        "development_fold": 0,
        "response_task": "born",
    })
    checkpoint = tmp_path / "bec.pt"
    torch.save({
        "architecture": BEC_RESPONSE_PRETRAINING_ARCHITECTURE,
        "objective": BEC_RESPONSE_PRETRAINING_OBJECTIVE,
        "born_tower": source.born_generator.state_dict(),
        "optimizer": {},
        "config": {
            **config,
            "electrostatic_encoder_width_multiplier": 0.56,
        },
        "pretraining_provenance": entry,
        "response_pretraining_contract": {
            "objective": BEC_RESPONSE_PRETRAINING_OBJECTIVE,
            "response_task": "born",
            "downstream_architecture": "a0_parameter_matched_irreps",
            "encoder_width_multiplier": 0.56,
        },
    }, checkpoint)
    target = make_model("a0_parameter_matched_irreps", config)
    piezo_before = {
        name: value.detach().clone()
        for name, value in target.piezo_generator.state_dict().items()
    }
    audit = load_bec_response_pretraining(
        target, "a0_parameter_matched_irreps", checkpoint, torch.device("cpu"),
        ["train-id"], ["development-id"], config,
    )
    assert audit["initialized_parameter_scope"] == "born_generator_only"
    for name, value in source.born_generator.state_dict().items():
        assert torch.equal(target.born_generator.state_dict()[name], value), name
    for name, value in piezo_before.items():
        assert torch.equal(target.piezo_generator.state_dict()[name], value), name
    with pytest.raises(ValueError, match="only defined for A0-PM"):
        load_bec_response_pretraining(
            make_model("a1_electromechanical_jet", config),
            "a1_electromechanical_jet", checkpoint, torch.device("cpu"),
            ["train-id"], ["development-id"], config,
        )


def test_bec_response_pretraining_rejects_development_panel_drift_and_resume_drift(tmp_path):
    split_source = tmp_path / "folds.json"
    split_source.write_text("{}", encoding="utf-8")
    entry = provenance(["train-id"], split_source, "train")
    entry.update({
        "development_material_id_sha256": sha256(b"development-id").hexdigest(),
        "development_formula_overlap_count": 0,
        "response_task": "born",
    })
    contract = {
        "objective": BEC_RESPONSE_PRETRAINING_OBJECTIVE,
        "response_task": "born",
        "downstream_architecture": "a0_parameter_matched_irreps",
        "encoder_width_multiplier": 0.56,
    }
    payload = {
        "architecture": BEC_RESPONSE_PRETRAINING_ARCHITECTURE,
        "objective": BEC_RESPONSE_PRETRAINING_OBJECTIVE,
        "born_tower": {}, "optimizer": {}, "pretraining_provenance": entry,
        "response_pretraining_contract": contract,
    }
    validate_bec_resume_payload(payload, entry, contract)
    with pytest.raises(ValueError, match="contract differs"):
        validate_bec_resume_payload(payload, entry, {**contract, "logical_batch_size": 32})
    from piezojet.pretraining_protocol import validate_bec_response_pretraining_checkpoint
    with pytest.raises(ValueError, match="development-panel hash"):
        validate_bec_response_pretraining_checkpoint(
            payload, ["train-id"], ["other-development-id"], {},
            expected_architecture="a0_parameter_matched_irreps",
            expected_width_multiplier=0.56,
            expected_development_ids=["other-development-id"],
        )


def test_electronic_response_pretraining_initializes_only_a0pm_piezo_tower(tmp_path):
    config = {
        "embedding_dim": 4, "cutoff": 3.0, "max_neighbors": 8, "lmax": 3,
        "num_blocks": 1, "radial_basis": 3, "radial_hidden": 8,
        "global_context_dim": 8, "spectral_channels": 2, "spectral_shells": 2,
        "polar_fluctuation_shells": 2, "reciprocal_cutoff": 3.0,
        "global_attention_dim": 4, "a0_parameter_matched_width_multiplier": 0.56,
        "code_commit": "b" * 40, "fold_identity": "electrostatic-development-fold-0",
    }
    split_source = tmp_path / "folds.json"
    split_source.write_text("{}", encoding="utf-8")
    source = make_model("a0_parameter_matched_irreps", config)
    entry = provenance(["train-id"], split_source, "train", config)
    entry.update({
        "development_material_id_sha256": sha256(b"development-id").hexdigest(),
        "development_formula_overlap_count": 0,
        "development_fold": 0,
        "response_task": "electronic",
    })
    checkpoint = tmp_path / "electronic.pt"
    torch.save({
        "architecture": ELECTRONIC_RESPONSE_PRETRAINING_ARCHITECTURE,
        "objective": ELECTRONIC_RESPONSE_PRETRAINING_OBJECTIVE,
        "piezo_tower": source.piezo_generator.state_dict(),
        "optimizer": {},
        "config": {**config, "electrostatic_encoder_width_multiplier": 0.56},
        "pretraining_provenance": entry,
        "response_pretraining_contract": {
            "objective": ELECTRONIC_RESPONSE_PRETRAINING_OBJECTIVE,
            "response_task": "electronic",
            "downstream_architecture": "a0_parameter_matched_irreps",
            "encoder_width_multiplier": 0.56,
        },
    }, checkpoint)
    target = make_model("a0_parameter_matched_irreps", config)
    born_before = {
        name: value.detach().clone()
        for name, value in target.born_generator.state_dict().items()
    }
    audit = load_electronic_response_pretraining(
        target, "a0_parameter_matched_irreps", checkpoint, torch.device("cpu"),
        ["train-id"], ["development-id"], config,
    )
    assert audit["initialized_parameter_scope"] == "piezo_generator_only"
    for name, value in source.piezo_generator.state_dict().items():
        assert torch.equal(target.piezo_generator.state_dict()[name], value), name
    for name, value in born_before.items():
        assert torch.equal(target.born_generator.state_dict()[name], value), name
    with pytest.raises(ValueError, match="only defined for A0-PM"):
        load_electronic_response_pretraining(
            make_model("a1_electromechanical_jet", config),
            "a1_electromechanical_jet", checkpoint, torch.device("cpu"),
            ["train-id"], ["development-id"], config,
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
    assert len(plan["steps"]) == 7
    candidate_steps = [
        step for step in plan["steps"] if step.get("architecture") is not None
    ]
    assert [step["architecture"] for step in candidate_steps] == [
        "a0_independent_irreps",
        "a0_parameter_matched_irreps",
        "a1_electromechanical_jet",
        "a16_hierarchical_electromechanical_jet",
    ]
    assert candidate_steps[0]["argv"][1:3] == [
        "-m", "piezojet.electrostatic_a0_fold_adjudication"
    ]
    assert candidate_steps[0]["argv"][
        candidate_steps[0]["argv"].index("--architecture") + 1
    ] == "a0_independent_irreps"
    assert "--early-stopping-patience-evaluations" in candidate_steps[0]["argv"]
    assert plan["comparison_contract"]["early_stopping_patience_updates"] == 50
    assert "--train-limit" not in plan["steps"][0]["argv"]


def test_stage_a_plan_separates_full_fold_pretraining_from_fixed_response_subset(tmp_path):
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({
        "frozen_validation_test_labels_read": False,
        "folds": [{"fold": 0, "development": ["dev"]}],
    }), encoding="utf-8")
    subset = tmp_path / "balanced.json"
    subset.write_text(json.dumps({"materials": 200}), encoding="utf-8")
    plan = build_plan(
        folds_path=folds,
        config_path=tmp_path / "config.yaml",
        cohort_root=tmp_path / "cohort",
        fold_index=0,
        seed=42,
        train_limit=0,
        development_limit=0,
        pretrain_epochs=1,
        updates=1,
        batch_size=1,
        eval_interval=1,
        response_subset_file=subset,
    )
    pretrain_argv = plan["steps"][0]["argv"]
    assert "--train-limit" not in pretrain_argv
    for step in (
        value for value in plan["steps"] if value.get("architecture") is not None
    ):
        assert step["argv"][step["argv"].index("--train-ids-file") + 1] == str(subset)
    assert plan["data_boundary"]["structure_pretraining_scope"] == (
        "complete fold-train structure universe"
    )


def test_stage_a_plan_reuses_audited_pretrain_without_planning_recompute(tmp_path):
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({
        "frozen_validation_test_labels_read": False,
        "folds": [{"fold": 0, "development": ["dev"]}],
    }), encoding="utf-8")
    checkpoint = tmp_path / "best_encoder.pt"
    checkpoint.write_bytes(b"audited-checkpoint-placeholder")
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
        pretrained_encoder=checkpoint,
        architectures=(
            "a0_independent_irreps",
            "a1_electromechanical_jet",
            "a16_hierarchical_electromechanical_jet",
        ),
    )
    assert len(plan["steps"]) == 4
    assert all(step.get("name") != "fold_train_only_structure_pretraining" for step in plan["steps"])
    assert plan["data_boundary"]["structure_pretraining_reused"] is True
    for step in (
        value for value in plan["steps"] if value.get("architecture") is not None
    ):
        argv = step["argv"]
        assert argv[argv.index("--pretrained-encoder") + 1] == str(checkpoint)


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
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[10], 5.0, 12)
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
    assert model.dielectric_adapter is not model.electronic_adapter
    assert float(model.born_adapter.residual_scale.detach()) == 0.0


def test_hierarchical_jet_preserves_shapes_and_bec_acoustic_sum_rule():
    graph = record_to_graph(load_gmtnet_records(gmtnet_root())[10], 5.0, 12)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    model = HierarchicalElectromechanicalJetHead(
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
    assert prediction.electronic_dielectric.shape == (1, 3, 3)
    assert torch.allclose(
        prediction.born_charges.mean(dim=0),
        torch.zeros(3, 3),
        atol=2e-6,
    )
    dielectric_eigenvalues = torch.linalg.eigvalsh(
        prediction.electronic_dielectric
    )
    assert torch.all(dielectric_eigenvalues >= 1.0 - 1e-6)
    assert model.born_adapter is not model.dielectric_adapter
    assert model.charge_screening_trunk is not model.polar_strain_trunk


def test_dielectric_audit_is_availability_masked_without_zero_fill_bias():
    target = torch.stack((torch.eye(3), 2.0 * torch.eye(3)))
    prediction = target.clone()
    prediction[1] = 0.0
    metrics = dielectric_capacity_metrics(
        prediction, target, torch.tensor([True, False])
    )
    assert metrics["available_materials"] == 1
    assert metrics["mean_stabilized_relative_frobenius_error"] == 0.0
