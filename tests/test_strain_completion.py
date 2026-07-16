import torch

from piezojet.data import load_gmtnet_records
from piezojet.jarvis_dfpt import JarvisDFPTCache
from piezojet.project_config import load_project_config
from piezojet.strain_completion import (
    CartesianSymmetryOperation,
    complete_internal_strain,
    invariant_basis,
    observed_identification_metrics,
    printed_block_rounding_bootstrap,
    vector_to_internal_tensor,
)


def test_invariant_basis_enforces_acoustic_nullspace():
    atoms = 3
    identity = CartesianSymmetryOperation(
        rotation=torch.eye(3, dtype=torch.float64),
        permutation=torch.arange(atoms),
        mapping_error_angstrom=0.0,
    )
    basis = invariant_basis(atoms, [identity])
    tensors = vector_to_internal_tensor(basis.T, atoms)
    assert basis.shape == (18 * atoms, 18 * (atoms - 1))
    assert tensors.sum(dim=1).abs().max() < 1e-10


def test_real_strict_completion_is_unique_and_asr_consistent():
    config = load_project_config("config.yaml")
    records = load_gmtnet_records(config["data_root"])
    record = next(item for item in records if str(item["JARVIS_ID"]) == "JVASP-22529")
    payload = JarvisDFPTCache(config["jarvis_dfpt_dir"]).load("JVASP-22529")
    assert payload is not None
    completed, audit = complete_internal_strain(record, payload)
    assert completed is not None
    assert audit["accepted"] and audit["uniquely_determined"]
    assert audit["observed_rank"] == audit["invariant_dimensions"]
    assert completed.sum(dim=0).abs().max() < 1e-5
    assert audit["ionic_closure_relative_error"] < 0.05
    assert audit["sigma_min_mb"] is not None
    assert audit["condition_number_mb"] is not None

    quarantined = dict(payload)
    quarantined["internal_strain_parse_audit"] = {
        "complete_observed_block_parse": False,
        "malformed_blocks": [{"reason": "numeric_overflow"}],
    }
    rejected, rejected_audit = complete_internal_strain(record, quarantined)
    assert rejected is None
    assert not rejected_audit["accepted"]
    assert not rejected_audit["source_internal_strain_block_parse_complete"]


def test_identification_conditioning_and_outcar_rounding_bootstrap_are_explicit():
    basis = torch.eye(2, dtype=torch.float64)
    observed = torch.tensor([[1.0, 0.0], [0.0, 1e-3]], dtype=torch.float64)
    metrics = observed_identification_metrics(observed, rank_tolerance=1e-7)
    assert abs(float(metrics["sigma_min_mb"]) - 1e-3) < 1e-12
    assert abs(float(metrics["condition_number_mb"]) - 1e3) < 1e-9
    assert abs(float(metrics["pseudoinverse_operator_norm_mb"]) - 1e3) < 1e-9
    zero_dimensional = observed_identification_metrics(
        torch.empty(12, 0, dtype=torch.float64), rank_tolerance=1e-7
    )
    assert zero_dimensional["identification_rank"] == 0
    assert zero_dimensional["condition_number_mb"] == 1.0
    assert zero_dimensional["pseudoinverse_operator_norm_mb"] == 0.0
    sensitivity = printed_block_rounding_bootstrap(
        basis=basis,
        observed_basis=observed,
        values=torch.tensor([2.0, 3.0], dtype=torch.float64),
        halfwidths=torch.tensor([5e-7, 5e-7], dtype=torch.float64),
        samples=32,
        seed=7,
    )
    assert sensitivity["status"] == "outcar_rounding_uniform"
    assert sensitivity["lambda_relative_p95"] > 0.0
