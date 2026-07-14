import torch

from piezojet.data import load_gmtnet_records
from piezojet.jarvis_dfpt import JarvisDFPTCache
from piezojet.strain_completion import (
    CartesianSymmetryOperation,
    complete_internal_strain,
    invariant_basis,
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
    records = load_gmtnet_records("data/raw/gmtnet")
    record = next(item for item in records if str(item["JARVIS_ID"]) == "JVASP-22529")
    payload = JarvisDFPTCache("data/processed/jarvis_dfpt_v1").load("JVASP-22529")
    assert payload is not None
    completed, audit = complete_internal_strain(record, payload)
    assert completed is not None
    assert audit["accepted"] and audit["uniquely_determined"]
    assert audit["observed_rank"] == audit["invariant_dimensions"]
    assert completed.sum(dim=0).abs().max() < 1e-5
    assert audit["ionic_closure_relative_error"] < 0.05
