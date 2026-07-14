from piezojet.data import load_gmtnet_records
from piezojet.hexagonal_completion_diagnostics import (
    fractional_cartesian_checks,
    synthetic_recovery,
)
from piezojet.jarvis_dfpt import JarvisDFPTCache


def test_no187_synthetic_recovery_and_skew_cell_round_trip():
    """A real P-6m2 structure is a regression test for the skew-cell branch."""
    records = load_gmtnet_records("data/raw/gmtnet")
    record = next(item for item in records if str(item["JARVIS_ID"]) == "JVASP-4696")
    payload = JarvisDFPTCache("data/processed/jarvis_dfpt_v1").load("JVASP-4696")
    assert payload is not None
    recovery = synthetic_recovery(record, payload)
    checks = fractional_cartesian_checks(record)
    assert recovery["observed_rank"] == recovery["invariant_dimensions"]
    assert recovery["cosine"] > 1.0 - 1e-12
    assert recovery["relative_recovery_error"] < 1e-12
    assert checks["affine_group_closure_failures"] == 0
    assert checks["maximum_fractional_to_cartesian_rotation_error"] < 1e-12
    assert checks["maximum_vector_round_trip_error"] < 1e-12
