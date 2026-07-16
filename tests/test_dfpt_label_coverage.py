import torch

from piezojet.data import load_gmtnet_records
from piezojet.dfpt_label_coverage import high_quality_partial_audit, summary
from piezojet.jarvis_dfpt import DFPT_CACHE_SCHEMA


def _payload(record):
    atoms = len(record["atoms"]["elements"])
    modes = 3 * atoms
    return {
        "schema": DFPT_CACHE_SCHEMA,
        "jid": str(record["JARVIS_ID"]),
        "provenance": {"source_archive": {"name": "unit-test.zip"}},
        "born_charges": torch.zeros(atoms, 3, 3),
        "dynamical_eigenvalues": torch.zeros(modes),
        "dynamical_eigenvectors": torch.zeros(modes, atoms, 3),
        "masses": torch.ones(atoms),
        "force_constants": torch.zeros(atoms, atoms, 3, 3),
        "dynamical_matrix": torch.zeros(atoms, atoms, 3, 3),
        "ionic_piezo_source": torch.zeros(3, 6),
        "total_piezo_source": torch.zeros(3, 6),
        "internal_strain_tensors": torch.zeros(1, 3, 3),
        "internal_strain_ions": torch.zeros(1, dtype=torch.long),
        "internal_strain_directions": torch.zeros(1, dtype=torch.long),
    }


def test_high_quality_partial_is_not_a_strict_completion():
    record = load_gmtnet_records("data/raw/gmtnet")[0]
    audit = high_quality_partial_audit(record, _payload(record))
    assert audit["partial_qualified"]
    assert audit["observed_internal_strain_blocks"] == 1
    # This audit intentionally cannot infer full Lambda acceptance.
    assert "accepted" not in audit


def test_partial_audit_rejects_nonfinite_observed_source_arrays():
    record = load_gmtnet_records("data/raw/gmtnet")[0]
    payload = _payload(record)
    payload["born_charges"][0, 0, 0] = float("nan")
    audit = high_quality_partial_audit(record, payload)
    assert not audit["partial_qualified"]
    assert "invalid_born_charges" in audit["partial_failures"]


def test_unrecoverable_lambda_block_does_not_discard_bec_phi_partial():
    record = load_gmtnet_records("data/raw/gmtnet")[0]
    payload = _payload(record)
    payload["internal_strain_tensors"] = torch.empty(0, 3, 3)
    payload["internal_strain_ions"] = torch.empty(0, dtype=torch.long)
    payload["internal_strain_directions"] = torch.empty(0, dtype=torch.long)
    payload["internal_strain_parse_audit"] = {
        "complete_observed_block_parse": False,
        "malformed_blocks": [{"reason": "numeric_overflow"}],
    }
    audit = high_quality_partial_audit(record, payload)
    assert audit["partial_qualified"]
    assert not audit["observed_internal_strain_qualified"]
    assert not audit["complete_observed_block_parse"]
    assert audit["malformed_internal_strain_blocks"] == 1


def test_selection_summary_reports_denominators_and_rates():
    rows = [
        {"raw_dfpt_available": True, "partial_qualified": True, "strict_complete": True,
         "observed_internal_strain_qualified": True, "crystal_system": "cubic", "atom_count_bin": "1-2", "gmtnet_response_bin": "1"},
        {"raw_dfpt_available": True, "partial_qualified": True, "strict_complete": False,
         "observed_internal_strain_qualified": False, "crystal_system": "cubic", "atom_count_bin": "1-2", "gmtnet_response_bin": "1"},
        {"raw_dfpt_available": False, "partial_qualified": False, "strict_complete": False,
         "observed_internal_strain_qualified": False, "crystal_system": "hexagonal", "atom_count_bin": "5-8", "gmtnet_response_bin": "3"},
    ]
    report = summary(rows)
    cubic = report["selection_bias"]["crystal_system"]["strict_complete"]["cubic"]
    assert cubic == {"denominator": 2, "numerator": 1, "rate": 0.5}
