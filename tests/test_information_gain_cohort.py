from collections import Counter

import pytest

from piezojet.rank_information_gain_cohort import (
    _gap,
    _select_test_crystal_coverage,
    _smoothed_rate,
)


def test_smoothed_completion_rate_shrinks_sparse_group_to_audited_prior():
    # A single accepted record is evidence, not a 100% completion guarantee.
    assert _smoothed_rate(1, 1, prior_rate=0.5, prior_strength=8.0) == pytest.approx(5.0 / 9.0)


def test_test_stratum_gap_prioritizes_test_present_train_missing_strata_only():
    train = Counter({"cubic": 9, "hexagonal": 1})
    test = Counter({"cubic": 2, "triclinic": 2})
    assert _gap(train, test, "triclinic") == pytest.approx(1.0)
    assert _gap(train, test, "cubic") == pytest.approx(0.0)
    assert _gap(train, test, "monoclinic") == pytest.approx(0.0)


def test_test_crystal_coverage_reserves_queue_slots_without_formula_duplicates():
    profiles = [
        {"jid": "c1", "formula": "A", "crystal_system": "cubic"},
        {"jid": "c2", "formula": "B", "crystal_system": "cubic"},
        {"jid": "h1", "formula": "C", "crystal_system": "hexagonal"},
        {"jid": "h2", "formula": "D", "crystal_system": "hexagonal"},
        # This is a higher-ranked polymorph of A and must not be duplicated.
        {"jid": "h3", "formula": "A", "crystal_system": "hexagonal"},
    ]
    selected, formulas, quotas = _select_test_crystal_coverage(
        profiles, queue_size=4, test_crystal=Counter({"cubic": 1, "hexagonal": 1}),
    )
    assert quotas == {"cubic": 2, "hexagonal": 2}
    assert [row["jid"] for row in selected] == ["c1", "c2", "h1", "h2"]
    assert formulas == {"A", "B", "C", "D"}
