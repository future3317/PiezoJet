from piezojet.audit_gmtnet_outcar_total import total_labels_are_consistent


def test_total_consistency_gate_requires_both_absolute_and_relative_excess():
    assert total_labels_are_consistent(0.051, 0.049)
    assert total_labels_are_consistent(0.049, 0.051)
    assert not total_labels_are_consistent(0.051, 0.051)
