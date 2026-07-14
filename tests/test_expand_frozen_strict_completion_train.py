import pytest

from piezojet.expand_frozen_strict_completion_train import expand_frozen_train_panel


def test_expand_preserves_frozen_panels_and_adds_only_new_train_ids():
    base = {"splits": {"train": ["a"], "val": ["b"], "test": ["c"]}}
    result = expand_frozen_train_panel(
        base, {"a", "d", "e"}, {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}, "new.json"
    )
    assert result["splits"] == {"train": ["a", "d", "e"], "val": ["b"], "test": ["c"]}
    assert result["added_train_ids"] == ["d", "e"]


def test_expand_rejects_new_formula_that_leaks_into_frozen_panel():
    base = {"splits": {"train": ["a"], "val": ["b"], "test": ["c"]}}
    with pytest.raises(ValueError, match="overlap frozen"):
        expand_frozen_train_panel(
            base, {"a", "d"}, {"a": "A", "b": "B", "c": "C", "d": "B"}, "new.json"
        )
