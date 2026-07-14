from pathlib import Path

from piezojet.reevaluate_protocol_checkpoints import _tag


def test_protocol_checkpoint_tag_is_derived_from_persisted_path():
    path = Path("outputs/feedback5/seed42/protocol_E/summary.json")
    assert _tag(path) == "E_seed42"
