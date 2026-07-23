from piezojet.checkpoint_runtime import atomic_link_or_copy


def test_atomic_link_or_copy_replaces_destination(tmp_path):
    source = tmp_path / "last.pt"
    destination = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint-v1")
    atomic_link_or_copy(source, destination)
    assert destination.read_bytes() == b"checkpoint-v1"
    source.write_bytes(b"checkpoint-v2")
    atomic_link_or_copy(source, destination)
    assert destination.read_bytes() == b"checkpoint-v2"
