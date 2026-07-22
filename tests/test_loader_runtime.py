import pytest

from piezojet.loader_runtime import loader_options


def test_single_process_loader_omits_multiprocessing_only_options():
    assert loader_options(0, cuda=True) == {
        "num_workers": 0,
        "pin_memory": True,
    }


def test_worker_loader_enables_persistent_prefetch_and_pinning():
    assert loader_options(4, cuda=True) == {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


def test_one_shot_worker_loader_can_disable_persistence():
    assert loader_options(2, cuda=False, persistent=False) == {
        "num_workers": 2,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }


@pytest.mark.parametrize("workers,prefetch", [(-1, 2), (0, 0)])
def test_invalid_loader_options_are_rejected(workers, prefetch):
    with pytest.raises(ValueError):
        loader_options(workers, cuda=False, prefetch_factor=prefetch)
