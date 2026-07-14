from collections import Counter

from piezojet.resample_stratified_subsets import apportion, choose_subsets


def test_apportion_conserves_requested_subset_size():
    quotas = apportion(35, Counter({"0": 15, "1": 13, "2": 16, "3": 14, "4": 11}))
    assert sum(quotas.values()) == 35
    assert quotas == {"0": 8, "1": 7, "2": 8, "3": 7, "4": 5}


def test_choose_subsets_keeps_exact_response_quotas_and_returns_distinct_panels():
    features = {
        f"m{index}": {
            "response_bin": str(index % 2),
            "crystal_system": "cubic" if index % 3 else "hexagonal",
            "polar": str(index % 4 == 0),
            "negative_optical_mode": str(index % 5 == 0),
            "atom_count_bin": str(index % 3),
            "elements": ("A", f"E{index % 4}"),
        }
        for index in range(12)
    }
    subsets, metadata = choose_subsets(list(features), features, size=6, count=3, seed=7, candidates=200)
    assert len({tuple(subset) for subset in subsets}) == 3
    assert metadata["response_bin_quotas"] == {"0": 3, "1": 3}
    for subset in subsets:
        assert Counter(features[item]["response_bin"] for item in subset) == Counter({"0": 3, "1": 3})
