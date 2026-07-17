import csv
import json

import pytest

from piezojet.summarize_global_l3_validation import summarize


def _physical_run(path, first_loss: float, second_loss: float, second_trs: float):
    path.mkdir(parents=True)
    fieldnames = [
        "epoch", "val_loss", "val_tensor_response_skill_vs_zero_loss",
        "val_displacement_response_loss", "val_ionic_piezo_loss",
        "val_displacement_first_order_consistency_loss",
        "val_electronic_piezo_loss", "val_branch_sum_loss",
    ]
    with (path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict(zip(fieldnames, [1, first_loss, -1, 1, 1, 1, 1, 1])))
        writer.writerow(dict(zip(fieldnames, [2, second_loss, second_trs, .2, .3, .4, .5, .6])))
    (path / "summary.json").write_text(json.dumps({"loss_best_epoch": 2}), encoding="utf-8")


def _direct_run(path, trs: float):
    path.mkdir(parents=True)
    with (path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "val_loss", "val_tensor_response_skill_vs_zero"]
        )
        writer.writeheader()
        writer.writerow({"epoch": 1, "val_loss": .8, "val_tensor_response_skill_vs_zero": trs})


def test_validation_summary_selects_loss_and_pairs_direct_control(tmp_path):
    replication = tmp_path / "replication"
    seed42 = tmp_path / "seed42"
    direct = tmp_path / "direct"
    _physical_run(seed42, 2.0, 1.0, .3)
    for seed, trs in ((7, .2), (1729, .1)):
        _physical_run(replication / f"factorized_seed{seed}", 2.0, 1.0, trs)
    for seed, trs in ((42, .1), (7, .05), (1729, 0.0)):
        _direct_run(direct / f"direct_seed{seed}", trs)

    result = summarize(replication, seed42, [42, 7, 1729], direct_root=direct)

    assert result["physical_individual"]["42"]["selected_epoch"] == 2
    assert result["physical_mean_sample_sd"]["total_trs"]["mean"] == .2
    assert result["paired_physical_minus_direct_total_trs"]["individual"]["42"] == pytest.approx(.2)
    assert result["validation_gate"]["positive_mean_total_trs"] is True
