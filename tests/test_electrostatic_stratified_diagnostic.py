from __future__ import annotations

import json

import pytest
import torch

from piezojet.electrostatic_stratified_diagnostic import build_report


def _checkpoint(tmp_path):
    ids = ["JVASP-1", "JVASP-2"]
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "checkpoint_provenance": {"frozen_validation_test_labels_read": False},
            "training_contract": {"development_ids": ids},
            "development_metrics": {
                "electronic": {
                    "per_material": [
                        {
                            "target_norm_c_per_m2": 0.0,
                            "prediction_norm_c_per_m2": 0.01,
                            "active": False,
                            "stabilized_relative_frobenius_error": 0.01,
                            "cosine": 0.0,
                            "stabilized_amplitude_ratio": 0.0,
                        },
                        {
                            "target_norm_c_per_m2": 1.0,
                            "prediction_norm_c_per_m2": 0.5,
                            "active": True,
                            "stabilized_relative_frobenius_error": 0.5,
                            "cosine": 0.8,
                            "stabilized_amplitude_ratio": 0.5,
                        },
                    ],
                    "per_irrep": {},
                },
                "born": {
                    "per_material": [
                        {
                            "atoms": 2,
                            "target_norm_e": 1.0,
                            "prediction_norm_e": 1.0,
                            "stabilized_relative_frobenius_error": 0.0,
                            "cosine": 1.0,
                        },
                        {
                            "atoms": 10,
                            "target_norm_e": 2.0,
                            "prediction_norm_e": 1.0,
                            "stabilized_relative_frobenius_error": 0.5,
                            "cosine": 0.5,
                        },
                    ]
                },
                "dielectric": {
                    "per_material": [
                        {"target_norm": 1.0, "stabilized_relative_frobenius_error": 0.1},
                        {"target_norm": 2.0, "stabilized_relative_frobenius_error": 0.2},
                    ]
                },
            },
        },
        checkpoint,
    )
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({"folds": [{"development": ids}]}), encoding="utf-8")
    return checkpoint, folds


def test_stratified_report_keeps_zero_and_active_panels_separate(tmp_path):
    checkpoint, folds = _checkpoint(tmp_path)
    report = build_report(checkpoint, folds, 0)
    electronic = report["overall"]["electronic"]
    assert electronic["materials"] == 2
    assert electronic["active_materials"] == 1
    assert electronic["active_mean_cosine"] == pytest.approx(0.8)
    assert report["strata"]["electronic_target_norm"]["zero"]["materials"] == 1


def test_stratified_report_rejects_frozen_checkpoint(tmp_path):
    checkpoint, folds = _checkpoint(tmp_path)
    payload = torch.load(checkpoint, weights_only=False)
    payload["checkpoint_provenance"]["frozen_validation_test_labels_read"] = True
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="frozen labels"):
        build_report(checkpoint, folds, 0)
