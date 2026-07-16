import json
from pathlib import Path

from piezojet.summarize_teacher_forced_probe import summarize


def _payload(value: float) -> dict:
    return {
        "formula_disjoint": False,
        "material_count": 1,
        "interpretation_boundary": "same ID",
        "factors": {
            "born_charge": {"macro_material_directional_cosine": value},
            "force_constant": {"macro_material_directional_cosine": value},
            "internal_strain_full": {"macro_material_directional_cosine": value},
        },
        "strict_symmetry_completed_lambda_oracle": {
            "displacement_response_target": {"macro_material_directional_cosine": value}
        },
        "ionic_response_aggregation": {"ionic_cosine_macro_material": value},
    }


def test_capacity_summary_requires_all_factors_and_preserves_same_id_boundary(tmp_path: Path) -> None:
    for label, value in (("samples1", 0.995), ("samples8", 0.991), ("samples32", 0.992)):
        directory = tmp_path / label
        directory.mkdir()
        payload = _payload(value)
        payload["material_count"] = int(label.removeprefix("samples"))
        (directory / "overfit_dfpt_train.json").write_text(json.dumps(payload), encoding="utf-8")

    report = summarize(tmp_path)

    assert report["capacity_probe_passes"]
    assert report["probes"]["samples1"]["material_count"] == 1
    assert "does not establish" in report["interpretation"]


def test_capacity_summary_rejects_formula_disjoint_mislabel(tmp_path: Path) -> None:
    for label in ("samples1", "samples8", "samples32"):
        directory = tmp_path / label
        directory.mkdir()
        payload = _payload(1.0)
        payload["material_count"] = int(label.removeprefix("samples"))
        payload["formula_disjoint"] = True
        (directory / "overfit_dfpt_train.json").write_text(json.dumps(payload), encoding="utf-8")

    try:
        summarize(tmp_path)
    except ValueError as error:
        assert "same-ID" in str(error)
    else:
        raise AssertionError("formula-disjoint mislabel was accepted")
