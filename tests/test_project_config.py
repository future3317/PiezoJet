from pathlib import Path

import pytest

from piezojet.project_config import apply_canonical_data_roles, load_project_config


def test_repository_config_uses_one_canonical_role_map(monkeypatch):
    monkeypatch.delenv("PIEZOJET_DATA_ROOT", raising=False)
    config = load_project_config(Path("config.yaml"))
    assert config["data_root"] == r"E:\DATA\PiezoJet\raw\gmtnet"
    assert config["processed_dir"] == r"E:\DATA\PiezoJet\processed"
    assert config["jarvis_dfpt_dir"].endswith("jarvis_dfpt_v9_full_public")
    assert config["jarvis_strain_completion_dir"].endswith(
        "jarvis_strain_completion_v10_zero_dimensional_fix"
    )
    assert config["canonical_data_schema"] == 1
    assert Path(config["canonical_data_manifest_path"]).name == "canonical_datasets.json"
    assert "canonical_data_manifest" not in config
    assert apply_canonical_data_roles(config, Path("config.yaml")) == config


def test_physical_data_root_can_be_rebased_outside_the_repository(
    monkeypatch, tmp_path
):
    external_root = (tmp_path / "PiezoJet-data").resolve()
    external_root.mkdir()
    monkeypatch.setenv("PIEZOJET_DATA_ROOT", str(external_root))
    config = load_project_config(Path("config.yaml"))
    assert Path(config["data_root"]) == external_root / "raw" / "gmtnet"
    assert Path(config["processed_dir"]) == external_root / "processed"
    assert Path(config["jarvis_dfpt_dir"]) == (
        external_root / "processed" / "jarvis_dfpt_v9_full_public"
    )
    assert Path(config["elastic_targets_path"]) == (
        external_root
        / "processed"
        / "jarvis_elastic_auxiliary_v2_reynolds"
        / "accepted_targets_gpa.pt"
    )
    assert config["strict_completion_split_file"].startswith("data/processed/")
    assert config["canonical_data_root_override"] == str(external_root)


def test_manifest_and_versioned_path_cannot_compete():
    manifest = Path("data/processed/canonical_datasets.json").resolve()
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_canonical_data_roles(
            {
                "canonical_data_manifest": str(manifest),
                "jarvis_dfpt_dir": "stale-versioned-cache",
            },
            Path("config.yaml"),
        )
