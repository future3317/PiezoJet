from pathlib import Path

import pytest

from piezojet.project_config import apply_canonical_data_roles, load_project_config


def test_repository_config_uses_one_canonical_role_map():
    config = load_project_config(Path("config.yaml"))
    assert config["data_root"] == r"E:\DATA\PiezoJet\raw\gmtnet"
    assert config["jarvis_dfpt_dir"].endswith("jarvis_dfpt_v9_full_public")
    assert config["jarvis_strain_completion_dir"].endswith(
        "jarvis_strain_completion_v10_zero_dimensional_fix"
    )
    assert config["canonical_data_schema"] == 1
    assert config["canonical_data_manifest_path"].endswith(
        "data\\processed\\canonical_datasets.json"
    )
    assert "canonical_data_manifest" not in config
    assert apply_canonical_data_roles(config, Path("config.yaml")) == config


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
