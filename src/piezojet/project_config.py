"""Load one project configuration and one canonical data-role manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


_ROLE_TO_CONFIG = {
    "gmtnet_source": "data_root",
    "jarvis_dfpt": "jarvis_dfpt_dir",
    "strict_internal_strain": "jarvis_strain_completion_dir",
    "strict_split": "strict_completion_split_file",
    "multitask_split": "pretrain_splits_file",
    "elastic_auxiliary": "elastic_targets_path",
}


def _manifest_path(value: str | Path, config_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    project_relative = Path.cwd() / candidate
    if project_relative.is_file():
        return project_relative
    return config_path.parent / candidate


def apply_canonical_data_roles(
    config: Mapping[str, Any], config_path: str | Path
) -> dict[str, Any]:
    """Resolve canonical dataset roles exactly once without directory scans."""

    resolved = dict(config)
    manifest_value = resolved.get("canonical_data_manifest")
    if manifest_value is None:
        return resolved
    path = _manifest_path(manifest_value, Path(config_path).resolve())
    if not path.is_file():
        raise FileNotFoundError(f"Canonical data manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    roles = payload.get("roles")
    if payload.get("schema") != 1 or not isinstance(roles, dict):
        raise ValueError(f"Unsupported canonical data manifest: {path}")
    conflicts = [key for key in _ROLE_TO_CONFIG.values() if key in resolved]
    if conflicts:
        raise ValueError(
            "Canonical data roles and explicit dataset paths are mutually "
            f"exclusive; remove: {', '.join(sorted(conflicts))}"
        )
    missing = [role for role in _ROLE_TO_CONFIG if role not in roles]
    if missing:
        raise ValueError(
            f"Canonical data manifest lacks required roles: {', '.join(missing)}"
        )
    for role, key in _ROLE_TO_CONFIG.items():
        resolved[key] = str(roles[role])
    # Resolved run configs contain explicit, auditable paths and must remain
    # reloadable without triggering a second manifest expansion.
    del resolved["canonical_data_manifest"]
    resolved["canonical_data_manifest_path"] = str(path)
    resolved["canonical_data_schema"] = int(payload["schema"])
    resolved["canonical_data_coverage"] = payload.get("coverage", {})
    return resolved


def load_project_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Project config must contain a mapping: {config_path}")
    return apply_canonical_data_roles(payload, config_path)
