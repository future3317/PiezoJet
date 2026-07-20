"""Load one project configuration and one canonical data-role manifest."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
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

_DATA_ROOT_ENV = "PIEZOJET_DATA_ROOT"


def _manifest_path(value: str | Path, config_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    project_relative = Path.cwd() / candidate
    if project_relative.is_file():
        return project_relative
    return config_path.parent / candidate


def _rebase_physical_roles(
    roles: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[dict[str, Any], str | None]:
    """Rebase physical data roles without changing repository-local roles."""

    override = os.environ.get(_DATA_ROOT_ENV)
    if override is None:
        return dict(roles), None
    override_path = Path(override).expanduser()
    if not override_path.is_absolute():
        raise ValueError(f"{_DATA_ROOT_ENV} must be an absolute path: {override}")
    source_root_value = payload.get("physical_data_root")
    if not isinstance(source_root_value, str):
        raise ValueError(
            "Canonical data manifest must declare physical_data_root before "
            f"{_DATA_ROOT_ENV} can be used"
        )
    path_type = PureWindowsPath if PureWindowsPath(source_root_value).drive else PurePosixPath
    source_root = path_type(source_root_value)
    rebased = dict(roles)
    for role, value in roles.items():
        if not isinstance(value, str):
            continue
        candidate = path_type(value)
        try:
            relative = candidate.relative_to(source_root)
        except ValueError:
            # Repository-local split/manifests remain relative to the checkout.
            continue
        rebased[role] = str(override_path.joinpath(*relative.parts))
    return rebased, str(override_path)


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
    roles, data_root_override = _rebase_physical_roles(roles, payload)
    for role, key in _ROLE_TO_CONFIG.items():
        resolved[key] = str(roles[role])
    # Resolved run configs contain explicit, auditable paths and must remain
    # reloadable without triggering a second manifest expansion.
    del resolved["canonical_data_manifest"]
    resolved["canonical_data_manifest_path"] = str(path)
    resolved["canonical_data_schema"] = int(payload["schema"])
    resolved["canonical_data_coverage"] = payload.get("coverage", {})
    if data_root_override is not None:
        resolved["canonical_data_root_override"] = data_root_override
    return resolved


def load_project_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Project config must contain a mapping: {config_path}")
    return apply_canonical_data_roles(payload, config_path)
