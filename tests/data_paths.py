"""Canonical external data paths used by data-backed tests."""

from functools import lru_cache

from piezojet.project_config import load_project_config


@lru_cache(maxsize=1)
def gmtnet_root() -> str:
    """Return the configured GMTNet root without assuming repository data."""

    return str(load_project_config("config.yaml")["data_root"])
