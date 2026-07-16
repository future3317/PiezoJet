"""Precompute point-group-projected piezoelectric supervision targets."""

from __future__ import annotations

import argparse
from pathlib import Path


from piezojet.data import load_gmtnet_records, precompute_symmetry_targets
from piezojet.project_config import load_project_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_project_config(args.config)
    records = load_gmtnet_records(config["data_root"])
    output = precompute_symmetry_targets(records, config["processed_dir"])
    print(output)


if __name__ == "__main__":
    main()
