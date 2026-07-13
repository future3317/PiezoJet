"""Precompute point-group-projected piezoelectric supervision targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from piezojet.data import load_gmtnet_records, precompute_symmetry_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records = load_gmtnet_records(config["data_root"])
    output = precompute_symmetry_targets(records, config["processed_dir"])
    print(output)


if __name__ == "__main__":
    main()
