"""Materialize versioned PBC graphs once for all later train/eval/inference runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .data import load_gmtnet_records, precompute_pbc_graphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records = load_gmtnet_records(config["data_root"])
    directory = precompute_pbc_graphs(records, config["processed_dir"], config["cutoff"], config["max_neighbors"])
    print(directory)


if __name__ == "__main__":
    main()
