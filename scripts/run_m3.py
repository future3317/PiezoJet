#!/usr/bin/env python
"""Launch the fixed-split three-seed M3 experiments; no test-set tuning."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--loss", choices=("full", "sketch", "hybrid"), default="full")
    args = parser.parse_args()
    root = Path(args.config).resolve().parent
    for seed in (42, 43, 44):
        output = root / "outputs" / "m3" / f"seed_{seed}"
        command = [sys.executable, "-m", "piezojet.train", "--config", str(args.config), "--loss", args.loss, "--seed", str(seed), "--output-dir", str(output)]
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        subprocess.run(command, check=True)
    print("M3 seeds 42, 43, 44 completed")


if __name__ == "__main__":
    main()
