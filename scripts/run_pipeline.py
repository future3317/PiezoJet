#!/usr/bin/env python
"""Run PiezoJet's required structural pretraining followed by response fine-tuning."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--loss", choices=("full", "sketch", "hybrid"), default="full")
    args = parser.parse_args()
    pretrain = [sys.executable, "-m", "piezojet.pretrain", "--config", str(args.config)]
    if args.pretrain_epochs is not None:
        pretrain.extend(["--epochs", str(args.pretrain_epochs)])
    subprocess.run(pretrain, check=True)
    finetune = [sys.executable, "-m", "piezojet.train", "--config", str(args.config), "--loss", args.loss]
    if args.finetune_epochs is not None:
        finetune.extend(["--epochs", str(args.finetune_epochs)])
    subprocess.run(finetune, check=True)


if __name__ == "__main__":
    main()
