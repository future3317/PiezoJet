"""Evaluate selected protocol checkpoints in one process under one frozen panel.

This utility only replays already selected checkpoints.  It does not train or
compare test values while a protocol is being selected.  A single Python
process avoids Windows shared-library start races during a many-checkpoint
audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate_dfpt import main as evaluate_dfpt_main


def _tag(summary_path: Path) -> str:
    protocol = summary_path.parent.name.removeprefix("protocol_")
    seed = summary_path.parent.parent.name.removeprefix("seed")
    if not protocol or not seed.isdigit():
        raise ValueError(f"Expected .../seedNN/protocol_X/summary.json, got {summary_path}")
    return f"{protocol}_seed{seed}"


def _evaluate(checkpoint: str, splits_file: Path, output: Path, device: str) -> None:
    previous = sys.argv
    sys.argv = [
        "piezojet.evaluate_dfpt", "--checkpoint", checkpoint,
        "--splits-file", str(splits_file), "--split", "test",
        "--device", device, "--output", str(output),
    ]
    try:
        evaluate_dfpt_main()
    finally:
        sys.argv = previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True, help="Protocol result root; repeatable")
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.splits_file.is_file():
        raise FileNotFoundError(args.splits_file)
    summaries = [path for root in args.root for path in root.glob("seed*/protocol_*/summary.json")]
    if not summaries:
        raise ValueError("No protocol summaries found")
    tagged = sorted((_tag(path), path) for path in summaries)
    if len({tag for tag, _ in tagged}) != len(tagged):
        raise ValueError("Duplicate protocol/seed checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for tag, summary_path in tagged:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        output = args.output_dir / f"{tag}.json"
        _evaluate(str(payload["selected_checkpoint"]), args.splits_file, output, args.device)
        manifest.append({"tag": tag, "summary": str(summary_path), "checkpoint": str(payload["selected_checkpoint"]), "output": str(output)})
    (args.output_dir / "manifest.json").write_text(json.dumps({"schema": 1, "scope": "frozen-panel post-selection replay", "runs": manifest}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
