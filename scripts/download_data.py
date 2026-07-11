#!/usr/bin/env python
"""Download the single approved GMTNet source repository."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SOURCE_URL = "https://github.com/YKQ98/GMTNet.git"


def write_blocked(output: Path, command: list[str], error: str) -> None:
    report = Path("DOWNLOAD_BLOCKED.md")
    report.write_text(
        "# Implementation blocked\n\n"
        "## Step\nDownload GMTNet data\n\n"
        "## Command\n```text\n" + " ".join(command) + "\n```\n\n"
        f"## Source\n{SOURCE_URL}\n\n"
        f"## Error\n```text\n{error}\n```\n\n"
        f"## Expected local path\n{output / 'data'}\n\n"
        "## What I checked\nThe official GMTNet repository was cloned once with depth 1 and its `data` directory was required.\n\n"
        "## What is needed from the user\nManually download the official GMTNet repository and place its `data` directory at the expected local path.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    command = ["git", "clone", "--depth", "1", SOURCE_URL, str(output)]

    if output.exists():
        if (output / "data").is_dir():
            revision = subprocess.run(
                ["git", "-C", str(output), "rev-parse", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            (output / "SOURCE_COMMIT.txt").write_text(revision + "\n", encoding="utf-8")
            print(f"GMTNet already present at {output}; commit {revision}")
            return
        error = f"Output path already exists but does not contain data/: {output}"
        write_blocked(output, command, error)
        raise RuntimeError(error)

    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        error = (result.stdout + result.stderr).strip()
        write_blocked(output, command, error)
        raise RuntimeError(error)
    if not (output / "data").is_dir():
        error = f"Clone completed but expected directory is absent: {output / 'data'}"
        write_blocked(output, command, error)
        raise RuntimeError(error)

    revision = subprocess.run(
        ["git", "-C", str(output), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    (output / "SOURCE_COMMIT.txt").write_text(revision + "\n", encoding="utf-8")
    print(f"Downloaded GMTNet to {output}; commit {revision}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
