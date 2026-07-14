"""Expand only the training side of an already frozen strict-completion panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import formula, load_gmtnet_records


def expand_frozen_train_panel(
    base: dict[str, Any],
    accepted_ids: set[str],
    formula_by_id: dict[str, str],
    source_completion_manifest: str,
) -> dict[str, Any]:
    """Add only new, formula-safe strict completions to the original train IDs."""
    splits = base.get("splits")
    if not isinstance(splits, dict) or any(name not in splits for name in ("train", "val", "test")):
        raise ValueError("Frozen benchmark must contain train/val/test splits")
    original = {name: [str(value) for value in splits[name]] for name in ("train", "val", "test")}
    all_original = original["train"] + original["val"] + original["test"]
    if len(all_original) != len(set(all_original)):
        raise ValueError("Frozen benchmark leaks material IDs across panels")
    missing = sorted((set(all_original) | accepted_ids) - set(formula_by_id))
    if missing:
        raise ValueError(f"Formula lookup missing IDs: {missing[:5]}")
    frozen_formulas = {formula_by_id[jid] for jid in original["val"] + original["test"]}
    additions = sorted(accepted_ids - set(all_original))
    unsafe = [jid for jid in additions if formula_by_id[jid] in frozen_formulas]
    if unsafe:
        raise ValueError(f"New strict completions overlap frozen validation/test formulas: {unsafe[:5]}")
    expanded = {
        "train": sorted(original["train"] + additions),
        "val": original["val"],
        "test": original["test"],
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared_formulae = {formula_by_id[jid] for jid in expanded[left]} & {formula_by_id[jid] for jid in expanded[right]}
        if shared_formulae:
            raise ValueError(f"Expanded split has formula leakage between {left}/{right}: {sorted(shared_formulae)[:5]}")
    return {
        "schema": 2,
        "frozen": True,
        "policy": (
            "original validation/test material IDs and formulas are immutable; newly strict-complete, "
            "formula-safe materials may be appended only to train"
        ),
        "base_benchmark": base.get("source_completion_manifest", "unknown"),
        "source_completion_manifest": source_completion_manifest,
        "added_train_ids": additions,
        "splits": expanded,
        "summary": {
            "base_train_materials": len(original["train"]),
            "added_train_materials": len(additions),
            "train_materials": len(expanded["train"]),
            "validation_materials_unchanged": len(expanded["val"]),
            "test_materials_unchanged": len(expanded["test"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--base-benchmark", type=Path, required=True)
    parser.add_argument("--completion-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite expanded frozen benchmark: {args.output}")
    base = json.loads(args.base_benchmark.read_text(encoding="utf-8"))
    completion = json.loads(args.completion_manifest.read_text(encoding="utf-8"))
    accepted = {str(value) for value in completion["material_ids"]}
    records = load_gmtnet_records(args.data_root)
    formula_by_id = {str(record["JARVIS_ID"]): formula(record) for record in records}
    payload = expand_frozen_train_panel(base, accepted, formula_by_id, str(args.completion_manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
