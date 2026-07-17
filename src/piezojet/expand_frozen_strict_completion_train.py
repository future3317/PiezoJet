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
    held_out_ids = set(original["val"] + original["test"])
    accepted_train_candidates = accepted_ids - held_out_ids
    unsafe = sorted(
        jid for jid in accepted_train_candidates
        if formula_by_id[jid] in frozen_formulas
    )
    safe_original_train = [
        jid for jid in original["train"]
        if jid in accepted_ids and formula_by_id[jid] not in frozen_formulas
    ]
    candidates = sorted(accepted_ids - set(all_original))
    additions = [
        jid for jid in candidates if formula_by_id[jid] not in frozen_formulas
    ]
    expanded = {
        "train": sorted(safe_original_train + additions),
        "val": original["val"],
        "test": original["test"],
    }
    for left, right in (("train", "val"), ("train", "test")):
        shared_formulae = {formula_by_id[jid] for jid in expanded[left]} & {formula_by_id[jid] for jid in expanded[right]}
        if shared_formulae:
            raise ValueError(f"Expanded split has formula leakage between {left}/{right}: {sorted(shared_formulae)[:5]}")
    held_out_formula_overlap = sorted(
        {formula_by_id[jid] for jid in expanded["val"]}
        & {formula_by_id[jid] for jid in expanded["test"]}
    )
    return {
        "schema": 3,
        "frozen": True,
        "policy": (
            "original validation/test material IDs and formulas are immutable; newly strict-complete, "
            "formula-safe materials may be appended only to train"
        ),
        "base_benchmark": base.get("source_completion_manifest", "unknown"),
        "source_completion_manifest": source_completion_manifest,
        "added_train_ids": additions,
        "excluded_frozen_formula_ids": unsafe,
        "removed_base_train_ids": sorted(set(original["train"]) - set(safe_original_train)),
        "validation_test_reduced_formula_overlap": held_out_formula_overlap,
        "splits": expanded,
        "summary": {
            "base_train_materials": len(original["train"]),
            "removed_base_train_materials": len(original["train"]) - len(safe_original_train),
            "added_train_materials": len(additions),
            "excluded_frozen_formula_materials": len(unsafe),
            "train_materials": len(expanded["train"]),
            "validation_materials_unchanged": len(expanded["val"]),
            "test_materials_unchanged": len(expanded["test"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-benchmark", type=Path, required=True)
    parser.add_argument("--completion-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite expanded frozen benchmark: {args.output}")
    base = json.loads(args.base_benchmark.read_text(encoding="utf-8"))
    completion = json.loads(args.completion_manifest.read_text(encoding="utf-8"))
    # Early completion manifests stored a compact ``material_ids`` list;
    # current strict audits retain one provenance row per requested material.
    # Support both representations, always filtering the latter by its strict
    # acceptance gate rather than treating every audited archive as a label.
    if isinstance(completion.get("material_ids"), list):
        accepted = {str(value) for value in completion["material_ids"]}
    elif isinstance(completion.get("rows"), list):
        accepted = {
            str(row["jid"])
            for row in completion["rows"]
            if isinstance(row, dict) and bool(row.get("accepted", False)) and row.get("jid")
        }
    else:
        raise ValueError("Completion manifest must contain material_ids or per-material strict-audit rows")
    if not accepted:
        raise ValueError("Completion manifest contains no accepted strict-completion labels")
    records = load_gmtnet_records(args.data_root)
    formula_by_id = {str(record["JARVIS_ID"]): formula(record) for record in records}
    payload = expand_frozen_train_panel(base, accepted, formula_by_id, str(args.completion_manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
