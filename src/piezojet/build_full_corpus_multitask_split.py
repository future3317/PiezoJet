"""Build the formula-disjoint full-GMTNet / strict-DFPT multitask split.

The train partition contains every GMTNet record whose formula is disjoint
from the immutable strict-completion validation and test formulas.  The latter
remain the only validation/test IDs.  Consequently, all 4,998 GMTNet labels
may be used where safe, while DFPT factor losses remain masked automatically
to records that carry JARVIS labels and strict Lambda completions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import formula, load_gmtnet_records
from .train import load_explicit_splits


def build_full_corpus_multitask_split(
    records: list[dict], strict_split_file: str | Path,
) -> dict:
    """Return a leakage-checked full-corpus train / frozen-panel val-test split."""
    record_by_id = {str(record["JARVIS_ID"]): record for record in records}
    strict = load_explicit_splits(Path(strict_split_file), set(record_by_id))
    held_out_ids = strict["val"] + strict["test"]
    held_out_formulas = {formula(record_by_id[material_id]) for material_id in held_out_ids}
    train = [
        str(record["JARVIS_ID"])
        for record in records
        if formula(record) not in held_out_formulas
    ]
    if set(train) & set(held_out_ids):
        raise RuntimeError("Full-corpus train split leaks a held-out material ID")
    train_formulas = {formula(record_by_id[material_id]) for material_id in train}
    if train_formulas & held_out_formulas:
        raise RuntimeError("Full-corpus train split leaks held-out formulas")
    if not set(strict["train"]).issubset(train):
        raise RuntimeError("A frozen strict training ID was unexpectedly removed")
    return {
        "schema": 1,
        "frozen": True,
        "policy": (
            "Train on all GMTNet records formula-disjoint from frozen strict val/test; "
            "DFPT/factor losses remain availability-masked and JARVIS-only."
        ),
        "source_strict_split": str(strict_split_file),
        "held_out_formula_count": len(held_out_formulas),
        "splits": {"train": train, "val": strict["val"], "test": strict["test"]},
        "summary": {
            "all_gmtnet_records": len(records),
            "train_gmtnet_records": len(train),
            "strict_train_factor_records": len(strict["train"]),
            "frozen_validation_records": len(strict["val"]),
            "frozen_test_records": len(strict["test"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--strict-split-file", required=True
    )
    parser.add_argument(
        "--output", required=True
    )
    args = parser.parse_args()
    payload = build_full_corpus_multitask_split(
        load_gmtnet_records(args.data_root), args.strict_split_file
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
