"""Read-only frozen-validation evaluator for an A0 electrostatic checkpoint.

The evaluator accepts an explicit validation-ID file and refuses any ID that
belongs to the frozen test panel in the canonical split.  It never selects or
modifies a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .project_config import load_project_config
from .data import graph_cache_key, load_gmtnet_records
from .electrostatic_a0_fold_adjudication import (
    _dataset,
    _evaluate_task,
    _tower,
    make_model,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ids(path: Path, key: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload.get("splits"), dict):
        payload = payload["splits"]
    if key == "validation" and "validation" not in payload and "val" in payload:
        key = "val"
    if key not in payload or not isinstance(payload[key], list):
        raise ValueError(f"{path} must contain a list field {key!r}")
    return [str(value) for value in payload[key]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-ids-file", type=Path, required=True)
    parser.add_argument("--canonical-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid batch/worker count")

    canonical = json.loads(args.canonical_split.read_text(encoding="utf-8-sig"))
    if isinstance(canonical.get("splits"), dict):
        canonical = canonical["splits"]
    test_ids = {str(v) for v in canonical.get("test", [])}
    validation_ids = _ids(args.validation_ids_file, "validation")
    if not validation_ids:
        raise ValueError("validation ID list is empty")
    overlap = set(validation_ids) & test_ids
    if overlap:
        raise ValueError(f"validation IDs overlap frozen test IDs: {sorted(overlap)}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "a0_parameter_matched_irreps":
        raise ValueError("formal evaluator is restricted to the final A0-PM candidate")
    if payload.get("checkpoint_provenance", {}).get("frozen_validation_test_labels_read", False):
        raise ValueError("checkpoint already reports frozen labels were read")

    config = load_project_config(args.config)
    records = load_gmtnet_records(config["data_root"])
    known = {str(record["JARVIS_ID"]) for record in records}
    missing = sorted(set(validation_ids) - known)
    if missing:
        raise KeyError(f"validation IDs missing from canonical records: {missing}")
    cache_key = graph_cache_key(records, float(config["cutoff"]), int(config["max_neighbors"]))
    dataset = _dataset(config, records, validation_ids, cache_key)
    dataset.warm_graph_cache()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        **({"persistent_workers": True} if args.num_workers else {}),
    )
    device = torch.device(args.device)
    model = make_model("a0_parameter_matched_irreps", config).to(device)
    for task in ("electronic", "born", "dielectric"):
        _tower(model, task).load_state_dict(payload["model"][task], strict=True)
    metrics = {
        task: _evaluate_task(_tower(model, task), loader, task)
        for task in ("electronic", "born", "dielectric")
    }
    output = {
        "schema": 1,
        "evaluation": "frozen_validation10_only",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "validation_ids_file": str(args.validation_ids_file),
        "validation_ids_sha256": _sha256(args.validation_ids_file),
        "canonical_split": str(args.canonical_split),
        "canonical_split_sha256": _sha256(args.canonical_split),
        "materials": len(validation_ids),
        "metrics": metrics,
        "frozen_validation_test_labels_read": True,
        "frozen_test20_labels_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
