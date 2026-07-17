"""Build the full-public, formula-safe electrostatic development stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .build_development_folds import assign_formula_groups
from .data import load_gmtnet_records
from .ood import reduced_formula
from .project_config import load_project_config


CONVENTION_QUARANTINE = {"JVASP-42995", "JVASP-28862"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def electrostatic_availability(payload: dict[str, object]) -> dict[str, bool]:
    """Validate finite source arrays rather than inferring coverage from keys."""
    def tensor(value: object) -> torch.Tensor:
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError):
            return torch.empty(0)

    born = tensor(payload.get("born_charges"))
    total = tensor(payload.get("total_piezo_source"))
    ionic = tensor(payload.get("ionic_piezo_source"))
    force = tensor(payload.get("force_constants"))
    epsilon_payload = payload.get("epsilon")
    epsilon = tensor(
        epsilon_payload.get("epsilon")
        if isinstance(epsilon_payload, dict) else None
    )
    atom_count = born.shape[0] if born.ndim == 3 else -1
    return {
        "born_charges": (
            born.ndim == 3 and born.shape[1:] == (3, 3)
            and bool(torch.isfinite(born).all())
        ),
        "electronic_piezo_from_same_outcar_total_minus_ionic": (
            total.shape == (3, 6) and ionic.shape == (3, 6)
            and bool(torch.isfinite(total).all()) and bool(torch.isfinite(ionic).all())
        ),
        "electronic_dielectric": (
            epsilon.shape == (3, 3) and bool(torch.isfinite(epsilon).all())
        ),
        "force_constants": (
            force.shape == (atom_count, atom_count, 3, 3)
            and bool(torch.isfinite(force).all())
        ),
    }


def electrostatic_fold_train_ids(
    payload: dict[str, object], fold_index: int
) -> list[str]:
    """Derive one fit panel from the non-redundant population/development map."""
    population = set(map(str, payload["material_ids"]))
    fold = next(
        (entry for entry in payload["folds"] if entry["fold"] == fold_index),
        None,
    )
    if fold is None:
        raise ValueError(f"Electrostatic fold {fold_index} is absent")
    development = set(map(str, fold["development"]))
    if not development or not development.issubset(population):
        raise ValueError("Electrostatic development IDs are not a population subset")
    return sorted(population - development)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--formula-safe-split",
        type=Path,
        default=Path("data/processed/full_corpus_multitask_train1595_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/electrostatic_development_folds_v2.json"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_project_config(args.config)
    split = json.loads(args.formula_safe_split.read_text(encoding="utf-8-sig"))
    source_train = list(map(str, split["splits"]["train"]))
    frozen_ids = set(map(str, split["splits"]["val"] + split["splits"]["test"]))
    records = load_gmtnet_records(config["data_root"])
    by_id = {str(record["JARVIS_ID"]): record for record in records}
    dfpt_root = Path(str(config["jarvis_dfpt_dir"]))
    payload_by_id: dict[str, dict[str, object]] = {}
    for path in dfpt_root.glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        jid = str(payload.get("jid", ""))
        if jid:
            payload_by_id[jid] = payload

    availability = {
        jid: electrostatic_availability(payload)
        for jid, payload in payload_by_id.items()
    }
    eligible = {
        jid for jid, flags in availability.items() if all(flags.values())
    }
    frozen_formulas = {reduced_formula(by_id[jid]) for jid in frozen_ids}
    candidate_ids = set(source_train) & eligible - CONVENTION_QUARANTINE
    reduced_formula_exclusions = sorted(
        jid for jid in candidate_ids
        if reduced_formula(by_id[jid]) in frozen_formulas
    )
    train_ids = sorted(candidate_ids - set(reduced_formula_exclusions))
    if set(train_ids) & frozen_ids:
        raise RuntimeError("Electrostatic train stream overlaps frozen IDs")
    train_formulas = {reduced_formula(by_id[jid]) for jid in train_ids}
    if train_formulas & frozen_formulas:
        raise RuntimeError("Electrostatic train stream overlaps frozen formulas")

    fold_ids = assign_formula_groups(
        [by_id[jid] for jid in train_ids], args.folds, args.seed
    )
    folds = []
    for index, development in enumerate(fold_ids):
        development_set = set(development)
        fit = sorted(set(train_ids) - development_set)
        fit_formulas = {reduced_formula(by_id[jid]) for jid in fit}
        dev_formulas = {reduced_formula(by_id[jid]) for jid in development}
        if fit_formulas & dev_formulas:
            raise RuntimeError(f"Electrostatic fold {index} has formula leakage")
        folds.append({
            "fold": index,
            "development": development,
            "train_materials": len(fit),
            "development_materials": len(development),
            "train_formulas": len(fit_formulas),
            "development_formulas": len(dev_formulas),
        })

    coverage = {
        "parsed_payloads": len(payload_by_id),
        "born_charges": sum(value["born_charges"] for value in availability.values()),
        "electronic_piezo_from_same_outcar_total_minus_ionic": sum(
            value["electronic_piezo_from_same_outcar_total_minus_ionic"]
            for value in availability.values()
        ),
        "electronic_dielectric": sum(
            value["electronic_dielectric"] for value in availability.values()
        ),
        "force_constants": sum(
            value["force_constants"] for value in availability.values()
        ),
        "all_electrostatic_labels_finite_and_shape_valid": len(eligible),
        "strict_lambda_not_required": True,
        "formula_safe_train_with_all_electrostatic_labels": len(train_ids),
        "excluded_missing_or_malformed_dfpt": len(set(source_train) - set(payload_by_id)),
        "excluded_invalid_electrostatic_payload": len(
            (set(source_train) & set(payload_by_id)) - eligible
        ),
        "excluded_convention_quarantine": sorted(CONVENTION_QUARANTINE & set(source_train)),
        "excluded_reduced_formula_overlap_with_frozen_panel": reduced_formula_exclusions,
    }
    result = {
        "schema": 2,
        "role": "electrostatic method development only; frozen val10/test20 labels unread",
        "source_formula_safe_split": str(args.formula_safe_split),
        "source_formula_safe_split_sha256": _sha256(args.formula_safe_split),
        "seed": args.seed,
        "fold_count": args.folds,
        "population": "full-public JARVIS/GMTNet overlap with verified same-OUTCAR electrostatic labels",
        "frozen_validation_test_labels_read": False,
        "availability_mask_policy": "BEC/electronic/dielectric do not require strict Lambda completion",
        "coverage": coverage,
        "material_ids": train_ids,
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "coverage": coverage, "fold_sizes": [len(v) for v in fold_ids]}, indent=2))


if __name__ == "__main__":
    main()
