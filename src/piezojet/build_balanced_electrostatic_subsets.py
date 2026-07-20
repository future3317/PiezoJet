"""Build nested, fold-train-only balanced electrostatic response subsets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Composition, Element
from scipy.stats import ks_2samp
import spglib
import torch

from .build_electrostatic_development_folds import electrostatic_fold_train_ids
from .checkpoint_provenance import file_sha256
from .data import PiezoDataset, deterministic_subset, graph_cache_key, load_gmtnet_records
from .electrostatic_subset import SUBSET_SCHEMA, material_id_sha256
from .project_config import load_project_config


def _crystal_system(space_group: int) -> str:
    boundaries = (
        (2, "triclinic"),
        (15, "monoclinic"),
        (74, "orthorhombic"),
        (142, "tetragonal"),
        (167, "trigonal"),
        (194, "hexagonal"),
        (230, "cubic"),
    )
    return next(name for upper, name in boundaries if space_group <= upper)


def _structure_metadata(record: dict[str, Any]) -> tuple[str, str, tuple[str, ...], int]:
    atoms = record["atoms"]
    elements = tuple(sorted(set(map(str, atoms["elements"]))))
    reduced_formula = Composition(Counter(map(str, atoms["elements"]))).reduced_formula
    numbers = [Element(symbol).Z for symbol in atoms["elements"]]
    dataset = spglib.get_symmetry_dataset(
        (np.asarray(atoms["lattice_mat"]), np.asarray(atoms["coords"]), numbers),
        symprec=1e-3,
    )
    if dataset is None:
        crystal_system = "unresolved"
    else:
        crystal_system = _crystal_system(int(dataset.number))
    return reduced_formula, crystal_system, elements, len(numbers)


def _atom_bin(atoms: int) -> str:
    for upper in (2, 4, 8, 16, 32, 64):
        if atoms <= upper:
            lower = 1 if upper == 2 else upper // 2 + 1
            return f"{lower}-{upper}"
    return "65+"


def _quantile_codes(values: np.ndarray, bins: int = 8) -> tuple[np.ndarray, list[float]]:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) == 1:
        return np.zeros(len(values), dtype=np.int64), edges.tolist()
    codes = np.searchsorted(edges[1:-1], values, side="right")
    return codes.astype(np.int64), edges.tolist()


def _incidence(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], np.ndarray]:
    categories: list[list[str]] = []
    weights: dict[str, float] = {}
    for row in rows:
        keys = [
            f"electronic_q={row['electronic_quantile']}",
            f"born_q={row['born_quantile']}",
            f"dielectric_q={row['dielectric_quantile']}",
            f"crystal={row['crystal_system']}",
            f"atoms={row['atom_bin']}",
        ]
        categories.append(keys)
        for key in keys:
            weights[key] = 2.0 if key.startswith(("electronic", "born", "dielectric")) else 1.0
    names = sorted(weights)
    index = {name: position for position, name in enumerate(names)}
    matrix = np.zeros((len(rows), len(names)), dtype=np.float64)
    for row_index, keys in enumerate(categories):
        matrix[row_index, [index[key] for key in keys]] = 1.0
    return matrix, names, np.asarray([weights[name] for name in names])


def balanced_order(rows: list[dict[str, Any]], limit: int, seed: int) -> list[int]:
    """Vectorized marginal-balancing coreset with element/formula diversity."""
    if not 0 < limit <= len(rows):
        raise ValueError("Balanced subset limit must be within the population")
    incidence, _, category_weights = _incidence(rows)
    population_counts = incidence.sum(axis=0)
    element_names = sorted({element for row in rows for element in row["elements"]})
    element_index = {name: index for index, name in enumerate(element_names)}
    elements = np.zeros((len(rows), len(element_names)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        elements[row_index, [element_index[value] for value in row["elements"]]] = 1.0
    element_frequency = elements.sum(axis=0)
    rare_element_weight = 1.0 / np.sqrt(element_frequency.clip(min=1.0))
    formulas = np.asarray([row["reduced_formula"] for row in rows], dtype=object)
    rng = np.random.default_rng(seed)
    tie_break = rng.uniform(0.0, 1e-10, size=len(rows))
    available = np.ones(len(rows), dtype=bool)
    category_counts = np.zeros(incidence.shape[1], dtype=np.float64)
    covered_elements = np.zeros(elements.shape[1], dtype=bool)
    used_formulas: set[str] = set()
    selected: list[int] = []
    for step in range(limit):
        desired = population_counts * ((step + 1) / len(rows))
        deficit = np.maximum(desired - category_counts, 0.0)
        marginal_score = incidence @ (
            category_weights * deficit / np.maximum(desired, 1.0)
        )
        unseen_element_score = (
            elements[:, ~covered_elements] @ rare_element_weight[~covered_elements]
            if bool((~covered_elements).any()) else np.zeros(len(rows))
        )
        unused_formula = np.asarray(
            [formula not in used_formulas for formula in formulas], dtype=np.float64
        )
        score = marginal_score + 3.0 * unseen_element_score + 0.25 * unused_formula + tie_break
        score[~available] = -np.inf
        choice = int(np.argmax(score))
        if not math.isfinite(float(score[choice])):
            raise RuntimeError("Balanced subset exhausted eligible candidates")
        selected.append(choice)
        category_counts += incidence[choice]
        covered_elements |= elements[choice].astype(bool)
        used_formulas.add(str(formulas[choice]))
        available[choice] = False
    return selected


def _distribution_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    result: dict[str, object] = {"materials": len(rows)}
    for key in ("electronic_norm", "born_norm", "dielectric_norm"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key] = {
            "median": float(np.median(values)),
            "q10": float(np.quantile(values, 0.1)),
            "q90": float(np.quantile(values, 0.9)),
            "maximum": float(values.max()),
        }
    result["crystal_system"] = dict(sorted(Counter(row["crystal_system"] for row in rows).items()))
    result["atom_bin"] = dict(sorted(Counter(row["atom_bin"] for row in rows).items()))
    result["elements"] = sorted({element for row in rows for element in row["elements"]})
    result["unique_reduced_formulas"] = len({row["reduced_formula"] for row in rows})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--folds", type=Path,
        default=Path("data/processed/electrostatic_development_folds_v2.json"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--limits", type=int, nargs="+", default=[200, 800])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/processed/electrostatic_balanced_subsets_v1"),
    )
    args = parser.parse_args()
    limits = sorted(set(args.limits))
    config = load_project_config(args.config)
    folds = json.loads(args.folds.read_text(encoding="utf-8-sig"))
    full_ids = electrostatic_fold_train_ids(folds, args.fold)
    records = load_gmtnet_records(config["data_root"])
    by_id = {str(record["JARVIS_ID"]): record for record in records}
    cache_key = graph_cache_key(records, float(config["cutoff"]), int(config["max_neighbors"]))
    dataset = PiezoDataset(
        records,
        full_ids,
        float(config["cutoff"]),
        int(config["max_neighbors"]),
        processed_dir=config["processed_dir"],
        cache_key=cache_key,
        project_targets=True,
        dfpt_dir=config["jarvis_dfpt_dir"],
        strain_completion_dir=None,
        dfpt_profile="electrostatic",
    )
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        graph = dataset[index]
        jid = str(graph.material_id)
        formula, crystal, elements, atoms = _structure_metadata(by_id[jid])
        rows.append({
            "material_id": jid,
            "reduced_formula": formula,
            "crystal_system": crystal,
            "elements": elements,
            "atoms": atoms,
            "atom_bin": _atom_bin(atoms),
            "electronic_norm": float(torch.linalg.vector_norm(graph.y_electronic_piezo)),
            "born_norm": float(torch.linalg.vector_norm(graph.y_born)),
            "dielectric_norm": float(torch.linalg.vector_norm(graph.y_dfpt_electronic_dielectric)),
        })
    for name in ("electronic", "born", "dielectric"):
        codes, edges = _quantile_codes(np.asarray([row[f"{name}_norm"] for row in rows]))
        for row, code in zip(rows, codes, strict=True):
            row[f"{name}_quantile"] = int(code)
        if not edges:
            raise RuntimeError(f"Could not construct {name} norm quantiles")
    order = balanced_order(rows, max(limits), args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_summary = _distribution_summary(rows)
    manifests: list[dict[str, object]] = []
    for limit in limits:
        selected_rows = [rows[index] for index in order[:limit]]
        ids = [row["material_id"] for row in selected_rows]
        original_ids = deterministic_subset(full_ids, limit, args.seed + 1000)
        original_rows = [rows[full_ids.index(jid)] for jid in original_ids]
        ks = {}
        for key in ("electronic_norm", "born_norm", "dielectric_norm"):
            full_values = [row[key] for row in rows]
            subset_values = [row[key] for row in selected_rows]
            statistic, pvalue = ks_2samp(full_values, subset_values)
            ks[key] = {"statistic": float(statistic), "pvalue": float(pvalue)}
        payload: dict[str, object] = {
            "schema": SUBSET_SCHEMA,
            "role": "fold-train-only balanced response supervision; no development or frozen-panel labels",
            "fold": args.fold,
            "seed": args.seed,
            "materials": limit,
            "material_ids": ids,
            "material_id_sha256": material_id_sha256(ids),
            "nested_prefix_of": max(limits) if limit != max(limits) else None,
            "source_folds": str(args.folds.resolve()),
            "source_folds_sha256": file_sha256(args.folds),
            "frozen_validation_test_labels_read": False,
            "selection": (
                "vectorized greedy marginal balance over electronic/BEC/dielectric norm octiles, "
                "GMTNet-pinned crystal system, atom count, element coverage, and reduced-formula diversity"
            ),
            "full_fold_train_summary": full_summary,
            "balanced_summary": _distribution_summary(selected_rows),
            "original_deterministic_subset_summary": _distribution_summary(original_rows),
            "balanced_vs_full_ks": ks,
        }
        path = args.output_dir / f"fold{args.fold}_balanced_n{limit}_seed{args.seed}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifests.append({"limit": limit, "path": str(path), "sha256": file_sha256(path)})
    (args.output_dir / "summary.json").write_text(
        json.dumps({
            "schema": 1,
            "fold": args.fold,
            "seed": args.seed,
            "population": len(rows),
            "manifests": manifests,
            "frozen_validation_test_labels_read": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
