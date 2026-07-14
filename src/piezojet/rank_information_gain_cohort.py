"""Rank a dry JARVIS DFPT queue by strict-completion propensity and information gain.

This selector never downloads an archive, creates a label, changes a strict
completion gate, or edits the frozen 69/10/20 benchmark.  It is deliberately a
retrieval queue.  Its score records what evidence was available for every
term, so missing phonon or BEC arrays cannot silently become a positive proxy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import torch
from pymatgen.core import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .data import _raw_cartesian_target, formula, load_gmtnet_records, response_norm_bin
from .strain_completion import _structure


PREFILTER_FRACTIONS = {0: 0.10, 1: 0.15, 2: 0.20, 3: 0.25, 4: 0.30}
WEIGHTS_WITHOUT_UNCERTAINTY = {
    "completion_propensity": 0.35,
    "diversity_relative_to_train": 0.25,
    "response_factor_proxy": 0.20,
    "test_stratum_gap": 0.20,
    "ensemble_uncertainty": 0.00,
}
WEIGHTS_WITH_UNCERTAINTY = {
    "completion_propensity": 0.30,
    "diversity_relative_to_train": 0.25,
    "response_factor_proxy": 0.20,
    "test_stratum_gap": 0.15,
    "ensemble_uncertainty": 0.10,
}


def _stable_key(record: dict) -> str:
    return hashlib.sha256(str(record["JARVIS_ID"]).encode()).hexdigest()


def _atom_bin(atoms: int) -> int:
    return 0 if atoms <= 2 else 1 if atoms <= 4 else 2 if atoms <= 8 else 3


def _polar(operations: Iterable) -> bool:
    rotations = torch.tensor([operation.rotation_matrix for operation in operations], dtype=torch.float64)
    return bool(torch.linalg.matrix_rank(rotations.mean(dim=0), tol=1e-7) > 0)


def _composition_vector(record: dict) -> torch.Tensor:
    vector = torch.zeros(118, dtype=torch.float64)
    elements = record["atoms"]["elements"]
    for symbol in elements:
        vector[Element(str(symbol)).Z - 1] += 1.0
    return vector / vector.sum().clamp_min(1.0)


def _min_train_chemical_distance(record: dict, train_vectors: list[torch.Tensor]) -> float:
    candidate = _composition_vector(record)
    # Half the L1 distance lies in [0, 1] for composition distributions.
    return min(float(0.5 * (candidate - train).abs().sum()) for train in train_vectors)


def _smoothed_rate(accepted: int, total: int, prior_rate: float, prior_strength: float = 8.0) -> float:
    """Beta-binomial shrinkage avoids ranking a one-row 100% group as certain."""
    return (accepted + prior_strength * prior_rate) / (total + prior_strength)


def _gap(train: Counter, test: Counter, value: str | int | bool) -> float:
    """One means a test-present stratum absent from the frozen training panel."""
    test_count = test[str(value)]
    if test_count == 0:
        return 0.0
    train_share = train[str(value)] / max(sum(train.values()), 1)
    test_share = test_count / max(sum(test.values()), 1)
    return max(0.0, min(1.0, (test_share - train_share) / test_share))


def _cache_path(directory: Path, jid: str) -> Path:
    return directory / (sha256(jid.encode("utf-8")).hexdigest()[:16] + ".pt")


def _cached_factor_proxy(directory: Path, jid: str) -> dict[str, float | str | bool | None]:
    """Read only already-parsed DFPT factors; never retrieve a new archive."""
    path = _cache_path(directory, jid)
    if not path.is_file():
        return {
            "raw_dfpt_available": False,
            "bec_mean_frobenius": None,
            "soft_optical_abs_eigenvalue": None,
            "factor_proxy_status": "unavailable_no_parsed_dfpt",
        }
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("jid") != jid:
        raise ValueError(f"DFPT cache identity mismatch for {jid}")
    born = torch.as_tensor(payload["born_charges"], dtype=torch.float64)
    eigenvalues = torch.sort(torch.as_tensor(payload["dynamical_eigenvalues"], dtype=torch.float64).abs()).values
    optical = eigenvalues[3:] if eigenvalues.numel() > 3 else eigenvalues
    return {
        "raw_dfpt_available": True,
        "bec_mean_frobenius": float(torch.linalg.vector_norm(born, dim=(-2, -1)).mean()),
        "soft_optical_abs_eigenvalue": float(optical.min()) if optical.numel() else None,
        "factor_proxy_status": "parsed_dfpt_cache",
    }


def _read_completion_history(path: Path | None) -> tuple[float, dict[str, tuple[int, int]], dict[str, tuple[int, int]], str]:
    if path is None or not path.is_file():
        return 0.5, {}, {}, "unavailable_no_audited_completion_profile"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return 0.5, {}, {}, "unavailable_empty_completion_profile"
    accepted = [str(row.get("accepted", "")).strip().lower() == "true" for row in rows]
    global_rate = sum(accepted) / len(accepted)
    by_atom: dict[str, list[bool]] = defaultdict(list)
    by_crystal: dict[str, list[bool]] = defaultdict(list)
    for row, value in zip(rows, accepted):
        by_atom[str(row.get("atom_count_bin", "unknown"))].append(value)
        by_crystal[str(row.get("crystal_system", "unknown"))].append(value)
    return (
        global_rate,
        {key: (sum(values), len(values)) for key, values in by_atom.items()},
        {key: (sum(values), len(values)) for key, values in by_crystal.items()},
        f"audited_profile:{path}",
    )


def _load_uncertainty(path: Path | None) -> tuple[dict[str, float], str]:
    if path is None:
        return {}, "unavailable_no_ensemble_scores_supplied"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("uncertainty", payload)
    if not isinstance(values, dict):
        raise ValueError("--ensemble-uncertainty must be a JSON object or contain an uncertainty object")
    parsed = {str(key): float(value) for key, value in values.items()}
    if not parsed or any(not math.isfinite(value) or value < 0 for value in parsed.values()):
        raise ValueError("Ensemble uncertainty values must be finite non-negative numbers")
    return parsed, f"supplied:{path}"


def _audited_material_ids(paths: Iterable[Path]) -> set[str]:
    """Return prior strict-audit IDs so known failures are never re-retrieved.

    An audit record, including a rejected one, is useful provenance.  It must
    not be treated as a missing label and put back into an acquisition queue.
    This does not change any strict-completion decision; it merely prevents
    duplicate downloads and duplicate attempts at the same strict gate.
    """
    audited: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"Prior audit manifest has no rows list: {path}")
        for row in rows:
            if not isinstance(row, dict) or "jid" not in row:
                raise ValueError(f"Prior audit manifest has an invalid row: {path}")
            audited.add(str(row["jid"]))
    return audited


def _select_one_formula_per_profile(profiles: list[dict], queue_size: int) -> tuple[list[dict], set[str]]:
    """Select ranked profiles while retaining at most one polymorph per formula."""
    selected, selected_formulas = [], set()
    for profile in profiles:
        if profile["formula"] in selected_formulas:
            continue
        selected.append(profile)
        selected_formulas.add(profile["formula"])
        if len(selected) == queue_size:
            break
    return selected, selected_formulas


def _select_test_crystal_coverage(
    profiles: list[dict], queue_size: int, test_crystal: Counter,
) -> tuple[list[dict], set[str], dict[str, int]]:
    """Reserve retrieval slots by frozen-test crystal-system frequency.

    The quotas are for *retrieval*, not accepted labels: low-completion systems
    remain subject to the unchanged strict gate.  They keep a high-propensity
    cubic class from exhausting a queue before structures represented in the
    frozen test panel have even been audited.
    """
    total_test = sum(test_crystal.values())
    if total_test <= 0:
        raise ValueError("Frozen test panel has no crystal-system metadata")
    quotas = {
        system: queue_size * count // total_test
        for system, count in test_crystal.items()
    }
    remainders = sorted(
        (
            (queue_size * count % total_test, str(system))
            for system, count in test_crystal.items()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for _, system in remainders[: queue_size - sum(quotas.values())]:
        quotas[system] += 1

    selected, selected_formulas = [], set()
    for system in sorted(quotas):
        for profile in profiles:
            if (
                profile["crystal_system"] != system
                or profile["formula"] in selected_formulas
            ):
                continue
            selected.append(profile)
            selected_formulas.add(profile["formula"])
            if sum(row["crystal_system"] == system for row in selected) == quotas[system]:
                break
    # A depleted structural class cannot justify a duplicate formula or an
    # unbounded queue.  Fill any such slot in the original evidence ranking.
    for profile in profiles:
        if len(selected) == queue_size:
            break
        if profile["formula"] in selected_formulas:
            continue
        selected.append(profile)
        selected_formulas.add(profile["formula"])
    return selected, selected_formulas, quotas


def _prefilter(records: list[dict], count: int) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[response_norm_bin(_raw_cartesian_target(record))].append(record)
    selected: list[dict] = []
    for response_bin, fraction in PREFILTER_FRACTIONS.items():
        quota = round(count * fraction)
        group = sorted(
            grouped[response_bin],
            key=lambda record: (len(record["atoms"]["elements"]), -float(torch.linalg.vector_norm(_raw_cartesian_target(record))), _stable_key(record)),
        )
        selected.extend(group[:quota])
    seen = {str(record["JARVIS_ID"]) for record in selected}
    remaining = [record for record in records if str(record["JARVIS_ID"]) not in seen]
    remaining.sort(key=lambda record: (len(record["atoms"]["elements"]), -response_norm_bin(_raw_cartesian_target(record)), _stable_key(record)))
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected[:count]


def rank(
    records: list[dict],
    frozen_splits: dict[str, list[str]],
    strict_completion_ids: set[str],
    preprofile_size: int,
    queue_size: int,
    excluded_space_groups: set[int],
    dfpt_dir: Path,
    completion_history: Path | None,
    uncertainty: dict[str, float],
    uncertainty_status: str,
    *,
    audited_ids: set[str] | None = None,
    selection_policy: str = "score",
) -> tuple[list[dict], dict]:
    """Return a deterministic, fully attributed retrieval queue and its audit."""
    by_id = {str(record["JARVIS_ID"]): record for record in records}
    frozen_ids = {jid for values in frozen_splits.values() for jid in values}
    if not frozen_ids.issubset(by_id):
        raise ValueError("Frozen split contains IDs missing from the GMTNet record source")
    frozen_formulas = {formula(by_id[jid]) for jid in frozen_ids}
    if selection_policy not in {"score", "test_crystal_coverage"}:
        raise ValueError("selection_policy must be score or test_crystal_coverage")
    audited_ids = audited_ids or set()
    candidates = [
        record for record in records
        if (
            str(record["JARVIS_ID"]) not in strict_completion_ids
            and str(record["JARVIS_ID"]) not in audited_ids
            and formula(record) not in frozen_formulas
        )
    ]
    if len(candidates) < preprofile_size:
        raise ValueError(f"Only {len(candidates)} eligible candidates, fewer than --preprofile-size={preprofile_size}")
    train_records = [by_id[jid] for jid in frozen_splits["train"]]
    train_vectors = [_composition_vector(record) for record in train_records]
    train_elements = {element for record in train_records for element in record["atoms"]["elements"]}
    preprofile = _prefilter(candidates, preprofile_size)
    global_rate, atom_rates, crystal_rates, history_status = _read_completion_history(completion_history)
    train_response = Counter(str(response_norm_bin(_raw_cartesian_target(record))) for record in train_records)
    test_records = [by_id[jid] for jid in frozen_splits["test"]]
    test_response = Counter(str(response_norm_bin(_raw_cartesian_target(record))) for record in test_records)

    profiles = []
    for record in preprofile:
        analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
        operations = analyzer.get_symmetry_operations(cartesian=True)
        space_group = int(analyzer.get_space_group_number())
        if space_group in excluded_space_groups:
            continue
        crystal = str(analyzer.get_crystal_system())
        point_group = str(analyzer.get_point_group_symbol())
        polar = _polar(operations)
        target = _raw_cartesian_target(record)
        factor_proxy = _cached_factor_proxy(dfpt_dir, str(record["JARVIS_ID"]))
        atom = _atom_bin(len(record["atoms"]["elements"]))
        atom_rate = _smoothed_rate(*atom_rates.get(str(atom), (0, 0)), global_rate)
        crystal_rate = _smoothed_rate(*crystal_rates.get(crystal, (0, 0)), global_rate)
        structural_completion = 0.5 * min(1.0, len(operations) / 48.0) + 0.5 / (1.0 + math.log2(len(record["atoms"]["elements"]) + 1.0))
        response_bin = response_norm_bin(target)
        profiles.append({
            "jid": str(record["JARVIS_ID"]),
            "formula": formula(record),
            "atoms": len(record["atoms"]["elements"]),
            "atom_count_bin": atom,
            "elements": sorted(set(record["atoms"]["elements"])),
            "space_group_number": space_group,
            "space_group_operations": len(operations),
            "crystal_system": crystal,
            "point_group": point_group,
            "polar": polar,
            "response_norm_c_per_m2": float(torch.linalg.vector_norm(target)),
            "response_bin": response_bin,
            "completion_propensity_score": 0.5 * (0.5 * atom_rate + 0.5 * crystal_rate) + 0.5 * structural_completion,
            "completion_history_status": history_status,
            "diversity_relative_to_train_score": 0.75 * _min_train_chemical_distance(record, train_vectors) + 0.25 * (len(set(record["atoms"]["elements"]) - train_elements) / max(len(set(record["atoms"]["elements"])), 1)),
            "raw_response_proxy_score": 0.5 * (response_bin / 4.0) + 0.5 * min(1.0, math.log1p(float(torch.linalg.vector_norm(target))) / math.log(11.0)),
            "response_proxy_source": "GMTNet JARVIS total piezoelectric tensor; not a new DFPT label",
            "factor_proxy": factor_proxy,
            "ensemble_uncertainty": uncertainty.get(str(record["JARVIS_ID"])),
            "ensemble_uncertainty_status": uncertainty_status,
        })

    available_bec = [
        float(row["factor_proxy"]["bec_mean_frobenius"])
        for row in profiles if row["factor_proxy"]["bec_mean_frobenius"] is not None
    ]
    available_soft = [
        float(row["factor_proxy"]["soft_optical_abs_eigenvalue"])
        for row in profiles if row["factor_proxy"]["soft_optical_abs_eigenvalue"] is not None
    ]
    bec_max = max(available_bec) if available_bec else 1.0
    soft_inverse_max = max((1.0 / max(value, 1e-12) for value in available_soft), default=1.0)
    for profile in profiles:
        factor_terms = []
        bec = profile["factor_proxy"]["bec_mean_frobenius"]
        soft = profile["factor_proxy"]["soft_optical_abs_eigenvalue"]
        if bec is not None:
            factor_terms.append(float(bec) / bec_max)
        if soft is not None:
            factor_terms.append((1.0 / max(float(soft), 1e-12)) / soft_inverse_max)
        profile["response_factor_proxy_score"] = (
            0.5 * float(profile["raw_response_proxy_score"]) + 0.5 * sum(factor_terms) / len(factor_terms)
            if factor_terms else float(profile["raw_response_proxy_score"])
        )
        profile["factor_proxy_score_status"] = (
            "combined_raw_response_and_parsed_bec_or_soft_mode" if factor_terms else "raw_response_only_no_parsed_bec_or_soft_mode"
        )

    train_crystal = Counter()
    train_point = Counter()
    train_polar = Counter()
    test_crystal = Counter()
    test_point = Counter()
    test_polar = Counter()
    for record, target_counter, crystal_counter, point_counter, polar_counter in (
        *[(record, train_response, train_crystal, train_point, train_polar) for record in train_records],
        *[(record, test_response, test_crystal, test_point, test_polar) for record in test_records],
    ):
        analyzer = SpacegroupAnalyzer(_structure(record), symprec=1e-5)
        operations = analyzer.get_symmetry_operations(cartesian=True)
        crystal_counter[str(analyzer.get_crystal_system())] += 1
        point_counter[str(analyzer.get_point_group_symbol())] += 1
        polar_counter[str(_polar(operations))] += 1
        # The response counters were constructed above from raw tensors; this
        # loop keeps the structural distribution computation explicit.
        del target_counter
    uncertainty_values = [value for value in uncertainty.values() if math.isfinite(value)]
    uncertainty_max = max(uncertainty_values) if uncertainty_values else 1.0
    use_uncertainty = bool(uncertainty_values)
    weights = WEIGHTS_WITH_UNCERTAINTY if use_uncertainty else WEIGHTS_WITHOUT_UNCERTAINTY
    for profile in profiles:
        profile["test_stratum_gap_score"] = (
            0.40 * _gap(train_response, test_response, profile["response_bin"])
            + 0.25 * _gap(train_crystal, test_crystal, profile["crystal_system"])
            + 0.25 * _gap(train_point, test_point, profile["point_group"])
            + 0.10 * _gap(train_polar, test_polar, profile["polar"])
        )
        raw_uncertainty = profile["ensemble_uncertainty"]
        profile["ensemble_uncertainty_score"] = (
            float(raw_uncertainty) / uncertainty_max if raw_uncertainty is not None and uncertainty_max > 0 else 0.0
        )
        profile["information_gain_score"] = sum(
            weights[name] * float(profile[f"{name}_score"]) for name in weights
        )
        profile["score_weights"] = weights
    profiles.sort(key=lambda row: (-float(row["information_gain_score"]), _stable_key({"JARVIS_ID": row["jid"]})))
    # A candidate queue dominated by polymorphs of one formula would repeat
    # the exact composition uncertainty that the resampling diagnosed.
    # Selection therefore keeps one representative per formula in every mode.
    if selection_policy == "score":
        selected, selected_formulas = _select_one_formula_per_profile(profiles, queue_size)
        retrieval_quotas: dict[str, int] = {}
    else:
        selected, selected_formulas, retrieval_quotas = _select_test_crystal_coverage(
            profiles, queue_size, test_crystal
        )
    if len(selected) < queue_size:
        raise ValueError("Insufficient distinct candidate formulas for the requested information-gain queue")
    return selected, {
        "eligible_after_frozen_formula_completed_label_and_prior_audit_exclusion": len(candidates),
        "excluded_prior_audits": len(audited_ids),
        "preprofiled": len(preprofile),
        "excluded_unresolved_space_group": preprofile_size - len(profiles),
        "completion_history_status": history_status,
        "ensemble_uncertainty_status": uncertainty_status,
        "score_weights": weights,
        "selection_policy": selection_policy,
        "retrieval_crystal_quotas": retrieval_quotas,
        "frozen_split_counts": {name: len(values) for name, values in frozen_splits.items()},
        "selected_response_bins": dict(Counter(str(row["response_bin"]) for row in selected)),
        "selected_crystal_systems": dict(Counter(str(row["crystal_system"]) for row in selected)),
        "selected_point_groups": dict(Counter(str(row["point_group"]) for row in selected)),
        "selected_polar": dict(Counter(str(row["polar"]) for row in selected)),
        "selected_cached_raw_dfpt": sum(bool(row["factor_proxy"]["raw_dfpt_available"]) for row in selected),
        "selected_unique_formulas": len(selected_formulas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/gmtnet"))
    parser.add_argument("--frozen-splits", type=Path, default=Path("data/processed/strict_completion_benchmark_v1.json"))
    parser.add_argument("--strict-completion-manifest", type=Path, default=Path("data/processed/jarvis_strain_completion_v4/manifest.json"))
    parser.add_argument("--dfpt-dir", type=Path, default=Path("data/processed/jarvis_dfpt_v1"))
    parser.add_argument("--completion-history", type=Path, default=Path("outputs/jarvis_dfpt_expansion_v1/completion_likelihood_report/completion_propensity_profiles.csv"))
    parser.add_argument("--ensemble-uncertainty", type=Path, help="Optional JSON of actually computed candidate ensemble uncertainties")
    parser.add_argument(
        "--prior-audit-manifest", type=Path, action="append", default=[],
        help="Strict audit manifest to exclude from retrieval, including rejected IDs; repeatable.",
    )
    parser.add_argument(
        "--selection-policy", choices=("score", "test_crystal_coverage"), default="score",
        help="Use test_crystal_coverage to reserve retrieval slots by frozen-test crystal-system frequency.",
    )
    parser.add_argument("--preprofile-size", type=int, default=500)
    parser.add_argument("--queue-size", type=int, default=100)
    parser.add_argument("--exclude-space-groups", default="187")
    parser.add_argument("--output", type=Path, default=Path("outputs/information_gain_cohort_v1/cohort.json"))
    args = parser.parse_args()
    if args.queue_size < 1 or args.preprofile_size < args.queue_size:
        raise ValueError("--preprofile-size must be at least the positive --queue-size")
    frozen = json.loads(args.frozen_splits.read_text(encoding="utf-8"))
    if not frozen.get("frozen") or set(frozen.get("splits", {})) != {"train", "val", "test"}:
        raise ValueError("--frozen-splits must be the registered frozen 69/10/20 split artifact")
    completion = json.loads(args.strict_completion_manifest.read_text(encoding="utf-8"))
    strict_ids = {str(jid) for jid in completion["material_ids"]}
    audited_ids = _audited_material_ids(args.prior_audit_manifest)
    uncertainty, uncertainty_status = _load_uncertainty(args.ensemble_uncertainty)
    selected, summary = rank(
        load_gmtnet_records(args.data_root), frozen["splits"], strict_ids,
        args.preprofile_size, args.queue_size,
        {int(value) for value in args.exclude_space_groups.split(",") if value.strip()},
        args.dfpt_dir, args.completion_history, uncertainty, uncertainty_status,
        audited_ids=audited_ids, selection_policy=args.selection_policy,
    )
    payload = {
        "schema": 1,
        "scope": "dry retrieval ranking only: no archive download, completion threshold change, label acceptance, or frozen-split reassignment",
        "policy": "completion propensity + diversity relative to frozen train + response/factor proxy + frozen-test stratum gaps; optional test-crystal retrieval quotas reserve audit slots without changing strict acceptance; ensemble uncertainty has zero weight unless supplied from an actual ensemble",
        "frozen_splits": str(args.frozen_splits),
        "strict_completion_manifest": str(args.strict_completion_manifest),
        "prior_audit_manifests": [str(path) for path in args.prior_audit_manifest],
        "excluded_space_groups": sorted({int(value) for value in args.exclude_space_groups.split(",") if value.strip()}),
        "summary": summary,
        "material_ids": [row["jid"] for row in selected],
        "materials": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [{key: value for key, value in row.items() if key not in {"factor_proxy", "score_weights"}} for row in selected]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
