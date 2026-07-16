"""Build and validate the immutable PiezoJet experiment ledger.

The ledger is deliberately an index over persisted artifacts, not a second
copy of their metrics.  Scientific interpretation is assigned at the cohort
level, while every run-like subdirectory and every file remains discoverable
through the generated JSON and JSONL inventories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


GENERATED_OUTPUT_NAMES = {
    "EXPERIMENT_REGISTRY.json",
    "EXPERIMENT_ARTIFACT_INDEX.jsonl",
}
RUN_MARKERS = {
    "config.resolved.yaml",
    "summary.json",
    "metrics.csv",
    "factor_metrics.csv",
    "dfpt_test.json",
    "test.json",
    "evaluation_test.json",
    "manifest.json",
    "history.json",
    "report.json",
    "INTERRUPTED.md",
    "BLOCKED.md",
    "experiment_plan.json",
    "experiment_status_history.jsonl",
}
KEY_NAME_RE = re.compile(
    r"(?:config|summary|metrics|report|manifest|provenance|test|audit|"
    r"INTERRUPTED|BLOCKED|driver\.(?:stdout|stderr))",
    re.IGNORECASE,
)
SEED_RE = re.compile(r"(?:^|[_-])seed[_-]?(\d+)(?:$|[_-])", re.IGNORECASE)
PASS_RE = re.compile(r"(?:^|[_-])passes?[_-]?(\d+)(?:$|[_-])", re.IGNORECASE)
HASHABLE_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".yaml", ".yml", ".txt"}


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cohort_policy(name: str) -> dict[str, Any]:
    """Return the auditable scientific classification for one output cohort."""

    policy: dict[str, Any] = {
        "family": "early_development",
        "archival_status": "historical",
        "execution_status": "completed_or_retained_partial",
        "result_disposition": "diagnostic",
        "convention_epoch": "historical_pre_v7_or_run_local",
        "comparability_group": "none_run_local_only",
        "paper_use": "historical_appendix_only",
        "interpretation_boundary": (
            "Development artifact; use only with its persisted run-local configuration. "
            "Do not pool it with current direct-U results."
        ),
    }

    if name == "exposure_matched_direct_u_v2_conditioning":
        policy.update(
            family="registered_direct_u_exposure_replay",
            archival_status="current",
            execution_status="running",
            result_disposition="pending_registered_decision",
            convention_epoch="v7_bec_transpose_regularized_direct_u",
            comparability_group="direct_u_exposure_v2_frozen_val10_test20",
            paper_use="appendix_after_registered_grid_completes",
            interpretation_boundary=(
                "Registered 1/5/10/20-pass, seed 42/7/1729 physical and matched-direct grid. "
                "No pass or checkpoint may be chosen from test results; incomplete points are diagnostics only."
            ),
            preregistered_passes=[1, 5, 10, 20],
            preregistered_seeds=[42, 7, 1729],
            split="full_corpus_multitask_train149_v1; frozen validation10/test20",
        )
    elif name == "direct_u_multistream_smoke_v1":
        policy.update(
            family="direct_u_implementation_smoke",
            archival_status="current",
            execution_status="completed",
            result_disposition="negative_one_pass_diagnostic",
            convention_epoch="v7_bec_transpose_regularized_direct_u",
            comparability_group="direct_u_smoke_only",
            paper_use="appendix_implementation_check_only",
            interpretation_boundary="One pass and one seed; validates execution/closure, not predictive performance.",
            split="full_corpus_multitask_train149_v1; frozen validation10/test20",
        )
    elif name.startswith("teacher_forced_zero_basin"):
        policy.update(
            family="identifiability_memorization_probe",
            archival_status="current",
            execution_status="planned_or_running",
            result_disposition="capacity_diagnostic_pending",
            convention_epoch="v7_bec_transpose_teacher_forced_direct_u",
            comparability_group="noninductive_memorization_only",
            paper_use="appendix_implementation_diagnostic_only",
            interpretation_boundary=(
                "1/8/32-material same-ID optimization with noninductive validation. "
                "It can falsify an optimization/capacity failure but can never support a holdout claim."
            ),
        )
    elif name.startswith("response_operator_action_capacity"):
        policy.update(
            family="response_operator_action_capacity_probe",
            archival_status="current",
            execution_status="planned_or_running",
            result_disposition="capacity_diagnostic_pending",
            convention_epoch="v7_bec_transpose_true_factor_operator_action",
            comparability_group="noninductive_memorization_only",
            paper_use="appendix_implementation_diagnostic_only",
            interpretation_boundary=(
                "1/8/32-material same-ID comparison of true-factor response-operator action supervision. "
                "It can test optimization/capacity only and cannot support a holdout claim."
            ),
        )
    elif name.startswith("full_corpus_multitask_detached_lift_v"):
        policy.update(
            family="historical_detached_macro_lift",
            archival_status="historical",
            execution_status="failed_superseded" if name.endswith("_v1") else "completed",
            result_disposition="negative_nonidentifiable_parameterization",
            convention_epoch="bec_transpose_with_removed_detached_lift",
            comparability_group="historical_detached_lift_only",
            paper_use="historical_appendix_only",
            interpretation_boundary=(
                "Retained negative evidence for the removed detached macro-to-Lambda lift; "
                "not comparable to independent direct-U training."
            ),
        )
    elif name in {"observable_subspace_v1", "observable_subspace_v2", "observable_lift_geometry_v1", "bec_transpose_observable_v1"}:
        policy.update(
            family="historical_observable_lift",
            result_disposition="negative_or_nonidentifiable",
            convention_epoch="run_local_observable_lift",
            comparability_group="historical_observable_lift_only",
            paper_use="historical_appendix_only",
            interpretation_boundary=(
                "pInv/ridge active-null chart has been removed. Persisted results are forensic evidence only; "
                "any incomplete e3nn subrun has no performance result."
            ),
        )
    elif name == "bec_transpose_cartesian_train149_pretrain_v1":
        policy.update(
            family="structural_pretraining",
            archival_status="current",
            execution_status="completed",
            result_disposition="support_artifact",
            convention_epoch="v7_bec_transpose",
            comparability_group="inductive_train149_pretraining",
            paper_use="appendix_provenance",
            interpretation_boundary="Train-only structural initialization; not a response-performance experiment.",
        )
    elif re.match(r"(?:dfpt_convention_audit|pymatgen_|symfc_|gmtnet_outcar_|schema4_|strict_v8_|audit$|response_audit)", name):
        policy.update(
            family="data_convention_and_provenance_audit",
            archival_status="current" if name not in {"dfpt_convention_audit_v1"} else "historical",
            execution_status="completed",
            result_disposition="support_audit",
            convention_epoch="source_audited_bec_transpose",
            comparability_group="not_a_performance_experiment",
            paper_use="appendix_provenance",
            interpretation_boundary="Data/parser/convention evidence; no model-performance claim.",
        )
    elif name.startswith("hessian_bond_laplacian_oracle"):
        policy.update(
            family="offline_hessian_model_class_oracle",
            archival_status="current",
            execution_status="completed",
            result_disposition="model_class_diagnostic",
            convention_epoch="v7_bec_transpose_train149",
            comparability_group="strict_train_only_model_class_diagnostic",
            paper_use="appendix_model_class_diagnostic",
            interpretation_boundary=(
                "Offline unrestricted symmetric edge-K least-squares projection on strict training IDs only. "
                "It is neither a learned checkpoint nor a held-out performance result."
            ),
        )
    elif re.match(r"(?:information_gain_|jarvis_dfpt_expansion)", name):
        policy.update(
            family="jarvis_data_expansion",
            execution_status="completed",
            result_disposition="retrieval_or_completion_audit",
            convention_epoch="historical_v4_v6_strict_gates",
            comparability_group="not_a_performance_experiment",
            paper_use="appendix_data_history",
            interpretation_boundary=(
                "Retrieval queues are not accepted-label cohorts. Historical strict gates remain intact; "
                "do not mix these convention variants with v7 labels."
            ),
        )
    elif re.match(r"(?:strict_learning_curve|optimization_ablation|feedback[45]_execution|factor_protected_|strict_completion_train\d+_protocol|stratified_subset|mode_aware)", name):
        policy.update(
            family="historical_registered_factor_forensics",
            execution_status="completed",
            result_disposition="mixed_or_negative_forensic",
            convention_epoch="historical_factorized_pre_direct_u",
            comparability_group="historical_factor_forensics_only",
            paper_use="historical_appendix_only",
            interpretation_boundary=(
                "Registered historical factor/protocol evidence. Its executors are no longer maintained and "
                "its values must not be pooled with v7 direct-U replay results."
            ),
        )
    elif re.match(r"(?:dfpt_128_|strict_completion_v4_)", name):
        policy.update(
            family="historical_dfpt_model_development",
            execution_status="completed",
            result_disposition="development_diagnostic",
            convention_epoch="historical_pre_v7",
            comparability_group="none_run_local_only",
            paper_use="historical_appendix_only",
            interpretation_boundary="Pre-v7 model-development run; never combine with convention-corrected results.",
        )
    elif re.match(r"(?:lambda_diagnostics|dfpt_pilot|dfpt_energy_refactor|dfpt_response_active)", name):
        policy.update(
            family="historical_dfpt_diagnostic",
            result_disposition="development_diagnostic",
            paper_use="historical_appendix_only",
        )
    elif name == "m3":
        policy.update(
            family="historical_milestone",
            execution_status="interrupted",
            result_disposition="no_generalization_result",
            paper_use="historical_appendix_only",
            interpretation_boundary="Three-seed run was not launched; one dry run was explicitly interrupted.",
        )
    elif name in {"baselines", "pretrain", "pretrain_cartesian"} or "pretrain" in name:
        policy.update(
            family="support_or_pretraining",
            execution_status="completed_or_partial",
            result_disposition="support_artifact",
            comparability_group="not_a_performance_experiment",
            paper_use="appendix_provenance_if_used",
        )
    elif re.match(r"(?:inference_|checkpoint_|m4$)", name):
        policy.update(
            family="engineering_validation",
            result_disposition="smoke_or_resource_diagnostic",
            comparability_group="not_a_performance_experiment",
            paper_use="project_ledger_only",
        )

    return policy


def _last_status_event(cohort: Path) -> dict[str, Any] | None:
    history = cohort / "experiment_status_history.jsonl"
    if not history.exists():
        return None
    events: list[dict[str, Any]] = []
    for line in history.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and "status" in event:
            events.append(event)
    return events[-1] if events else None


def _run_status(path: Path, *, cohort_running: bool) -> str:
    names = {item.name for item in path.iterdir() if item.is_file()}
    if "INTERRUPTED.md" in names:
        return "interrupted"
    if "BLOCKED.md" in names:
        return "blocked"
    if any("failed" in name.lower() for name in names):
        return "failed_or_has_failure_record"
    if {"dfpt_test.json", "test.json", "evaluation_test.json"} & names:
        return "completed_with_evaluation"
    if cohort_running and (
        "summary.json" in names
        or "metrics.csv" in names
        or any(name.endswith(".pt") for name in names)
    ):
        return "running_or_pending_evaluation"
    if "summary.json" in names or "report.json" in names:
        return "completed_with_summary"
    if "history.json" in names and any(name.endswith(".pt") for name in names):
        return "completed_training_or_pretraining"
    if any(name.endswith(".pt") for name in names):
        return "running" if cohort_running else "partial_no_evaluation"
    return "support_record"


def _run_records(cohort: Path, *, cohort_running: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for directory in [cohort, *(p for p in cohort.rglob("*") if p.is_dir())]:
        files = [p for p in directory.iterdir() if p.is_file()]
        names = {p.name for p in files}
        has_marker = bool(names & RUN_MARKERS) or any(p.suffix == ".pt" for p in files)
        if not has_marker:
            continue
        relative = directory.relative_to(cohort).as_posix() or "."
        seed_match = SEED_RE.search(relative.replace("/", "_"))
        pass_match = PASS_RE.search(relative.replace("/", "_"))
        records.append(
            {
                "path": relative,
                "execution_status": _run_status(directory, cohort_running=cohort_running),
                "seed": int(seed_match.group(1)) if seed_match else None,
                "passes": int(pass_match.group(1)) if pass_match else None,
                "markers": sorted(names & RUN_MARKERS),
                "checkpoint_files": sorted(p.name for p in files if p.suffix == ".pt"),
            }
        )
    return records


def _key_artifacts(files: Iterable[Path], root: Path, limit: int = 80) -> list[str]:
    selected = [p.relative_to(root).as_posix() for p in files if KEY_NAME_RE.search(p.name)]
    return sorted(selected)[:limit]


def build_registry(outputs: Path) -> dict[str, Any]:
    cohorts: list[dict[str, Any]] = []
    all_files = [
        p for p in outputs.rglob("*")
        if p.is_file() and p.name not in GENERATED_OUTPUT_NAMES
    ]
    for cohort in sorted((p for p in outputs.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        files = [p for p in all_files if cohort in p.parents]
        policy = _cohort_policy(cohort.name)
        last_status = _last_status_event(cohort)
        if last_status is not None:
            policy["last_status_event"] = last_status
            if last_status["status"] in {"failed_or_interrupted", "incomplete_after_process_exit"}:
                policy["execution_status"] = "failed_or_interrupted"
                policy["result_disposition"] = "incomplete_no_registered_result"
                policy["paper_use"] = "appendix_incomplete_only"
            elif last_status["status"] == "completed":
                policy["execution_status"] = "completed"
        runs = _run_records(cohort, cohort_running=policy["execution_status"] == "running")
        completed = sum(run["execution_status"].startswith("completed") for run in runs)
        partial = sum(
            run["execution_status"]
            in {
                "partial_no_evaluation",
                "running_or_pending_evaluation",
                "interrupted",
                "blocked",
                "failed_or_has_failure_record",
            }
            for run in runs
        )
        if cohort.name == "exposure_matched_direct_u_v2_conditioning":
            expected_runs = 2 * len(policy["preregistered_passes"]) * len(policy["preregistered_seeds"])
            complete_grid = completed == expected_runs
            policy["planned_run_records"] = expected_runs
            policy["registered_grid_complete"] = complete_grid
            if complete_grid:
                policy["execution_status"] = "completed"
                policy["result_disposition"] = "registered_result_available"
                policy["paper_use"] = "appendix_registered_result"
        if cohort.name.startswith("teacher_forced_zero_basin"):
            completed_probes = len(list(cohort.glob("samples*/overfit_dfpt_train.json")))
            # The old CPU plumbing smoke intentionally stopped at 1/2 examples;
            # the registered GPU falsification ladder is 1/8/32 as prescribed.
            expected_probes = 2 if "cpu_smoke" in cohort.name else 3
            policy["expected_completed_probes"] = expected_probes
            policy["completed_probes"] = completed_probes
            if completed_probes == expected_probes:
                policy["execution_status"] = "completed"
                policy["result_disposition"] = (
                    "cpu_implementation_smoke" if "cpu_smoke" in cohort.name
                    else "noninductive_capacity_diagnostic_available"
                )
            elif completed_probes:
                policy["execution_status"] = "partial"
                policy["result_disposition"] = "partial_noninductive_diagnostic"
            elif files:
                policy["execution_status"] = "partial"
                policy["result_disposition"] = "failed_or_incomplete_noninductive_diagnostic"
        if cohort.name.startswith("response_operator_action_capacity"):
            completed_probes = len(list(cohort.glob("samples*/overfit_dfpt_train.json")))
            policy["expected_completed_probes"] = 3
            policy["completed_probes"] = completed_probes
            if completed_probes == 3:
                policy["execution_status"] = "completed"
                policy["result_disposition"] = "noninductive_capacity_diagnostic_available"
            elif completed_probes:
                policy["execution_status"] = "partial"
                policy["result_disposition"] = "partial_noninductive_diagnostic"
            elif files:
                policy["execution_status"] = "partial"
                policy["result_disposition"] = "failed_or_incomplete_noninductive_diagnostic"
        modified = max((p.stat().st_mtime for p in files), default=cohort.stat().st_mtime)
        entry = {
            "experiment_id": cohort.name,
            "artifact_root": f"outputs/{cohort.name}",
            **policy,
            "artifact_summary": {
                "file_count": len(files),
                "bytes": sum(p.stat().st_size for p in files),
                "last_modified_utc": _iso_utc(modified),
                "run_records": len(runs),
                "completed_run_records": completed,
                "partial_or_blocked_run_records": partial,
            },
            "key_artifacts": _key_artifacts(files, outputs),
            "runs": runs,
        }
        cohorts.append(entry)

    root_files = sorted(
        p for p in outputs.iterdir()
        if p.is_file() and p.name not in GENERATED_OUTPUT_NAMES
    )
    return {
        "schema_version": 1,
        "scope": "Every top-level outputs cohort plus run-like subdirectories and root artifacts",
        "preservation_policy": {
            "never_overwrite_prior_cohort": True,
            "record_negative_failed_interrupted_and_running": True,
            "test_selection_forbidden": True,
            "historical_conventions_must_not_be_pooled": True,
            "artifact_index": "outputs/EXPERIMENT_ARTIFACT_INDEX.jsonl",
        },
        "cohort_count": len(cohorts),
        "cohorts": cohorts,
        "root_artifacts": [
            {
                "path": f"outputs/{p.name}",
                "bytes": p.stat().st_size,
                "last_modified_utc": _iso_utc(p.stat().st_mtime),
                "scope": (
                    "active exposure replay log" if p.name.startswith("exposure_matched_direct_u_v2")
                    else "unscoped early root-level artifact retained for provenance"
                ),
            }
            for p in root_files
        ],
    }


def build_artifact_rows(outputs: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((p for p in outputs.rglob("*") if p.is_file()), key=lambda p: p.as_posix().lower()):
        if path.name in GENERATED_OUTPUT_NAMES:
            continue
        relative = path.relative_to(outputs)
        parts = relative.parts
        cohort = parts[0] if len(parts) > 1 else "__root__"
        stat = path.stat()
        row: dict[str, Any] = {
            "cohort": cohort,
            "path": f"outputs/{relative.as_posix()}",
            "bytes": stat.st_size,
            "last_modified_utc": _iso_utc(stat.st_mtime),
        }
        if path.suffix.lower() in HASHABLE_SUFFIXES and stat.st_size <= 20 * 1024 * 1024 and path.suffix.lower() != ".log":
            row["sha256"] = _sha256(path)
        rows.append(row)
    return rows


def render_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# PiezoJet experiment registry",
        "",
        "This is the human-readable index of every top-level cohort under `outputs/`. "
        "The machine-readable registry contains subrun markers, seeds, passes, artifact pointers, "
        "and interpretation boundaries; the JSONL index inventories every persisted file.",
        "",
        "Negative, failed, interrupted, partial, running, and historical runs are intentionally retained. "
        "A directory's existence never implies a valid performance result.",
        "",
        f"Registered top-level cohorts: **{registry['cohort_count']}**.",
        "",
        "| Cohort | Family | Execution | Result | Convention | Paper use | Runs (complete/partial) |",
        "|---|---|---|---|---|---|---:|",
    ]
    for cohort in registry["cohorts"]:
        summary = cohort["artifact_summary"]
        run_count = f"{summary['run_records']} ({summary['completed_run_records']}/{summary['partial_or_blocked_run_records']})"
        values = [
            f"`{cohort['experiment_id']}`",
            cohort["family"],
            cohort["execution_status"],
            cohort["result_disposition"],
            cohort["convention_epoch"],
            cohort["paper_use"],
            run_count,
        ]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")

    lines.extend(
        [
            "",
            "## Non-comparable families",
            "",
            "Historical pre-v7, observable-lift/pInv, protocol A--G, sketch/mode-aware, and early-development "
            "cohorts remain evidence about prior hypotheses only. They must not be pooled with the "
            "v7 BEC-transpose regularized direct-U replay.",
            "",
            "## Current registered replay",
            "",
            "`exposure_matched_direct_u_v2_conditioning` is the registered 1/5/10/20-pass by "
            "seed 42/7/1729 grid. While it is running, every completed point remains an intermediate "
            "diagnostic. The final summary must include all planned points and may not select a pass from test data.",
            "",
            "## File-level preservation",
            "",
            "`outputs/EXPERIMENT_ARTIFACT_INDEX.jsonl` records every artifact path, byte size, and modification "
            "time. JSON, CSV, Markdown, YAML, and text records up to 20 MiB also receive SHA-256 hashes. "
            "Large checkpoints are retained and indexed by path/size/time without being duplicated.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_registry(registry: dict[str, Any], outputs: Path) -> list[str]:
    errors: list[str] = []
    actual = {p.name for p in outputs.iterdir() if p.is_dir()}
    registered = {entry["experiment_id"] for entry in registry["cohorts"]}
    if actual != registered:
        errors.append(f"cohort coverage mismatch: missing={sorted(actual - registered)}, stale={sorted(registered - actual)}")
    for entry in registry["cohorts"]:
        required = {"execution_status", "result_disposition", "convention_epoch", "paper_use", "interpretation_boundary"}
        missing = sorted(required - entry.keys())
        if missing:
            errors.append(f"{entry['experiment_id']}: missing fields {missing}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--json", type=Path, default=Path("outputs/EXPERIMENT_REGISTRY.json"))
    parser.add_argument("--markdown", type=Path, default=Path("EXPERIMENT_REGISTRY.md"))
    parser.add_argument("--artifact-index", type=Path, default=Path("outputs/EXPERIMENT_ARTIFACT_INDEX.jsonl"))
    parser.add_argument("--check", action="store_true", help="Validate coverage without writing files")
    args = parser.parse_args()

    registry = build_registry(args.outputs)
    errors = validate_registry(registry, args.outputs)
    if errors:
        raise SystemExit("\n".join(errors))
    if args.check:
        print(f"experiment registry coverage valid: {registry['cohort_count']} cohorts")
        return

    artifact_rows = build_artifact_rows(args.outputs)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_index.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(registry), encoding="utf-8")
    args.artifact_index.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in artifact_rows),
        encoding="utf-8",
    )
    print(f"registered {registry['cohort_count']} cohorts and {len(artifact_rows)} artifacts")


if __name__ == "__main__":
    main()
