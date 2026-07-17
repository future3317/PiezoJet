from pathlib import Path

from piezojet.experiment_registry import (
    build_artifact_rows,
    build_registry,
    validate_registry,
)


def test_registry_covers_cohorts_runs_and_artifacts(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    complete = outputs / "direct_u_multistream_smoke_v1" / "seed42"
    complete.mkdir(parents=True)
    (complete / "summary.json").write_text("{}", encoding="utf-8")
    (complete / "dfpt_test.json").write_text("{}", encoding="utf-8")
    interrupted = outputs / "m3" / "dry_run_seed_42"
    interrupted.mkdir(parents=True)
    (interrupted / "INTERRUPTED.md").write_text("stopped", encoding="utf-8")

    registry = build_registry(outputs)

    assert validate_registry(registry, outputs) == []
    assert registry["cohort_count"] == 2
    direct = next(item for item in registry["cohorts"] if item["experiment_id"] == "direct_u_multistream_smoke_v1")
    assert direct["result_disposition"] == "negative_one_pass_diagnostic"
    assert direct["runs"][0]["seed"] == 42
    assert direct["runs"][0]["execution_status"] == "completed_with_evaluation"
    m3 = next(item for item in registry["cohorts"] if item["experiment_id"] == "m3")
    assert m3["execution_status"] == "interrupted"
    assert m3["runs"][0]["execution_status"] == "interrupted"

    rows = build_artifact_rows(outputs)
    assert {row["path"] for row in rows} == {
        "outputs/direct_u_multistream_smoke_v1/seed42/summary.json",
        "outputs/direct_u_multistream_smoke_v1/seed42/dfpt_test.json",
        "outputs/m3/dry_run_seed_42/INTERRUPTED.md",
    }
    assert all("sha256" in row for row in rows)


def test_registered_exposure_grid_is_pending_not_a_result(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run = outputs / "exposure_matched_direct_u_v2_conditioning" / "physical" / "passes1_seed42"
    run.mkdir(parents=True)
    (run / "loss_best.pt").write_bytes(b"checkpoint")

    registry = build_registry(outputs)
    entry = registry["cohorts"][0]

    assert entry["execution_status"] == "running"
    assert entry["result_disposition"] == "pending_registered_decision"
    assert entry["preregistered_passes"] == [1, 5, 10, 20]
    run_entry = next(item for item in entry["runs"] if item["path"] == "physical/passes1_seed42")
    assert run_entry["execution_status"] == "running_or_pending_evaluation"
    assert run_entry["passes"] == 1
    assert run_entry["seed"] == 42


def test_failure_json_preserves_explicit_user_interruption(tmp_path: Path) -> None:
    run = (
        tmp_path / "outputs" / "electromechanical_jet_fold_adjudication"
        / "pilot_n100_fold0_a0_seed42"
    )
    run.mkdir(parents=True)
    (run / "failure.json").write_text(
        '{"status":"interrupted","reason":"user_requested_training_pause"}\n',
        encoding="utf-8",
    )

    entry = build_registry(tmp_path / "outputs")["cohorts"][0]

    assert entry["runs"][0]["execution_status"] == "interrupted"
    assert entry["artifact_summary"]["partial_or_blocked_run_records"] == 1


def test_exited_incomplete_replay_is_not_left_running(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    cohort = outputs / "exposure_matched_direct_u_v2_conditioning"
    run = cohort / "physical" / "passes1_seed42"
    run.mkdir(parents=True)
    (run / "loss_best.pt").write_bytes(b"checkpoint")
    (cohort / "experiment_status_history.jsonl").write_text(
        '{"status":"running"}\n'
        '{"status":"incomplete_after_process_exit","detail":"1/24"}\n',
        encoding="utf-8",
    )

    entry = build_registry(outputs)["cohorts"][0]

    assert entry["execution_status"] == "failed_or_interrupted"
    assert entry["result_disposition"] == "incomplete_no_registered_result"
    run_entry = next(item for item in entry["runs"] if item["path"] == "physical/passes1_seed42")
    assert run_entry["execution_status"] == "partial_no_evaluation"


def test_teacher_forced_capacity_ladder_requires_all_1_8_32_probes(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    cohort = outputs / "teacher_forced_zero_basin_v1"
    for count in (1, 8):
        run = cohort / f"samples{count}"
        run.mkdir(parents=True)
        (run / "overfit_dfpt_train.json").write_text("{}", encoding="utf-8")

    entry = build_registry(outputs)["cohorts"][0]

    assert entry["expected_completed_probes"] == 3
    assert entry["completed_probes"] == 2
    assert entry["execution_status"] == "partial"
