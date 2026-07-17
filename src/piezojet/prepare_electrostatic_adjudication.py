"""Prepare, but never execute, matched first-order jet adjudications."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .electrostatic_protocol import ARCHITECTURES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(
    *,
    folds_path: Path,
    config_path: Path,
    cohort_root: Path,
    fold_index: int,
    seed: int,
    train_limit: int,
    development_limit: int,
    pretrain_epochs: int,
    updates: int,
    batch_size: int,
    eval_interval: int,
    pretrain_batch_size: int = 0,
    microbatch_size: int = 0,
    evaluation_batch_size: int = 0,
    diagnostic_batch_size: int = 0,
) -> dict[str, object]:
    """Return a leak-safe argv plan whose commands require later execution."""
    if min(pretrain_epochs, updates, batch_size, eval_interval) < 1:
        raise ValueError("Epochs, updates, batch size, and eval interval must be positive")
    if train_limit < 0 or development_limit < 0:
        raise ValueError("Material limits cannot be negative")
    if min(pretrain_batch_size, microbatch_size, evaluation_batch_size, diagnostic_batch_size) < 0:
        raise ValueError("Microbatch and evaluation batch sizes cannot be negative")
    effective_microbatch = microbatch_size or batch_size
    if effective_microbatch > batch_size or batch_size % effective_microbatch:
        raise ValueError("Batch size must be divisible by microbatch size")
    folds = json.loads(folds_path.read_text(encoding="utf-8-sig"))
    fold = next(
        (entry for entry in folds["folds"] if entry["fold"] == fold_index),
        None,
    )
    if fold is None:
        raise ValueError(f"Fold {fold_index} is absent from {folds_path}")
    if folds.get("frozen_validation_test_labels_read") is not False:
        raise ValueError("Fold manifest must explicitly certify frozen labels unread")

    python = r"D:\Anaconda\envs\EGNN\python.exe"
    pretrain_dir = cohort_root / (
        f"stage_a_n{train_limit or 'full'}_fold{fold_index}_seed{seed}_pretrain"
    )
    candidate_dirs = {
        architecture: cohort_root
        / f"stage_a_n{train_limit or 'full'}_fold{fold_index}_{architecture}_seed{seed}"
        for architecture in ARCHITECTURES
    }
    all_output_dirs = [pretrain_dir, *candidate_dirs.values()]
    existing = [str(path) for path in all_output_dirs if path.exists()]
    if existing:
        raise FileExistsError(
            "Planned run directories must be fresh: " + ", ".join(existing)
        )

    common = [
        "--config", str(config_path),
        "--electrostatic-folds", str(folds_path),
        "--fold", str(fold_index),
        "--train-limit", str(train_limit),
        "--seed", str(seed),
    ]
    pretrain_command = [
        python, "-m", "piezojet.pretrain_e3nn", *common,
        "--output-dir", str(pretrain_dir),
        "--epochs", str(pretrain_epochs),
        "--batch-size", str(pretrain_batch_size or batch_size),
    ]
    candidate_commands = []
    for architecture in ARCHITECTURES:
        candidate_commands.append({
            "architecture": architecture,
            "output_dir": str(candidate_dirs[architecture]),
            "argv": [
                python, "-m", "piezojet.electrostatic_fold_adjudication",
                "--config", str(config_path),
                "--folds", str(folds_path),
                "--fold", str(fold_index),
                "--architecture", architecture,
                "--output-dir", str(candidate_dirs[architecture]),
                "--updates", str(updates),
                "--batch-size", str(batch_size),
                "--microbatch-size", str(effective_microbatch),
                "--eval-batch-size", str(evaluation_batch_size or effective_microbatch),
                "--diagnostic-batch-size", str(diagnostic_batch_size or effective_microbatch),
                "--eval-interval", str(eval_interval),
                "--train-limit", str(train_limit),
                "--development-limit", str(development_limit),
                "--pretrained-encoder", str(pretrain_dir / "best_encoder.pt"),
                "--seed", str(seed),
                "--device", "cuda",
            ],
        })

    return {
        "schema": 1,
        "status": "planned_not_executed",
        "purpose": "matched formula-disjoint A0/A1/A1.5 first-order jet adjudication",
        "execution_authorization": "requires a later explicit user request to resume training",
        "environment": {
            "python": python,
            "pythonpath": str(Path.cwd() / "src"),
            "num_workers": 0,
        },
        "data_boundary": {
            "fold_manifest": str(folds_path),
            "fold_manifest_sha256": _sha256(folds_path),
            "fold": fold_index,
            "development_population": len(fold["development"]),
            "train_limit": train_limit,
            "development_limit": development_limit,
            "frozen_validation_test_labels_read": False,
            "samples32_checkpoint_used": False,
        },
        "comparison_contract": {
            "same_fold": True,
            "same_seed": True,
            "same_train_and_development_subsets": True,
            "same_structure_pretraining_state": True,
            "random_response_heads": True,
            "same_updates_and_batches": True,
            "same_logical_batch_and_microbatch_schedule": True,
            "fixed_response_active_gradient_panel": True,
            "gradient_panel_norm_stratified": (diagnostic_batch_size or effective_microbatch) > 1,
            "primary_selection": (
                "minimum development electronic stabilized-relative error plus "
                "Born mean-relative error"
            ),
            "report_together": [
                "target and prediction norms",
                "relative Frobenius error",
                "cosine",
                "amplitude ratio",
                "irrep-resolved electronic error",
                "BEC acoustic leakage",
                "all/shared task gradient norms and cosine",
                "peak CUDA allocation and optimizer seconds",
            ],
            "automatic_production_promotion": False,
            "automatic_dataset_expansion": False,
            "logical_batch_size": batch_size,
            "pretrain_batch_size": pretrain_batch_size or batch_size,
            "microbatch_size": effective_microbatch,
            "evaluation_batch_size": evaluation_batch_size or effective_microbatch,
            "diagnostic_batch_size": diagnostic_batch_size or effective_microbatch,
        },
        "steps": [
            {"name": "fold_train_only_structure_pretraining", "argv": pretrain_command},
            *candidate_commands,
            {
                "name": "register_and_compare_after_all_candidates_finish",
                "argv": [
                    python, "-m", "piezojet.experiment_registry"
                ],
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds", type=Path,
        default=Path("data/processed/electrostatic_development_folds_v2.json"),
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--cohort-root", type=Path,
        default=Path("outputs/electromechanical_jet_fold_adjudication"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--development-limit", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pretrain-batch-size", type=int, default=0)
    parser.add_argument("--microbatch-size", type=int, default=0)
    parser.add_argument("--evaluation-batch-size", type=int, default=0)
    parser.add_argument("--diagnostic-batch-size", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=25)
    args = parser.parse_args()
    plan = build_plan(
        folds_path=args.folds,
        config_path=args.config,
        cohort_root=args.cohort_root,
        fold_index=args.fold,
        seed=args.seed,
        train_limit=args.train_limit,
        development_limit=args.development_limit,
        pretrain_epochs=args.pretrain_epochs,
        updates=args.updates,
        batch_size=args.batch_size,
        pretrain_batch_size=args.pretrain_batch_size,
        microbatch_size=args.microbatch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        diagnostic_batch_size=args.diagnostic_batch_size,
        eval_interval=args.eval_interval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
