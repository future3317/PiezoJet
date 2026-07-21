#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/workspace/lrh/PiezoJet
PY=/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python
COHORT="$ROOT/outputs/vnext_stage_a_hierarchical_fairness_server_v1_correct"
FOLDS="$ROOT/data/processed/electrostatic_development_folds_v2.json"
SUBSET=/home/workspace/lrh/DATA/PiezoJet/processed/electrostatic_balanced_subsets_v1/fold0_balanced_n800_seed42.json
FULL="$COHORT/stage_a_full_fold0_seed42_pretrain/best_encoder.pt"
PM="$COHORT/stage_a_full_fold0_seed42_pretrain_a0_parameter_matched/best_encoder.pt"
SHA=9c35cb5799525bbab68a760c5cb3332a45992f0c

export PYTHONPATH="$ROOT/src"
export PIEZOJET_DATA_ROOT=/home/workspace/lrh/DATA/PiezoJet
export CUDA_VISIBLE_DEVICES=3
mkdir -p "$COHORT/logs"

full_pid="${1:?full pretraining PID is required}"
while kill -0 "$full_pid" >/dev/null 2>&1; do sleep 30; done
test -f "$FULL"

"$PY" -m piezojet.pretrain_e3nn \
  --config "$ROOT/config.yaml" --electrostatic-folds "$FOLDS" --fold 0 --seed 42 \
  --output-dir "$COHORT/stage_a_full_fold0_seed42_pretrain_a0_parameter_matched" \
  --epochs 20 --batch-size 2 --logical-batch-size 32 \
  --encoder-width-multiplier 0.56 --code-commit "$SHA" \
  > "$COHORT/logs/01_parameter_matched_pretrain.log" 2>&1
test -f "$PM"

run_candidate() {
  local name="$1" module="$2" architecture="$3" checkpoint="$4" output="$5"
  "$PY" -m "$module" --config "$ROOT/config.yaml" --folds "$FOLDS" --fold 0 \
    --architecture "$architecture" --output-dir "$COHORT/$output" \
    --updates 1500 --batch-size 16 --microbatch-size 8 --eval-batch-size 32 \
    --diagnostic-batch-size 16 --eval-interval 100 \
    --early-stopping-patience-evaluations 0 \
    --early-stopping-minimum-improvement 0.0 \
    --train-ids-file "$SUBSET" --development-limit 0 \
    --pretrained-encoder "$COHORT/$checkpoint" --seed 42 --device cuda \
    --code-commit "$SHA" > "$COHORT/logs/$name.log" 2>&1
}

run_candidate 02_a0_independent_irreps piezojet.electrostatic_a0_fold_adjudication \
  a0_independent_irreps stage_a_full_fold0_seed42_pretrain/best_encoder.pt \
  stage_a_n800_fold0_a0_independent_irreps_seed42
run_candidate 03_a0_parameter_matched_irreps piezojet.electrostatic_a0_fold_adjudication \
  a0_parameter_matched_irreps stage_a_full_fold0_seed42_pretrain_a0_parameter_matched/best_encoder.pt \
  stage_a_n800_fold0_a0_parameter_matched_irreps_seed42
run_candidate 04_a1_electromechanical_jet piezojet.electrostatic_fold_adjudication \
  a1_electromechanical_jet stage_a_full_fold0_seed42_pretrain/best_encoder.pt \
  stage_a_n800_fold0_a1_electromechanical_jet_seed42
run_candidate 05_a16_hierarchical_electromechanical_jet piezojet.electrostatic_fold_adjudication \
  a16_hierarchical_electromechanical_jet stage_a_full_fold0_seed42_pretrain/best_encoder.pt \
  stage_a_n800_fold0_a16_hierarchical_electromechanical_jet_seed42

"$PY" -m piezojet.experiment_registry > "$COHORT/logs/06_registry.log" 2>&1
