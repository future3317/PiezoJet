#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/workspace/lrh/PiezoJet
PY=/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python
COHORT="$ROOT/outputs/vnext_stage_a_hierarchical_fairness_server_v1_correct"
FOLDS="$ROOT/data/processed/electrostatic_development_folds_v2.json"
SUBSET=/home/workspace/lrh/DATA/PiezoJet/processed/electrostatic_balanced_subsets_v1/fold0_balanced_n800_seed42.json
FULL="$COHORT/provenance/full_width_best_encoder_rebound_split_snapshot.pt"
PM="$COHORT/stage_a_full_fold0_seed42_pretrain_a0_parameter_matched_v2/best_encoder.pt"
SHA=9c35cb5799525bbab68a760c5cb3332a45992f0c
WORKERS=${PIEZOJET_NUM_WORKERS:-0}
MICROBATCH=${PIEZOJET_MICROBATCH_SIZE:-16}
EVAL_BATCH=${PIEZOJET_EVAL_BATCH_SIZE:-64}
MATMUL_PRECISION=${PIEZOJET_MATMUL_PRECISION:-high}

export PYTHONPATH="$ROOT/src"
export PIEZOJET_DATA_ROOT=/home/workspace/lrh/DATA/PiezoJet
mkdir -p "$COHORT/logs"

run_candidate() {
  local gpu="$1" name="$2" module="$3" architecture="$4" checkpoint="$5" output="$6"
  local output_dir="$COHORT/$output"
  local resume_args=()
  if test -f "$output_dir/progress_runtime_microbatch16.pt"; then
    resume_args=(--resume "$output_dir/progress_runtime_microbatch16.pt")
  elif test -f "$output_dir/progress.pt"; then
    resume_args=(--resume "$output_dir/progress.pt")
  fi
  env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m "$module" \
    --config "$ROOT/config.yaml" --folds "$FOLDS" --fold 0 \
    --architecture "$architecture" --output-dir "$output_dir" \
    --updates 1500 --batch-size 16 --microbatch-size "$MICROBATCH" \
    --eval-batch-size "$EVAL_BATCH" --num-workers "$WORKERS" \
    --matmul-precision "$MATMUL_PRECISION" \
    --diagnostic-batch-size 16 --eval-interval 100 \
    --early-stopping-patience-evaluations 0 \
    --early-stopping-minimum-improvement 0.0 \
    --train-ids-file "$SUBSET" --development-limit 0 \
    --pretrained-encoder "$checkpoint" --seed 42 --device cuda \
    --code-commit "$SHA" "${resume_args[@]}" \
    > "$COHORT/logs/${name}_optimized.log" 2>&1
}

run_gpu3() {
  test -f "$FULL"
  run_candidate 3 02_a0_independent_irreps piezojet.electrostatic_a0_fold_adjudication \
    a0_independent_irreps "$FULL" stage_a_n800_fold0_a0_independent_irreps_seed42
  run_candidate 3 04_a1_electromechanical_jet piezojet.electrostatic_fold_adjudication \
    a1_electromechanical_jet "$FULL" stage_a_n800_fold0_a1_electromechanical_jet_seed42
}

run_gpu4() {
  local last="$COHORT/stage_a_full_fold0_seed42_pretrain_a0_parameter_matched_v2/last_encoder.pt"
  local epoch
  epoch=$("$PY" -c "import torch; print(torch.load('$last', map_location='cpu', weights_only=False)['epoch'])")
  if (( epoch < 20 )); then
    env CUDA_VISIBLE_DEVICES=4 "$PY" -m piezojet.pretrain_e3nn \
      --config "$ROOT/config.yaml" --electrostatic-folds "$FOLDS" --fold 0 \
      --seed 42 --output-dir "$COHORT/stage_a_full_fold0_seed42_pretrain_a0_parameter_matched_v2" \
      --epochs 20 --batch-size 2 --logical-batch-size 32 \
      --encoder-width-multiplier 0.56 --num-workers "$WORKERS" \
      --matmul-precision "$MATMUL_PRECISION" --resume "$last" \
      --code-commit "$SHA" \
      > "$COHORT/logs/01_parameter_matched_pretrain_v2_optimized.log" 2>&1
  fi
  test -f "$PM"
  run_candidate 4 03_a0_parameter_matched_irreps piezojet.electrostatic_a0_fold_adjudication \
    a0_parameter_matched_irreps "$PM" stage_a_n800_fold0_a0_parameter_matched_irreps_seed42
  run_candidate 4 05_a16_hierarchical_electromechanical_jet piezojet.electrostatic_fold_adjudication \
    a16_hierarchical_electromechanical_jet "$FULL" stage_a_n800_fold0_a16_hierarchical_electromechanical_jet_seed42
}

run_gpu3 &
gpu3_pid=$!
run_gpu4 &
gpu4_pid=$!
wait "$gpu3_pid"
wait "$gpu4_pid"

"$PY" -m piezojet.experiment_registry > "$COHORT/logs/06_registry.log" 2>&1
