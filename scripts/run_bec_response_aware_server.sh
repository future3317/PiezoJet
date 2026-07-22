#!/usr/bin/env bash
# One-seed, development-only BEC response-aware initialization gate.
set -euo pipefail

ROOT=${PIEZOJET_ROOT:-/home/workspace/lrh/PiezoJet}
PY=${PIEZOJET_PYTHON:-/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python}
DATA_ROOT=${PIEZOJET_DATA_ROOT:-/home/workspace/lrh/DATA/PiezoJet}
COHORT=${PIEZOJET_COHORT:-$ROOT/outputs/bec_response_pretraining_a0pm_fold0_seed42_v1}
FOLDS=$ROOT/data/processed/electrostatic_development_folds_v2.json
SUBSET=$DATA_ROOT/processed/electrostatic_balanced_subsets_v1/fold0_balanced_n800_seed42.json
STRUCTURAL=$ROOT/outputs/vnext_stage_a_hierarchical_fairness_server_v1_correct/stage_a_full_fold0_seed42_pretrain_a0_parameter_matched_v2/best_encoder.pt
COMMIT=${PIEZOJET_CODE_COMMIT:?Set PIEZOJET_CODE_COMMIT to the committed source SHA.}
GPU=${PIEZOJET_GPU:-0}

export PYTHONPATH=$ROOT/src
export PIEZOJET_DATA_ROOT=$DATA_ROOT

test -f "$FOLDS"
test -f "$SUBSET"
test -f "$STRUCTURAL"
test -d "$COHORT"
test ! -e "$COHORT/bec_pretrain_full"
test ! -e "$COHORT/a0_pm_bec_response_aware_n800"
mkdir -p "$COHORT/logs"

CUDA_VISIBLE_DEVICES=$GPU "$PY" -m piezojet.pretrain_bec_e3nn \
  --config "$ROOT/config.yaml" --folds "$FOLDS" --fold 0 \
  --output-dir "$COHORT/bec_pretrain_full" --epochs 20 \
  --batch-size 16 --logical-batch-size 32 --learning-rate 1e-3 \
  --seed 42 --device cuda --code-commit "$COMMIT" \
  --num-workers 0 --matmul-precision high \
  > "$COHORT/logs/bec_pretrain.log" 2>&1

test -f "$COHORT/bec_pretrain_full/best_bec_tower.pt"

CUDA_VISIBLE_DEVICES=$GPU "$PY" -m piezojet.electrostatic_a0_fold_adjudication \
  --config "$ROOT/config.yaml" --folds "$FOLDS" --fold 0 \
  --architecture a0_parameter_matched_irreps \
  --output-dir "$COHORT/a0_pm_bec_response_aware_n800" \
  --updates 1500 --batch-size 16 --microbatch-size 16 --eval-batch-size 64 \
  --diagnostic-batch-size 16 --eval-interval 50 \
  --early-stopping-patience-evaluations 4 \
  --early-stopping-minimum-improvement 0.0 \
  --train-ids-file "$SUBSET" --development-limit 0 \
  --pretrained-encoder "$STRUCTURAL" \
  --bec-pretrained-tower "$COHORT/bec_pretrain_full/best_bec_tower.pt" \
  --seed 42 --device cuda --num-workers 0 --matmul-precision high \
  --code-commit "$COMMIT" \
  > "$COHORT/logs/a0_pm_bec_response_aware.log" 2>&1
