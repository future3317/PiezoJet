#!/usr/bin/env bash
# One-seed electronic-only response-aware initialization gate for A0-PM.
set -euo pipefail

ROOT=${PIEZOJET_ROOT:-/home/workspace/lrh/PiezoJet}
PY=${PIEZOJET_PYTHON:-/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python}
DATA_ROOT=${PIEZOJET_DATA_ROOT:-/home/workspace/lrh/DATA/PiezoJet}
COHORT=${PIEZOJET_COHORT:-$ROOT/outputs/electronic_response_pretraining_a0pm_fold0_seed42_v1}
FOLDS=$ROOT/data/processed/electrostatic_development_folds_v2.json
SUBSET=$DATA_ROOT/processed/electrostatic_balanced_subsets_v1/fold0_balanced_n800_seed42.json
STRUCTURAL=$ROOT/outputs/vnext_stage_a_hierarchical_fairness_server_v1_correct/stage_a_full_fold0_seed42_pretrain_a0_parameter_matched_v2/best_encoder.pt
BEC=$ROOT/outputs/bec_response_pretraining_a0pm_fold0_seed42_v1/bec_pretrain_full/best_bec_tower.pt
COMMIT=${PIEZOJET_CODE_COMMIT:?Set PIEZOJET_CODE_COMMIT to the committed source SHA.}
GPU=${PIEZOJET_GPU:-5}
GRAPH_CACHE_KEY=${PIEZOJET_GRAPH_CACHE_KEY:-333fe701095f973afbdb}

export PYTHONPATH=$ROOT/src
export PIEZOJET_DATA_ROOT=$DATA_ROOT

test -f "$FOLDS"
test -f "$SUBSET"
test -f "$STRUCTURAL"
test -f "$BEC"
test ! -e "$COHORT"
mkdir -p "$COHORT/logs"

CUDA_VISIBLE_DEVICES=$GPU "$PY" -m piezojet.pretrain_electronic_e3nn \
  --config "$ROOT/config.yaml" --folds "$FOLDS" --fold 0 \
  --output-dir "$COHORT/electronic_pretrain_full" --epochs 20 \
  --train-ids-file "$SUBSET" \
  --graph-cache-key "$GRAPH_CACHE_KEY" \
  --batch-size 16 --logical-batch-size 32 --learning-rate 1e-3 \
  --seed 42 --device cuda --code-commit "$COMMIT" \
  --num-workers 0 --matmul-precision highest \
  > "$COHORT/logs/electronic_pretrain.log" 2>&1

test -f "$COHORT/electronic_pretrain_full/best_electronic_tower.pt"

CUDA_VISIBLE_DEVICES=$GPU "$PY" -m piezojet.electrostatic_a0_fold_adjudication \
  --config "$ROOT/config.yaml" --folds "$FOLDS" --fold 0 \
  --architecture a0_parameter_matched_irreps \
  --output-dir "$COHORT/a0_pm_electronic_response_aware_n800" \
  --updates 1500 --batch-size 16 --microbatch-size 16 --eval-batch-size 64 \
  --diagnostic-batch-size 16 --eval-interval 50 \
  --early-stopping-patience-evaluations 4 \
  --early-stopping-minimum-improvement 0.0 \
  --train-ids-file "$SUBSET" --development-limit 0 \
  --pretrained-encoder "$STRUCTURAL" \
  --bec-pretrained-tower "$BEC" \
  --electronic-pretrained-tower \
    "$COHORT/electronic_pretrain_full/best_electronic_tower.pt" \
  --seed 42 --device cuda --num-workers 0 --matmul-precision highest \
  --code-commit "$COMMIT" \
  > "$COHORT/logs/a0_pm_electronic_response_aware.log" 2>&1
