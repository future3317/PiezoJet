#!/usr/bin/env bash
# Fold-specific, single-seed A0-PM gate.  This script never overwrites a run.
set -euo pipefail

ROOT=${PIEZOJET_ROOT:-/home/workspace/lrh/PiezoJet}
PY=${PIEZOJET_PYTHON:-/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python}
DATA_ROOT=${PIEZOJET_DATA_ROOT:-/home/workspace/lrh/DATA/PiezoJet}
FOLD=${1:?usage: $0 FOLD GPU}
GPU=${2:?usage: $0 FOLD GPU}
COMMIT=${PIEZOJET_CODE_COMMIT:?Set PIEZOJET_CODE_COMMIT to a 40-character source SHA}
GRAPH_CACHE_KEY=${PIEZOJET_GRAPH_CACHE_KEY:-333fe701095f973afbdb}
COHORT=${PIEZOJET_COHORT:-$ROOT/outputs/electrostatic_a0pm_threefold_seed42_v1}
FOLDS=$ROOT/data/processed/electrostatic_development_folds_v2.json
SUBSET=$COHORT/subsets/fold${FOLD}_balanced_n800_seed42.json
FOLD_ROOT=$COHORT/fold${FOLD}
STRUCTURAL=$FOLD_ROOT/structure_pretrain/best_encoder.pt
BEC=$FOLD_ROOT/bec_pretrain/best_bec_tower.pt
ELECTRONIC=$FOLD_ROOT/electronic_pretrain/best_electronic_tower.pt
A0=$FOLD_ROOT/a0_pm
LOG=$FOLD_ROOT/logs

export PYTHONPATH=$ROOT/src
export PIEZOJET_DATA_ROOT=$DATA_ROOT
mkdir -p "$FOLD_ROOT" "$LOG"

test -f "$FOLDS"
test -f "$SUBSET"
if [[ ! -f "$STRUCTURAL" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -m piezojet.pretrain_e3nn \
    --config "$ROOT/config.yaml" --electrostatic-folds "$FOLDS" --fold "$FOLD" \
    --seed 42 --output-dir "$FOLD_ROOT/structure_pretrain" --epochs 20 \
    --batch-size 2 --logical-batch-size 32 --encoder-width-multiplier 0.56 \
    --num-workers 0 --prefetch-factor 2 --matmul-precision highest \
    --code-commit "$COMMIT" > "$LOG/structure_pretrain.log" 2>&1
fi
test -f "$STRUCTURAL"

if [[ ! -f "$BEC" ]]; then
  BEC_RESUME=()
  if [[ -f "$FOLD_ROOT/bec_pretrain/last_bec_tower.pt" && ! -f "$FOLD_ROOT/bec_pretrain/history.json" ]]; then
    BEC_RESUME=(--resume "$FOLD_ROOT/bec_pretrain/last_bec_tower.pt")
  fi
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -m piezojet.pretrain_bec_e3nn \
    --config "$ROOT/config.yaml" --folds "$FOLDS" --fold "$FOLD" \
    --output-dir "$FOLD_ROOT/bec_pretrain" --epochs 20 --batch-size 16 \
    --logical-batch-size 32 --learning-rate 1e-3 --seed 42 --device cuda \
    --num-workers 0 --prefetch-factor 2 --cache-graphs --matmul-precision highest \
    "${BEC_RESUME[@]}" \
    --code-commit "$COMMIT" > "$LOG/bec_pretrain.log" 2>&1
fi
test -f "$BEC"

if [[ ! -f "$ELECTRONIC" ]]; then
  ELECTRONIC_RESUME=()
  if [[ -f "$FOLD_ROOT/electronic_pretrain/last_electronic_tower.pt" && ! -f "$FOLD_ROOT/electronic_pretrain/history.json" ]]; then
    ELECTRONIC_RESUME=(--resume "$FOLD_ROOT/electronic_pretrain/last_electronic_tower.pt")
  fi
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -m piezojet.pretrain_electronic_e3nn \
    --config "$ROOT/config.yaml" --folds "$FOLDS" --fold "$FOLD" \
    --output-dir "$FOLD_ROOT/electronic_pretrain" --epochs 20 \
    --train-ids-file "$SUBSET" --graph-cache-key "$GRAPH_CACHE_KEY" \
    --batch-size 16 --logical-batch-size 32 --learning-rate 1e-3 --seed 42 \
    --device cuda --num-workers 0 --prefetch-factor 2 --cache-graphs \
    "${ELECTRONIC_RESUME[@]}" \
    --matmul-precision highest --code-commit "$COMMIT" \
    > "$LOG/electronic_pretrain.log" 2>&1
fi
test -f "$ELECTRONIC"

if [[ ! -f "$A0/selected.pt" ]]; then
  A0_RESUME=()
  if [[ -f "$A0/progress.pt" ]]; then
    A0_RESUME=(--resume "$A0/progress.pt" --allow-runtime-resume)
  elif [[ -d "$A0" ]]; then
    # A process can be terminated before the first checkpoint.  Remove only
    # an actually empty staging directory; never erase a partial run that has
    # evidence which should be resumed or audited.
    rmdir "$A0" 2>/dev/null || {
      echo "A0 output exists without selected.pt/progress.pt; refusing overwrite: $A0" >&2
      exit 1
    }
  fi
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -m piezojet.electrostatic_a0_fold_adjudication \
  --config "$ROOT/config.yaml" --folds "$FOLDS" --fold "$FOLD" \
  --architecture a0_parameter_matched_irreps --output-dir "$A0" \
  --updates 1500 --batch-size 16 --microbatch-size 16 --eval-batch-size 64 \
  --diagnostic-batch-size 16 --eval-interval 50 \
  --train-eval-interval 250 --early-stopping-patience-evaluations 4 \
  --early-stopping-minimum-improvement 0.0 --train-ids-file "$SUBSET" \
  --development-limit 0 --pretrained-encoder "$STRUCTURAL" \
  --bec-pretrained-tower "$BEC" --electronic-pretrained-tower "$ELECTRONIC" \
  --graph-cache-key "$GRAPH_CACHE_KEY" --seed 42 --device cuda --num-workers 0 \
  --matmul-precision highest --code-commit "$COMMIT" \
  "${A0_RESUME[@]}" > "$LOG/a0_pm.log" 2>&1
fi

test -f "$A0/selected.pt"
printf 'fold=%s status=complete output=%s\n' "$FOLD" "$A0"
