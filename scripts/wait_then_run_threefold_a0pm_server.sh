#!/usr/bin/env bash
# Wait for an already-running fold structure pretrainer, then continue safely.
set -euo pipefail

ROOT=${PIEZOJET_ROOT:-/home/workspace/lrh/PiezoJet}
COHORT=${PIEZOJET_COHORT:-$ROOT/outputs/electrostatic_a0pm_threefold_seed42_v1}
FOLD=${1:?usage: $0 FOLD GPU}
GPU=${2:?usage: $0 FOLD GPU}
STRUCTURAL=$COHORT/fold${FOLD}/structure_pretrain/best_encoder.pt
LOG=$COHORT/fold${FOLD}/logs
mkdir -p "$LOG"
while [[ ! -f "$STRUCTURAL" ]]; do
  sleep 60
done
exec "$ROOT/scripts/run_threefold_a0pm_seed42_server.sh" "$FOLD" "$GPU"
