#!/usr/bin/env bash
# PLAN 2.5: Phase 2 grid from configs/exp/p1_best.yaml (fill that in first).
#   scripts/p2_grid.sh a                 E2.0-E2.3, seed 1
#   scripts/p2_grid.sh seeds <exp> ...   seeds 2-3 for the given configs (best lambda and lambda = 0)
#   scripts/p2_grid.sh e4                E2.4 (optional; set lambda/schedule to the winner first)
set -euo pipefail
STAGE=${1:-a}
case $STAGE in
  a) for c in p2_e0_lambda0 p2_e1_lambda0.1 p2_e2_lambda0.3 p2_e3_lambda0.3to0.05; do
       scripts/train.sh configs/exp/$c.yaml 1
     done ;;
  e4) scripts/train.sh configs/exp/p2_e4_transcribe_then_classify.yaml 1 ;;
  seeds) for c in "${@:2}"; do for s in 2 3; do scripts/train.sh configs/exp/$c.yaml $s; done; done ;;
esac
