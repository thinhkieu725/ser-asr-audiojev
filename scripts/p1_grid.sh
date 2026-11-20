#!/usr/bin/env bash
# PLAN 1.10: Phase 1 grid, seed 1. Edit the E1.3+ configs to carry the winners forward
# before running them (see the comments inside each config).
set -euo pipefail
STAGE=${1:-a}
case $STAGE in
  a) for c in p1_e1_frozen_smoothce p1_e2_frozen_smoothce_vi; do scripts/train.sh configs/exp/$c.yaml 1; done ;;
  b) for c in p1_e3_frozen_ce p1_e4_smallloss_0.1 p1_e4_smallloss_0.2; do scripts/train.sh configs/exp/$c.yaml 1; done ;;
  c) for c in p1_e5_unfreeze4 p1_e6_unfreeze8; do scripts/train.sh configs/exp/$c.yaml 1; done ;;
  d) for c in p1_e7_noperm p1_e8_lora_r16 p1_e8_lora_r64; do scripts/train.sh configs/exp/$c.yaml 1; done ;;
  seeds) for c in "${@:2}"; do for s in 2 3; do scripts/train.sh configs/exp/$c.yaml $s; done; done ;;
esac
