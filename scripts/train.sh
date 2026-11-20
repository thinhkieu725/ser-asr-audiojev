#!/usr/bin/env bash
# Usage: scripts/train.sh configs/exp/p1_e1_frozen_smoothce.yaml [seed] [--set key=value ...]
set -euo pipefail
source scripts/env.sh
CFG=$1; SEED=${2:-1}; shift; shift || true
accelerate launch --num_processes 1 --mixed_precision bf16 -m sersys.train --config "$CFG" --seed "$SEED" "$@"
