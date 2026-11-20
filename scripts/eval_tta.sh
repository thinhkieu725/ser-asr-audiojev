#!/usr/bin/env bash
# PLAN 1.9: dev metrics of a trained run for K in {1, 3, 7}, using its fitted temperature.
# Usage: scripts/eval_tta.sh runs/<exp>/<seed>
set -euo pipefail
source scripts/env.sh
RUN=$1
for K in 1 3 7; do
  python -m sersys.predict --config $RUN/config.yaml --adapter $RUN/best --split dev --tta $K --out $RUN/best/dev_tta$K
done
