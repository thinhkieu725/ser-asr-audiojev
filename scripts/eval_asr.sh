#!/usr/bin/env bash
# PLAN 2.4: greedy ASR on dev (CER, syllable WER) for a trained run, or zero-shot without a run.
# Usage: scripts/eval_asr.sh runs/<exp>/<seed>      |   scripts/eval_asr.sh zeroshot
set -euo pipefail
source scripts/env.sh
if [ "$1" = "zeroshot" ]; then
  python -m sersys.asr --split dev --out runs/zeroshot_asr/dev
else
  python -m sersys.asr --config $1/config.yaml --adapter $1/best --split dev --out $1/best/dev_asr
fi
