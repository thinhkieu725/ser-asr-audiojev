#!/usr/bin/env bash
# PLAN 1.1: zero-shot AudioJev over the whole dev set ("step 0" row), K = 1 and 3,
# with both prompt languages.
set -euo pipefail
source scripts/env.sh
for LANG in en vi; do
  for K in 1 3; do
    python -m sersys.predict --split dev --tta $K --set ser.prompt_lang=$LANG         --out runs/zeroshot_$LANG/dev_tta$K 2>&1 | tee runs/zeroshot_$LANG.dev_tta$K.log
  done
done
