#!/usr/bin/env bash
# PLAN 1.1 + 1.3: smoke test, unit tests, and the parity gate (must pass before training).
set -euo pipefail
source scripts/env.sh
DEV=${1:-data/manifests/dev_gold.jsonl}
python scripts/smoke_test.py --manifest "$DEV" --n 5
python -m pytest -q -m "not gpu"
SER_DEV_MANIFEST="$DEV" python -m pytest -q -s -m gpu tests/test_scoring_parity.py
