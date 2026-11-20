#!/usr/bin/env bash
# PLAN 4.1-4.3 for the frozen final configuration.
#   scripts/p4_final.sh train            train configs/exp/final.yaml with seeds 1-3 (T is fitted on dev)
#   scripts/p4_final.sh test [adapter]   test evaluation of each seed, ONCE (adapter: best | avg5)
#   scripts/p4_final.sh ablations        test evaluation of the ablation rows listed below
#   scripts/p4_final.sh tables           tables/test_results.{md,tex}
#   scripts/p4_final.sh export [seed] [adapter]   merged model in exports/final (+ validation, latency)
set -euo pipefail
source scripts/env.sh
STAGE=${1:-}
case $STAGE in
  train)
    for s in 1 2 3; do scripts/train.sh configs/exp/final.yaml $s; done ;;
  test)
    ADAPTER=${2:-best}
    for s in 1 2 3; do python -m sersys.evaluate --run_dir runs/final/$s --adapter $ADAPTER; done ;;
  ablations)
    # PLACEHOLDER: list the ablation runs (runs/<exp>/<seed>) that go into the paper table.
    python -m sersys.evaluate --run_dir runs/zeroshot_en/1 --zeroshot --config configs/base.yaml \
        --set ser.prompt_lang=en --fit_temperature_on_dev
    for r in runs/p1_best/1 runs/p2_e0_lambda0/1 runs/p2_best/1 runs/p3_round1/1; do
      [ -d "$r" ] && python -m sersys.evaluate --run_dir "$r" --adapter best
    done ;;
  tables)
    python scripts/make_tables.py --out tables ;;
  export)
    SEED=${2:-1}; ADAPTER=${3:-best}
    python -m sersys.export --run_dir runs/final/$SEED --adapter $ADAPTER --out exports/final
    python scripts/benchmark_latency.py --model exports/final --out exports/final/latency.json ;;
  *) echo "usage: $0 train|test|ablations|tables|export"; exit 2 ;;
esac
