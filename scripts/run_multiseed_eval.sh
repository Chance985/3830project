#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/resnet18_cifar10_50ep_best.pt}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SUBSET_SIZE="${SUBSET_SIZE:-2000}"
SEEDS="${SEEDS:-0,1,2}"
OUTPUT="${OUTPUT:-results/tables/raw_results_multiseed_eval.csv}"
EXTRA_ARGS=()
if [[ "${FULL_TEST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--full-test)
else
  EXTRA_ARGS+=(--subset-size "${SUBSET_SIZE}")
fi

python -m src.evaluate \
  --checkpoint "${CHECKPOINT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --seeds "${SEEDS}" \
  --include-clean \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
