#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/resnet18_cifar10_50ep_best.pt}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SUBSET_SIZE="${SUBSET_SIZE:-2000}"
SEED="${SEED:-3830}"
THRESHOLDS="${THRESHOLDS:-0.5,0.7,0.9}"
OUTPUT="${OUTPUT:-results/tables/confidence_gated_results.csv}"
EXTRA_ARGS=()
if [[ "${FULL_TEST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--full-test)
else
  EXTRA_ARGS+=(--subset-size "${SUBSET_SIZE}")
fi

python -m src.evaluate \
  --checkpoint "${CHECKPOINT}" \
  --methods gated \
  --thresholds "${THRESHOLDS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --include-clean \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
