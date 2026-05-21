#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"

for SEED in 0 1 2; do
  python -m src.train \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --run-name "resnet18_cifar10_${EPOCHS}ep_seed${SEED}"
done
