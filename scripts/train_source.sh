#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-3830}"
RUN_NAME="${RUN_NAME:-resnet18_cifar10_${EPOCHS}ep}"

python -m src.train \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --run-name "${RUN_NAME}"
