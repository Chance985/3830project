#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SUBSET_SIZE="${SUBSET_SIZE:-2000}"
TABLE_DIR="${TABLE_DIR:-results/tables}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-${TABLE_DIR}/raw_results_multiseed_trained.csv}"
EXTRA_ARGS=()
if [[ "${FULL_TEST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--full-test)
else
  EXTRA_ARGS+=(--subset-size "${SUBSET_SIZE}")
fi

for SEED in 0 1 2; do
  python -m src.evaluate \
    --checkpoint "checkpoints/resnet18_cifar10_${EPOCHS}ep_seed${SEED}_best.pt" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --include-clean \
    --output "${TABLE_DIR}/raw_results_seed${SEED}.csv" \
    "${EXTRA_ARGS[@]}"
done

TABLE_DIR="${TABLE_DIR}" COMBINED_OUTPUT="${COMBINED_OUTPUT}" python - <<'PY'
import os
from pathlib import Path

import pandas as pd

table_dir = Path(os.environ["TABLE_DIR"])
combined_output = Path(os.environ["COMBINED_OUTPUT"])
paths = [table_dir / f"raw_results_seed{seed}.csv" for seed in (0, 1, 2)]
missing = [str(path) for path in paths if not path.exists()]
if missing:
    raise SystemExit(f"Missing per-seed result files: {missing}")
combined = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
combined.to_csv(combined_output, index=False)
PY
