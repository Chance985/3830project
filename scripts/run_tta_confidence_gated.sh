#!/usr/bin/env bash
set -euo pipefail

python -m src.evaluate \
  --checkpoint checkpoints/resnet18_cifar10_5ep_best.pt \
  --methods gated \
  --thresholds 0.5,0.7,0.9 \
  --batch-size 256 \
  --num-workers 0 \
  --subset-size 2000 \
  --include-clean \
  --output results/tables/confidence_gated_results.csv
