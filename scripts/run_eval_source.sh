#!/usr/bin/env bash
set -euo pipefail

python -m src.evaluate \
  --checkpoint checkpoints/resnet18_cifar10_5ep_best.pt \
  --methods source \
  --batch-size 256 \
  --num-workers 0 \
  --subset-size 2000 \
  --include-clean \
  --output results/tables/source_results.csv
