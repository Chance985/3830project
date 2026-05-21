#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --epochs 5 \
  --batch-size 256 \
  --num-workers 0 \
  --run-name resnet18_cifar10_5ep
