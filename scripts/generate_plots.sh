#!/usr/bin/env bash
set -euo pipefail

python -m src.plot_results \
  --input results/tables/raw_results.csv \
  --tables-dir results/tables \
  --figures-dir results/figures
