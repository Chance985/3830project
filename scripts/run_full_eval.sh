#!/usr/bin/env bash
set -euo pipefail

FULL_TEST=1 OUTPUT="${OUTPUT:-results/tables/raw_results.csv}" bash "$(dirname "$0")/run_all_eval.sh"
