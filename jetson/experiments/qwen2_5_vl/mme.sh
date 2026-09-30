#!/usr/bin/env bash
# Main comparison: MME (2,374 yes/no questions) for every framework and precision, 3B and 7B.
# Smoke test first (SMOKE=1: 8 samples each), then the full runs. Results: $RESULTS_DIR/<model>/mme/...
# Usage: jetson/experiments/qwen2_5_vl/mme.sh            (TensorRT Edge-LLM needs its engines, see docs/tensorrt-edgellm.md)
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
source "$(dirname "$0")/matrix.sh"
[ "${SMOKE:-0}" = 1 ] && exec "$REPO/jetson/scripts/run_matrix.sh" mme 8 -- "${MATRIX[@]}"
exec "$REPO/jetson/scripts/run_matrix.sh" mme "" -- "${MATRIX[@]}"
