#!/usr/bin/env bash
# GQA testdev-balanced (12,578 short-answer questions on 398 images, exact match), 3B and 7B.
# Needs dataset:lmms-lab-encoder/GQA:testdev_balanced_instructions and :testdev_balanced_images (download_assets.sh).
# SMOKE=1: 8 samples each. Results: $RESULTS_DIR/<model>/gqa/...
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
source "$(dirname "$0")/matrix.sh"
[ "${SMOKE:-0}" = 1 ] && exec "$REPO/jetson/scripts/run_matrix.sh" gqa 8 -- "${MATRIX[@]}"
exec "$REPO/jetson/scripts/run_matrix.sh" gqa "" -- "${MATRIX[@]}"
