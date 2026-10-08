#!/usr/bin/env bash
# Phase 1 of the text-only study: is TensorRT Edge-LLM also the best framework for a text-only LLM?
# Qwen2.5-1.5B / 3B-Instruct on GSM8K (5-shot, decode-heavy), MMLU (5-shot generative, prefill-heavy) and IFEval
# (long free-form answers), every framework at 16, 8 and 4 bits (matrix.sh). 3B first, then 1.5B.
# Resumable: valid finished runs are skipped (run_matrix.sh SKIP_DONE=1); missing Edge-LLM engines are built first.
# SMOKE=1: 8 questions per task (MMLU: per subject). SIZES / TASKS override the lists, e.g. SIZES=3b TASKS=ifeval.
# Results: $RESULTS_DIR/Qwen2.5-<size>-Instruct/<task>/<framework>-<precision>/<run id>/
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
source "$(dirname "$0")/matrix.sh"
export MODEL=qwen2_5
LIMIT=
[ "${SMOKE:-0}" = 1 ] && LIMIT=8
status=0
for size in ${SIZES:-$SIZES_PHASE1}; do
  [ "${SKIP_ENGINES:-0}" = 1 ] || "$REPO/jetson/frameworks/trt_edgellm/ensure_engines.sh" "$size" $EDGELLM_PHASE1 || status=1
  matrix_for "$size"
  for task in ${TASKS:-$TASKS_PHASE1}; do
    SKIP_DONE=1 "$REPO/jetson/scripts/run_matrix.sh" "$task" $LIMIT -- "${MATRIX[@]}" || status=1
  done
done
exit $status
