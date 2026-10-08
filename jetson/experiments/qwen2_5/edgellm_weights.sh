#!/usr/bin/env bash
# Phase 2 of the text-only study: which TensorRT Edge-LLM weight configuration is best for Qwen2.5-1.5B / 3B?
# Main configurations on all three tasks; the extras (one change each against a main one) on GSM8K and MMLU only
# (scope decision, results/thor/NOTES_qwen2_5.md). Recipes: frameworks/trt_edgellm/recipes.sh.
#   main:   fp16, fp8, int4_awq (Phase 1 runs are reused), int8_sq, nvfp4
#   extras: fp8_kv (fp8 KV cache), fp8_lmhead / int4_awq_lmhead (quantized LM head), int4_awq_taskcal (calibrated on
#           GSM8K train + MMLU validation), int4_gptq_qwen / int4_awq_qwen_v1 (Qwen's int4 checkpoints imported), mxfp8
# Engines are exported and built first, one at a time, never next to a run (build_info.txt per engine: times, peak
# memory, size). An engine that fails to build is reported and its runs are skipped. Resumable (SKIP_DONE=1).
# SMOKE=1: 8 questions per task. SIZES / MAIN / EXTRAS / TASKS_MAIN / TASKS_EXTRAS override the lists.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
export MODEL=qwen2_5
LIMIT=
[ "${SMOKE:-0}" = 1 ] && LIMIT=8
MAIN=${MAIN:-fp16 fp8 int4_awq int8_sq nvfp4}
EXTRAS=${EXTRAS:-fp8_kv fp8_lmhead int4_awq_lmhead int4_awq_taskcal int4_gptq_qwen int4_awq_qwen_v1 mxfp8}
TASKS_MAIN=${TASKS_MAIN:-gsm8k_jetson mmlu_jetson ifeval}
TASKS_EXTRAS=${TASKS_EXTRAS:-gsm8k_jetson mmlu_jetson}
TRT_WORKSPACE=${TRT_WORKSPACE:-/opt/models/trt-edgellm}
status=0

# built <size> <precision...>: the precisions whose engine exists after ensure_engines (one build per call, so one
# failure does not stop the others).
built() {
  local size=$1 p
  shift
  for p in "$@"; do
    "$REPO/jetson/frameworks/trt_edgellm/ensure_engines.sh" "$size" "$p" >&2
    [ -d "$TRT_WORKSPACE/$MODEL_TAG-$p/engines/llm" ] && echo "$p"
  done
}

source "$REPO/jetson/frameworks/common.sh"
for size in ${SIZES:-3b 1.5b}; do
  load_model "$size" || exit 1
  main=($(built "$size" $MAIN))
  extras=($(built "$size" $EXTRAS))
  want=($MAIN $EXTRAS)
  [ $(( ${#main[@]} + ${#extras[@]} )) -eq ${#want[@]} ] || { echo "$size: engines missing for some precisions (see above)"; status=1; }
  matrix=()
  for p in "${main[@]}"; do matrix+=(trt_edgellm "$size-$p"); done
  for task in $TASKS_MAIN; do
    [ ${#matrix[@]} -gt 0 ] && { SKIP_DONE=1 "$REPO/jetson/scripts/run_matrix.sh" "$task" $LIMIT -- "${matrix[@]}" || status=1; }
  done
  matrix=()
  for p in "${extras[@]}"; do matrix+=(trt_edgellm "$size-$p"); done
  for task in $TASKS_EXTRAS; do
    [ ${#matrix[@]} -gt 0 ] && { SKIP_DONE=1 "$REPO/jetson/scripts/run_matrix.sh" "$task" $LIMIT -- "${matrix[@]}" || status=1; }
  done
done
exit $status
