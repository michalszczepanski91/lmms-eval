#!/usr/bin/env bash
# Controlled-length sweep of the text-only study: how prefill and decode scale with prompt length in each framework.
# Prompts of exactly 128 / 512 / 2,048 tokens and 256 generated tokens everywhere (jetson/tasks/length_sweep: the
# prompt asks for a count to 1000, so every framework stops at the same cap). 16 prompts per length.
# Configurations: 16-bit and 8-bit in every framework (fp8; llama.cpp has no fp8, so Q8_0), both sizes, 3B first.
# Resumable (SKIP_DONE=1). SMOKE=1: 4 prompts per length. SIZES / TASKS override the lists.
# Results: $RESULTS_DIR/Qwen2.5-<size>-Instruct/length_sweep_<N>/<framework>-<precision>/<run id>/
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
export MODEL=qwen2_5
LIMIT=
[ "${SMOKE:-0}" = 1 ] && LIMIT=4
status=0
for size in ${SIZES:-3b 1.5b}; do
  [ "${SKIP_ENGINES:-0}" = 1 ] || "$REPO/jetson/frameworks/trt_edgellm/ensure_engines.sh" "$size" fp16 fp8 || status=1
  matrix=(hf "$size" vllm "$size" vllm "$size-fp8" llamacpp "$size-f16" llamacpp "$size-q8_0" trt_edgellm "$size-fp16" trt_edgellm "$size-fp8")
  for task in ${TASKS:-length_sweep_128 length_sweep_512 length_sweep_2048}; do
    SKIP_DONE=1 "$REPO/jetson/scripts/run_matrix.sh" "$task" $LIMIT -- "${matrix[@]}" || status=1
  done
done
exit $status
