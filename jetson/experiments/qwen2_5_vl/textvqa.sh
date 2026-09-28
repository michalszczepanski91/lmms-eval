#!/usr/bin/env bash
# TextVQA val (5,000 questions about text in 3,166 photos, e.g. signs, labels, screens; VQA accuracy against 10 human
# answers), 3B and 7B. OCR-heavy, so it tests what quantization and image handling do to reading fine text.
# Needs dataset:lmms-lab-encoder/textvqa (download_assets.sh). Builds the TensorRT Edge-LLM 3B engines if they are
# missing (SKIP_ENGINES=1 to skip; the trt_edgellm runs then fail if there are none).
# SMOKE=1: 8 samples each. Results: $RESULTS_DIR/<model>/textvqa_val/...
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
MATRIX=(
  hf 3b hf 7b
  vllm 3b vllm 3b-awq vllm 7b vllm 7b-awq
  llamacpp 3b-q8_0 llamacpp 3b-q4_k_m llamacpp 7b-q8_0 llamacpp 7b-q4_k_m
  trt_edgellm 3b-fp16 trt_edgellm 3b-fp8 trt_edgellm 3b-int4_awq
)
[ "${SKIP_ENGINES:-0}" = 1 ] || "$REPO/jetson/frameworks/trt_edgellm/ensure_engines.sh" 3b fp16 fp8 int4_awq
[ "${SMOKE:-0}" = 1 ] && exec "$REPO/jetson/scripts/run_matrix.sh" textvqa_val 8 -- "${MATRIX[@]}"
exec "$REPO/jetson/scripts/run_matrix.sh" textvqa_val "" -- "${MATRIX[@]}"
