#!/usr/bin/env bash
# Export + build TensorRT Edge-LLM engines that are missing (idempotent; engines that exist are skipped).
# Used by the experiment scripts so that a matrix with trt_edgellm entries can start from an empty workspace.
#
# Usage: ensure_engines.sh <3b|7b> <precision> [<precision> ...]
#   jetson/frameworks/trt_edgellm/ensure_engines.sh 3b fp16 fp8 int4_awq
# ONNX exports are deleted after a successful build (REMOVE_ONNX=1, disk). A failed precision is reported and
# skipped; the exit status is 1 if any failed. Don't run it next to a benchmark (it uses the GPU and docker build).
set -uo pipefail

SIZE=${1:?usage: $0 <3b|7b> <precision> [...]}
shift
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
source "$REPO/jetson/frameworks/common.sh"
load_model "$SIZE" || exit 1
TRT_WORKSPACE=${TRT_WORKSPACE:-/opt/models/trt-edgellm}

failed=()
for precision in "$@"; do
  dir=$TRT_WORKSPACE/$MODEL_TAG-$precision
  if [ -d "$dir/engines/llm" ] && [ -d "$dir/engines/visual" ]; then
    echo "$(date +%H:%M:%S) $MODEL_TAG-$precision: engines exist"
    continue
  fi
  echo "$(date +%H:%M:%S) $MODEL_TAG-$precision: export + build"
  if "$JETSON/frameworks/trt_edgellm/export.sh" "$SIZE" "$precision" "$TRT_WORKSPACE" &&
    REMOVE_ONNX=1 "$JETSON/frameworks/trt_edgellm/build_engines.sh" "$SIZE" "$precision"; then
    echo "$(date +%H:%M:%S) $MODEL_TAG-$precision: done"
  else
    echo "$(date +%H:%M:%S) $MODEL_TAG-$precision: FAILED"
    failed+=("$precision")
  fi
done
[ ${#failed[@]} -eq 0 ] || { echo "failed: ${failed[*]}"; exit 1; }
