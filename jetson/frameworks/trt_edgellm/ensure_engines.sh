#!/usr/bin/env bash
# Export + build TensorRT Edge-LLM engines that are missing (idempotent; engines that exist are skipped).
# Used by the experiment scripts so that a matrix with trt_edgellm entries can start from an empty workspace.
#
# Usage: ensure_engines.sh <3b|7b> <precision> [<precision> ...]
#   jetson/frameworks/trt_edgellm/ensure_engines.sh 3b fp16 fp8 int4_awq
# ONNX exports are deleted after a successful build (REMOVE_ONNX=1, disk). A failed precision is reported and
# skipped; the exit status is 1 if any failed. Don't run it next to a benchmark (it uses the GPU and docker build).
# Each new engine gets <workspace>/<model>-<precision>/build_info.txt: quantize / export / build seconds, peak system
# memory during each step (Thor's memory is shared by CPU and GPU), engine size; logs in export.log / build.log.
set -uo pipefail

SIZE=${1:?usage: $0 <3b|7b> <precision> [...]}
shift
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
source "$REPO/jetson/frameworks/common.sh"
load_model "$SIZE" || exit 1
TRT_WORKSPACE=${TRT_WORKSPACE:-/opt/models/trt-edgellm}

# measure <name> <log> <command...>: run the command (output to the terminal and <log>), then append
# <name>_seconds and <name>_peak_used_gb (system memory in use, sampled every 2 s) to $info.
measure() {
  local name=$1 log=$2 start peak_file status
  shift 2
  peak_file=$(mktemp)
  ( peak=0; while :; do u=$(free -m | awk '/^Mem:/ {print $3}'); [ "$u" -gt "$peak" ] && { peak=$u; echo "$peak" > "$peak_file"; }; sleep 2; done ) &
  local sampler=$!
  start=$(date +%s)
  "$@" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  echo "${name}_seconds=$(( $(date +%s) - start ))" >> "$info"
  echo "${name}_peak_used_gb=$(awk '{printf "%.1f", $1 / 1024}' "$peak_file")" >> "$info"
  rm -f "$peak_file"
  return "$status"
}

failed=()
for precision in "$@"; do
  dir=$TRT_WORKSPACE/$MODEL_TAG-$precision
  if [ -d "$dir/engines/llm" ] && { [ "${MODEL_MODALITY:-vision}" = text ] || [ -d "$dir/engines/visual" ]; }; then
    echo "$(date +%H:%M:%S) $MODEL_TAG-$precision: engines exist"
    continue
  fi
  echo "$(date +%H:%M:%S) $MODEL_TAG-$precision: export + build"
  mkdir -p "$dir"
  info=$dir/build_info.txt
  echo "date=$(date -Iseconds)" > "$info"
  if measure export "$dir/export.log" "$JETSON/frameworks/trt_edgellm/export.sh" "$SIZE" "$precision" "$TRT_WORKSPACE" &&
    measure build "$dir/build.log" env REMOVE_ONNX=1 "$JETSON/frameworks/trt_edgellm/build_engines.sh" "$SIZE" "$precision"; then
    grep -h '^quantize_seconds=' "$dir/export.log" >> "$info"
    echo "engine_bytes=$(du -sb "$dir/engines" | cut -f1)" >> "$info"
    echo "$(date +%H:%M:%S) $MODEL_TAG-$precision: done ($(tr '\n' ' ' < "$info"))"
  else
    echo "$(date +%H:%M:%S) $MODEL_TAG-$precision: FAILED"
    failed+=("$precision")
  fi
done
[ ${#failed[@]} -eq 0 ] || { echo "failed: ${failed[*]}"; exit 1; }
