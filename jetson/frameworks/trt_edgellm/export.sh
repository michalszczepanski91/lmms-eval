#!/usr/bin/env bash
# Export Qwen2.5-VL to ONNX for TensorRT Edge-LLM. Run on an x86 Linux host with an NVIDIA GPU
# (compute capability 8.0+) or on Jetson Thor - NOT on Orin. UNTESTED here (no such host available).
#
# Usage: export.sh <3b|7b> <fp16|int4_awq> [workspace]
#   workspace default: ./trt-edgellm-workspace -> <workspace>/<model>-<precision>/onnx/{llm,visual}
# Then copy <workspace>/<model>-<precision>/onnx to the device under
#   /opt/models/trt-edgellm/<model>-<precision>/onnx   and run build_engines.sh there.
# Memory (Edge-LLM docs): ~8-16 GB GPU for 3B, 20-48 GB for 7B.
set -euo pipefail

SIZE=${1:?usage: $0 <3b|7b> <fp16|int4_awq> [workspace]}
PRECISION=${2:?usage: $0 <3b|7b> <fp16|int4_awq> [workspace]}
WORKSPACE=$(realpath -m "${3:-./trt-edgellm-workspace}")
EDGELLM_REF=${EDGELLM_REF:-v0.6.0}
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}

case "${SIZE,,}" in 3b) MODEL=Qwen2.5-VL-3B-Instruct ;; 7b) MODEL=Qwen2.5-VL-7B-Instruct ;; *) echo "unknown size $SIZE" >&2; exit 1 ;; esac
case "$PRECISION" in fp16|int4_awq) ;; *) echo "precision must be fp16 or int4_awq (Orin runs FP16/INT8/INT4 only)" >&2; exit 1 ;; esac

OUT=$WORKSPACE/$MODEL-$PRECISION
mkdir -p "$OUT" "$WORKSPACE/hf-cache"

docker run --rm --gpus all --ipc=host \
  -v "$WORKSPACE":/workspace -w /workspace -e HF_HOME=/workspace/hf-cache \
  -e MODEL="$MODEL" -e PRECISION="$PRECISION" -e EDGELLM_REF="$EDGELLM_REF" \
  "$IMAGE" bash -euo pipefail -c '
    git clone --depth 1 --branch "$EDGELLM_REF" https://github.com/NVIDIA/TensorRT-Edge-LLM.git /tmp/edgellm
    (cd /tmp/edgellm && git submodule update --init --recursive --depth 1 && pip install -q .)
    OUT=/workspace/$MODEL-$PRECISION
    LLM_SRC=Qwen/$MODEL
    if [ "$PRECISION" = int4_awq ]; then
      tensorrt-edgellm-quantize-llm --model_dir "Qwen/$MODEL" --quantization int4_awq --output_dir "$OUT/quantized"
      LLM_SRC=$OUT/quantized
    fi
    tensorrt-edgellm-export-llm --model_dir "$LLM_SRC" --output_dir "$OUT/onnx/llm"
    # The vision encoder stays FP16 for both precisions (the only dtype export_visual supports).
    tensorrt-edgellm-export-visual --model_dir "Qwen/$MODEL" --output_dir "$OUT/onnx/visual"
    echo "exported: $OUT/onnx"
  '
echo "Copy $OUT/onnx to the device: /opt/models/trt-edgellm/$MODEL-$PRECISION/onnx"
