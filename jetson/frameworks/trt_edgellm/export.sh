#!/usr/bin/env bash
# Export Qwen2.5-VL to ONNX for TensorRT Edge-LLM. Run on an x86 Linux host with an NVIDIA GPU
# (compute capability 8.0+) or on Jetson Thor - NOT on Orin. UNTESTED here (no such host available).
#
# Usage: export.sh <3b|7b> <fp16|int4_awq|fp8|nvfp4> [workspace]
#   fp8 / nvfp4 run only on Thor (SM110); Orin supports fp16 and int4_awq.
#   workspace default: ./trt-edgellm-workspace -> <workspace>/<model>-<precision>/onnx/{llm,visual}
# Then copy <workspace>/<model>-<precision>/onnx to the device under
#   /opt/models/trt-edgellm/<model>-<precision>/onnx   and run build_engines.sh there.
# Memory (Edge-LLM docs): ~8-16 GB GPU for 3B, 20-48 GB for 7B.
set -euo pipefail

SIZE=${1:?usage: $0 <3b|7b> <fp16|int4_awq|fp8|nvfp4> [workspace]}
PRECISION=${2:?usage: $0 <3b|7b> <fp16|int4_awq|fp8|nvfp4> [workspace]}
WORKSPACE=$(realpath -m "${3:-./trt-edgellm-workspace}")
EDGELLM_REF=${EDGELLM_REF:-v0.6.0}
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}
# x86 hosts: --gpus all; on Jetson Thor use GPU_FLAGS="--runtime nvidia" (and a newer IMAGE / EDGELLM_REF, see jetson/THOR.md).
GPU_FLAGS=${GPU_FLAGS:---gpus all}

case "${SIZE,,}" in 3b) MODEL=Qwen2.5-VL-3B-Instruct ;; 7b) MODEL=Qwen2.5-VL-7B-Instruct ;; *) echo "unknown size $SIZE" >&2; exit 1 ;; esac
case "$PRECISION" in fp16|int4_awq|fp8|nvfp4) ;; *) echo "precision must be fp16, int4_awq, fp8 or nvfp4 (fp8/nvfp4: Thor only)" >&2; exit 1 ;; esac

OUT=$WORKSPACE/$MODEL-$PRECISION
mkdir -p "$OUT" "$WORKSPACE/hf-cache"

docker run --rm $GPU_FLAGS --ipc=host \
  -v "$WORKSPACE":/workspace -w /workspace -e HF_HOME=/workspace/hf-cache \
  -e MODEL="$MODEL" -e PRECISION="$PRECISION" -e EDGELLM_REF="$EDGELLM_REF" \
  "$IMAGE" bash -euo pipefail -c '
    git clone --depth 1 --branch "$EDGELLM_REF" https://github.com/NVIDIA/TensorRT-Edge-LLM.git /tmp/edgellm
    (cd /tmp/edgellm && git submodule update --init --recursive --depth 1 && pip install -q .)
    OUT=/workspace/$MODEL-$PRECISION
    LLM_SRC=Qwen/$MODEL
    if [ "$PRECISION" != fp16 ]; then
      tensorrt-edgellm-quantize-llm --model_dir "Qwen/$MODEL" --quantization "$PRECISION" --output_dir "$OUT/quantized"
      LLM_SRC=$OUT/quantized
    fi
    tensorrt-edgellm-export-llm --model_dir "$LLM_SRC" --output_dir "$OUT/onnx/llm"
    # The vision encoder stays FP16 for every precision (export_visual only supports FP16 here).
    tensorrt-edgellm-export-visual --model_dir "Qwen/$MODEL" --output_dir "$OUT/onnx/visual"
    echo "exported: $OUT/onnx"
  '
echo "Copy $OUT/onnx to the device: /opt/models/trt-edgellm/$MODEL-$PRECISION/onnx"
