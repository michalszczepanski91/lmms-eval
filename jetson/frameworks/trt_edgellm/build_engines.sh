#!/usr/bin/env bash
# Build TensorRT Edge-LLM engines on the Jetson from ONNX produced by export.sh. UNTESTED here.
#
# Usage: build_engines.sh <3b|7b> <fp16|int4_awq|fp8|nvfp4>   (fp8/nvfp4: Thor only)
# Expects  $TRT_WORKSPACE/<model>-<precision>/onnx/{llm,visual}   (TRT_WORKSPACE default /opt/models/trt-edgellm)
# Writes   $TRT_WORKSPACE/<model>-<precision>/engines/{llm,visual}
#
# Image-token limits match the HF/vLLM runs (min_pixels 200704 / max_pixels 1605632 = 256..2048 tokens
# of 28x28 patches), so every framework sees the same image resolution range.
set -euo pipefail

SIZE=${1:?usage: $0 <3b|7b> <fp16|int4_awq|fp8|nvfp4>}
PRECISION=${2:?usage: $0 <3b|7b> <fp16|int4_awq|fp8|nvfp4>}
TRT_WORKSPACE=${TRT_WORKSPACE:-/opt/models/trt-edgellm}
IMAGE=${IMAGE:-lmms-eval-trt-edgellm:latest}

case "${SIZE,,}" in 3b) MODEL=Qwen2.5-VL-3B-Instruct ;; 7b) MODEL=Qwen2.5-VL-7B-Instruct ;; *) echo "unknown size $SIZE" >&2; exit 1 ;; esac
DIR=$TRT_WORKSPACE/$MODEL-$PRECISION
[ -d "$DIR/onnx/llm" ] && [ -d "$DIR/onnx/visual" ] || { echo "missing $DIR/onnx/{llm,visual} - run export.sh on an x86 GPU host or Thor first" >&2; exit 1; }

docker run --rm --runtime nvidia --ipc=host \
  --user "$(id -u):$(id -g)" $(getent group "${SHARED_GROUP:-mlusers}" | cut -d: -f3 | sed 's/^/--group-add /') -e HOME=/tmp \
  -v "$DIR":"$DIR" "$IMAGE" bash -euo pipefail -c "
    umask 002
    llm_build --onnxDir '$DIR/onnx/llm' --engineDir '$DIR/engines/llm' \
      --maxBatchSize 1 --maxInputLen 2560 --maxKVCacheCapacity 3072
    visual_build --onnxDir '$DIR/onnx/visual' --engineDir '$DIR/engines/visual' \
      --minImageTokens 256 --maxImageTokens 2048 --maxImageTokensPerImage 2048
  "
echo "engines: $DIR/engines/{llm,visual}"
