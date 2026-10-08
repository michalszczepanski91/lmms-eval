# TensorRT Edge-LLM (C++ runner) via lmms-eval's trt_edgellm backend.
# Engines first: trt_edgellm/ensure_engines.sh <size> <precision>... (export on Thor or an x86 GPU host, build on the
# device; docs/tensorrt-edgellm.md). Latency comes from the runner's profile (trt_profile.json): per-stage GPU-time
# medians and an average TTFT, not per-sample timings.
source "$JETSON/frameworks/trt_edgellm/recipes.sh"
FW_PRECISIONS=$EDGELLM_PRECISIONS  # first = default; fp8 / nvfp4 / mxfp8: Thor only
TRT_WORKSPACE=${TRT_WORKSPACE:-/opt/models/trt-edgellm}

fw_assets() {
  : # Engines are built locally from exported ONNX; nothing to download from the Hub.
}

fw_setup() {
  IMAGE=${TRT_IMAGE:-lmms-eval-trt-edgellm:latest}
  BACKEND=trt_edgellm
  local dir=$TRT_WORKSPACE/$MODEL_TAG-$PRECISION/engines
  if [ "${MODEL_MODALITY:-vision}" = text ]; then
    # Text only: LLM engine alone, no image-side limits.
    [ -d "$dir/llm" ] || { echo "missing engine in $dir/llm - see jetson/frameworks/trt_edgellm/" >&2; return 1; }
    # No system message of our own: the runner applies the chat template's default (Qwen2.5: "You are Qwen, created by
    # Alibaba Cloud. ..."), as HF, vLLM and llama.cpp do; the backend's VL default would differ.
    MODEL_ARGS="engine_dir=$dir/llm,profile_output=$OUT_REL/trt_profile.json,system_prompt=none"
    DOCKER_ARGS+=(-v "$TRT_WORKSPACE":"$TRT_WORKSPACE":ro)
    return 0
  fi
  [ -d "$dir/llm" ] && [ -d "$dir/visual" ] || { echo "missing engines in $dir - see jetson/frameworks/trt_edgellm/" >&2; return 1; }
  MODEL_ARGS="engine_dir=$dir/llm,multimodal_engine_dir=$dir/visual,profile_output=$OUT_REL/trt_profile.json"
  # No cross-request reuse (as for vLLM / llama.cpp): Edge-LLM >= 0.10 caches vision-encoder outputs by default,
  # which would skip the encoder for MME's second question on every image. TRT_ENCODER_CACHE= (empty) for older runners.
  MODEL_ARGS+=${TRT_ENCODER_CACHE-,encoder_cache_budget_bytes=0}
  # Edge-LLM (v0.10) rejects raw images with a side above 4096 px ("GPU-resize budget"), e.g. 128 of MME's landmark
  # questions; downscale those first. Every framework shrinks them to <= 2048 visual tokens anyway.
  MODEL_ARGS+=",max_image_side=${TRT_MAX_IMAGE_SIDE:-4096}"
  DOCKER_ARGS+=(-v "$TRT_WORKSPACE":"$TRT_WORKSPACE":ro)
}
