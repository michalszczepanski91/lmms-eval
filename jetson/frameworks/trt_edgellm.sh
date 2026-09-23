# TensorRT Edge-LLM (C++ runner) via lmms-eval's trt_edgellm backend. UNTESTED on this board:
# ONNX export needs an x86 GPU host or Jetson Thor (trt_edgellm/export.sh), then trt_edgellm/build_engines.sh.
# Latency is the runner's aggregate profile (trt_profile.json), not per sample.
FW_PRECISIONS="fp16 int4_awq fp8 nvfp4"  # fp8 / nvfp4: Thor only
TRT_WORKSPACE=${TRT_WORKSPACE:-/opt/models/trt-edgellm}

fw_assets() {
  : # Engines are built locally from exported ONNX; nothing to download from the Hub.
}

fw_setup() {
  IMAGE=${TRT_IMAGE:-lmms-eval-trt-edgellm:latest}
  BACKEND=trt_edgellm
  local dir=$TRT_WORKSPACE/$MODEL_TAG-$PRECISION/engines
  [ -d "$dir/llm" ] && [ -d "$dir/visual" ] || { echo "missing engines in $dir - see jetson/frameworks/trt_edgellm/" >&2; return 1; }
  MODEL_ARGS="engine_dir=$dir/llm,multimodal_engine_dir=$dir/visual,profile_output=$OUT_REL/trt_profile.json"
  DOCKER_ARGS+=(-v "$TRT_WORKSPACE":"$TRT_WORKSPACE":ro)
}
