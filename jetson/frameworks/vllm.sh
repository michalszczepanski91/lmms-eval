# vLLM (in process, lmms-eval `vllm` backend). awq = the model's official AWQ 4-bit checkpoint ($AWQ_REPO);
# fp8 / w8a8 = RedHatAI's 8-bit W8A8 checkpoints ($FP8_REPO / $W8A8_REPO, language model only). On Thor (SM110),
# vLLM 0.19.0's default Cutlass kernels fail; pick an alternative kernel (results/thor/NOTES.md, problem 17):
#   EVAL_ENV=VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel RUN_TAG=torch   (fp8 via torch._scaled_mm)
#   EVAL_ENV=VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel RUN_TAG=triton (int8 via Triton)
FW_PRECISIONS="bf16 awq fp8 w8a8"

_vllm_repo() {
  case "$PRECISION" in
    bf16) echo "$HF_REPO" ;;
    awq)  echo "$AWQ_REPO" ;;
    fp8)  echo "$FP8_REPO" ;;
    w8a8) echo "$W8A8_REPO" ;;
  esac
}

fw_assets() {
  echo "model:$(_vllm_repo)"
}

fw_setup() {
  BACKEND=vllm
  # Fraction of the board's unified memory vLLM may take (weights + KV cache + activations). The default is a fixed
  # budget in GB (13.8, or 21.0 for 7B bf16: ~0.45 / 0.70 of the Orin 32GB), so boards with more memory (Thor) keep the
  # same KV-cache budget and memory footprint instead of reserving e.g. 57-74 GB.
  # 7B bf16 needs 21.0: with 18.4 (0.6 on Orin) nothing is left for the KV cache after profiling and vLLM refuses to start.
  # 7B 8-bit (~9.5 GB of weights) gets 16.0, the same headroom over its weights as 7B bf16.
  local budget_gb=13.8
  [ "$SIZE" = 7b ] && [ "$PRECISION" = bf16 ] && budget_gb=21.0
  [ "$SIZE" = 7b ] && { [ "$PRECISION" = fp8 ] || [ "$PRECISION" = w8a8 ]; } && budget_gb=16.0
  local default_mem
  default_mem=$(awk -v gb="$budget_gb" '/^MemTotal:/ {printf "%.2f", gb * 1048576 / $2}' /proc/meminfo)
  # Same image resolution range as the other frameworks (IMAGE_MIN/MAX_PIXELS from the model file).
  if [ "${MODEL_MODALITY:-vision}" = text ]; then
    # Text only: no image arguments; context from the model file. The backend generates max(task, own) tokens, so
    # its own default goes to 1 and the task's max_new_tokens applies, as in the other frameworks.
    MODEL_ARGS="model=$(hf_snapshot "$(_vllm_repo)"),gpu_memory_utilization=${VLLM_GPU_MEM:-$default_mem},max_model_len=$MAX_MODEL_LEN,max_new_tokens=1"
  else
    MODEL_ARGS="model=$(hf_snapshot "$(_vllm_repo)"),gpu_memory_utilization=${VLLM_GPU_MEM:-$default_mem},max_model_len=4096,max_pixels=$IMAGE_MAX_PIXELS"
    MODEL_ARGS+=",mm_processor_kwargs={\"min_pixels\":$IMAGE_MIN_PIXELS,\"max_pixels\":$IMAGE_MAX_PIXELS}"
  fi
  # No cross-request reuse: MME asks two questions per image, so prefix/image caches would skip most of the
  # vision + prefill work for every second sample, which the HF reference cannot do. Text tasks share their few-shot
  # prefix across questions, so prefix caching stays off there too.
  if [ "${MODEL_MODALITY:-vision}" = text ]; then
    MODEL_ARGS+=",enable_prefix_caching=False"
  else
    MODEL_ARGS+=",enable_prefix_caching=False,mm_processor_cache_gb=0"
  fi
  # Keep torch.compile / CUDA graph caches between runs (the container is ephemeral).
  DOCKER_ARGS+=(-e VLLM_CACHE_ROOT="$JETSON/.cache/vllm")
}
