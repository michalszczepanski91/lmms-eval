# vLLM (in process, lmms-eval `vllm` backend). awq = Qwen's official AWQ 4-bit checkpoint.
FW_PRECISIONS="bf16 awq"

_vllm_repo() {
  case "$PRECISION" in
    bf16) echo "Qwen/$MODEL_TAG" ;;
    awq)  echo "Qwen/$MODEL_TAG-AWQ" ;;
  esac
}

fw_assets() {
  echo "model:$(_vllm_repo)"
}

fw_setup() {
  BACKEND=vllm
  # Fraction of the board's unified memory vLLM may take (weights + KV cache + activations).
  local default_mem=0.45
  [ "$SIZE" = 7b ] && [ "$PRECISION" = bf16 ] && default_mem=0.7  # 0.6 leaves no KV cache after profiling (tested)
  # Same image resolution range as the HF run (256..2048 visual tokens).
  MODEL_ARGS="model=$(hf_snapshot "$(_vllm_repo)"),gpu_memory_utilization=${VLLM_GPU_MEM:-$default_mem},max_model_len=4096,max_pixels=1605632"
  MODEL_ARGS+=',mm_processor_kwargs={"min_pixels":200704,"max_pixels":1605632}'
  # No cross-request reuse: MME asks two questions per image, so prefix/image caches would skip most of the
  # vision + prefill work for every second sample, which the HF reference cannot do.
  MODEL_ARGS+=",enable_prefix_caching=False,mm_processor_cache_gb=0"
  # Keep torch.compile / CUDA graph caches between runs (the container is ephemeral).
  DOCKER_ARGS+=(-e VLLM_CACHE_ROOT="$REPO/jetson/.cache/vllm")
}
