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
  # Fraction of the board's unified memory vLLM may take (weights + KV cache + activations). The default is a fixed
  # budget in GB (13.8, or 21.0 for 7B bf16: ~0.45 / 0.70 of the Orin 32GB), so boards with more memory (Thor) keep the
  # same KV-cache budget and memory footprint instead of reserving e.g. 57-74 GB.
  # 7B bf16 needs 21.0: with 18.4 (0.6 on Orin) nothing is left for the KV cache after profiling and vLLM refuses to start.
  local budget_gb=13.8
  [ "$SIZE" = 7b ] && [ "$PRECISION" = bf16 ] && budget_gb=21.0
  local default_mem
  default_mem=$(awk -v gb="$budget_gb" '/^MemTotal:/ {printf "%.2f", gb * 1048576 / $2}' /proc/meminfo)
  # Same image resolution range as the HF run (256..2048 visual tokens).
  MODEL_ARGS="model=$(hf_snapshot "$(_vllm_repo)"),gpu_memory_utilization=${VLLM_GPU_MEM:-$default_mem},max_model_len=4096,max_pixels=1605632"
  MODEL_ARGS+=',mm_processor_kwargs={"min_pixels":200704,"max_pixels":1605632}'
  # No cross-request reuse: MME asks two questions per image, so prefix/image caches would skip most of the
  # vision + prefill work for every second sample, which the HF reference cannot do.
  MODEL_ARGS+=",enable_prefix_caching=False,mm_processor_cache_gb=0"
  # Keep torch.compile / CUDA graph caches between runs (the container is ephemeral).
  DOCKER_ARGS+=(-e VLLM_CACHE_ROOT="$REPO/jetson/.cache/vllm")
}
