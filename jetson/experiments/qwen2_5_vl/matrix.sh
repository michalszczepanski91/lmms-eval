# The framework / precision combinations every Qwen2.5-VL experiment runs (sourced by mme.sh, gqa.sh, textvqa.sh).
# Edge-LLM engines are built locally first (docs/tensorrt-edgellm.md).
MATRIX=(
  hf 3b hf 7b
  vllm 3b vllm 3b-awq vllm 7b vllm 7b-awq
  llamacpp 3b-q8_0 llamacpp 3b-q4_k_m llamacpp 3b-f16 llamacpp 7b-q8_0 llamacpp 7b-q4_k_m llamacpp 7b-f16
  trt_edgellm 3b-fp16 trt_edgellm 3b-fp8 trt_edgellm 3b-int4_awq trt_edgellm 7b-fp16 trt_edgellm 7b-fp8 trt_edgellm 7b-int4_awq
)
