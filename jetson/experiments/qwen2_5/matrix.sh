# The framework / precision combinations of the text-only Qwen2.5 framework comparison (Phase 1; sourced by
# frameworks.sh). One list per size: 16-bit, 8-bit and 4-bit in every framework that has them.
# vLLM w8a8 is left out (Triton int8 was slower than bf16 in the Qwen2.5-VL study); vLLM fp8 runs with the
# torch._scaled_mm kernel on Thor (frameworks/vllm.sh). Edge-LLM int4_awq_qwen (Qwen's AWQ checkpoint exported directly)
# is left out: it produced garbage text in the 3B smoke test (results/thor/NOTES_qwen2_5.md, problem 6).
matrix_for() {
  local s=$1
  MATRIX=(
    hf "$s"
    vllm "$s" vllm "$s-fp8" vllm "$s-awq"
    llamacpp "$s-f16" llamacpp "$s-q8_0" llamacpp "$s-q4_k_m"
    trt_edgellm "$s-fp16" trt_edgellm "$s-fp8" trt_edgellm "$s-int4_awq"
  )
}
EDGELLM_PHASE1="fp16 fp8 int4_awq"
TASKS_PHASE1="gsm8k_jetson mmlu_jetson ifeval"
SIZES_PHASE1="3b 1.5b"
