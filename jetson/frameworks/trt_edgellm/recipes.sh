# TensorRT Edge-LLM weight configurations ("precisions"): how export.sh produces each LLM checkpoint.
# Sourced by export.sh and frameworks/trt_edgellm.sh (FW_PRECISIONS). Needs the model variables (HF_REPO, AWQ_REPO, ...).
#
# edgellm_recipe <precision> sets:
#   EDGELLM_QUANT        tensorrt-edgellm-quantize --quantization (empty: export the checkpoint as is)
#   EDGELLM_LM_HEAD      --lm_head_quantization (empty: LM head stays fp16)
#   EDGELLM_KV           --kv_cache_quantization (empty: fp16 KV cache)
#   EDGELLM_DATASET      --text_dataset (empty: the quantizer's default, cnn_dailymail)
#   EDGELLM_IMPORT_REPO  Hub repo exported directly instead of the base checkpoint (pre-quantized weights)
# Flags as listed by `tensorrt-edgellm-quantize llm --help` in Edge-LLM v0.10.1.

EDGELLM_PRECISIONS="fp16 int4_awq fp8 nvfp4 int4_awq_qwen fp8_kv int8_sq mxfp8 fp8_lmhead int4_awq_lmhead int4_awq_wikitext"

edgellm_recipe() {
  EDGELLM_QUANT= EDGELLM_LM_HEAD= EDGELLM_KV= EDGELLM_DATASET= EDGELLM_IMPORT_REPO=
  case "$1" in
    fp16) ;;
    fp8 | nvfp4 | int8_sq | mxfp8) EDGELLM_QUANT=$1 ;;
    int4_awq)
      # AWQ_SOURCE=qwen (kept for the Qwen2.5-VL runs) exports Qwen's AWQ checkpoint instead of quantizing.
      if [ "${AWQ_SOURCE:-quantize}" = qwen ]; then EDGELLM_IMPORT_REPO=$AWQ_REPO; else EDGELLM_QUANT=int4_awq; fi ;;
    int4_awq_qwen) EDGELLM_IMPORT_REPO=$AWQ_REPO ;;
    fp8_kv) EDGELLM_QUANT=fp8 EDGELLM_KV=fp8 ;;
    fp8_lmhead) EDGELLM_QUANT=fp8 EDGELLM_LM_HEAD=fp8 ;;
    int4_awq_lmhead) EDGELLM_QUANT=int4_awq EDGELLM_LM_HEAD=int4_awq ;;
    int4_awq_wikitext) EDGELLM_QUANT=int4_awq EDGELLM_DATASET=wikitext ;;
    *) echo "unknown Edge-LLM precision '$1' (choose: $EDGELLM_PRECISIONS)" >&2; return 1 ;;
  esac
}
