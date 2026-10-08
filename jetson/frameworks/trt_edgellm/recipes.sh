# TensorRT Edge-LLM weight configurations ("precisions"): how export.sh produces each LLM checkpoint.
# Sourced by export.sh and frameworks/trt_edgellm.sh (FW_PRECISIONS). Needs the model variables (HF_REPO, AWQ_REPO, ...).
#
# edgellm_recipe <precision> sets:
#   EDGELLM_QUANT        tensorrt-edgellm-quantize --quantization (empty: export the checkpoint as is)
#   EDGELLM_LM_HEAD      --lm_head_quantization (empty: LM head stays fp16)
#   EDGELLM_KV           --kv_cache_quantization (empty: fp16 KV cache)
#   EDGELLM_DATASET      --text_dataset (empty: the quantizer's default, cnn_dailymail)
#   EDGELLM_IMPORT_REPO  Hub repo exported directly instead of the base checkpoint (pre-quantized weights)
#   EDGELLM_CALIB_FILE   local JSONL ("text" field) that replaces the calibration dataset, through Edge-LLM's documented
#                        local override of a built-in (EDGELLM_QUANT_DATASET_<NAME>); path relative to this directory
#   EDGELLM_EXPORT_ARGS  extra tensorrt-edgellm-export flags
# Flags as listed by `tensorrt-edgellm-quantize llm --help` in Edge-LLM v0.10.1.

EDGELLM_PRECISIONS="fp16 int4_awq fp8 nvfp4 int4_awq_qwen int4_awq_qwen_v1 int4_gptq_qwen fp8_kv int8_sq mxfp8 fp8_lmhead int4_awq_lmhead int4_awq_taskcal"

edgellm_recipe() {
  EDGELLM_QUANT= EDGELLM_LM_HEAD= EDGELLM_KV= EDGELLM_DATASET= EDGELLM_IMPORT_REPO= EDGELLM_CALIB_FILE= EDGELLM_EXPORT_ARGS=
  case "$1" in
    fp16) ;;
    fp8 | nvfp4 | int8_sq | mxfp8) EDGELLM_QUANT=$1 ;;
    int4_awq)
      # AWQ_SOURCE=qwen (kept for the Qwen2.5-VL runs) exports Qwen's AWQ checkpoint instead of quantizing.
      if [ "${AWQ_SOURCE:-quantize}" = qwen ]; then EDGELLM_IMPORT_REPO=$AWQ_REPO; else EDGELLM_QUANT=int4_awq; fi ;;
    int4_awq_qwen) EDGELLM_IMPORT_REPO=$AWQ_REPO ;;
    # The same import with the legacy int4 GEMM plugin (AWQ-swizzled weights) instead of the default cuteDSL V2.
    int4_awq_qwen_v1) EDGELLM_IMPORT_REPO=$AWQ_REPO EDGELLM_EXPORT_ARGS="--int4-gemm-plugin-version 1" ;;
    # Qwen's GPTQ int4 checkpoint. (The exporter reads every GPTQ checkpoint as int4 and ignores "bits", so Qwen's
    # GPTQ-Int8 cannot be imported.)
    int4_gptq_qwen) EDGELLM_IMPORT_REPO=$GPTQ_INT4_REPO ;;
    fp8_kv) EDGELLM_QUANT=fp8 EDGELLM_KV=fp8 ;;
    fp8_lmhead) EDGELLM_QUANT=fp8 EDGELLM_LM_HEAD=fp8 ;;
    int4_awq_lmhead) EDGELLM_QUANT=int4_awq EDGELLM_LM_HEAD=int4_awq ;;
    # AWQ calibrated on the benchmark's own format (GSM8K train + MMLU validation, jetson/datasets/edgellm_calibration.py)
    # instead of cnn_dailymail. The file replaces the built-in "wikitext" dataset, which is only the name it runs under.
    int4_awq_taskcal) EDGELLM_QUANT=int4_awq EDGELLM_DATASET=wikitext EDGELLM_CALIB_FILE=calibration/gsm8k_mmlu.jsonl ;;
    *) echo "unknown Edge-LLM precision '$1' (choose: $EDGELLM_PRECISIONS)" >&2; return 1 ;;
  esac
}
