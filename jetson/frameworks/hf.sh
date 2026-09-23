# Hugging Face transformers (PyTorch, in process) - the reference implementation.
# Sourced by run_eval.sh / download_assets.sh; see run_eval.sh for the variables available here.
FW_PRECISIONS="bf16"

fw_assets() {
  echo "model:Qwen/$MODEL_TAG"
}

fw_setup() {
  BACKEND=qwen2_5_vl
  MODEL_ARGS="pretrained=$(hf_snapshot "Qwen/$MODEL_TAG"),attn_implementation=${ATTN:-flash_attention_2}"
}
