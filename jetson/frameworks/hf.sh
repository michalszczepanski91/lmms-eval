# Hugging Face transformers (PyTorch, in process) - the reference implementation.
# Sourced by run_eval.sh / download_assets.sh; the framework interface is described in frameworks/common.sh.
FW_PRECISIONS="bf16"

fw_assets() {
  echo "model:$HF_REPO"
}

fw_setup() {
  BACKEND=$HF_BACKEND
  MODEL_ARGS="pretrained=$(hf_snapshot "$HF_REPO"),attn_implementation=${ATTN:-flash_attention_2}"
  # Vision-encoder time per sample (CUDA events, no extra syncs) for the latency breakdown in SUMMARY.md.
  # Text-only models (MODEL_MODALITY=text) use the generic chat backend: no vision stage, so no stage_timing.
  [ "${MODEL_MODALITY:-vision}" = text ] || MODEL_ARGS+=",stage_timing=${HF_STAGE_TIMING:-True}"
}
