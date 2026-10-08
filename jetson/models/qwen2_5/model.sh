# Qwen2.5-Instruct (text only): everything model-specific the harness needs (sourced by frameworks/common.sh).
#
# model_resolve <size> sets:
#   MODEL_TAG       result-dir name and Hub model name, e.g. Qwen2.5-3B-Instruct
#   HF_REPO         base checkpoint (hf, vllm bf16, Edge-LLM export)
#   AWQ_REPO        official AWQ 4-bit checkpoint (vllm awq, Edge-LLM AWQ_SOURCE=qwen)
#   GPTQ_INT4_REPO / GPTQ_INT8_REPO   official GPTQ checkpoints
#   FP8_REPO        RedHatAI FP8 W8A8 checkpoint (vllm fp8)
#   W8A8_REPO / W8A16_REPO            RedHatAI INT8 W8A8 / W8A16 checkpoints
#   GGUF_REPO       official GGUF weights (llamacpp); file names from model_gguf_files
#   HF_BACKEND      lmms-eval model name for the hf framework (generic causal-LM chat backend)
# MODEL_MODALITY=text: no vision part (frameworks skip mmproj, image pixels and the visual engine).
# MAX_MODEL_LEN: context for vLLM / llama.cpp and the Edge-LLM engine; the longest task prompt plus its max output.
# FLOPs per sample: models/qwen2_5/flops.py. Task variants used by the experiments: TASK_INCLUDE_PATH.

MODEL_SIZES="1.5b 3b"
MODEL_MODALITY=text
MAX_MODEL_LEN=4096
HF_BACKEND=huggingface
TASK_INCLUDE_PATH=jetson/tasks

model_resolve() {
  case "${1,,}" in
    1.5b) MODEL_TAG=Qwen2.5-1.5B-Instruct ;;
    3b) MODEL_TAG=Qwen2.5-3B-Instruct ;;
    *) echo "unknown model size '$1' (qwen2_5: $MODEL_SIZES)" >&2; return 1 ;;
  esac
  HF_REPO=Qwen/$MODEL_TAG
  AWQ_REPO=Qwen/$MODEL_TAG-AWQ
  GPTQ_INT4_REPO=Qwen/$MODEL_TAG-GPTQ-Int4
  GPTQ_INT8_REPO=Qwen/$MODEL_TAG-GPTQ-Int8
  FP8_REPO=RedHatAI/$MODEL_TAG-FP8-dynamic
  W8A8_REPO=RedHatAI/$MODEL_TAG-quantized.w8a8
  W8A16_REPO=RedHatAI/$MODEL_TAG-quantized.w8a16
  GGUF_REPO=Qwen/$MODEL_TAG-GGUF
}

# model_gguf_files <precision>: the GGUF file(s) of one llama.cpp precision, first file first (llama-server loads
# split files from the first part). Official names are lowercase; the 3B fp16 is split in two.
model_gguf_files() {
  local base=${MODEL_TAG,,}
  case "$1" in
    q4_k_m) echo "$base-q4_k_m.gguf" ;;
    q8_0)   echo "$base-q8_0.gguf" ;;
    f16)
      if [ "$MODEL_TAG" = Qwen2.5-3B-Instruct ]; then
        echo "$base-fp16-00001-of-00002.gguf $base-fp16-00002-of-00002.gguf"
      else
        echo "$base-fp16.gguf"
      fi ;;
  esac
}
