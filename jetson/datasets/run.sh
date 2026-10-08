#!/usr/bin/env bash
# Dataset statistics and examples (image, prompt, reference, model answer), in the eval container: offline, CPU only.
#
# Usage: jetson/datasets/run.sh <textvqa|mme|gqa> [--run <run dir>] [--examples <doc id> ...] [--out DIR] [--limit N]
#        jetson/datasets/run.sh text_report <gsm8k_jetson|mmlu_jetson|ifeval> [--run <run dir> ...]      # text-only tasks
#        jetson/datasets/run.sh questions <textvqa|mme|gqa> [-n 50] [--run <run dir>]     # one row per question
#   jetson/datasets/run.sh mme
#   jetson/datasets/run.sh textvqa --examples 10 4042
#   jetson/datasets/run.sh gqa --run jetson/results/thor/Qwen2.5-VL-3B-Instruct/gqa/hf-bf16/20260925-062406
# Without --examples it shows the shortest and longest answer (the model's with --run, else the reference).
# Writes <out>/<task>.md plus thumbnails (default out: jetson/datasets/reports).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${DATASETS_IMAGE:-lmms-eval-jetson:latest}
source "$REPO/jetson/frameworks/common.sh"
NAME=${1:?usage: $0 <textvqa|mme|gqa> [args]}
shift
[ -f "$REPO/jetson/datasets/$NAME.py" ] || { echo "no jetson/datasets/$NAME.py" >&2; exit 1; }
docker run --rm --user "$(id -u):$(id -g)" $(shared_group_args) \
  -e HOME=/tmp -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HF_HOME="$HF_CACHE" -e HF_HUB_CACHE="$HF_CACHE/hub" -e HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub" \
  -e HF_HUB_OFFLINE="$OFFLINE" -e HF_DATASETS_OFFLINE="$OFFLINE" -e TRANSFORMERS_OFFLINE="$OFFLINE" -e CUDA_VISIBLE_DEVICES= \
  -v "$REPO":"$REPO" -w "$REPO" -e PYTHONPATH="$REPO:$REPO/jetson/datasets:$REPO/jetson/.cache/text-deps" -e NLTK_DATA="$REPO/jetson/.cache/nltk_data" -v "$HF_CACHE":"$HF_CACHE" \
  "$IMAGE" python "jetson/datasets/$NAME.py" "$@"
