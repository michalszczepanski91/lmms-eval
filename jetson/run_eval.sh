#!/usr/bin/env bash
# Run one lmms-eval evaluation on Jetson Orin inside the lmms-eval-jetson container.
#
# Usage: jetson/run_eval.sh <3b|7b|HF model id> [tasks=mme] [limit]
#   jetson/run_eval.sh 3b mme 8     # smoke test on 8 samples
#   jetson/run_eval.sh 7b mme       # full MME
# Env overrides: BACKEND (default qwen2_5_vl), ATTN (default flash_attention_2),
#                EXTRA_MODEL_ARGS (appended to --model_args), IMAGE (docker image),
#                OFFLINE (default 1: use only what download_assets.sh put in /opt/hf-cache).
#
# Each run writes to jetson/results/<model>/<tasks>/<timestamp>[_limitN]/:
#   run.log (full console), tegrastats.log (RAM/GPU sampling), run_info.txt (command, versions),
#   plus lmms-eval's own *_results.json and *_samples_*.jsonl.
set -uo pipefail

MODEL=${1:?usage: $0 <3b|7b|HF model id> [tasks] [limit]}
TASKS=${2:-mme}
LIMIT=${3:-}
BACKEND=${BACKEND:-qwen2_5_vl}
ATTN=${ATTN:-flash_attention_2}
IMAGE=${IMAGE:-lmms-eval-jetson:latest}
HF_CACHE=${HF_CACHE:-/opt/hf-cache}
OFFLINE=${OFFLINE:-1}

case "$MODEL" in
  3b|3B) PRETRAINED=Qwen/Qwen2.5-VL-3B-Instruct ;;
  7b|7B) PRETRAINED=Qwen/Qwen2.5-VL-7B-Instruct ;;
  *)     PRETRAINED=$MODEL ;;
esac

REPO=$(cd "$(dirname "$0")/.." && pwd)
MODEL_TAG=$(basename "$PRETRAINED")
RUN_ID=$(date +%Y%m%d-%H%M%S)${LIMIT:+_limit$LIMIT}
OUT_REL=jetson/results/$MODEL_TAG/${TASKS//,/+}/$RUN_ID
OUT=$REPO/$OUT_REL
mkdir -p "$OUT"

# Load from the cached snapshot path: transformers 4.57.3's tokenizer loader queries the Hub API
# for repo ids even when HF_HUB_OFFLINE=1, but skips that for local paths.
SNAPSHOT_REF=$HF_CACHE/hub/models--${PRETRAINED//\//--}/refs/main
if [ "$OFFLINE" = 1 ] && [ -f "$SNAPSHOT_REF" ]; then
  PRETRAINED=$(dirname "$(dirname "$SNAPSHOT_REF")")/snapshots/$(cat "$SNAPSHOT_REF")
fi

MODEL_ARGS="pretrained=$PRETRAINED,attn_implementation=$ATTN${EXTRA_MODEL_ARGS:+,$EXTRA_MODEL_ARGS}"
EVAL_CMD=(python -m lmms_eval --model "$BACKEND" --model_args "$MODEL_ARGS" --tasks "$TASKS"
          --batch_size 1 --log_samples --output_path "$OUT_REL" ${LIMIT:+--limit "$LIMIT"})

{
  echo "date:        $(date -Is)"
  echo "board:       $(tr -d '\0' </proc/device-tree/model)"
  echo "l4t:         $(head -1 /etc/nv_tegra_release)"
  echo "power mode:  $(nvpmodel -q 2>/dev/null | head -1)"
  echo "git commit:  $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo ' (dirty)')"
  echo "model:       $PRETRAINED"
  echo "image:       $IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c1-19))"
  echo "other containers: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
  echo "command:     ${EVAL_CMD[*]}"
} >"$OUT/run_info.txt"

tegrastats --interval 1000 --logfile "$OUT/tegrastats.log" &
TEGRA_PID=$!
trap 'kill $TEGRA_PID 2>/dev/null || true' EXIT

# Run as the calling user (group mlusers) so files in the shared HF cache stay group-writable.
docker run --rm --runtime nvidia --ipc=host \
  --user "$(id -u):$(id -g)" --group-add "$(getent group mlusers | cut -d: -f3)" \
  -e HOME=/tmp -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HF_HOME="$HF_CACHE" -e HF_HUB_CACHE="$HF_CACHE/hub" -e HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub" \
  -e TRANSFORMERS_CACHE="$HF_CACHE/hub" -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HF_HUB_OFFLINE="$OFFLINE" -e HF_DATASETS_OFFLINE="$OFFLINE" -e TRANSFORMERS_OFFLINE="$OFFLINE" \
  -v "$REPO":"$REPO" -w "$REPO" -e PYTHONPATH="$REPO" -v "$HF_CACHE":"$HF_CACHE" \
  "$IMAGE" bash -c 'umask 002; python -c "import torch,transformers;print(\"torch\",torch.__version__,\"transformers\",transformers.__version__)"; exec "$@"' _ "${EVAL_CMD[@]}" \
  2>&1 | tee "$OUT/run.log"
STATUS=${PIPESTATUS[0]}
# lmms-eval logs evaluation errors but still exits 0.
if [ "$STATUS" -eq 0 ] && grep -q "Error during evaluation" "$OUT/run.log"; then STATUS=1; fi

# lmms-eval names its output dir after the model path; with a cache snapshot that is "snapshots__<hash>".
for d in "$OUT"/snapshots__*; do [ -d "$d" ] && mv "$d" "$OUT/$MODEL_TAG"; done

grep -m1 '^torch ' "$OUT/run.log" | sed 's/^/versions:    /' >>"$OUT/run_info.txt" || true
echo "results in: $OUT (exit $STATUS)"
exit "$STATUS"
