#!/usr/bin/env bash
# Fixed-size latency sweep: every MME image resized to one square (jetson/tasks/mme_fixed), so each framework
# processes identical inputs with a constant visual token count. 200 samples per point (latency, not accuracy).
#   side 448 / 672 / 896 / 1260 px  ->  256 / 576 / 1024 / 2025 visual tokens
# Frameworks: HF bf16, vLLM bf16 (+pil), llama.cpp Q8_0 (+png1); model size from $1 (default 3b).
# Then: python3 jetson/experiments/fixed_size_summary.py  ->  $RESULTS_DIR/fixed_size_sweep.md
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN=$REPO/jetson/run_eval.sh
SIZE=${1:-3b}
N=${N:-200}
LOG=$REPO/${RESULTS_DIR:-jetson/results}/fixed_size_sweep.log
mkdir -p "$(dirname "$LOG")"

run() {
  echo "=== $(date +%H:%M:%S) $*" | tee -a "$LOG"
  if env "$@" >/dev/null 2>&1; then echo "    ok" | tee -a "$LOG"; else echo "    FAILED" | tee -a "$LOG"; fi
}

for side in 448 672 896 1260; do
  run RUN_TAG=side$side EVAL_ENV="FIXED_IMAGE_SIDE=$side" $RUN hf "$SIZE" mme_fixed "$N"
  run RUN_TAG=pil-side$side EVAL_ENV="FIXED_IMAGE_SIDE=$side" EXTRA_MODEL_ARGS=pass_pil_images=True $RUN vllm "$SIZE" mme_fixed "$N"
  run RUN_TAG=png1-side$side EVAL_ENV="FIXED_IMAGE_SIDE=$side LMMS_IMAGE_PNG_COMPRESS_LEVEL=1" $RUN llamacpp "$SIZE-q8_0" mme_fixed "$N"
done
python3 "$REPO/jetson/experiments/fixed_size_summary.py" | tee -a "$LOG"
