#!/usr/bin/env bash
# Pre-fetch models and datasets into the shared HF cache (/opt/hf-cache) so evals can run offline.
# All repos here are public; no HF token needed. Datasets are also prepared into $HF_HOME/datasets
# (the Arrow cache that `datasets` requires offline); this also sidesteps task YAMLs with `token: True`.
#
# Usage: jetson/download_assets.sh [repo ...]   (default: Qwen2.5-VL 3B + 7B and MME)
set -euo pipefail

HF_CACHE=${HF_CACHE:-/opt/hf-cache}
IMAGE=${IMAGE:-lmms-eval-jetson:latest}
REPOS=("$@")
[ ${#REPOS[@]} -gt 0 ] || REPOS=(
  dataset:lmms-lab-encoder/MME
  model:Qwen/Qwen2.5-VL-3B-Instruct
  model:Qwen/Qwen2.5-VL-7B-Instruct
)

docker run --rm -i \
  --user "$(id -u):$(id -g)" --group-add "$(getent group mlusers | cut -d: -f3)" \
  -e HOME=/tmp -e HF_HOME="$HF_CACHE" -e HF_HUB_CACHE="$HF_CACHE/hub" -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -v "$HF_CACHE":"$HF_CACHE" \
  "$IMAGE" bash -c 'umask 002; python - "$@"' _ "${REPOS[@]}" <<'EOF'
import sys
import os
import datasets
from huggingface_hub import snapshot_download

for spec in sys.argv[1:]:
    kind, _, repo = spec.rpartition(":")
    path = snapshot_download(repo, repo_type=kind or "model", token=False)
    if kind == "dataset":
        ds = datasets.load_dataset(repo, token=False, cache_dir=os.path.join(os.environ["HF_HOME"], "datasets"))
        print(f"    {ds}", flush=True)
    print(f"ok  {spec} -> {path}", flush=True)
EOF
