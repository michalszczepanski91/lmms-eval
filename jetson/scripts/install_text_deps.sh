#!/usr/bin/env bash
# Extra Python packages and NLTK data the text-only tasks need (IFEval: langdetect, immutabledict, nltk punkt), installed
# into jetson/.cache on the repo mount instead of the images, so the images (and the Qwen2.5-VL runs) stay unchanged.
# run_eval.sh adds them to PYTHONPATH / NLTK_DATA when the model file sets EXTRA_PYTHONPATH (models/qwen2_5/model.sh).
# Needs uv (https://docs.astral.sh/uv/) and network once; runs are offline afterwards.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${IMAGE:-lmms-eval-jetson:latest}
DEPS=$REPO/jetson/.cache/text-deps
NLTK=$REPO/jetson/.cache/nltk_data
PY=$(docker run --rm "$IMAGE" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
uv pip install --target "$DEPS" --python-platform aarch64-unknown-linux-gnu --python-version "$PY" langdetect immutabledict
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$REPO":"$REPO" "$IMAGE" \
  python -c "import nltk; [nltk.download(r, download_dir='$NLTK') for r in ('punkt', 'punkt_tab')]"
echo "installed: $DEPS, $NLTK"
