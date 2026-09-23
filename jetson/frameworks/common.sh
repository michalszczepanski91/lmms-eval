# Shared helpers for run_eval.sh and download_assets.sh (sourced, not executed).
#
# resolve_framework <framework> <size>[-<precision>] sources jetson/frameworks/<framework>.sh and sets:
#   FRAMEWORK, SIZE (3b|7b), PRECISION, MODEL_TAG (e.g. Qwen2.5-VL-3B-Instruct)
# A framework file defines FW_PRECISIONS (first = default), fw_assets (Hub specs to download),
# fw_setup (sets BACKEND, MODEL_ARGS, appends to DOCKER_ARGS; may override IMAGE) and optionally
# fw_start / fw_stop (for server-based frameworks). It can use hf_snapshot / hf_file below and
# REPO, OUT, OUT_REL, HF_CACHE, OFFLINE.

HF_CACHE=${HF_CACHE:-/opt/hf-cache}
OFFLINE=${OFFLINE:-1}
# Containers run as the calling user; if this group exists it is added so shared caches stay group-writable.
SHARED_GROUP=${SHARED_GROUP:-mlusers}

shared_group_args() {
  local gid
  gid=$(getent group "$SHARED_GROUP" | cut -d: -f3)
  [ -z "$gid" ] || echo "--group-add $gid"
}

resolve_framework() {
  FRAMEWORK=$1
  local spec=${2,,}
  local fw_file=$REPO/jetson/frameworks/$FRAMEWORK.sh
  if [ ! -f "$fw_file" ]; then
    echo "unknown framework '$FRAMEWORK' (available: $(cd "$REPO/jetson/frameworks" && ls *.sh | grep -v common.sh | sed 's/\.sh$//' | tr '\n' ' '))" >&2
    return 1
  fi
  source "$fw_file"
  SIZE=${spec%%-*}
  PRECISION=${spec#"$SIZE"}
  PRECISION=${PRECISION#-}
  PRECISION=${PRECISION:-${FW_PRECISIONS%% *}}
  case "$SIZE" in
    3b) MODEL_TAG=Qwen2.5-VL-3B-Instruct ;;
    7b) MODEL_TAG=Qwen2.5-VL-7B-Instruct ;;
    *) echo "unknown model size '$SIZE' (use 3b or 7b)" >&2; return 1 ;;
  esac
  if [[ " $FW_PRECISIONS " != *" $PRECISION "* ]]; then
    echo "precision '$PRECISION' not available for $FRAMEWORK (choose: $FW_PRECISIONS)" >&2
    return 1
  fi
}

# Local snapshot dir for a cached Hub repo when running offline, else the repo id.
# (transformers 4.57.3's tokenizer loader queries the Hub for repo ids even with HF_HUB_OFFLINE=1.)
hf_snapshot() {
  local ref=$HF_CACHE/hub/models--${1//\//--}/refs/main
  if [ "$OFFLINE" = 1 ] && [ -f "$ref" ]; then
    echo "$HF_CACHE/hub/models--${1//\//--}/snapshots/$(cat "$ref")"
  else
    echo "$1"
  fi
}

# Path of one file inside a cached Hub repo snapshot (download it first with download_assets.sh).
hf_file() {
  local ref=$HF_CACHE/hub/models--${1//\//--}/refs/main
  local path=$HF_CACHE/hub/models--${1//\//--}/snapshots/$(cat "$ref" 2>/dev/null)/$2
  if [ ! -f "$path" ]; then
    echo "missing $1/$2 in $HF_CACHE - run jetson/download_assets.sh $FRAMEWORK $SIZE-$PRECISION" >&2
    return 1
  fi
  echo "$path"
}
