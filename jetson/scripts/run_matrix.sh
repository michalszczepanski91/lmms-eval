#!/usr/bin/env bash
# Run several framework/model combinations one after another, then refresh results/SUMMARY.md.
# A failed run is reported and skipped; the rest continue.
#
# Usage: jetson/scripts/run_matrix.sh <tasks> [limit] -- <framework> <size-precision> [<framework> <size-precision> ...]
#   jetson/scripts/run_matrix.sh mme -- vllm 3b llamacpp 3b-q8_0
#   jetson/scripts/run_matrix.sh mme 8 -- hf 3b vllm 3b             # smoke test, 8 samples each
# The experiment scripts (jetson/experiments/<model>/*.sh) call it with their full matrix.
# After each run, check_run.py scans the answers (marks runs with "????" or many empty answers INVALID).
# Watchdog: a run whose run.log has not grown for STALL_MINUTES (default 30; 0 = off) is stopped and gets a STALLED
# file (a llama.cpp MME run on Thor once hung on one request for 27 h); the matrix goes on with the next run.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/jetson/frameworks/common.sh"  # RESULTS_DIR
USAGE="usage: $0 <tasks> [limit] -- <framework> <size-precision> [...]"
TASKS=${1:?$USAGE}
shift
LIMIT=
if [ $# -gt 0 ] && [ "$1" != "--" ]; then LIMIT=$1; shift; fi
[ "${1:-}" = "--" ] && shift
MATRIX=("$@")
[ ${#MATRIX[@]} -gt 0 ] && [ $((${#MATRIX[@]} % 2)) -eq 0 ] || { echo "$USAGE" >&2; exit 1; }

STALL_MINUTES=${STALL_MINUTES:-30}

# run_eval.sh in its own process group; returns its exit status, or 124 if the watchdog stopped it.
run_one() {
  local marker log pid status
  marker=$(mktemp)
  setsid "$REPO/jetson/scripts/run_eval.sh" "$@" >/dev/null 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    for _ in $(seq 12); do kill -0 "$pid" 2>/dev/null && sleep 5; done
    [ "$STALL_MINUTES" -gt 0 ] || continue
    # This run's log: the newest run.log written since the run started (none yet while waiting for an idle GPU).
    log=$(find "$REPO/$RESULTS_DIR" -name run.log -newer "$marker" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$log" ] && [ $(($(date +%s) - $(stat -c %Y "$log"))) -gt $((STALL_MINUTES * 60)) ]; then
      echo "    STALLED: $log has not grown for $STALL_MINUTES min, stopping the run"
      echo "watchdog (run_matrix.sh): run.log did not grow for $STALL_MINUTES min; run stopped at $(date -Is)" >"$(dirname "$log")/STALLED"
      for c in $(docker ps --format '{{.Names}}' | grep '^llamacpp-bench-'); do docker logs "$c" >"$(dirname "$log")/server-stalled.log" 2>&1; done
      kill -TERM -- "-$pid" 2>/dev/null  # docker run forwards TERM to the eval process; run_eval.sh's trap stops servers
      for _ in $(seq 12); do kill -0 "$pid" 2>/dev/null || break; sleep 10; done
      kill -KILL -- "-$pid" 2>/dev/null
      docker ps --format '{{.Names}}' | grep '^llamacpp-bench-' | xargs -r docker rm -f >/dev/null
      wait "$pid" 2>/dev/null
      rm -f "$marker"
      return 124
    fi
  done
  wait "$pid"
  status=$?
  rm -f "$marker"
  return $status
}

failed=()
for ((i = 0; i < ${#MATRIX[@]}; i += 2)); do
  framework=${MATRIX[i]} spec=${MATRIX[i + 1]}
  echo "=== $(date +%H:%M:%S) $framework $spec $TASKS ${LIMIT:+limit=$LIMIT}"
  run_one "$framework" "$spec" "$TASKS" $LIMIT
  status=$?
  run_dir=$(ls -dt "$REPO/$RESULTS_DIR"/*/"${TASKS//,/+}"/"$framework"-*/*/ 2>/dev/null | head -1)
  if [ "$status" -ne 0 ]; then
    echo "    FAILED (exit $status, see ${run_dir:-the newest run dir under $RESULTS_DIR/*/*/$framework-*/})"
    failed+=("$framework $spec")
  elif [ -n "$run_dir" ] && ! python3 "$REPO/jetson/scripts/check_run.py" "$run_dir" | sed 's/^/    /'; then
    failed+=("$framework $spec (INVALID)")
  fi
done

python3 "$REPO/jetson/scripts/summarize.py" ${LIMIT:+--include-smoke} >/dev/null
echo "summary: $REPO/$RESULTS_DIR/SUMMARY.md"
[ ${#failed[@]} -eq 0 ] || { printf 'failed: %s\n' "${failed[@]}"; exit 1; }
