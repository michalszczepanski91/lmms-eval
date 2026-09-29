#!/usr/bin/env python3
"""Sanity-check a finished run dir's answers (stdlib only; run_matrix.sh calls it after every run).

- Answers made only of "?" (llama.cpp fp16 overflow: every later answer is "????" until the server restarts):
  at >= 1% of the samples the run gets an INVALID file with the reason (summarize.py skips such runs).
- Empty answers (e.g. Edge-LLM fp16 vision-encoder overflow: NaN logits decode to nothing): INVALID at >= 2%.
  Healthy runs of the short-answer tasks here have none.
- "[LMMS_EVAL_REQUEST_FAILED ...]" answers are counted and reported, not marked invalid.

Usage: python3 jetson/scripts/check_run.py <run dir> [<run dir> ...]    exit 1 if any run was marked INVALID
"""

import glob
import gzip
import json
import os
import re
import sys

QUESTION_MARKS = re.compile(r"^\s*\?{2,}\s*$")


def answer(sample):
    resp = sample.get("filtered_resps")
    while isinstance(resp, list):
        resp = resp[0] if resp else ""
    return str(resp or "")


def check(run_dir):
    counts = {"samples": 0, "question_marks": 0, "empty": 0, "request_failed": 0}
    for path in glob.glob(os.path.join(run_dir, "lmms_eval", "*_samples_*.jsonl*")):
        for line in gzip.open(path, "rt") if path.endswith(".gz") else open(path):
            text = answer(json.loads(line))
            counts["samples"] += 1
            counts["question_marks"] += bool(QUESTION_MARKS.match(text))
            counts["empty"] += not text.strip()
            counts["request_failed"] += text.startswith("[LMMS_EVAL_REQUEST_FAILED")
    print(f"{run_dir}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    reason = None
    if counts["samples"] and counts["question_marks"] >= max(1, 0.01 * counts["samples"]):
        reason = f"{counts['question_marks']} of {counts['samples']} answers are only '?' (llama.cpp numeric overflow); scores are not valid.\n"
    elif counts["samples"] and counts["empty"] >= max(1, 0.02 * counts["samples"]):
        reason = f"{counts['empty']} of {counts['samples']} answers are empty (e.g. NaN logits from a numeric overflow); scores are not valid.\n"
    if reason:
        with open(os.path.join(run_dir, "INVALID"), "w") as f:
            f.write(f"check_run.py: {reason}")
        print(f"{run_dir}: marked INVALID - {reason.strip()}")
        return False
    return True


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ok = [check(run_dir) for run_dir in sys.argv[1:]]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
