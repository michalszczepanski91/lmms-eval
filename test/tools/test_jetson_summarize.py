"""jetson/scripts/summarize.py: run markers (SUPERSEDED, TIMING_INVALID, INVALID) and the "Excluded runs" list."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("jetson_summarize_markers", REPO / "jetson" / "scripts" / "summarize.py")
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)


def make_run(results, run, score, marker=None, reason=""):
    run_dir = os.path.join(results, "Model-3B", "gqa", run.split("/")[0], run.split("/")[1])
    os.makedirs(os.path.join(run_dir, "lmms_eval"))
    result = {"results": {"gqa": {"alias": "gqa", "exact_match,none": score, "exact_match_stderr,none": 0.01}}, "n-samples": {"gqa": {"effective": 100}}, "total_evaluation_time_seconds": "600"}
    with open(os.path.join(run_dir, "lmms_eval", "x_results.json"), "w") as f:
        json.dump(result, f)
    with open(os.path.join(run_dir, "run_info.txt"), "w") as f:
        f.write("board:       Test Board\npower mode:  NV Power Mode: MAXN\nother containers: \n")
    if marker:
        with open(os.path.join(run_dir, marker), "w") as f:
            f.write(reason + "\n")


class TestSummarizeMarkers(unittest.TestCase):
    def summary(self):
        with tempfile.TemporaryDirectory() as results:
            make_run(results, "hf-bf16/20260101-000000", 0.601234)
            make_run(results, "vllm-bf16/20260101-010000", 0.5, "SUPERSEDED", "Replaced by a later run.")
            make_run(results, "vllm-awq/20260101-020000", 0.59, "TIMING_INVALID", "Foreign CPU load.")
            make_run(results, "llamacpp-q8_0/20260101-030000", 0.1, "INVALID", "check_run.py: 50 of 100 answers are empty.")
            with mock.patch.object(summarize, "RESULTS", results), mock.patch.object(summarize.sys, "argv", ["summarize.py"]), contextlib.redirect_stdout(io.StringIO()):
                summarize.main()
            return open(os.path.join(results, "SUMMARY.md")).read()

    def test_main_table_keeps_valid_runs_only(self):
        text = self.summary()
        main_table = text.split("## Latency breakdown")[0]
        self.assertIn("| hf-bf16 | gqa | exact_match: 0.6012 |", main_table)
        self.assertIn("| MAXN |", main_table)
        self.assertNotIn("vllm-bf16", main_table)
        self.assertNotIn("llamacpp-q8_0", main_table)

    def test_timing_invalid_keeps_score_drops_timings(self):
        text = self.summary()
        main_table, rest = text.split("## Latency breakdown")
        row = next(line for line in main_table.splitlines() if "| vllm-awq |" in line)
        self.assertIn("exact_match: 0.5900", row)
        self.assertIn("timings not valid: Foreign CPU load.", row)
        self.assertNotIn("| vllm-awq |", rest.split("## Excluded runs")[0])  # no latency row

    def test_excluded_runs_list_reasons(self):
        excluded = self.summary().split("## Excluded runs")[1]
        self.assertIn("| SUPERSEDED | exact_match: 0.5000 | Replaced by a later run. |", excluded)
        self.assertIn("| INVALID | exact_match: 0.1000 | 50 of 100 answers are empty. |", excluded)


if __name__ == "__main__":
    unittest.main()
