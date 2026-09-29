"""jetson/scripts/check_run.py: which runs get an INVALID file ("????" answers, empty answers)."""

import contextlib
import gzip
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("jetson_check_run", REPO / "jetson" / "scripts" / "check_run.py")
check_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_run)


class TestCheckRun(unittest.TestCase):
    def check(self, answers):
        with tempfile.TemporaryDirectory() as run_dir:
            os.makedirs(os.path.join(run_dir, "lmms_eval"))
            with gzip.open(os.path.join(run_dir, "lmms_eval", "x_samples_textvqa_val.jsonl.gz"), "wt") as f:
                for doc_id, answer in enumerate(answers):
                    f.write(json.dumps({"doc_id": doc_id, "filtered_resps": [answer]}) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                ok = check_run.check(run_dir)
            invalid = os.path.join(run_dir, "INVALID")
            return ok, open(invalid).read() if os.path.exists(invalid) else None

    def test_healthy_run_passes(self):
        self.assertEqual(self.check(["yes"] * 99 + [""]), (True, None))

    def test_question_marks_invalid(self):
        ok, reason = self.check(["yes"] * 98 + ["????"] * 2)
        self.assertFalse(ok)
        self.assertIn("only '?'", reason)

    def test_many_empty_answers_invalid(self):
        ok, reason = self.check(["yes"] * 80 + [""] * 20)
        self.assertFalse(ok)
        self.assertIn("20 of 100 answers are empty", reason)

    def test_few_empty_answers_pass(self):
        self.assertEqual(self.check(["yes"] * 99 + [" "]), (True, None))


if __name__ == "__main__":
    unittest.main()
