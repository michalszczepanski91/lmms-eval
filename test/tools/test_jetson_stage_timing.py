"""Per-stage latency (vision encoder / LLM prefill) used by the Jetson benchmarks.

- jetson/scripts/summarize.py: llama-server log parsing (verbosity 4) and TensorRT Edge-LLM profile stages
- lmms_eval/models/chat/qwen2_5_vl.py: _VisionTimer event pairing (CUDA events replaced by a fake clock)
"""

import gzip
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("jetson_summarize", REPO / "jetson" / "scripts" / "summarize.py")
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)

# One image request as llama-server b11135 logs it with -lv 4 (other lines trimmed), then a text-only request.
_SERVER_LOG = """\
0.03.512.296 I slot launch_slot_: id  0 | task 15 | processing task, is_child = 0
0.03.521.591 I slot process_mtmd: id  0 | task 15 | encoding mtmd batch from idx = 15, n_chunks = 1
0.04.192.191 I decoding image batch 1/1, n_tokens_batch = 999
0.04.354.303 I image decoded (batch 1/1) in 162 ms
0.04.699.810 I slot print_timing: id  0 | task 15 | prompt eval time =     988.52 ms /  1024 tokens (    0.97 ms per token,  1035.89 tokens per second)
0.04.699.817 I slot print_timing: id  0 | task 15 |        eval time =     198.95 ms /    13 tokens (   16.58 ms per token,    60.32 tokens per second)
"""
_TEXT_ONLY_LOG = """\
1.04.770.138 I slot launch_slot_: id  0 | task 30 | processing task, is_child = 0
1.05.000.000 I slot print_timing: id  0 | task 30 | prompt eval time =      20.00 ms /    25 tokens (    0.80 ms per token,  1250.00 tokens per second)
"""


class TestLlamaCppServerStats(unittest.TestCase):
    def stats(self, log):
        with tempfile.TemporaryDirectory() as run_dir:
            with gzip.open(os.path.join(run_dir, "server.log.gz"), "wt") as f:
                f.write(log)
            return summarize.llamacpp_server_stats(run_dir)

    def test_vision_and_prefill_from_log(self):
        stats = self.stats(_SERVER_LOG)
        self.assertAlmostEqual(stats["vision_p50"], 4.192191 - 3.521591, places=6)
        # LLM prefill = prompt eval time (which includes the encoder) - encoder.
        self.assertAlmostEqual(stats["prefill_p50"], 0.98852 - (4.192191 - 3.521591), places=6)
        self.assertEqual(stats["stage_source"], "server log")

    def test_text_only_request_keeps_full_prompt_time(self):
        stats = self.stats(_SERVER_LOG + _TEXT_ONLY_LOG + _TEXT_ONLY_LOG)  # prefills: 0.318, 0.02, 0.02
        self.assertAlmostEqual(stats["prefill_p50"], 0.02)

    def test_no_encoder_lines(self):
        with tempfile.TemporaryDirectory() as run_dir:
            Path(run_dir, "server.log").write_text("0.00.000.520 I srv  llama_server: initializing ...\n")
            self.assertEqual(summarize.llamacpp_server_stats(run_dir), {})


class TestTrtProfileStages(unittest.TestCase):
    def test_stage_medians(self):
        profile = {
            "prefill": {"average_time_per_run_ms": 39.1},
            "generation": {"average_time_per_token_ms": 15.2, "average_tokens_per_run": 2.3},
            "multimodal": {"average_time_per_token_ms": 0.1, "total_multimodal_tokens": 3500, "total_runs": 10},
            "stages": [
                {"stage_id": "vision_encoder", "gpu_time_stats": {"median_ms": 33.7}},
                {"stage_id": "llm_prefill", "gpu_time_stats": {"median_ms": 38.9}},
            ],
        }
        with tempfile.TemporaryDirectory() as run_dir:
            path = os.path.join(run_dir, "trt_profile.json")
            json.dump([{"profile": profile}], open(path, "w"))
            stats = summarize.trt_profile_stats(path)
        self.assertAlmostEqual(stats["vision_p50"], 0.0337)
        self.assertAlmostEqual(stats["prefill_p50"], 0.0389)
        self.assertAlmostEqual(stats["ttft_avg"], (35 + 39.1) / 1000)
        self.assertEqual(stats["out_tokens_mean"], 2.3)


class TestVisionTimer(unittest.TestCase):
    def test_pairs_pre_and_post_hooks(self):
        torch = __import__("torch")
        from lmms_eval.models.chat.qwen2_5_vl import _VisionTimer

        clock = iter([10.0, 35.0, 100.0, 104.0])  # ms: two encoder calls of 25 ms and 4 ms

        class FakeEvent:
            def __init__(self, enable_timing=False):
                self.t = None

            def record(self):
                self.t = next(clock)

            def synchronize(self):
                pass

            def elapsed_time(self, end):
                return end.t - self.t

        visual = torch.nn.Identity()
        with mock.patch.object(torch.cuda, "Event", FakeEvent):
            timer = _VisionTimer(visual)
            self.assertIsNone(timer.pop_seconds())
            visual(torch.zeros(1))
            visual(torch.zeros(1))
            self.assertAlmostEqual(timer.pop_seconds(), 0.029)
            self.assertIsNone(timer.pop_seconds())


if __name__ == "__main__":
    unittest.main()
