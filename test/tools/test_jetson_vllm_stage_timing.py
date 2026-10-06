"""vLLM backend stage_timing=True: vision-encoder time per sample (CPU only; the engine and CUDA events are faked).

- lmms_eval/models/model_utils/stage_timing.py: VisionTimer, shared with the HF backend
- lmms_eval/models/chat/vllm.py: in-process engine requirement, encoder hook via LLM.apply_model, and
  TokenCounts.vision_seconds per request (start-up encoder calls are dropped)
"""

import os
import types
import unittest
from unittest import mock

import torch

from lmms_eval.models.chat import vllm as vllm_chat
from lmms_eval.models.model_utils import stage_timing


class FakeClock:
    """Stand-in for torch.cuda.Event: each record() takes the next time (ms) from a list."""

    def __init__(self, times):
        self.times = iter(times)
        clock = self

        class Event:
            def __init__(self, enable_timing=False):
                self.t = None

            def record(self):
                self.t = next(clock.times)

            def synchronize(self):
                pass

            def elapsed_time(self, end):
                return end.t - self.t

        self.Event = Event


class FakeOutput:
    def __init__(self, text, ttft):
        self.prompt_token_ids = [1] * 10
        self.outputs = [types.SimpleNamespace(text=text, token_ids=[7, 8, 9])]
        self.metrics = types.SimpleNamespace(first_token_latency=ttft)


class FakeLLM:
    """Engine stand-in: apply_model runs on a model with a `visual` module; chat runs that module once per request."""

    def __init__(self):
        self.model = types.SimpleNamespace(visual=torch.nn.Identity())

    def apply_model(self, func):
        return [func(self.model)]

    def chat(self, sampling_params, messages, chat_template=None):
        self.model.visual(torch.zeros(1))
        return [FakeOutput("answer", 0.300)]


def make_backend(client, timer):
    backend = vllm_chat.VLLM.__new__(vllm_chat.VLLM)
    backend._rank = 0
    backend.batch_size_per_gpu = 1
    backend.disable_log_stats = True
    backend.chat_template = None
    backend.client = client
    backend._vision_timer = timer
    backend.make_one_request = lambda request: ([{"role": "user", "content": "q"}], {"max_tokens": 4})
    backend._run_tp_synced = lambda inputs, run_fn: run_fn(inputs)
    return backend


class TestSharedVisionTimer(unittest.TestCase):
    def test_hf_backend_uses_the_shared_timer(self):
        from lmms_eval.models.chat import qwen2_5_vl

        self.assertIs(qwen2_5_vl._VisionTimer, stage_timing.VisionTimer)


class TestVLLMStageTiming(unittest.TestCase):
    def test_vision_seconds_per_request(self):
        client = FakeLLM()
        # start-up encoder call (profiling): 1 -> 5 ms, then one call per request: 10 -> 35 ms and 100 -> 140 ms
        clock = FakeClock([1.0, 5.0, 10.0, 35.0, 100.0, 140.0])
        with mock.patch.object(torch.cuda, "Event", clock.Event), mock.patch.object(vllm_chat, "SamplingParams", lambda **kw: kw):
            timer = client.apply_model(lambda model: stage_timing.VisionTimer(model.visual))[0]
            client.model.visual(torch.zeros(1))  # profiling run before the first request
            results = make_backend(client, timer).generate_until([object(), object()])
        counts = [r.token_counts for r in results]
        self.assertAlmostEqual(counts[0].vision_seconds, 0.025)
        self.assertAlmostEqual(counts[1].vision_seconds, 0.040)
        self.assertAlmostEqual(counts[0].time_to_first_token_seconds, 0.300)
        self.assertIsNotNone(counts[0].generation_seconds)

    def test_without_stage_timing_no_vision_seconds(self):
        with mock.patch.object(vllm_chat, "SamplingParams", lambda **kw: kw):
            results = make_backend(FakeLLM(), None).generate_until([object()])
        self.assertIsNone(results[0].token_counts.vision_seconds)


class TestVLLMStageTimingSetup(unittest.TestCase):
    def init(self, **kwargs):
        def fake_simple_init(self, *args, **kw):
            self.client = FakeLLM()

        backend = vllm_chat.VLLM.__new__(vllm_chat.VLLM)
        with mock.patch.object(vllm_chat.VLLMSimple, "__init__", fake_simple_init):
            vllm_chat.VLLM.__init__(backend, **kwargs)
        return backend

    def test_runs_engine_in_process_and_hooks_encoder(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
            backend = self.init(stage_timing="True")
            self.assertEqual(os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"], "0")
        self.assertIsInstance(backend._vision_timer, stage_timing.VisionTimer)

    def test_refuses_multiprocessing_engine(self):
        with mock.patch.dict(os.environ, {"VLLM_ENABLE_V1_MULTIPROCESSING": "1"}):
            with self.assertRaises(ValueError):
                self.init(stage_timing=True)

    def test_refuses_multi_gpu(self):
        with mock.patch.dict(os.environ, {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}):
            with self.assertRaises(ValueError):
                self.init(stage_timing=True, tensor_parallel_size=2)

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
            backend = self.init()
            self.assertNotIn("VLLM_ENABLE_V1_MULTIPROCESSING", os.environ)
        self.assertIsNone(backend._vision_timer)


if __name__ == "__main__":
    unittest.main()
