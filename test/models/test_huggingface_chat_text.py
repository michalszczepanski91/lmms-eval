"""Generic HF chat backend with a text-only causal LM (CPU only; model and tokenizer are faked).

- text-only requests go through the tokenizer, not the multimodal processor
- per-sample TokenCounts: input / output tokens, preprocess, time to first token (generation streamer), generation
"""

import types
import unittest

import torch

from lmms_eval.models.chat.huggingface import Huggingface


class FakeBatch(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return self


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, texts, padding=True, return_tensors="pt", **kwargs):
        self.calls.append(texts)
        return FakeBatch(input_ids=torch.tensor([[5, 6, 7, 8]]), attention_mask=torch.tensor([[1, 1, 1, 1]]))


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Like a text-model template (Qwen2.5-Instruct): content must be a string.
        return "<chat>" + messages[0]["content"]

    def __call__(self, **kwargs):
        raise AssertionError("text-only requests must not go through the multimodal processor")

    def batch_decode(self, ids, **kwargs):
        return [" ".join(str(int(t)) for t in row) for row in ids]


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def generate(self, input_ids, attention_mask, streamer=None, max_new_tokens=None, **kwargs):
        self.kwargs = kwargs
        if streamer is not None:
            streamer.put(input_ids)  # prompt
            for _ in range(3):
                streamer.put(None)  # one call per generated token
            streamer.end()
        return torch.cat([input_ids, torch.tensor([[11, 12, 13]])], dim=1)


def make_backend():
    backend = Huggingface.__new__(Huggingface)
    backend._model = FakeModel()
    backend._tokenizer = FakeTokenizer()
    backend.processor = FakeProcessor()
    backend.batch_size_per_gpu = 1
    backend._rank = 0
    backend.system_prompt = None
    backend.device_map = "auto"
    backend.use_cache = True
    backend.cache_hook = types.SimpleNamespace(add_partial=lambda *args: None)
    backend.task_dict = {"demo": {"test": [{"q": "What is 2+2?"}]}}
    return backend


def text_request():
    def doc_to_messages(doc):
        return [{"role": "user", "content": [{"type": "text", "text": doc["q"]}]}]

    return types.SimpleNamespace(args=("ctx", doc_to_messages, {"max_new_tokens": 8}, 0, "demo", "test"))


class TestHuggingfaceTextOnly(unittest.TestCase):
    def test_text_request_uses_tokenizer_and_records_timing(self):
        backend = make_backend()
        result = backend.generate_until([text_request()])[0]
        self.assertEqual(backend.tokenizer.calls, [["<chat>What is 2+2?"]])
        self.assertEqual(result.text, "11 12 13")
        counts = result.token_counts
        self.assertEqual((counts.input_tokens, counts.output_tokens), (4, 3))
        for field in ("preprocess_seconds", "time_to_first_token_seconds", "generation_seconds"):
            self.assertIsNotNone(getattr(counts, field), field)
        self.assertLessEqual(counts.time_to_first_token_seconds, counts.generation_seconds)

    def test_greedy_has_no_repetition_penalty_from_the_checkpoint(self):
        # generate() would otherwise take repetition_penalty from the checkpoint's generation_config.json.
        backend = make_backend()
        backend.generate_until([text_request()])
        self.assertEqual(backend.model.kwargs["repetition_penalty"], 1.0)
        self.assertFalse(backend.model.kwargs["do_sample"])


class TestModelClass(unittest.TestCase):
    def test_text_model_gets_causal_lm_head(self):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, Qwen2_5_VLConfig, Qwen2Config

        self.assertIs(Huggingface.model_class_for(Qwen2Config()), AutoModelForCausalLM)
        self.assertIs(Huggingface.model_class_for(Qwen2_5_VLConfig()), AutoModelForImageTextToText)


if __name__ == "__main__":
    unittest.main()
