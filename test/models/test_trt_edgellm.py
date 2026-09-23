from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from types import SimpleNamespace

from PIL import Image

from lmms_eval.models.chat.trt_edgellm import TRTEdgeLLM

# Stand-in for Edge-LLM's llm_inference: echoes each request's text and writes a profile.
_FAKE_RUNNER = """#!{python}
import json, sys
args = {{a: sys.argv[i + 1] if i + 1 < len(sys.argv) else None for i, a in enumerate(sys.argv) if a.startswith("--")}}
data = json.load(open(args["--inputFile"]))
responses = []
for i, req in enumerate(data["requests"]):
    text = [c["text"] for m in req["messages"] if isinstance(m["content"], list) for c in m["content"] if c["type"] == "text"][0]
    responses.append({{"request_idx": i, "batch_idx": 0, "output_text": "echo:" + text}})
json.dump({{"responses": responses}}, open(args["--outputFile"], "w"))
json.dump({{"prefill": {{"average_time_per_run_ms": 12.5}}}}, open(args["--profileOutputFile"], "w"))
"""


def _request(text: str, max_new_tokens: int, doc_id: int) -> SimpleNamespace:
    def doc_to_messages(doc):
        return [{"role": "user", "content": [{"type": "image", "url": doc["image"]}, {"type": "text", "text": text}]}]

    return SimpleNamespace(args=("", doc_to_messages, {"max_new_tokens": max_new_tokens}, doc_id, "demo", "test"))


class TestTRTEdgeLLM(unittest.TestCase):
    def test_generate_until_runs_edgellm_and_maps_responses(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = os.path.join(tmp, "llm_inference")
            with open(runner, "w") as f:
                f.write(_FAKE_RUNNER.format(python=sys.executable))
            os.chmod(runner, os.stat(runner).st_mode | stat.S_IEXEC)
            profile_output = os.path.join(tmp, "out", "trt_profile.json")

            model = TRTEdgeLLM(engine_dir="engines/llm", multimodal_engine_dir="engines/visual", inference_bin=runner, profile_output=profile_output)
            image = Image.new("RGB", (8, 8))
            model.task_dict = {"demo": {"test": [{"image": image}, {"image": image}, {"image": image}]}}

            results = model.generate_until([_request("a", 16, 0), _request("b", 32, 1), _request("c", 16, 2)])

            self.assertEqual([r.text for r in results], ["echo:a", "echo:b", "echo:c"])
            profiles = json.load(open(profile_output))
            self.assertEqual(sorted(p["max_new_tokens"] for p in profiles), [16, 32])
            self.assertEqual(profiles[0]["profile"]["prefill"]["average_time_per_run_ms"], 12.5)
