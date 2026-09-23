"""TensorRT Edge-LLM backend (NVIDIA Jetson / DRIVE).

Edge-LLM ships a C++ runner (``llm_inference``) that reads a JSON file of chat requests,
so this backend writes every request of a ``generate_until`` call into one input file,
runs the binary once (engines load once) and maps responses back by ``request_idx``.

Latency is only available as the runner's aggregate profile (vision encoder, prefill,
per-token decode), not per sample; pass ``profile_output`` to keep it.

``max_image_side`` downscales larger images (aspect ratio kept) before they reach the runner: Edge-LLM
rejects raw images with a side above its GPU-resize budget (4096 px in v0.10), e.g. MME's landmark photos.

Requests the runner fails (``finish_reason == "error"``) are scored as empty answers and
counted in the profile (``failed_requests``) instead of aborting the whole evaluation.

Engines are built with Edge-LLM's ``llm_build`` / ``visual_build`` (see jetson/frameworks/trt_edgellm/).
"""

import json
import os
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages


@register_model("trt_edgellm")
class TRTEdgeLLM(lmms):
    is_simple = False

    def __init__(
        self,
        engine_dir: str,
        multimodal_engine_dir: Optional[str] = None,
        inference_bin: str = "llm_inference",
        system_prompt: Optional[str] = "You are a helpful assistant.",
        profile_output: Optional[str] = None,
        warmup: int = 1,
        encoder_cache_budget_bytes: Optional[int] = None,
        max_image_side: Optional[int] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        assert int(batch_size) == 1, "trt_edgellm engines are built for batch size 1 here"
        self.engine_dir = engine_dir
        self.multimodal_engine_dir = multimodal_engine_dir
        self.inference_bin = inference_bin
        self.system_prompt = system_prompt
        self.profile_output = profile_output
        self.warmup = int(warmup)
        # Edge-LLM >= 0.10 caches vision-encoder outputs across requests (256 MiB by default); 0 disables it.
        # None = don't pass the flag (older runners don't know it).
        self.encoder_cache_budget_bytes = encoder_cache_budget_bytes
        self.max_image_side = int(max_image_side) if max_image_side else None

    def _to_edgellm_messages(self, chat_messages: ChatMessages, media_dir: str, request_index: int) -> list:
        messages = []
        if self.system_prompt and not any(m.role == "system" for m in chat_messages.messages):
            messages.append({"role": "system", "content": self.system_prompt})
        image_count = 0
        for message in chat_messages.messages:
            content = []
            for item in message.content:
                if item.type == "text":
                    content.append({"type": "text", "text": item.text})
                elif item.type == "image":
                    image = item.url
                    if not isinstance(image, Image.Image) and self.max_image_side:
                        image = Image.open(image)
                    if isinstance(image, Image.Image):
                        image = image.convert("RGB")
                        if self.max_image_side and max(image.size) > self.max_image_side:
                            scale = self.max_image_side / max(image.size)
                            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.BICUBIC)
                        path = os.path.join(media_dir, f"{request_index}_{image_count}.png")
                        image.save(path)
                        image = path
                    image_count += 1
                    content.append({"type": "image", "image": image})
                else:
                    raise NotImplementedError(f"trt_edgellm backend does not support {item.type} inputs yet")
            messages.append({"role": message.role, "content": content})
        return messages

    def _run(self, requests: List[dict], max_new_tokens: int, workdir: str) -> Tuple[List[str], Optional[dict]]:
        input_file = os.path.join(workdir, "input.json")
        output_file = os.path.join(workdir, "output.json")
        profile_file = os.path.join(workdir, "profile.json")
        # Greedy decoding to match the other backends (temperature 0 in lmms-eval task configs).
        with open(input_file, "w") as f:
            json.dump({"batch_size": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 1, "max_generate_length": max_new_tokens, "requests": requests}, f)
        cmd = [self.inference_bin, "--engineDir", self.engine_dir, "--inputFile", input_file, "--outputFile", output_file, "--dumpProfile", "--profileOutputFile", profile_file, "--warmup", str(self.warmup)]
        if self.multimodal_engine_dir:
            cmd += ["--multimodalEngineDir", self.multimodal_engine_dir]
        if self.encoder_cache_budget_bytes is not None:
            cmd += ["--encoderCacheBudgetBytes", str(self.encoder_cache_budget_bytes)]
        eval_logger.info(f"Running TensorRT Edge-LLM: {' '.join(cmd)}")
        # The runner exits non-zero when any request failed but still writes every response.
        returncode = subprocess.run(cmd).returncode
        if not os.path.exists(output_file):
            raise RuntimeError(f"TensorRT Edge-LLM runner failed (exit {returncode}) without writing {output_file}")

        with open(output_file) as f:
            responses = json.load(f)["responses"]
        texts = [""] * len(requests)
        failed = 0
        for response in responses:
            if response.get("finish_reason") == "error":
                failed += 1
                eval_logger.warning(f"TensorRT Edge-LLM request {response['request_idx']} failed: {response['output_text']}")
                continue
            texts[response["request_idx"]] = response["output_text"]
        profile = None
        if os.path.exists(profile_file):
            with open(profile_file) as f:
                profile = json.load(f)
            profile["failed_requests"] = failed
        return texts, profile

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        results: List[Optional[GenerationResult]] = [None] * len(requests)
        # One runner invocation per max_new_tokens value (a single file shares one generation length).
        groups: dict = {}
        for index, request in enumerate(requests):
            gen_kwargs = request.args[2]
            groups.setdefault(int(gen_kwargs.get("max_new_tokens", 128)), []).append(index)

        profiles = []
        started_at = time.perf_counter()
        for max_new_tokens, indices in groups.items():
            with tempfile.TemporaryDirectory(prefix="trt_edgellm_") as workdir:
                edgellm_requests = []
                for position, index in enumerate(indices):
                    _, doc_to_messages, _, doc_id, task, split = requests[index].args
                    chat_messages = ChatMessages(messages=doc_to_messages(self.task_dict[task][split][doc_id]))
                    edgellm_requests.append({"messages": self._to_edgellm_messages(chat_messages, workdir, position)})
                texts, profile = self._run(edgellm_requests, max_new_tokens, workdir)
            if profile is not None:
                profiles.append({"max_new_tokens": max_new_tokens, "num_requests": len(indices), "profile": profile})
            for index, text in zip(indices, texts):
                results[index] = GenerationResult(text=text, token_counts=TokenCounts())
        elapsed = time.perf_counter() - started_at

        if self.profile_output and profiles:
            os.makedirs(os.path.dirname(os.path.abspath(self.profile_output)), exist_ok=True)
            with open(self.profile_output, "w") as f:
                json.dump(profiles, f, indent=2)
            eval_logger.info(f"TensorRT Edge-LLM profile saved to {self.profile_output}")
        log_metrics(total_elapsed_time=elapsed, total_gen_tokens=0, avg_speed=0)
        return results

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("trt_edgellm backend does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("trt_edgellm backend does not support multi-round generation")
