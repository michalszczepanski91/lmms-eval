#!/usr/bin/env python3
"""Analytical FLOPs of a text-only Qwen2.5 decoder for every sample of a run_eval.sh run -> <run dir>/flops.json.

Same interface and output as jetson/models/qwen2_5_vl/flops.py (so compute_flops.sh and summarize.py work unchanged),
with no vision encoder: image tokens and vision FLOPs are 0. The language-model count is the VL file's `llm_flops`
(matrix multiplies at 2 FLOPs per multiply-accumulate: q, k, v, o projections, SwiGLU MLP, causal attention as
L(L+1)/2 query-key pairs, the LM head for every generated token), validated against torch's FlopCounterMode there.
  prefill - the whole prompt (chat template + task text) + the LM head for the first output token
  decode  - one forward per further output token against the KV cache + the LM head
Prompt / output token counts come from the backend (input_tokens / output_tokens); TensorRT Edge-LLM records none,
so they are recounted with the tokenizer and the model's chat template.

Usage: python jetson/models/qwen2_5/flops.py <run dir> [<run dir> ...]
"""

import glob
import gzip
import importlib.util
import json
import os
import re
import sys
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location("qwen2_5_vl_flops", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen2_5_vl", "flops.py"))
_vl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vl)
llm_flops = _vl.llm_flops


def load_config(model_tag):
    from huggingface_hub import snapshot_download

    path = snapshot_download(f"Qwen/{model_tag}", allow_patterns=["*.json"], local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
    config = json.load(open(os.path.join(path, "config.json")))
    return SimpleNamespace(**config), path


def per_token_flops(text_cfg):
    """(linear FLOPs per token without attention, LM head FLOPs) - for the docs and the bandwidth figure."""
    h, layers = text_cfg.hidden_size, text_cfg.num_hidden_layers
    kv = text_cfg.num_key_value_heads * (h // text_cfg.num_attention_heads)
    return 2 * layers * (2 * h * h + 2 * h * kv + 3 * h * text_cfg.intermediate_size), 2 * h * text_cfg.vocab_size


def run_flops(run_dir):
    model_tag = os.path.relpath(run_dir).split(os.sep)[-4]
    text_cfg, model_path = load_config(model_tag)
    tokenizer = None
    tasks = {}
    # max_new_tokens per task: an answer cut at the cap has no end token, so the recount adds none there.
    caps = {}
    for results_path in glob.glob(os.path.join(run_dir, "lmms_eval", "*_results.json")):
        for name, config in json.load(open(results_path)).get("configs", {}).items():
            caps[name] = (config.get("generation_kwargs") or {}).get("max_new_tokens")
    for samples_path in sorted(glob.glob(os.path.join(run_dir, "lmms_eval", "*_samples_*.jsonl*"))):
        task_name = re.sub(r"\.jsonl(\.gz)?$", "", re.sub(r"^.*?_samples_", "", os.path.basename(samples_path)))
        rows, recounted = [], 0
        for line in gzip.open(samples_path, "rt") if samples_path.endswith(".gz") else open(samples_path):
            sample = json.loads(line)
            counts = (sample.get("token_counts") or [None])[0] or {}
            if counts.get("input_tokens") is not None and counts.get("output_tokens") is not None:
                prompt_tokens, output_tokens = counts["input_tokens"], counts["output_tokens"]
            else:
                if tokenizer is None:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                prompt_tokens = len(tokenizer.apply_chat_template([{"role": "user", "content": sample["input"]}], add_generation_prompt=True))
                response = sample.get("resps", sample.get("filtered_resps", ""))
                while isinstance(response, list):
                    response = response[0] if response else ""
                output_tokens = len(tokenizer(str(response))["input_ids"]) + 1  # + <|im_end|>
                if caps.get(task_name):
                    output_tokens = min(output_tokens, int(caps[task_name]))
                recounted += 1
            prefill, decode = llm_flops(text_cfg, prompt_tokens, output_tokens)
            rows.append(
                {
                    "doc_id": sample["doc_id"],
                    "image_grids": [],
                    "image_tokens": 0,
                    "text_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "vision_flops": 0,
                    "prefill_flops": prefill,
                    "decode_flops": decode,
                    "total_flops": prefill + decode,
                }
            )
        n = len(rows)
        keys = ("image_tokens", "text_tokens", "output_tokens", "vision_flops", "prefill_flops", "decode_flops", "total_flops")
        tasks[task_name] = {
            "n": n,
            "token_counts": "recorded" if not recounted else ("tokenizer" if recounted == n else f"tokenizer for {recounted} of {n}"),
            "mean": {k: sum(r[k] for r in rows) / max(1, n) for k in keys},
            "total": {k: sum(r[k] for r in rows) for k in keys},
            "samples": rows,
        }
        mean = tasks[task_name]["mean"]
        print(f"{run_dir} {task_name}: n={n} mean GFLOPs prefill {mean['prefill_flops'] / 1e9:.1f} decode {mean['decode_flops'] / 1e9:.1f} total {mean['total_flops'] / 1e9:.1f}", flush=True)

    out = {
        "model": model_tag,
        "method": "analytical, 2 FLOPs per multiply-accumulate; decoder linear layers + attention matmuls + LM head (see jetson/models/qwen2_5/flops.py)",
        "tasks": tasks,
    }
    json.dump(out, open(os.path.join(run_dir, "flops.json"), "w"), indent=1)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for run_dir in sys.argv[1:]:
        run_flops(run_dir.rstrip("/"))


if __name__ == "__main__":
    main()
