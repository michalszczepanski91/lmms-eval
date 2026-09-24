#!/usr/bin/env python3
"""Latency vs fixed image size, per framework, from fixed_size_sweep.sh runs -> <results>/fixed_size_sweep.md."""

import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, os.environ.get("RESULTS_DIR", "jetson/results"))


def pct(values, q):
    values = sorted(values)
    return values[int(q * (len(values) - 1))] if values else None


rows = []
for samples in sorted(glob.glob(os.path.join(RESULTS, "*", "mme_fixed", "*", "*", "lmms_eval", "*_samples_*.jsonl"))):
    run_dir = os.path.dirname(os.path.dirname(samples))
    model, _, variant, _ = os.path.relpath(run_dir, RESULTS).split(os.sep)
    side = int(re.search(r"side(\d+)", variant).group(1))
    framework = re.sub(r"[+-]?side\d+$", "", variant)
    counts = [json.loads(line)["token_counts"][0] for line in open(samples)][1:]  # drop the warm-up sample
    ttft = [c["time_to_first_token_seconds"] * 1000 for c in counts if c.get("time_to_first_token_seconds")]
    gen = [c["generation_seconds"] * 1000 for c in counts if c.get("generation_seconds")]
    tokens = sorted({c.get("input_tokens") for c in counts if c.get("input_tokens")})
    rows.append((model, framework, side, (side // 28) ** 2, tokens, len(ttft), pct(ttft, 0.5), pct(ttft, 0.9), pct(gen, 0.5)))

lines = [
    "# Fixed-size latency sweep",
    "",
    "Every image resized to side x side pixels (MME, first 200 samples, warm-up sample excluded). Visual tokens = (side/28)^2;",
    "input tokens = visual + text tokens as counted by each backend. TTFT = model call to first token; answer = whole call.",
    "",
    "| Model | Framework | Side (px) | Visual tokens | Input tokens (min-max) | N | TTFT p50 / p90 (ms) | Answer p50 (ms) |",
    "|---|---|---:|---:|---|---:|---:|---:|",
]
for model, framework, side, vis, tokens, n, t50, t90, g50 in sorted(rows):
    tok = f"{tokens[0]}-{tokens[-1]}" if tokens else "-"
    lat = f"{t50:.0f} / {t90:.0f}" if t50 is not None else "-"
    ans = f"{g50:.0f}" if g50 is not None else "-"
    lines.append(f"| {model} | {framework} | {side} | {vis} | {tok} | {n} | {lat} | {ans} |")
out = os.path.join(RESULTS, "fixed_size_sweep.md")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
