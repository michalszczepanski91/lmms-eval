#!/usr/bin/env python3
"""Figures of the text-only Qwen2.5 study (FINDINGS_qwen2_5.md), from the result files and the model artifacts.

  bandwidth  measured decode ms/token (GSM8K, median) against the bytes one decode step reads (weights without the
             embedding table + KV cache at the mean context), with the floor of Thor's 273 GB/s.
             Bytes come from each framework's own artifact: safetensors headers (HF, vLLM), the Edge-LLM llm.engine
             file, GGUF tensor headers (llama.cpp).
  sweep      controlled-length sweep: TTFT and decode ms/token against prompt length (128 / 512 / 2,048 tokens,
             256 generated), 16- and 8-bit per framework, both sizes.
  pareto     Edge-LLM weight configurations (Phase 2): accuracy against energy per question, per task and size.

Usage: python3 jetson/experiments/qwen2_5/figures.py bandwidth|pareto [-o out.svg]
"""

import argparse
import glob
import json
import os
import struct
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402

HF_CACHE = os.environ.get("HF_HUB_CACHE", "/opt/hf-cache/hub")
TRT_WORKSPACE = os.environ.get("TRT_WORKSPACE", "/opt/models/trt-edgellm")
BANDWIDTH = 273e9
# plot_stages.py's ink; framework colours validated for all pairs (CVD ΔE 9.2, normal 18.0); green is below 3:1 on
# white, so every point carries a text label.
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
FRAMEWORK = {"hf": ("HF transformers", "#b0304f"), "vllm": ("vLLM", "#2a78d6"), "llamacpp": ("llama.cpp", "#eb6834"), "trt_edgellm": ("TensorRT Edge-LLM", "#1baf7a")}
KV_BYTES = {"1.5b": 28 * 1024, "3b": 36 * 1024}  # 16-bit K + V per token (models/qwen2_5/docs)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I32": 4, "I8": 1, "U8": 1, "I64": 8}
# GGML block sizes: type id -> (bytes per block, values per block)
GGML = {0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32), 8: (34, 32), 10: (84, 256), 11: (110, 256), 12: (144, 256), 13: (176, 256), 14: (210, 256), 15: (292, 256), 30: (2, 1)}


def safetensors_step_bytes(repo):
    """Bytes of every tensor except the embedding table; a tied model (no lm_head) reads the embedding as its head."""
    sizes = {}
    for path in glob.glob(f"{HF_CACHE}/models--{repo.replace('/', '--')}/snapshots/*/*.safetensors"):
        with open(path, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for name, t in header.items():
            if name != "__metadata__":
                sizes[name] = t["data_offsets"][1] - t["data_offsets"][0]
    total = sum(sizes.values())
    return total - sizes.get("model.embed_tokens.weight", 0) if "lm_head.weight" in sizes else total


def gguf_step_bytes(paths):
    """Sum of the GGUF tensors read per token: all but token_embd (unless the model has no separate output.weight)."""
    sizes = {}
    for path in paths:
        with open(path, "rb") as f:

            def u(fmt):
                return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

            def string():
                return f.read(u("Q")).decode()

            def skip(t):
                if t == 8:
                    string()
                elif t == 9:
                    et, n = u("I"), u("Q")
                    for _ in range(n):
                        skip(et)
                else:
                    f.read({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t])

            assert f.read(4) == b"GGUF"
            u("I")
            n_tensors, n_kv = u("Q"), u("Q")
            for _ in range(n_kv):
                string()
                skip(u("I"))
            for _ in range(n_tensors):
                name = string()
                dims = [u("Q") for _ in range(u("I"))]
                ttype = u("I")
                u("Q")
                n = 1
                for d in dims:
                    n *= d
                block_bytes, block = GGML[ttype]
                sizes[name] = n // block * block_bytes
    total = sum(sizes.values())
    return total - sizes.get("token_embd.weight", 0) if "output.weight" in sizes else total


def step_bytes(size, config):
    """Weight bytes one decode step reads for a Phase 1 configuration."""
    tag = A.SIZES[size]
    fw, precision = config.split("-", 1)
    if fw == "trt_edgellm":
        return os.path.getsize(f"{TRT_WORKSPACE}/{tag}-{precision}/engines/llm/llm.engine")
    if fw == "llamacpp":
        prefix = {"f16": "fp16", "q8_0": "q8_0", "q4_k_m": "q4_k_m"}[precision]
        return gguf_step_bytes(sorted(glob.glob(f"{HF_CACHE}/models--Qwen--{tag}-GGUF/snapshots/*/{tag.lower()}-{prefix}*.gguf")))
    repo = {"bf16": f"Qwen/{tag}", "fp8": f"RedHatAI/{tag}-FP8-dynamic", "awq": f"Qwen/{tag}-AWQ"}[precision]
    return safetensors_step_bytes(repo)


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def bandwidth(out, task="gsm8k_jetson"):
    """Dumbbell per configuration: the bandwidth floor (bytes per step / 273 GB/s) and the measured decode time."""
    table = A.collect(list(A.SIZES), [task])
    order = [c for c in A.WIDTH]  # 16-bit, then 8-bit, then 4-bit
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True, gridspec_kw=dict(left=0.15, right=0.97, top=0.83, bottom=0.11, wspace=0.12))
    rows = []
    for ax, size in zip(axes, ("1.5b", "3b")):
        style(ax)
        ax.grid(axis="y", visible=False)
        y, ticks, labels = 0, [], []
        for width in (16, 8, 4):
            for config in [c for c in order if A.WIDTH[c] == width]:
                m = (table.get((size, task)) or {}).get(config)
                fw = A.framework_of(config)
                ticks.append(y)
                labels.append(f"{FRAMEWORK[fw][0].replace('TensorRT ', '').replace(' transformers', '')} {config.split('-', 1)[1]}")
                if m and m.get("decode_ms"):
                    context = 752 + (m["out_tokens"] or 0) / 2  # GSM8K mean prompt + half the answer
                    gb = (step_bytes(size, config) + context * KV_BYTES[size]) / 1e9
                    floor = 1e3 * gb * 1e9 / BANDWIDTH
                    ms = m["decode_ms"]
                    ax.plot([floor, ms], [y, y], color=GRID, lw=3, solid_capstyle="butt", zorder=1)
                    ax.plot([floor], [y], marker="|", markersize=12, mew=2, color=MUTED, zorder=2)
                    ax.scatter([ms], [y], s=60, color=FRAMEWORK[fw][1], edgecolor="white", linewidth=1.5, zorder=3)
                    ax.text(ms + 0.8, y, f"{ms:.1f} ms · {100 * floor / ms:.0f}%", va="center", fontsize=8.5, color=INK)
                    rows.append((size, config, gb, ms, floor))
                y += 1
            y += 0.6
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=9, color=INK)
        ax.set_xlim(0, 48)
        ax.set_title(A.SIZES[size], loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.set_xlabel("Decode, ms per token (GSM8K median)", fontsize=9, color=MUTED)
    axes[0].invert_yaxis()  # shared y axis: invert once, 16-bit on top
    handles = [plt.Line2D([], [], ls="", marker="|", markersize=12, mew=2, color=MUTED, label="floor: bytes per step / 273 GB/s")]
    handles += [plt.Line2D([], [], ls="", marker="o", color=c, markersize=8, label=f"measured, {n}") for n, c in FRAMEWORK.values()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.15, 0.915), ncol=5, frameon=False, fontsize=8.5)
    fig.text(0.15, 0.955, "Decode time against the memory-bandwidth floor", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.035, "Bytes per step: weights without the embedding table (safetensors / GGUF headers, Edge-LLM llm.engine) + KV cache at the mean GSM8K context.", fontsize=8, color=MUTED)
    fig.text(0.02, 0.012, "% = floor / measured, the share of Thor's bandwidth reached. Edge-LLM: median GPU time per step; the others: per-request wall time.", fontsize=8, color=MUTED)
    fig.text(0.97, 0.012, "python3 jetson/experiments/qwen2_5/figures.py bandwidth", fontsize=7, color="#8a8a8a", ha="right")
    fig.savefig(out, facecolor="white")
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=150, facecolor="white")
    print("| Size | Configuration | GB per step | Floor (ms) | Measured (ms) | Share of bandwidth |\n|---|---|---:|---:|---:|---:|")
    for size, config, gb, ms, floor in rows:
        print(f"| {size} | {config} | {gb:.2f} | {floor:.1f} | {ms:.1f} | {100 * floor / ms:.0f}% |")
    print("wrote", out)

SWEEP = [128, 512, 2048]
SWEEP_CONFIGS = {"hf-bf16": 16, "vllm-bf16": 16, "llamacpp-f16": 16, "trt_edgellm-fp16": 16, "vllm-fp8": 8, "llamacpp-q8_0": 8, "trt_edgellm-fp8": 8}


def sweep(out):
    tasks = [f"length_sweep_{n}" for n in SWEEP]
    table = A.collect(list(A.SIZES), tasks, include_smoke=os.environ.get("INCLUDE_SMOKE") == "1")  # INCLUDE_SMOKE=1: test on smoke runs
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, gridspec_kw=dict(left=0.08, right=0.8, top=0.88, bottom=0.08, hspace=0.18, wspace=0.18))
    rows = []
    for col, size in enumerate(("1.5b", "3b")):
        for row, (key, label) in enumerate((("ttft_ms", "Time to first token (ms, median)"), ("decode_ms", "Decode (ms per token, median)"))):
            ax = axes[row][col]
            style(ax)
            for config, bits in SWEEP_CONFIGS.items():
                xs, ys = [], []
                for n, task in zip(SWEEP, tasks):
                    m = (table.get((size, task)) or {}).get(config)
                    if m and m.get(key):
                        xs.append(n)
                        ys.append(m[key])
                        if row == 0:
                            rows.append((size, config, n, m.get("ttft_ms"), m.get("decode_ms"), m.get("out_tokens")))
                if xs:
                    fw = A.framework_of(config)
                    ax.plot(xs, ys, color=FRAMEWORK[fw][1], lw=2, ls="-" if bits == 16 else (0, (4, 2)), marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=f"{FRAMEWORK[fw][0]} {config.split('-', 1)[1]}")
            ax.set_xscale("log", base=2)
            ax.set_xticks(SWEEP)
            ax.set_xticklabels([f"{n:,}" for n in SWEEP])
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(A.SIZES[size], loc="left", fontsize=11, fontweight="bold", color=INK)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, color=MUTED)
            if row == 1:
                ax.set_xlabel("Prompt tokens (chat template included); 256 tokens generated", fontsize=9, color=MUTED)
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.81, 0.88), frameon=False, fontsize=8.5, handlelength=3.5, title="solid 16-bit, dashed 8-bit", title_fontsize=8.5)
    fig.text(0.08, 0.945, "Prefill and decode against prompt length", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.012, "16 prompts per length. Edge-LLM TTFT: median prefill GPU time of its runner (no host overhead); the others: per-request wall time.", fontsize=8, color=MUTED)
    fig.text(0.98, 0.012, "python3 jetson/experiments/qwen2_5/figures.py sweep", fontsize=7, color="#8a8a8a", ha="right")
    fig.savefig(out, facecolor="white")
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=150, facecolor="white")
    print("| Size | Configuration | Prompt tokens | TTFT p50 (ms) | Decode (ms/token) | Output tokens |\n|---|---|---:|---:|---:|---:|")
    for size, config, n, ttft, dec, outt in rows:
        print(f"| {size} | {config} | {n:,} | {ttft:.0f} | {dec:.1f} | {outt:.0f} |")
    print("wrote", out)

EDGE = ["fp16", "fp8", "fp8_lmhead", "fp8_kv", "int8_sq", "int4_awq", "int4_awq_taskcal", "int4_gptq_qwen"]
EDGE_BITS = {"fp16": 16, "fp8": 8, "fp8_lmhead": 8, "fp8_kv": 8, "int8_sq": 8, "int4_awq": 4, "int4_awq_taskcal": 4, "int4_gptq_qwen": 4}
BIT_MARKER = {16: "o", 8: "s", 4: "^"}
# label offsets (points) where configurations sit on top of each other; default (6, 4)
LABEL_OFFSET = {
    ("1.5b", "gsm8k_jetson", "fp8_lmhead"): (-62, -12),
    ("1.5b", "gsm8k_jetson", "fp8"): (6, 2),
    ("3b", "gsm8k_jetson", "fp8_lmhead"): (-66, -4),
    ("3b", "gsm8k_jetson", "fp8_kv"): (8, -4),
    ("3b", "mmlu_jetson", "fp8"): (8, -12),
    ("3b", "mmlu_jetson", "fp8_lmhead"): (-30, 8),
    ("3b", "mmlu_jetson", "int4_gptq_qwen"): (-20, -14),
    ("1.5b", "mmlu_jetson", "fp8_lmhead"): (-12, -15),
}


def pareto(out):
    """Edge-LLM weight configurations: accuracy against energy per question; filled = passes the accuracy rule."""
    tasks = ["gsm8k_jetson", "mmlu_jetson"]
    table = A.collect(list(A.SIZES), tasks)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), gridspec_kw=dict(left=0.08, right=0.95, top=0.85, bottom=0.1, hspace=0.32, wspace=0.16))
    green = FRAMEWORK["trt_edgellm"][1]
    rows = []
    for r, task in enumerate(tasks):
        for c, size in enumerate(("1.5b", "3b")):
            ax = axes[r][c]
            style(ax)
            data = table.get((size, task)) or {}
            hf = data.get("hf-bf16")
            if hf:
                ax.axhline(100 * hf["acc"], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
                ax.text(0.99, 100 * hf["acc"], " HF bf16", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=MUTED)
            pts = []
            for cfg in EDGE:
                m = data.get(f"trt_edgellm-{cfg}")
                if not m or not m.get("j_per_q"):
                    continue
                x, y = m["j_per_q"], 100 * m["acc"]
                pts.append((x, y, cfg))
                rows.append((size, task, cfg, y, m["delta"], m["se"], m["passes"], x, m.get("s_per_q"), m.get("decode_ms")))
                ax.scatter(x, y, s=75, marker=BIT_MARKER[EDGE_BITS[cfg]], facecolor=green if m["passes"] else "white", edgecolor=green, linewidth=1.8, zorder=3)
            for x, y, cfg in pts:
                ax.annotate(cfg, (x, y), xytext=LABEL_OFFSET.get((size, task, cfg), (6, 4)), textcoords="offset points", fontsize=8, color=INK)
            front, best = [], -1
            for x, y, cfg in sorted(pts):
                if y > best:
                    front.append((x, y))
                    best = y
            if len(front) > 1:
                ax.plot(*zip(*front), color=green, lw=1, alpha=0.5, zorder=2)
            ax.set_title(f"{A.SIZES[size]} · {'GSM8K' if task == 'gsm8k_jetson' else 'MMLU'}", loc="left", fontsize=11, fontweight="bold", color=INK)
            ax.set_xlabel("Energy per question (J, whole board)", fontsize=9, color=MUTED)
            if c == 0:
                ax.set_ylabel("Accuracy (%)", fontsize=9, color=MUTED)
            ax.set_xlim(left=0)
    handles = [plt.Line2D([], [], ls="", marker=mk, color=green, markersize=8, label=f"{b}-bit") for b, mk in BIT_MARKER.items()]
    handles += [plt.Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=green, markersize=8, label="hollow: fails the accuracy rule")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.925), ncol=4, frameon=False, fontsize=9)
    fig.text(0.08, 0.955, "TensorRT Edge-LLM weight configurations: accuracy against energy", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.035, "Line: configurations that no other one beats on both axes. Not shown: nvfp4, int4 LM head and Qwen's AWQ import (garbage text), mxfp8 (engine build fails),", fontsize=8, color=MUTED)
    fig.text(0.02, 0.014, "GPTQ-Int8 (not importable). 1.5B MMLU: quantized engines answer with the letter first more often, which this scoring rewards (FINDINGS_qwen2_5.md).", fontsize=8, color=MUTED)
    fig.text(0.97, 0.014, "python3 jetson/experiments/qwen2_5/figures.py pareto", fontsize=7, color="#8a8a8a", ha="right")
    fig.savefig(out, facecolor="white")
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=150, facecolor="white")
    print("| Size | Task | Configuration | Accuracy | Δ vs HF | Passes | Energy (J/q) | Time (s/q) | Decode (ms/token) |\n|---|---|---|---:|---:|---|---:|---:|---:|")
    for size, task, cfg, y, d, se, ok, x, t, dec in rows:
        print(f"| {size} | {task} | {cfg} | {y:.1f} | {100 * d:+.1f} ± {100 * se:.1f} | {'yes' if ok else 'no'} | {x:.1f} | {t:.2f} | {dec:.1f} |")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("figure", choices=["bandwidth", "sweep", "pareto"])
    p.add_argument("-o", "--out")
    a = p.parse_args()
    out = a.out or os.path.join(A.RESULTS, f"qwen2_5_{a.figure}.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if a.figure == "bandwidth":
        bandwidth(out)
    elif a.figure == "sweep":
        sweep(out)
    else:
        pareto(out)


if __name__ == "__main__":
    main()
