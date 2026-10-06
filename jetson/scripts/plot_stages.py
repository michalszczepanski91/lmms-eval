#!/usr/bin/env python3
"""Plot time per question split into vision encoder / prefill / decode, as an SVG, for runs you pick.

Usage: jetson/scripts/plot_stages.py [--task textvqa_val] [--by-size] [--title T] [--xmax MS] [-o out.svg] <run> [<run> ...]
  <run>  <size>:<framework>-<precision>[+tag], e.g. 3b:hf-bf16 7b:llamacpp-q8_0 3b:trt_edgellm-fp8
         (newest valid full run of results/<board>/<model>/<task>/<dir>), or a run directory path.

  jetson/scripts/plot_stages.py 3b:hf-bf16 3b:llamacpp-q8_0 3b:trt_edgellm-fp8 7b:hf-bf16 -o same_images.svg
  jetson/scripts/plot_stages.py --by-size 3b:hf-bf16 7b:hf-bf16 -o by_size.svg
  # 3B and 7B as separate images on the same time scale:
  jetson/scripts/plot_stages.py --title "Qwen2.5-VL-3B, 16-bit" --xmax 1600 3b:hf-bf16 3b:vllm-bf16 3b:llamacpp-f16 3b:trt_edgellm-fp16 -o 3b.svg
  jetson/scripts/plot_stages.py --title "Qwen2.5-VL-7B, 16-bit" --xmax 1600 7b:hf-bf16 7b:vllm-bf16 7b:llamacpp-f16 7b:trt_edgellm-fp16 -o 7b.svg

Default: one bar per run, mean per stage over all questions. --by-size: one panel per run, one bar per
image-token group (--bins), median per stage. Decode is drawn for --decode-tokens tokens after the first.

Stage sources: HF samples (vision_seconds, needs stage_timing=True), llama.cpp server.log (verbosity 4),
Edge-LLM trt_profile.json (means only, so not with --by-size), vLLM samples (vision_seconds with stage_timing=True;
runs without it have no vision/prefill split, and their time to first token is drawn as one grey segment).
"""

import argparse
import glob
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import EXCLUDE_MARKERS, LLAMA_TS, RESULTS, open_text  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a8981"]  # vision, prefill, decode, vision + prefill (not split)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def find_run(spec, task):
    if os.path.isdir(spec):
        return spec.rstrip("/")
    size, _, fw = spec.partition(":")
    models = glob.glob(os.path.join(RESULTS, f"*-{size.upper()}-*"))
    runs = sorted(r for m in models for r in glob.glob(os.path.join(m, task, fw, "*")) if "_limit" not in r and not any(os.path.exists(os.path.join(r, x)) for x in EXCLUDE_MARKERS + ("TIMING_INVALID",)))
    if not runs:
        sys.exit(f"no valid full run for {spec} ({task}) under {RESULTS}")
    return runs[-1]


def label(run_dir):
    parts = run_dir.split(os.sep)
    size = re.search(r"-(\d+B)-", parts[-4])
    return f"{size[1] if size else parts[-4]} {parts[-2]}"


def hf_stages(run_dir):
    """Per question: (image tokens, vision ms, prefill ms, decode ms/token or None)."""
    flops = {d["doc_id"]: d for d in map(json.loads, open_text(os.path.join(run_dir, "flops_samples.jsonl.gz")))}
    out = []
    for path in glob.glob(os.path.join(run_dir, "lmms_eval", "*_samples_*.jsonl*")):
        for d in map(json.loads, open_text(path)):
            c = d["token_counts"][0]
            if "vision_seconds" not in c:
                sys.exit(f"{run_dir}: no vision_seconds (HF needs stage_timing=True)")
            ttft = c["time_to_first_token_seconds"] * 1e3
            n = c["output_tokens"] - 1
            dec = (c["generation_seconds"] * 1e3 - ttft) / n if n > 0 else None
            out.append((flops[d["doc_id"]]["image_tokens"], c["vision_seconds"] * 1e3, ttft - c["vision_seconds"] * 1e3, dec))
    return out


def vllm_stages(run_dir):
    """Like hf_stages. Runs with stage_timing=True have vision_seconds; without, vision is None and TTFT is one segment."""
    flops = {d["doc_id"]: d for d in map(json.loads, open_text(os.path.join(run_dir, "flops_samples.jsonl.gz")))}
    out = []
    for path in glob.glob(os.path.join(run_dir, "lmms_eval", "*_samples_*.jsonl*")):
        for d in map(json.loads, open_text(path)):
            c = d["token_counts"][0]
            ttft = c["time_to_first_token_seconds"] * 1e3
            n = c["output_tokens"] - 1
            vis = c["vision_seconds"] * 1e3 if c.get("vision_seconds") is not None else None
            out.append((flops[d["doc_id"]]["image_tokens"], vis, ttft if vis is None else ttft - vis, (c["generation_seconds"] * 1e3 - ttft) / n if n > 0 else None))
    return out


def llamacpp_stages(run_dir):
    """Same as hf_stages, from llama-server's log: encoder = 'encoding mtmd batch' to the first 'decoding image batch'."""
    out, cur = [], {}
    for line in open_text(os.path.join(run_dir, "server.log")):
        ts = LLAMA_TS.match(line)
        t = (int(ts[1]) * 60 + int(ts[2]) + int(ts[3]) / 1e3 + int(ts[4]) / 1e6) * 1e3 if ts else None
        if "encoding mtmd batch" in line:
            cur = {"start": t}
        elif "decoding image batch 1/" in line and "start" in cur:
            cur["img"] = int(re.search(r"n_tokens_batch = (\d+)", line)[1])
            cur["vis"] = t - cur["start"]
        elif (m := re.search(r"prompt eval time =\s+([\d.]+) ms", line)) and "vis" in cur:
            cur["pre"] = float(m[1]) - cur["vis"]
        elif (m := re.search(r"\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens", line)) and "pre" in cur:
            # "eval time / N tokens" counts the first token too, but it comes from the prompt eval: N - 1 decode steps.
            n = int(m[2]) - 1
            out.append((cur["img"], cur["vis"], cur["pre"], float(m[1]) / n if n > 0 else None))
            cur = {}
    return out


def edgellm_means(run_dir):
    """(vision, prefill, decode ms/token) GPU-time means from the runner profile."""
    stages = {}
    for r in json.load(open(os.path.join(run_dir, "trt_profile.json"))):
        for s in r["profile"]["stages"]:
            stages.setdefault(s["stage_id"], []).append(s["gpu_time_stats"]["mean_ms"])
    return tuple(st.mean(stages[k]) for k in ("vision_encoder", "llm_prefill", "llm_generation"))


def per_question(run_dir):
    if os.path.exists(os.path.join(run_dir, "trt_profile.json")):
        return None
    if os.path.exists(os.path.join(run_dir, "server.log")) or os.path.exists(os.path.join(run_dir, "server.log.gz")):
        return llamacpp_stages(run_dir)
    if "/vllm-" in run_dir:
        return vllm_stages(run_dir)
    return hf_stages(run_dir)


def segments(vis, pre, dec):
    """Bar segments as (ms, color index); vis None = vision + prefill not split."""
    first = [(pre, 3)] if vis is None else [(vis, 0), (pre, 1)]
    return first + [(dec, 2)]


def stage_rows(q, agg):
    vis = None if q[0][1] is None else agg(x[1] for x in q)
    return segments(vis, agg(x[2] for x in q), agg(x[3] for x in q if x[3]))


def svg(panels, out, decode_tokens, foot, title=None, xmax=None):
    names = ["Vision encoder", "Prefill (prompt + image tokens, first token)", f"Decode ({decode_tokens} more tokens)", "Vision + prefill (not split)"]
    used = sorted({c for _, rows in panels for _, segs in rows for _, c in segs})
    xmax_ms = xmax or max(sum(v for v, _ in segs) for _, rows in panels for _, segs in rows)
    step = next(s for s in (50, 100, 200, 250, 500, 1000, 2000, 5000) if xmax_ms / s <= 8)
    xmax_ms = xmax or step * (int(xmax_ms // step) + 1)
    width, lx0 = 920, 250
    px = (width - lx0 - 80) / xmax_ms
    o = []

    def text(x, y, s, fs=12, c=INK, a="start", w="normal"):
        o.append(f'<text x="{x:.1f}" y="{y}" font-size="{fs}" fill="{c}" text-anchor="{a}" font-weight="{w}">{s}</text>')

    y = 0
    if title:
        text(10, 22, title, fs=15, w="bold")
        y = 26
    lx, ly = 10, y
    for i in used:
        s = names[i]
        if lx + 17 + 6.2 * len(s) > width:
            lx, ly = 10, ly + 20
        o.append(f'<rect x="{lx}" y="{ly + 12}" width="12" height="12" rx="2" fill="{COLORS[i]}"/>')
        text(lx + 17, ly + 22, s)
        lx += 17 + 6.2 * len(s) + 28
    y = top = ly + 44
    for title, rows in panels:
        if title:
            y += 18
            text(10, y, title, fs=13, w="bold")
        y += 10
        for lab, segs in rows:
            text(lx0 - 8, y + 15, lab, fs=11.5, a="end")
            x = lx0
            for v, c in segs:
                w = v * px
                o.append(f'<rect x="{x:.1f}" y="{y + 3}" width="{max(w - 2, 1):.1f}" height="18" rx="3" fill="{COLORS[c]}"/>')
                if w > 28:
                    text(x + w / 2 - 1, y + 16, f"{v:.0f}", fs=10.5, c="#ffffff", a="middle", w="bold")
                x += w
            text(x + 6, y + 16, f"{sum(v for v, _ in segs):.0f} ms", fs=11)
            y += 26
        y += 6
    grid = []
    for ms in range(0, int(xmax_ms) + 1, step):
        gx = lx0 + ms * px
        grid.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{y}" stroke="{GRID}"/>')
        text(gx, y + 16, ms, fs=11, c=MUTED, a="middle")
    text(lx0 + xmax_ms * px / 2, y + 34, "ms per question", fs=11.5, c=MUTED, a="middle")
    for k, line in enumerate(foot):
        text(10, y + 56 + 16 * k, line, fs=11, c=MUTED)
    h = y + 66 + 16 * (len(foot) - 1)
    with open(out, "w") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" viewBox="0 0 {width} {h}" font-family="Helvetica, Arial, sans-serif">'
            f'<rect width="{width}" height="{h}" fill="#ffffff"/>' + "".join(grid) + "".join(o) + "</svg>"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--task", default="textvqa_val")
    p.add_argument("--by-size", action="store_true")
    p.add_argument("--bins", default="400,800,1200,1600", help="image-token group edges (default: %(default)s)")
    p.add_argument("--decode-tokens", type=int, default=2)
    p.add_argument("--title", help="title above the chart")
    p.add_argument("--xmax", type=int, help="time axis end in ms (same value = same scale across images)")
    p.add_argument("-o", "--out", default="stages.svg")
    a = p.parse_args()
    dirs = [find_run(r, a.task) for r in a.runs]
    for d in dirs:
        print(f"{label(d)}: {os.path.relpath(d)}")
    k = a.decode_tokens
    foot = [f"{a.task}, same questions for every run. Decode drawn for a {k + 1}-token answer."]
    if not a.by_size:
        rows, gpu_only = [], False
        for d in dirs:
            q = per_question(d)
            if q is None:
                vis, pre, dec = edgellm_means(d)
                rows.append((label(d) + " †", segments(vis, pre, k * dec)))
                gpu_only = True
            else:
                segs = stage_rows(q, st.mean)
                segs[-1] = (k * segs[-1][0], 2)
                rows.append((label(d), segs))
        foot[0] = foot[0].replace("Decode", "Mean per stage. Decode")
        if gpu_only:
            foot.append("† Edge-LLM: GPU time only, no host overhead.")
        svg([("", rows)], a.out, k, foot, a.title, a.xmax)
    else:
        edges = [int(e) for e in a.bins.split(",")]
        groups = list(zip(edges, edges[1:]))
        panels = []
        for d in dirs:
            q = per_question(d)
            if q is None:
                sys.exit(f"{d}: Edge-LLM reports means only, it can't be grouped by image size")
            rows = []
            for lo, hi in groups:
                g = [x for x in q if lo <= x[0] < hi]
                if g:
                    segs = stage_rows(g, st.median)
                    segs[-1] = (k * segs[-1][0], 2)
                    rows.append((f"{lo}–{hi - 1} image tok · {len(g):,} q", segs))
            panels.append((label(d), rows))
        foot[0] = foot[0].replace("Decode", "Median per stage and image-token group. Decode")
        svg(panels, a.out, k, foot, a.title, a.xmax)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
