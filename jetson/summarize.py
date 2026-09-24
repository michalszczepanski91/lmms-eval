#!/usr/bin/env python3
"""Collect run_eval.sh outputs into <results dir>/SUMMARY.md (stdlib only; run on the host).

Expects <results dir>/<model>/<task>/<framework>-<precision>[+tag]/<run_id>/lmms_eval/*_results.json.
The results dir is $RESULTS_DIR relative to the repo (default jetson/results), as in run_eval.sh.

Usage: [RESULTS_DIR=jetson/results-thor] python3 jetson/summarize.py [--include-smoke]
"""

import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, os.environ.get("RESULTS_DIR", "jetson/results"))


# Power rails differ by board. GPU rail: Orin VDD_GPU_SOC (GPU + SoC), Thor VDD_GPU (GPU only).
# Module total = sum of the sub-rails, comparable across boards: Orin VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0,
# Thor VDD_GPU + VDD_CPU_SOC_MSS + VIN_SYS_5V0 (Thor's VIN, the board input, is not included).
GPU_RAILS = ("VDD_GPU_SOC", "VDD_GPU")
TOTAL_RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VDD_GPU", "VDD_CPU_SOC_MSS", "VIN_SYS_5V0")


def tegrastats_stats(path):
    ram, gpu_mw, total_mw = [], [], []
    if os.path.exists(path):
        for line in open(path):
            if m := re.search(r"RAM (\d+)/", line):
                ram.append(int(m.group(1)))
            rails = {name: int(mw) for name, mw in re.findall(r"\b([A-Z][A-Z0-9_]+) (\d+)mW/", line)}
            if gpu := next((rails[r] for r in GPU_RAILS if r in rails), None):
                gpu_mw.append(gpu)
            if any(r in rails for r in TOTAL_RAILS):
                total_mw.append(sum(rails.get(r, 0) for r in TOTAL_RAILS))
    if not ram:
        return {}

    def avg_w(values):
        return sum(values) / len(values) / 1000 if values else None

    return {"base_ram": ram[0], "peak_ram": max(ram), "avg_gpu_w": avg_w(gpu_mw), "avg_total_w": avg_w(total_mw)}


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))] if values else None


def latency_stats(samples_path):
    """Per-sample latency recorded by backends that fill TokenCounts timing fields."""
    ttft, gen, decode_ms = [], [], []
    for line in open(samples_path):
        counts = (json.loads(line).get("token_counts") or [None])[0] or {}
        if counts.get("time_to_first_token_seconds") is None or counts.get("generation_seconds") is None:
            continue
        ttft.append(counts["time_to_first_token_seconds"])
        gen.append(counts["generation_seconds"])
        if counts.get("output_tokens", 0) > 1:
            decode_ms.append(1000 * (counts["generation_seconds"] - counts["time_to_first_token_seconds"]) / (counts["output_tokens"] - 1))
    if not ttft:
        return {}
    return {
        "ttft_p50": percentile(ttft, 50),
        "ttft_p90": percentile(ttft, 90),
        "gen_p50": percentile(gen, 50),
        "gen_p90": percentile(gen, 90),
        "decode_ms_p50": percentile(decode_ms, 50),
    }


def trt_profile_stats(path):
    """TensorRT Edge-LLM only reports averages: TTFT ~ vision encoder + prefill per request."""
    if not os.path.exists(path):
        return {}
    profile = json.load(open(path))[0]["profile"]
    vision = profile.get("multimodal", {})
    vision_ms = vision.get("average_time_per_token_ms", 0) * vision.get("total_multimodal_tokens", 0) / max(1, vision.get("total_runs", 1))
    ttft = (vision_ms + profile.get("prefill", {}).get("average_time_per_run_ms", 0)) / 1000
    decode = profile.get("generation", {}).get("average_time_per_token_ms")
    return {"ttft_avg": ttft, "decode_ms_p50": decode}


def run_info_field(run_dir, field):
    info = os.path.join(run_dir, "run_info.txt")
    for line in open(info) if os.path.exists(info) else []:
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip()
    return None


def other_containers(run_dir):
    """Other running containers and (if recorded) other GPU processes, e.g. another user's host job."""
    containers = run_info_field(run_dir, "other containers")
    if containers is None:
        return "?"
    containers = containers.split("(")[0].strip()
    procs = [p.split(",")[1].strip().split("/")[-1] for p in (run_info_field(run_dir, "other GPU processes") or "").split(";") if p.count(",") >= 2]
    return ", ".join(filter(None, [containers, *(f"GPU: {p}" for p in procs)])) or "none"


def fmt_gb(mb):
    return f"{mb / 1024:.1f}" if mb else "-"


def fmt_ms(row, p50_key, p90_key):
    if row.get(p50_key) is not None:
        return f"{1000 * row[p50_key]:.0f} / {1000 * row[p90_key]:.0f}"
    if p50_key == "ttft_p50" and row.get("ttft_avg") is not None:
        return f"~{1000 * row['ttft_avg']:.0f} (avg)"
    return "-"


def main():
    include_smoke = "--include-smoke" in sys.argv
    rows = []
    for results_json in sorted(glob.glob(os.path.join(RESULTS, "*", "*", "*", "*", "lmms_eval", "*_results.json"))):
        run_dir = os.path.dirname(os.path.dirname(results_json))
        model, _task_dir, framework, run_id = os.path.relpath(run_dir, RESULTS).split(os.sep)
        if "_limit" in run_id and not include_smoke:
            continue
        if os.path.exists(os.path.join(run_dir, "INVALID")):  # kept as evidence; the file says why
            continue
        r = json.load(open(results_json))
        n = sum(r["n-samples"][t]["effective"] for t in r["n-samples"])
        samples = glob.glob(os.path.join(os.path.dirname(results_json), "*_samples_*.jsonl"))
        stats = {
            **tegrastats_stats(os.path.join(run_dir, "tegrastats.log")),
            **(latency_stats(samples[0]) if samples else {}),
            **trt_profile_stats(os.path.join(run_dir, "trt_profile.json")),
        }
        for task, metrics in r["results"].items():
            scores = {k.split(",")[0]: v for k, v in metrics.items() if k.endswith(",none") and "stderr" not in k}
            rows.append(
                {
                    "model": model,
                    "framework": framework,
                    "task": task,
                    "run": os.path.relpath(run_dir, RESULTS),
                    "scores": scores,
                    "n": n,
                    "minutes": float(r["total_evaluation_time_seconds"]) / 60,
                    "others": other_containers(run_dir),
                    "board": run_info_field(run_dir, "board"),
                    "power_mode": (run_info_field(run_dir, "power mode") or "").replace("NV Power Mode:", "").strip(),
                    **stats,
                }
            )

    lines = [
        f"# {' / '.join(sorted({row['board'] for row in rows if row['board']})) or 'Jetson'} eval results",
        "",
        f"Generated by `python3 jetson/summarize.py`. Batch size 1, power mode {' / '.join(sorted({row['power_mode'] for row in rows if row['power_mode']})) or '?'}, same images (256..2048 visual tokens) for every framework.",
        "RAM is whole-board unified memory from tegrastats (base = before model load; other processes included).",
        "Power (tegrastats, averaged over the run): GPU rail = VDD_GPU_SOC on Orin (GPU + SoC), VDD_GPU on Thor (GPU only);",
        "module = sum of the GPU, CPU/SoC and 5V system rails (comparable across boards).",
        "Latency is per sample: TTFT = model call to first generated token (vision encoder + prefill + 1 step);",
        "answer = whole model call; decode = time per token after the first. p50 / p90 over all samples.",
        "llama.cpp is measured client side over HTTP (streaming); TensorRT Edge-LLM only reports averages.",
        "",
        "| Model | Framework | Task | Scores | N | Wall time (min) | TTFT p50 / p90 (ms) | Answer p50 / p90 (ms) | Decode (ms/token) | Base RAM (GB) | Peak RAM (GB) | Avg GPU rail (W) | Avg module (W) | Other containers / GPU processes | Run dir |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        scores = "<br>".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in row["scores"].items())
        power = " | ".join(f"{row[k]:.1f}" if row.get(k) else "-" for k in ("avg_gpu_w", "avg_total_w"))
        decode = f"{row['decode_ms_p50']:.0f}" if row.get("decode_ms_p50") is not None else "-"
        lines.append(
            f"| {row['model']} | {row['framework']} | {row['task']} | {scores} | {row['n']} | {row['minutes']:.1f} | "
            f"{fmt_ms(row, 'ttft_p50', 'ttft_p90')} | {fmt_ms(row, 'gen_p50', 'gen_p90')} | {decode} | "
            f"{fmt_gb(row.get('base_ram'))} | {fmt_gb(row.get('peak_ram'))} | {power} | {row['others']} | [{row['run']}]({row['run']}) |"
        )
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "SUMMARY.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
