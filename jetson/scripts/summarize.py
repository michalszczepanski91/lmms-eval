#!/usr/bin/env python3
"""Collect run_eval.sh outputs into <results dir>/SUMMARY.md (stdlib only; run on the host).

Expects <results dir>/<model>/<task>/<framework>-<precision>[+tag]/<run_id>/lmms_eval/*_results.json.
The results dir is $RESULTS_DIR relative to the repo (default jetson/results/<board>: orin, thor), as in run_eval.sh.

Usage: [RESULTS_DIR=jetson/results/thor] python3 jetson/scripts/summarize.py [--include-smoke]
"""

import glob
import gzip
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def board_name():
    """Same as board_name in jetson/frameworks/common.sh."""
    try:
        model = open("/proc/device-tree/model").read().strip("\0\n")
    except OSError:
        model = ""
    return "thor" if "Thor" in model else "orin" if "Orin" in model else (model or "unknown").lower().replace(" ", "-")


RESULTS = os.path.join(REPO, os.environ.get("RESULTS_DIR") or f"jetson/results/{board_name()}")


# Power rails differ by board. GPU rail: Orin VDD_GPU_SOC (GPU + SoC), Thor VDD_GPU (GPU only).
# Module total = sum of the sub-rails, comparable across boards: Orin VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0,
# Thor VDD_GPU + VDD_CPU_SOC_MSS + VIN_SYS_5V0 (Thor's VIN, the board input, is not included).
GPU_RAILS = ("VDD_GPU_SOC", "VDD_GPU")
TOTAL_RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VDD_GPU", "VDD_CPU_SOC_MSS", "VIN_SYS_5V0")


def open_text(path):
    """Open a text file that slim_run.py may have gzipped (path or path + ".gz")."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def tegrastats_stats(path):
    ram, gpu_mw, total_mw = [], [], []
    if os.path.exists(path) or os.path.exists(path + ".gz"):
        for line in open_text(path):
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
    ttft, gen, decode_ms, pre, vision, prefill, out_tokens = [], [], [], [], [], [], []
    for line in open_text(samples_path):
        counts = (json.loads(line).get("token_counts") or [None])[0] or {}
        if counts.get("output_tokens") is not None:
            out_tokens.append(counts["output_tokens"])
        if counts.get("time_to_first_token_seconds") is None or counts.get("generation_seconds") is None:
            continue
        ttft.append(counts["time_to_first_token_seconds"])
        gen.append(counts["generation_seconds"])
        if counts.get("preprocess_seconds") is not None:
            pre.append(counts["preprocess_seconds"])
        if counts.get("vision_seconds") is not None:  # HF with stage_timing: prefill = TTFT - vision encoder
            vision.append(counts["vision_seconds"])
            prefill.append(counts["time_to_first_token_seconds"] - counts["vision_seconds"])
        if counts.get("output_tokens", 0) > 1:
            decode_ms.append(1000 * (counts["generation_seconds"] - counts["time_to_first_token_seconds"]) / (counts["output_tokens"] - 1))
    stats = {"out_tokens_mean": sum(out_tokens) / len(out_tokens)} if out_tokens else {}
    if not ttft:
        return stats
    return {
        **stats,
        "ttft_p50": percentile(ttft, 50),
        "ttft_p90": percentile(ttft, 90),
        "ttft_p99": percentile(ttft, 99),
        "gen_p50": percentile(gen, 50),
        "gen_p90": percentile(gen, 90),
        "gen_p99": percentile(gen, 99),
        "decode_ms_p50": percentile(decode_ms, 50),
        "ttft_mean": sum(ttft) / len(ttft),
        "gen_mean": sum(gen) / len(gen),
        **({"pre_p50": percentile(pre, 50)} if pre else {}),
        **({"vision_p50": percentile(vision, 50), "prefill_p50": percentile(prefill, 50), "stage_source": "CUDA events"} if vision else {}),
    }


LLAMA_TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")


def llamacpp_server_stats(run_dir):
    """Per-request stages from llama-server's log (verbosity 4, llamacpp.sh), in seconds.

    Vision encoder = "encoding mtmd batch" to the first "decoding image batch" (the image embeddings are ready);
    LLM prefill = "prompt eval time" (which includes the encoder) minus the encoder. Server side, without HTTP.
    """
    path = os.path.join(run_dir, "server.log")
    if not (os.path.exists(path) or os.path.exists(path + ".gz")):
        return {}
    vision, prefill = [], []
    encode_start = current_vision = None
    for line in open_text(path):
        ts = LLAMA_TS.match(line)
        t = int(ts[1]) * 60 + int(ts[2]) + int(ts[3]) / 1e3 + int(ts[4]) / 1e6 if ts else None
        if "encoding mtmd batch" in line:
            encode_start = t
        elif "decoding image batch 1/" in line and encode_start is not None and t is not None:
            current_vision = t - encode_start
            vision.append(current_vision)
            encode_start = None
        elif m := re.search(r"prompt eval time =\s+([\d.]+) ms", line):
            prefill.append(float(m[1]) / 1000 - (current_vision or 0))
            current_vision = None
    if not vision:
        return {}
    return {"vision_p50": percentile(vision, 50), "prefill_p50": percentile(prefill, 50), "stage_source": "server log"}


def trt_profile_stats(path):
    """TensorRT Edge-LLM only reports averages: TTFT ~ vision encoder + prefill per request."""
    if not os.path.exists(path):
        return {}
    profile = json.load(open(path))[0]["profile"]
    vision = profile.get("multimodal", {})
    vision_ms = vision.get("average_time_per_token_ms", 0) * vision.get("total_multimodal_tokens", 0) / max(1, vision.get("total_runs", 1))
    ttft = (vision_ms + profile.get("prefill", {}).get("average_time_per_run_ms", 0)) / 1000
    decode = profile.get("generation", {}).get("average_time_per_token_ms")
    stats = {"ttft_avg": ttft, "decode_ms_p50": decode, "out_tokens_mean": profile.get("generation", {}).get("average_tokens_per_run")}
    # Per-stage GPU time distributions (Edge-LLM >= 0.10): vision_encoder, llm_prefill, llm_generation.
    stages = {stage["stage_id"]: stage.get("gpu_time_stats", {}) for stage in profile.get("stages", [])}
    if "vision_encoder" in stages and "llm_prefill" in stages:
        stats.update(
            vision_p50=stages["vision_encoder"]["median_ms"] / 1000,
            prefill_p50=stages["llm_prefill"]["median_ms"] / 1000,
            stage_source="runner profile (GPU time)",
        )
    return stats


def flops_stats(run_dir, task):
    """Mean analytical FLOPs per sample from flops.json (jetson/models/<model>/flops.py)."""
    path = os.path.join(run_dir, "flops.json")
    task_flops = json.load(open(path))["tasks"].get(task) if os.path.exists(path) else None
    return {f"flops_{k}": v for k, v in task_flops["mean"].items()} if task_flops else {}


def flops_rates(row):
    """Achieved TFLOP/s: (vision + prefill) FLOPs over TTFT, all FLOPs over the whole answer."""
    if row.get("flops_total_flops") is None:
        return "-", "-"
    ttft, answer = row.get("ttft_mean"), row.get("gen_mean")
    if ttft is None and row.get("ttft_avg") is not None:  # TensorRT Edge-LLM: averages only
        ttft = row["ttft_avg"]
        answer = ttft + (row.get("decode_ms_p50") or 0) / 1000 * max(0, row["flops_output_tokens"] - 1)
    first = (row["flops_vision_flops"] + row["flops_prefill_flops"]) / ttft / 1e12 if ttft else None
    total = row["flops_total_flops"] / answer / 1e12 if answer else None
    return tuple(f"{v:.1f}" if v is not None else "-" for v in (first, total))


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
        samples = glob.glob(os.path.join(os.path.dirname(results_json), "*_samples_*.jsonl*"))
        stats = {
            **tegrastats_stats(os.path.join(run_dir, "tegrastats.log")),
            **(latency_stats(samples[0]) if samples else {}),
            **trt_profile_stats(os.path.join(run_dir, "trt_profile.json")),
            **llamacpp_server_stats(run_dir),
        }
        for task, metrics in r["results"].items():
            # Metrics without a value (e.g. TextVQA's "submission" file writer) are left out.
            scores = {k.split(",")[0]: v for k, v in metrics.items() if k.endswith(",none") and "stderr" not in k and v is not None}
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
                    **flops_stats(run_dir, task),
                }
            )

    lines = [
        f"# {' / '.join(sorted({row['board'] for row in rows if row['board']})) or 'Jetson'} eval results",
        "",
        f"Generated by `python3 jetson/scripts/summarize.py`. Batch size 1, power mode {' / '.join(sorted({row['power_mode'] for row in rows if row['power_mode']})) or '?'}, same images (256..2048 visual tokens) for every framework.",
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
    lines += [
        "",
        "## Latency breakdown and efficiency",
        "",
        "Where the time to the first token goes, per sample (p50). Preprocess = prompt and image preparation before the model call",
        "(client side; for llama.cpp this includes the PNG/base64 encoding). Vision encoder and LLM prefill (text + image tokens,",
        "first output token) come from: HF - CUDA events around the vision encoder (`stage_timing=True`), prefill = TTFT - encoder;",
        "llama.cpp - llama-server's log (server side, without HTTP); TensorRT Edge-LLM - the runner's per-stage GPU time. vLLM runs",
        "the encoder inside its prefill step and reports only TTFT. TTFT p99 shows the tail. Throughput = samples / eval wall time;",
        "energy per sample = average module power x eval wall time / samples (includes preprocessing and idle gaps).",
        "",
        "| Model | Framework | Task | N | Preprocess (ms) | Vision encoder (ms) | LLM prefill (ms) | TTFT p50 / p99 (ms) | Answer p99 (ms) | Output tokens (mean) | Decode (tokens/s) | Throughput (samples/s) | Energy (J/sample) | Stage timing source |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:

        def ms(key):
            return f"{1000 * row[key]:.0f}" if row.get(key) is not None else "-"

        seconds = row["minutes"] * 60
        energy = f"{row['avg_total_w'] * seconds / row['n']:.1f}" if row.get("avg_total_w") and row["n"] else "-"
        decode_tps = f"{1000 / row['decode_ms_p50']:.0f}" if row.get("decode_ms_p50") else "-"
        out_tokens = f"{row['out_tokens_mean']:.1f}" if row.get("out_tokens_mean") is not None else "-"
        lines.append(
            f"| {row['model']} | {row['framework']} | {row['task']} | {row['n']} | {ms('pre_p50')} | {ms('vision_p50')} | {ms('prefill_p50')} | "
            f"{fmt_ms(row, 'ttft_p50', 'ttft_p99')} | {ms('gen_p99')} | {out_tokens} | {decode_tps} | {row['n'] / seconds:.2f} | {energy} | {row.get('stage_source', '-')} |"
        )
    lines += [
        "",
        "## Compute (FLOPs)",
        "",
        "Analytical FLOPs per sample from `flops.json` (`jetson/models/<model>/flops.py`): matrix multiplies at 2 FLOPs per multiply-accumulate",
        "(linear layers + attention), for each sample's actual image grid and prompt / output token counts; identical for every",
        "framework and precision. Vision = ViT encoder + patch merger; prefill = LLM over the prompt (text + image tokens) incl. the",
        "first token; decode = the remaining output tokens. Achieved TFLOP/s: (vision + prefill) / mean TTFT, and total / mean answer",
        "time (TensorRT Edge-LLM: from its averages). Means per sample; total = whole run.",
        "",
        "| Model | Framework | Task | N | Image / text / output tokens | Vision (GFLOP) | Prefill (GFLOP) | Decode (GFLOP) | Total (GFLOP) | Run total (TFLOP) | TFLOP/s to 1st token | TFLOP/s answer |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("flops_total_flops") is None:
            continue
        gflop = " | ".join(f"{row[f'flops_{k}_flops'] / 1e9:.0f}" for k in ("vision", "prefill", "decode", "total"))
        tokens = " / ".join(f"{row[f'flops_{k}_tokens']:.0f}" for k in ("image", "text", "output"))
        first, answer = flops_rates(row)
        lines.append(f"| {row['model']} | {row['framework']} | {row['task']} | {row['n']} | {tokens} | {gflop} | {row['flops_total_flops'] * row['n'] / 1e12:.0f} | {first} | {answer} |")
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "SUMMARY.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
