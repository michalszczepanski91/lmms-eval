#!/usr/bin/env python3
"""Tables of the text-only Qwen2.5 study, from the result files (the source of every number in FINDINGS_qwen2_5.md).

For each size, task and configuration (framework-precision) it takes the first valid full run (no _limit, no
INVALID / STALLED / SUPERSEDED marker; later runs of the same configuration are repeats, compared with --repeats) and reports accuracy with the paired difference to HF bf16 (same questions,
standard error of the per-question differences), time and energy per question, TTFT and decode medians, output
tokens, energy per output token, peak memory and FLOPs. --q1 applies the Q1 decision rule of NOTES_qwen2_5.md.

Usage: python3 jetson/experiments/qwen2_5/analyze.py [--tasks ...] [--sizes ...] [--q1] [--json out.json] [--include-smoke]
"""

import argparse
import glob
import gzip
import json
import math
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import summarize as S  # noqa: E402

RESULTS = S.RESULTS
SIZES = {"3b": "Qwen2.5-3B-Instruct", "1.5b": "Qwen2.5-1.5B-Instruct"}
TASKS = ["gsm8k_jetson", "mmlu_jetson", "ifeval"]
REFERENCE = "hf-bf16"
# Bit width of each Phase 1 configuration (the Q1 rule compares frameworks within a bit width).
WIDTH = {
    "hf-bf16": 16,
    "vllm-bf16": 16,
    "llamacpp-f16": 16,
    "trt_edgellm-fp16": 16,
    "vllm-fp8": 8,
    "llamacpp-q8_0": 8,
    "trt_edgellm-fp8": 8,
    "vllm-awq": 4,
    "llamacpp-q4_k_m": 4,
    "trt_edgellm-int4_awq": 4,
}
Q1_METRICS = [("j_per_q", "energy / question"), ("s_per_q", "time / question"), ("ttft_ms", "TTFT median"), ("decode_ms", "decode ms/token median")]


def framework_of(config):
    return config.split("-")[0]


def valid_runs(model, task, config, include_smoke=False):
    runs = []
    for run in sorted(glob.glob(os.path.join(RESULTS, model, task, config, "2*"))):
        if "_limit" in run and not include_smoke:
            continue
        if any(os.path.exists(os.path.join(run, m)) for m in S.EXCLUDE_MARKERS):
            continue
        if glob.glob(os.path.join(run, "lmms_eval", "*_results.json")):
            runs.append(run)
    return runs


def samples(run):
    for path in sorted(glob.glob(os.path.join(run, "lmms_eval", "*_samples_*.jsonl*"))):
        name = os.path.basename(path).split("_samples_")[1].split(".jsonl")[0]
        for line in S.open_text(path):
            yield name, json.loads(line)


def question_scores(run, task):
    """{question key: score} with the headline metric of each task. GSM8K writes one record per filter, in filter
    order (strict-match, then flexible-extract): the last record of a question is the flexible-extract one."""
    out = {}
    for name, s in samples(run):
        key = (name, s["doc_id"])
        if task == "ifeval":
            out[key] = float(s["prompt_level_strict_acc"])
        elif task.startswith("length_sweep"):
            out[key] = float(s["counts"])
        else:
            out[key] = float(s["exact_match"])
    return out


def output_tokens(run):
    """Generated tokens per question: the backend's count, else the tokenizer recount of flops.py (Edge-LLM)."""
    out = [((s.get("token_counts") or [{}])[0] or {}).get("output_tokens") for _, s in samples(run)]
    out = [t for t in out if t is not None]
    if not out and os.path.exists(os.path.join(run, "flops_samples.jsonl.gz")):
        out = [json.loads(line).get("output_tokens") for line in gzip.open(os.path.join(run, "flops_samples.jsonl.gz"), "rt")]
        out = [t for t in out if t is not None]
    return out


def paired(scores, ref):
    keys = [k for k in ref if k in scores]
    if not keys or len(keys) < 2:
        return None, None
    d = [scores[k] - ref[k] for k in keys]
    mean = sum(d) / len(d)
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1) / len(d))
    return mean, se


def run_metrics(run, task):
    results_json = glob.glob(os.path.join(run, "lmms_eval", "*_results.json"))[0]
    r = json.load(open(results_json))
    n = sum(v["effective"] for v in r["n-samples"].values())
    seconds = float(r["total_evaluation_time_seconds"])
    text_only = S.run_info_field(run, "modality") == "text"
    paths = sorted(glob.glob(os.path.join(run, "lmms_eval", "*_samples_*.jsonl*")))
    stats = {
        **S.tegrastats_stats(os.path.join(run, "tegrastats.log")),
        **S.latency_stats(paths, text_only),
        **S.trt_profile_stats(os.path.join(run, "trt_profile.json")),
        **S.llamacpp_server_stats(run, text_only),
        **S.flops_stats(run, task),
    }
    timing_invalid = S.marker_reason(run, "TIMING_INVALID")
    out = output_tokens(run)
    m = {
        "run": os.path.relpath(run, RESULTS),
        "date": os.path.basename(run)[:8],
        "n": n,
        "out_tokens": sum(out) / len(out) if out else stats.get("out_tokens_mean"),
        "peak_ram_gb": stats["peak_ram"] / 1024 if stats.get("peak_ram") else None,
        "power_w": stats.get("avg_total_w"),
        "gflops_q": stats["flops_total_flops"] / 1e9 if stats.get("flops_total_flops") else None,
        "timing_invalid": timing_invalid,
    }
    if not timing_invalid:
        m["s_per_q"] = seconds / n
        m["j_per_q"] = stats["avg_total_w"] * seconds / n if stats.get("avg_total_w") else None
        # TTFT: per-request median where the backend records it; Edge-LLM: median prefill GPU time of its runner profile
        # (it reports no per-request TTFT; its prefill produces the first token).
        m["ttft_ms"] = 1000 * (stats.get("ttft_p50") if stats.get("ttft_p50") is not None else stats.get("prefill_p50") or 0) or None
        m["ttft_source"] = "per request" if stats.get("ttft_p50") is not None else stats.get("stage_source", "-")
        m["prefill_ms"] = 1000 * stats["prefill_p50"] if stats.get("prefill_p50") is not None else None
        m["decode_ms"] = stats.get("decode_ms_p50")
        m["j_per_token"] = m["j_per_q"] / m["out_tokens"] if m.get("j_per_q") and m["out_tokens"] else None
    return m


def collect(sizes, tasks, include_smoke=False, min_tolerance=0.0):
    """{(size, task): OrderedDict(config -> metrics)}, configs in Phase 1 order then the rest alphabetically."""
    table = {}
    for size in sizes:
        model = SIZES[size]
        for task in tasks:
            configs = sorted(os.listdir(os.path.join(RESULTS, model, task))) if os.path.isdir(os.path.join(RESULTS, model, task)) else []
            configs = [c for c in WIDTH if c in configs] + [c for c in configs if c not in WIDTH]
            ref_runs = valid_runs(model, task, REFERENCE, include_smoke)
            ref = question_scores(ref_runs[0], task) if ref_runs else {}
            rows = OrderedDict()
            for config in configs:
                runs = valid_runs(model, task, config, include_smoke)
                if not runs:
                    continue
                scores = question_scores(runs[0], task)
                m = run_metrics(runs[0], task)
                m["acc"] = sum(scores.values()) / len(scores) if scores else None
                m["delta"], m["se"] = paired(scores, ref) if ref and config != REFERENCE else (None, None)
                m["same"] = sum(scores[k] == ref[k] for k in ref if k in scores) / len(ref) if ref and config != REFERENCE else None
                # Rule 1: within 2 paired SE of HF, or above. min_tolerance (sensitivity check, NOTES decision 16):
                # also pass when the difference is smaller than this many points.
                m["passes"] = config == REFERENCE or (m["delta"] is not None and m["delta"] >= -max(2 * m["se"], min_tolerance / 100))
                m["repeats"] = [run_metrics(r, task) | {"acc": (lambda sc: sum(sc.values()) / len(sc))(question_scores(r, task))} for r in runs[1:]]
                rows[config] = m
            table[(size, task)] = rows
    return table


def fmt(v, nd=0, scale=1):
    return "-" if v is None else f"{v * scale:,.{nd}f}"


def print_tables(table):
    for (size, task), rows in table.items():
        if not rows:
            continue
        print(f"\n### {SIZES[size]} · {task}\n")
        print("| Configuration | Accuracy | Δ vs HF (± paired SE) | Same score | Time / question (s) | Energy / question (J) | Energy / output token (J) | TTFT p50 (ms) | Decode (ms/token) | Output tokens | Peak RAM (GB) | Power (W) | Run |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for config, m in rows.items():
            delta = "reference" if config == REFERENCE else ("-" if m["delta"] is None else f"{100 * m['delta']:+.1f} ± {100 * m['se']:.1f}" + ("" if m["passes"] else " ✗"))
            ttft = fmt(m.get("ttft_ms")) + ("*" if m.get("ttft_source") not in (None, "per request") else "")
            print(
                f"| {config} | {fmt(m['acc'], 1, 100)} | {delta} | {fmt(m['same'], 0, 100)}{'%' if m['same'] is not None else ''} | {fmt(m.get('s_per_q'), 2)} | {fmt(m.get('j_per_q'), 1)} | "
                f"{fmt(m.get('j_per_token'), 3)} | {ttft} | {fmt(m.get('decode_ms'), 1)} | {fmt(m['out_tokens'], 0)} | {fmt(m['peak_ram_gb'], 1)} | {fmt(m['power_w'], 1)} | [{m['date']}]({m['run']}) |"
            )
    print("\n✗ = fails the accuracy rule (more than 2 paired standard errors below HF bf16). * = Edge-LLM: median prefill GPU time from its runner profile (no per-request TTFT).")


def q1(table, sizes, tasks):
    """The Q1 decision rule (NOTES_qwen2_5.md), per size, task and bit width."""
    print("\n## Q1 rule\n")
    print("Per size, task and bit width: frameworks that pass the accuracy rule, the best value of each metric and who is best or tied (within 5%; 15% when the two runs are from different days).\n")
    print("| Size | Task | Bits | Metric | Best | Best or tied | Edge-LLM |")
    print("|---|---|---:|---|---|---|---|")
    edge_task_wins = {size: {} for size in sizes}
    edge_fails = []
    for size in sizes:
        for task in tasks:
            rows = table.get((size, task)) or {}
            wins = []
            for width in (16, 8, 4):
                cands = {c: m for c, m in rows.items() if WIDTH.get(c) == width and m["passes"] and not m.get("timing_invalid")}
                edge = next((c for c in rows if c.startswith("trt_edgellm") and WIDTH.get(c) == width), None)
                if edge and not rows[edge]["passes"] and any(m["passes"] for c, m in rows.items() if WIDTH.get(c) == width and c != edge):
                    edge_fails.append((size, task, edge))
                if not cands:
                    continue
                edge_hits = 0
                for key, label in Q1_METRICS:
                    vals = {c: m[key] for c, m in cands.items() if m.get(key)}
                    if not vals:
                        continue
                    best = min(vals, key=vals.get)
                    tied = [c for c, v in vals.items() if v <= vals[best] * (1.05 if cands[c]["date"] == cands[best]["date"] else 1.15)]
                    is_edge = edge in tied if edge else False
                    edge_hits += is_edge
                    print(f"| {size} | {task} | {width} | {label} | {best} ({vals[best]:.3g}) | {', '.join(tied)} | {'✓' if is_edge else '–'} |")
                if edge in cands:
                    wins.append((width, edge_hits))
            edge_task_wins[size][task] = wins
    print("\nEdge-LLM best or tied on ≥ 3 of 4 metrics, per size and task (bit width: metrics won):\n")
    verdict = True
    for size in sizes:
        tasks_won = 0
        for task in tasks:
            wins = edge_task_wins[size].get(task, [])
            won = bool(wins) and all(h >= 3 for _, h in wins)
            tasks_won += won
            print(f"- {size} {task}: " + (", ".join(f"{w}-bit {h}/4" for w, h in wins) or "no comparison") + (" → wins" if won else ""))
        verdict &= tasks_won >= 2
        print(f"- **{size}: Edge-LLM wins {tasks_won} of {len(tasks)} tasks**")
    if edge_fails:
        print("- Edge-LLM fails the accuracy rule where another framework passes: " + ", ".join(f"{s} {t} {c}" for s, t, c in edge_fails))
    print(f"\n**Q1 rule: {'Edge-LLM wins' if verdict and not edge_fails else 'not a win (split or negative), see the rows above'}.**")


def print_repeats(table):
    """Repeat runs (later valid runs of a configuration) against the first run."""
    print("\n## Repeats\n")
    print("| Size | Task | Configuration | Runs (dates) | Accuracy | Time / question | Energy / question | TTFT p50 | Decode ms/token |")
    print("|---|---|---|---|---|---|---|---|---|")

    def change(a, b, nd=2):
        return "-" if a is None or b is None else f"{a:.{nd}f} → {b:.{nd}f} ({100 * (b - a) / a:+.0f}%)"

    for (size, task), rows in table.items():
        for config, m in rows.items():
            for r in m["repeats"]:
                print(
                    f"| {size} | {task} | {config} | {m['date']} / {r['date']} | {100 * m['acc']:.1f} → {100 * r['acc']:.1f} | {change(m.get('s_per_q'), r.get('s_per_q'))} | "
                    f"{change(m.get('j_per_q'), r.get('j_per_q'), 1)} | {change(m.get('ttft_ms'), r.get('ttft_ms'), 0)} | {change(m.get('decode_ms'), r.get('decode_ms'), 1)} |"
                )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", nargs="*", default=list(SIZES))
    p.add_argument("--tasks", nargs="*", default=TASKS)
    p.add_argument("--q1", action="store_true", help="apply the Q1 decision rule")
    p.add_argument("--repeats", action="store_true", help="compare repeat runs with the first run")
    p.add_argument("--json", help="also write the metrics to this file (for the figures)")
    p.add_argument("--include-smoke", action="store_true")
    p.add_argument("--min-tolerance", type=float, default=0.0, help="sensitivity check: also pass accuracy differences smaller than this many points (default 0: the rule as fixed)")
    a = p.parse_args()
    table = collect(a.sizes, a.tasks, a.include_smoke, a.min_tolerance)
    if a.min_tolerance:
        print(f"Sensitivity check: accuracy rule passes Δ ≥ −max(2 SE, {a.min_tolerance:g} points) (NOTES_qwen2_5.md decision 16).")
    print_tables(table)
    if a.q1:
        q1(table, a.sizes, a.tasks)
    if a.repeats:
        print_repeats(table)
    if a.json:
        json.dump({f"{s}|{t}": rows for (s, t), rows in table.items()}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
