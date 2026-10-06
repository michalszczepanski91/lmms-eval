#!/usr/bin/env python3
"""Per-question time per stage (vision encoder / prefill / decode) for chosen runs, as a Markdown table.

Usage: jetson/scripts/stages_table.py [--task textvqa_val] [-n 50] [--start 0] [-o out.md] <run> [<run> ...]
  <run>  <size>:<framework>-<precision>[+tag] (newest valid full run) or a run directory, as in plot_stages.py.

  jetson/scripts/stages_table.py -n 50 3b:hf-bf16 3b:vllm-bf16 3b:llamacpp-f16 3b:trt_edgellm-fp16 \\
      7b:hf-bf16 7b:vllm-bf16 7b:llamacpp-f16 7b:trt_edgellm-fp16 -o jetson/results/thor/stages_textvqa_50.md

One table per model size, one row per question: image and prompt tokens, the model's answer, and per run
"vision / prefill / decode" in ms (decode = all tokens after the first). vLLM has no vision/prefill split: its cell
is "TTFT / decode". Edge-LLM records only means: it appears in the summary rows, not per question.
"""

import argparse
import glob
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_stages import edgellm_means, find_run, label  # noqa: E402
from summarize import LLAMA_TS, open_text  # noqa: E402


def samples(run_dir):
    out = {}
    for path in glob.glob(os.path.join(run_dir, "lmms_eval", "*_samples_*.jsonl*")):
        for d in map(json.loads, open_text(path)):
            out[d["doc_id"]] = d
    return out


def answer(d):
    r = d["filtered_resps"]
    return r if isinstance(r, str) else r[0]


def client_stages(run_dir):
    """HF / vLLM, per doc: (vision ms or None, prefill or TTFT ms, decode ms, generated tokens)."""
    out = {}
    for i, d in samples(run_dir).items():
        c = d["token_counts"][0]
        ttft, gen = c["time_to_first_token_seconds"] * 1e3, c["generation_seconds"] * 1e3
        vis = c["vision_seconds"] * 1e3 if "vision_seconds" in c else None
        out[i] = (vis, ttft - vis if vis is not None else ttft, gen - ttft, c["output_tokens"])
    return out


def llamacpp_stages(run_dir):
    """Per doc (requests run one at a time in doc order): (vision, prefill, decode ms, generated tokens), server side."""
    out, cur = [], {}
    for line in open_text(os.path.join(run_dir, "server.log")):
        ts = LLAMA_TS.match(line)
        t = (int(ts[1]) * 60 + int(ts[2]) + int(ts[3]) / 1e3 + int(ts[4]) / 1e6) * 1e3 if ts else None
        if "encoding mtmd batch" in line:
            cur = {"start": t}
        elif "decoding image batch 1/" in line and "start" in cur:
            cur["vis"] = t - cur["start"]
        elif (m := re.search(r"prompt eval time =\s+([\d.]+) ms", line)) and "vis" in cur:
            cur["pre"] = float(m[1]) - cur["vis"]
        elif (m := re.search(r"\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens", line)) and "pre" in cur:
            out.append((cur["vis"], cur["pre"], float(m[1]), int(m[2])))
            cur = {}
    return dict(enumerate(out))


def kind(run_dir):
    if os.path.exists(os.path.join(run_dir, "trt_profile.json")):
        return "edgellm"
    if glob.glob(os.path.join(run_dir, "server.log*")):
        return "llamacpp"
    return "client"


def cell(s):
    return s.replace("|", "\\|").replace("\n", " ")


def fmt(v):
    vis, pre, dec = v[:3]
    return f"{pre:.0f} / {dec:.0f}" if vis is None else f"{vis:.0f} / {pre:.0f} / {dec:.0f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--task", default="textvqa_val")
    p.add_argument("-n", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("-o", "--out", default="stages_table.md")
    a = p.parse_args()

    runs = [find_run(r, a.task) for r in a.runs]
    sizes = []
    for d in runs:
        s = label(d).split()[0]
        if s not in sizes:
            sizes.append(s)
    ids = list(range(a.start, a.start + a.n))
    md = [
        f"# Time per stage per question: {a.task}, questions {ids[0]}–{ids[-1]}",
        "",
        "- Cells: **vision / prefill / decode** in ms; decode = all generated tokens after the first. **vLLM: TTFT / decode** (it doesn't report the vision encoder separately).",
        "- HF and vLLM: measured by lmms-eval around the model call. llama.cpp: server log, without HTTP. Edge-LLM: GPU time, means only (summary rows).",
        "- Text tokens: prompt + chat template (21). Answer and generated tokens (end token included) from the first run of each table; **≠** lists runs whose answer differs.",
        "- The first question of a run includes warm-up (e.g. vLLM's first TTFT of several seconds); the means include it too.",
        "",
    ]
    for d in runs:
        md.append(f"- {label(d)}: `{os.path.relpath(d)}`")
    md.append("")
    for size in sizes:
        group = [d for d in runs if label(d).split()[0] == size]
        timed = [d for d in group if kind(d) != "edgellm"]
        stages = {d: (llamacpp_stages(d) if kind(d) == "llamacpp" else client_stages(d)) for d in timed}
        answers = {d: {i: answer(x) for i, x in samples(d).items()} for d in group}
        ref = group[0]
        flops = {x["doc_id"]: x for x in map(json.loads, open_text(os.path.join(ref, "flops_samples.jsonl.gz")))}
        heads = ["Doc", "Image tokens", "Text tokens", "Answer", "Gen. tokens"] + [label(d).split(" ", 1)[1] for d in timed] + ["≠"]
        md += [f"## {size}", "", "| " + " | ".join(heads) + " |", "|" + "|".join("---" if h in ("Answer", "≠") else "---:" for h in heads) + "|"]
        for i in ids:
            f = flops.get(i, {})
            gen = stages[ref][i][3] if ref in stages and i in stages[ref] else f.get("output_tokens", "–")
            differ = [label(d).split(" ", 1)[1].split("-")[0] for d in group[1:] if answers[d].get(i, "").strip().lower() != answers[ref].get(i, "").strip().lower()]
            md.append(
                "| "
                + " | ".join([str(i), str(f.get("image_tokens", "–")), str(f.get("text_tokens", "–")), cell(answers[ref].get(i, "–")), str(gen)] + [fmt(stages[d][i]) if i in stages[d] else "–" for d in timed] + [", ".join(differ)])
                + " |"
            )
        # summary rows: mean over these questions and over all questions
        for name, sel in (("mean, these questions", ids), ("mean, all questions", None)):
            row = [f"**{name}**", "", "", "", ""]
            for d in timed:
                v = [stages[d][i] for i in (sel or stages[d]) if i in stages[d]]
                row.append(fmt((None if v[0][0] is None else st.mean(x[0] for x in v), st.mean(x[1] for x in v), st.mean(x[2] for x in v))))
            md.append("| " + " | ".join(row + [""]) + " |")
        for d in group:
            if kind(d) == "edgellm":
                vis, pre, dec = edgellm_means(d)
                md.append(f"\n{label(d)}, all questions (GPU time, means): vision {vis:.0f} / prefill {pre:.0f} ms, decode {dec:.0f} ms per token.")
        md.append("")
    with open(a.out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
