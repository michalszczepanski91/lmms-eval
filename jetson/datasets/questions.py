#!/usr/bin/env python3
"""One row per question: prompt, image, image tokens and the model's answer, with token counts.

Usage: jetson/datasets/run.sh questions <textvqa|mme|gqa> [-n 50] [--start 0] [--run <run dir>] [--out DIR]
  jetson/datasets/run.sh questions gqa -n 50
  jetson/datasets/run.sh questions textvqa -n 20 --run jetson/results/thor/Qwen2.5-VL-7B-Instruct/textvqa_val/vllm-bf16/20260929-020010

--run defaults to the newest full HF bf16 3B run of the task under jetson/results/<board>/. Writes <out>/<task>_questions.md.
"""

import argparse
import glob
import importlib
import os

from common import PATCH, SUFFIX_NOTE, image_tokens, load_run, load_task, response, tokenizer

EXCLUDED = ("INVALID", "STALLED", "SUPERSEDED")


def default_run(task):
    runs = [r for r in glob.glob(f"jetson/results/*/Qwen2.5-VL-3B-Instruct/{task}/hf-bf16/*") if "_limit" not in r and not any(os.path.exists(os.path.join(r, x)) for x in EXCLUDED)]
    return max(runs, key=os.path.basename) if runs else None


def cell(s):
    return s.replace("|", "\\|").replace("\n", "<br>")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="textvqa, mme or gqa (a module in jetson/datasets)")
    p.add_argument("-n", type=int, default=50, help="number of questions (default: %(default)s)")
    p.add_argument("--start", type=int, default=0, help="first doc id (default: %(default)s)")
    p.add_argument("--run", help="lmms-eval run dir with the model's answers (default: newest HF bf16 3B run)")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="tokenizer (default: %(default)s)")
    p.add_argument("--out", default="jetson/datasets/reports")
    a = p.parse_args()

    ds = importlib.import_module(a.dataset).DATASET
    run_dir = a.run or default_run(ds.task)
    run = load_run(run_dir) if run_dir else {}
    tk = tokenizer(a.model)

    def ntok(s):
        return len(tk(s, add_special_tokens=False)["input_ids"])

    task, docs = load_task(ds.task)

    head = ["Doc", "Prompt text", "Prompt tokens", "Image size (px)", "Resized (px)", "Image tokens W × H", "Image tokens", "Model answer", "Answer tokens", "Generated tokens", "Reference", "Score"]
    rows = []
    for i in range(a.start, min(a.start + a.n, len(docs))):
        doc = docs[i]
        w, h = task.doc_to_visual(doc)[0].size
        n_img, rw, rh = image_tokens(w, h)
        prompt = task.doc_to_text(doc)
        s = run.get(i)
        ans = response(s) if s else ""
        gen = (s.get("token_counts") or [{}])[0].get("output_tokens") if s else None
        score = next((v for k, v in (s or {}).items() if k in ("exact_match", "mme_percetion_score", "mme_cognition_score")), None)
        score = score.get("score") if isinstance(score, dict) else score
        refs = ds.references(doc)
        rows.append(
            [
                str(i),
                cell(prompt),
                str(ntok(prompt)),
                f"{w}×{h}",
                f"{rw}×{rh}",
                f"{rw // PATCH} × {rh // PATCH}",
                str(n_img),
                cell(ans) if s else "–",
                str(ntok(ans)) if s else "–",
                str(gen) if gen is not None else "–",
                cell(", ".join(sorted(set(refs), key=refs.count, reverse=True)[:3])),
                f"{score:g}" if isinstance(score, (int, float)) else "–",
            ]
        )

    md = [
        f"# {ds.title}: questions {a.start}–{a.start + len(rows) - 1}",
        "",
        f"Task `{ds.task}`, tokenizer `{a.model}`" + (f", answers from `{run_dir}`" if run_dir else ", no run (no answers)") + ".",
        "",
        f"- **Image tokens:** the image is resized to multiples of {PATCH} px within 256–2,048 tokens; each {PATCH}×{PATCH} px block is one token (W × H = tokens across × down).",
        "- **Prompt tokens:** the prompt text only; the chat template adds 21 tokens. The model reads prompt + image tokens.",
        f"- **Answer tokens:** the model's answer tokenized; **generated tokens:** as logged by the run ({SUFFIX_NOTE}; empty for Edge-LLM).",
        "- **Reference:** up to 3 of the most common reference answers.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join("---:" if h in ("Doc", "Prompt tokens", "Image tokens", "Answer tokens", "Generated tokens", "Score") else "---" for h in head) + "|",
    ]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{ds.task}_questions.md")
    open(path, "w").write("\n".join(md) + "\n")
    print(f"answers: {run_dir or 'none'}\nwrote {path} ({len(rows)} questions)")


if __name__ == "__main__":
    main()
