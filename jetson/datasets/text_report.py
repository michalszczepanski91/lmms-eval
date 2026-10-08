#!/usr/bin/env python3
"""Text-only tasks (GSM8K, MMLU, IFEval of jetson/models/qwen2_5): statistics and one example per task.

Prompts come from lmms-eval's chat path exactly as the evaluation builds them (jetson/tasks variants, few-shot sampler
seeded like the evaluator), with the Qwen2.5 chat template applied, so the token counts are what every framework
receives. With --run, adds the generated tokens of each run (token_counts.output_tokens; Edge-LLM runs record none).

Usage (eval container, offline, CPU): jetson/datasets/run.sh text_report <gsm8k_jetson|mmlu_jetson|ifeval> [--run DIR ...]
Writes <out>/<task>.md (default jetson/datasets/reports).
"""

import argparse
import glob
import gzip
import json
import os
import statistics as st

from common import tokenizer

REFERENCE = {
    # task: (reference text of a doc, description of the scoring)
    "gsm8k_jetson": (lambda d: d["answer"].split("####")[-1].strip(), "exact match of the last number in the answer (flexible-extract)"),
    "mmlu_jetson": (lambda d: "ABCD"[d["answer"]], "the first A-D letter of the answer; accuracy weighted by subject size"),
    "ifeval": (None, "fraction of prompts whose verifiable instructions are all followed (prompt-level, strict)"),
}
QUESTION = {
    "gsm8k_jetson": lambda d: d["question"],
    "mmlu_jetson": lambda d: d["question"],
    "ifeval": lambda d: d["prompt"],
}


def leaves(tasks):
    for v in tasks.values():
        if isinstance(v, dict):
            yield from leaves(v)
        else:
            yield v


def fmt(xs, nd=0):
    def f(v):
        return f"{v:,.{nd}f}" if isinstance(v, float) else f"{v:,}"

    return f"{f(float(st.mean(xs)))} ({f(min(xs))}–{f(max(xs))})"


def run_output_tokens(run_dir):
    out = []
    for path in glob.glob(os.path.join(run_dir, "lmms_eval", "*_samples_*.jsonl*")):
        for line in gzip.open(path, "rt") if path.endswith(".gz") else open(path):
            counts = (json.loads(line).get("token_counts") or [{}])[0]
            if counts.get("output_tokens") is not None:
                out.append(counts["output_tokens"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", choices=sorted(REFERENCE))
    p.add_argument("--run", nargs="*", default=[], help="run dirs: adds their generated tokens per framework")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="tokenizer and chat template (default: %(default)s)")
    p.add_argument("--out", default="jetson/datasets/reports")
    a = p.parse_args()

    from lmms_eval.tasks import TaskManager, get_task_dict

    tk = tokenizer(a.model)

    def ntok(s):
        return len(tk(s, add_special_tokens=False)["input_ids"])

    reference, scoring = REFERENCE[a.task]
    rows, example = [], None
    for task in leaves(get_task_dict([a.task], TaskManager(include_path="jetson/tasks"), task_type="chat")):
        task.set_fewshot_seed(seed=1234)  # as the evaluator does
        for doc in task.eval_docs:
            text = "".join(c["text"] for m in task.doc_to_messages(doc) for c in m["content"] if c.get("type") == "text")
            prompt = len(tk.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True))
            row = {"prompt": prompt, "question": ntok(QUESTION[a.task](doc))}
            if reference:
                ref = reference(doc)
                row.update(ref_words=len(ref.split()), ref_tokens=ntok(ref))
            if a.task == "ifeval":
                row["instructions"] = len(doc["instruction_id_list"])
            rows.append(row)
            if example is None:
                example = (text, reference(doc) if reference else ", ".join(doc["instruction_id_list"]))

    def col(k):
        return [r[k] for r in rows]

    stats = [
        ("Questions", f"{len(rows):,}"),
        ("Prompt tokens (chat template + few-shot context), avg (min–max)", fmt(col("prompt"))),
        ("Question tokens (the question alone), avg (min–max)", fmt(col("question"))),
    ]
    if reference:
        stats += [("Reference answer words / tokens, avg (min–max)", f"{fmt(col('ref_words'), 1)} / {fmt(col('ref_tokens'), 1)}")]
    if a.task == "ifeval":
        stats += [("Instructions per prompt, avg (min–max)", fmt(col("instructions"), 1))]
    for run_dir in a.run:
        out = run_output_tokens(run_dir)
        if out:
            stats.append((f"Generated tokens (end token included), {os.path.basename(os.path.dirname(run_dir.rstrip('/')))}", fmt(out, 1)))
    stats.append(("Scoring", scoring))

    md = [f"# {a.task}", "", f"Tokenizer and chat template: `{a.model}`.", "", "| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in stats]
    md += ["", "## Example (first question)", "", "```", example[0], "```", "", f"Reference: {example[1]}", ""]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.task}.md")
    open(path, "w").write("\n".join(md))
    print("\n".join(f"{k}: {v}" for k, v in stats))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
