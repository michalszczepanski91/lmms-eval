#!/usr/bin/env python3
"""Task-matched calibration text for TensorRT Edge-LLM quantization (Phase 2 of the text-only Qwen2.5 study).

Edge-LLM calibrates on 512 texts of up to 512 tokens, from cnn_dailymail by default. This builds the same number of
texts in the benchmark's own format: a question as the Jetson task variants ask it, then the reference answer, in
the Qwen2.5 chat template. Half GSM8K (train split), half MMLU (validation split, all subjects); never a test split,
and nothing from IFEval (it has only its evaluation split). Used through Edge-LLM's documented local override of a
built-in dataset (EDGELLM_QUANT_DATASET_WIKITEXT=<jsonl>, field "text"); see frameworks/trt_edgellm/recipes.sh.

Usage (eval container, offline, CPU): jetson/datasets/run.sh edgellm_calibration [--n 512]
Writes jetson/frameworks/trt_edgellm/calibration/gsm8k_mmlu.jsonl.
"""

import argparse
import glob
import json
import os
import random

from common import tokenizer

MMLU_INSTRUCTION = "Answer the last question with the letter of the correct option only."


def gsm8k_texts(n):
    from datasets import load_dataset

    for doc in load_dataset("gsm8k", "main", split="train").select(range(n)):
        yield f"Question: {doc['question']}\nAnswer:", doc["answer"]


def mmlu_texts(n):
    from datasets import load_dataset

    subjects = sorted(os.path.basename(f)[len("mmlu_jetson_") : -len(".yaml")] for f in glob.glob("jetson/tasks/mmlu_jetson/mmlu_jetson_*.yaml"))
    docs = [(subject, doc) for subject in subjects for doc in load_dataset("hails/mmlu_no_train", subject, split="validation")]
    random.Random(0).shuffle(docs)  # mix the subjects
    for subject, doc in docs[:n]:
        subject = subject.replace("_", " ")
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", doc["choices"]))
        prompt = f"The following are multiple choice questions (with answers) about {subject}. {MMLU_INSTRUCTION}\n\n{doc['question'].strip()}\n{options}\nAnswer:"
        yield prompt, "ABCD"[doc["answer"]]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=512, help="texts in all, half per task (default %(default)s)")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="chat template (same for 1.5B and 3B)")
    p.add_argument("--out", default="jetson/frameworks/trt_edgellm/calibration/gsm8k_mmlu.jsonl")
    a = p.parse_args()
    tk = tokenizer(a.model)
    rows = []
    for source in (gsm8k_texts(a.n // 2), mmlu_texts(a.n - a.n // 2)):
        for prompt, answer in source:
            text = tk.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], tokenize=False)
            rows.append({"text": text})
    random.Random(0).shuffle(rows)  # the quantizer takes the first texts in order
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    tokens = [len(tk(r["text"])["input_ids"]) for r in rows]
    print(f"wrote {a.out}: {len(rows)} texts, {sum(tokens) / len(tokens):.0f} tokens on average (max {max(tokens)})")


if __name__ == "__main__":
    main()
