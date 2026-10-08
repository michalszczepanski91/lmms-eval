#!/usr/bin/env python3
"""Prompts of an exact length for the controlled-length sweep (jetson/tasks/length_sweep).

Each prompt is a block of synthetic notes followed by an instruction that makes the model generate far more than
256 tokens ("count from 1 to 1000, one number per line"). Every framework then stops at the same cap of 256 new
tokens, without any backend option: the Edge-LLM runner has no ignore-EOS setting, so the prompt does that job.
The notes are cut so that the whole request (Qwen2.5 chat template with its default system prompt) is exactly
128, 512 or 2,048 tokens. Different questions start at different notes, so no two prompts are equal.

Usage (eval container, offline, CPU): jetson/datasets/run.sh length_sweep_data [--n 16]
Writes jetson/tasks/length_sweep/data/length_<N>.jsonl.
"""

import argparse
import json
import os

from common import tokenizer

LENGTHS = (128, 512, 2048)
INSTRUCTION = "\n\nIgnore the notes above. Count from 1 to 1000, one number per line, with no other text."
ITEMS = ["valves", "cables", "lamps", "pumps", "filters", "sensors", "brackets", "gears", "hoses", "panels", "relays"]
CITIES = ["Oslo", "Lyon", "Porto", "Graz", "Leeds", "Turin", "Bergen", "Ghent", "Malmo", "Brno", "Cork", "Split", "Basel"]


def note(i):
    return f"Note {i}: shipment {i} of {(i * 37) % 900 + 100} {ITEMS[i % len(ITEMS)]} left {CITIES[i % len(CITIES)]} on day {i % 28 + 1} and arrived after {i % 9 + 2} days."


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=16, help="prompts per length (default %(default)s)")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="tokenizer and chat template (same for 1.5B and 3B)")
    p.add_argument("--out", default="jetson/tasks/length_sweep/data")
    a = p.parse_args()
    tk = tokenizer(a.model)

    def length(text):
        return len(tk.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True))

    os.makedirs(a.out, exist_ok=True)
    for n_tokens in LENGTHS:
        rows = []
        for q in range(a.n):
            notes = " ".join(note(i) for i in range(q * 200 + 1, q * 200 + 200))
            ids = tk(notes, add_special_tokens=False)["input_ids"]
            # Cut the notes at the longest length that fits, then pad with one-token words: re-tokenizing a cut
            # text can merge tokens at the cut, so a cut alone cannot hit every length.
            k = n_tokens - length(INSTRUCTION)
            while length(tk.decode(ids[:k]).rstrip() + INSTRUCTION) > n_tokens:
                k -= 1
            notes = tk.decode(ids[:k]).rstrip()
            while length(notes + INSTRUCTION) < n_tokens:
                cur = length(notes + INSTRUCTION)
                notes += next(pad for pad in (" x", " ok", " -", ".", " the", " and") if length(notes + pad + INSTRUCTION) == cur + 1)
            text = notes + INSTRUCTION
            assert length(text) == n_tokens, (n_tokens, q, length(text))
            rows.append({"id": f"len{n_tokens}_{q:02d}", "prompt_tokens": n_tokens, "prompt": text})
        path = os.path.join(a.out, f"length_{n_tokens}.jsonl")
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"wrote {path}: {len(rows)} prompts of exactly {n_tokens} tokens")


if __name__ == "__main__":
    main()
