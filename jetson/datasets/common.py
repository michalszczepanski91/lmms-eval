"""Shared code for the per-dataset scripts (textvqa.py, mme.py, gqa.py): statistics and examples.

Docs, prompts and images come from lmms-eval's own task (doc_to_text / doc_to_visual), so they are exactly what
the evaluation sends to the model. A dataset script only says how to read the reference answers and image ids.

Output (stdout and <out>/<task>.md): the dataset table row of FINDINGS.md, and examples with image, prompt,
reference and, with --run, the model's answer. Run through jetson/datasets/run.sh (eval container, offline, CPU).
"""

import argparse
import glob
import gzip
import json
import os
import statistics as st
from dataclasses import dataclass
from typing import Callable

# Qwen2.5-VL image range used by every framework (jetson/models/qwen2_5_vl/model.sh): 256-2048 tokens of 28x28 px.
PATCH = 28
MIN_PIXELS, MAX_PIXELS = 256 * PATCH * PATCH, 2048 * PATCH * PATCH
SUFFIX_NOTE = "end token included"


@dataclass
class Dataset:
    task: str  # lmms-eval task name
    title: str
    references: Callable[[dict], list]  # all reference answers of a doc
    image_id: Callable[[dict], str]  # to count distinct images
    reference: Callable[[dict], str] = None  # the one reference used for length stats (default: first)
    scoring: str = ""


def tokenizer(model):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # Local snapshot path: transformers queries the Hub for repo ids even when offline.
    return AutoTokenizer.from_pretrained(snapshot_download(model, local_files_only=True))


def image_tokens(w, h):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    rh, rw = smart_resize(h, w, factor=PATCH, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return (rh // PATCH) * (rw // PATCH), rw, rh


def load_task(task):
    from lmms_eval.tasks import TaskManager, get_task_dict

    t = get_task_dict([task], TaskManager())[task]
    return t, list(t.test_docs())


def load_run(run_dir):
    """doc_id -> sample record of an lmms-eval run (samples jsonl)."""
    paths = glob.glob(os.path.join(run_dir, "lmms_eval", "*_samples_*.jsonl*"))
    if not paths:
        raise SystemExit(f"no samples file in {run_dir}/lmms_eval")
    out = {}
    for p in paths:
        for line in gzip.open(p, "rt") if p.endswith(".gz") else open(p):
            d = json.loads(line)
            out[d["doc_id"]] = d
    return out


def response(sample):
    r = sample["filtered_resps"]
    return r if isinstance(r, str) else r[0]


def fmt(xs, nd=1):
    def f(v):
        return f"{v:,.{nd}f}" if isinstance(v, float) else f"{v:,}"

    return f"{f(float(st.mean(xs)))} ({f(min(xs))}–{f(max(xs))})"


def main(ds: Dataset):
    p = argparse.ArgumentParser(description=f"{ds.title}: dataset statistics and examples.")
    p.add_argument("--examples", type=int, nargs="*", help="doc ids to show (default: shortest and longest answer)")
    p.add_argument("--run", help="lmms-eval run dir: adds the model's answers (examples, generated tokens)")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="tokenizer (default: %(default)s)")
    p.add_argument("--out", default="jetson/datasets/reports", help="writes <out>/<task>.md and example images")
    p.add_argument("--limit", type=int, help="first N docs only (quick test)")
    a = p.parse_args()
    reference = ds.reference or (lambda d: ds.references(d)[0])

    tk = tokenizer(a.model)

    def ntok(s):
        return len(tk(s, add_special_tokens=False)["input_ids"])

    def tok_list(s):
        return " ".join(f"`{t.replace('Ġ', '␣')}`" for t in tk.convert_ids_to_tokens(tk(s, add_special_tokens=False)["input_ids"]))

    task, docs = load_task(ds.task)
    docs = docs[: a.limit] if a.limit else docs
    run = load_run(a.run) if a.run else None

    rows = []
    for i, doc in enumerate(docs):
        img = task.doc_to_visual(doc)[0]
        w, h = img.size
        n_img, _, _ = image_tokens(w, h)
        prompt = task.doc_to_text(doc)
        ref = reference(doc)
        r = dict(id=i, w=w, h=h, mp=w * h / 1e6, img_tok=n_img, q_tok=ntok(prompt), ref_words=len(ref.split()), ref_tok=ntok(ref), digit=any(c.isdigit() for c in ref), image=ds.image_id(doc))
        if run and i in run:
            ans = response(run[i])
            r.update(ans_tok=ntok(ans), gen_tok=run[i]["token_counts"][0].get("output_tokens"))
        rows.append(r)

    def col(k):
        return [r[k] for r in rows]

    stats = [
        ("Questions (distinct images)", f"{len(rows):,} ({len(set(col('image'))):,})"),
        ("Image size, MP, avg (min–max)", f"{fmt(col('mp'), 2)}, avg {st.mean(col('w')):.0f}×{st.mean(col('h')):.0f} px"),
        ("Image tokens, avg (min–max)", fmt(col("img_tok"))),
        ("Question tokens (prompt text), avg (min–max)", fmt(col("q_tok"))),
        ("Reference answer words, avg (min–max)", fmt(col("ref_words"), 2)),
        ("Reference answer tokens, avg (min–max)", fmt(col("ref_tok"), 2)),
        ("Reference answers of one word / with a digit", f"{sum(x == 1 for x in col('ref_words')) / len(rows):.0%} / {sum(col('digit')) / len(rows):.0%}"),
    ]
    with_run = [r for r in rows if "ans_tok" in r]
    if with_run:
        stats += [
            ("Model answer tokens, avg (min–max)", fmt([r["ans_tok"] for r in with_run], 2)),
            (f"Generated tokens ({SUFFIX_NOTE}), avg (min–max)", fmt([r["gen_tok"] for r in with_run if r["gen_tok"] is not None], 2)),
        ]
    if ds.scoring:
        stats.append(("Scoring", ds.scoring))

    key = "ans_tok" if with_run else "ref_tok"
    pool = with_run or rows
    ids = a.examples if a.examples else [min(pool, key=lambda r: (r[key], r["id"]))["id"], max(pool, key=lambda r: (r[key], r["id"]))["id"]]

    os.makedirs(a.out, exist_ok=True)
    md = [f"# {ds.title}", "", f"Task `{ds.task}`, tokenizer `{a.model}`" + (f", answers from `{a.run}`" if run else "") + ".", "", "| | |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in stats]
    md += ["", "## Examples", ""]
    for i in ids:
        doc, r = docs[i], rows[i]
        img = task.doc_to_visual(doc)[0].convert("RGB")
        n_img, rw, rh = image_tokens(r["w"], r["h"])
        fn = f"{ds.task}_{i}.jpg"
        thumb = img.copy()
        thumb.thumbnail((480, 480))
        thumb.save(os.path.join(a.out, fn), quality=85)
        refs = ds.references(doc)
        ref_txt = ", ".join(f'"{x}"' + (f" ×{refs.count(x)}" if len(refs) > 1 else "") for x in sorted(set(refs), key=lambda x: -refs.count(x)))
        md += [
            f"### doc {i}",
            "",
            f"![{ds.task} doc {i}]({fn})",
            "",
            "| | |",
            "|---|---|",
            f"| Image | {r['w']}×{r['h']} px → resized {rw}×{rh} px → {n_img} image tokens |",
            f"| Prompt ({r['q_tok']} tokens) | {task.doc_to_text(doc).replace(chr(10), '<br>')} |",
            f"| Reference | {ref_txt} |",
            f"| Reference tokens | {tok_list(reference(doc))} |",
        ]
        if run and i in run:
            ans = response(run[i])
            score = {k: v for k, v in run[i].items() if k not in ("doc_id", "doc", "target", "filtered_resps", "resps", "token_counts", "doc_hash", "input_media", "input", "submission", "arguments")}
            md += [
                f"| Model answer | {ans} |",
                f"| Answer tokens | {tok_list(ans) if r['ans_tok'] <= 30 else str(r['ans_tok']) + ' tokens'} (generated: {r['gen_tok']}, {SUFFIX_NOTE}) |",
                f"| Score | {json.dumps(score, ensure_ascii=False)} |",
            ]
        md.append("")
    path = os.path.join(a.out, f"{ds.task}.md")
    open(path, "w").write("\n".join(md))
    for k, v in stats:
        print(f"{k:50s} {v}")
    print(f"examples: doc {', '.join(map(str, ids))}\nwrote {path}")
