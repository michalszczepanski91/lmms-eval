#!/usr/bin/env python3
"""Teaching diagrams of the Qwen2.5-1.5B / 3B-Instruct (text-only) architecture -> figures/*.png, figures/*.svg and one PDF.

Every size in the figures is read from the checkpoints' config.json (copies in configs/). Parameter counts are
derived from the layer shapes and, when the weights are in the HF cache, checked against the safetensors headers.
The structure follows transformers 4.57 (models/qwen2/modeling_qwen2.py). The worked example ("What is 2+2?" ->
36 prompt tokens -> "2 + 2 equals 4.") was produced with the real tokenizer and chat template; see README.md.
The drawing kit (boxes, arrows, panels, colours) is shared with the Qwen2.5-VL diagrams.

Usage: python3 jetson/models/qwen2_5/docs/make_diagrams.py      (needs only matplotlib + numpy)
"""

import glob
import importlib.util
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
VL_DOCS = HERE.parents[1] / "qwen2_5_vl" / "docs"
HF_CACHE = os.environ.get("HF_HUB_CACHE", "/opt/hf-cache/hub")

_spec = importlib.util.spec_from_file_location("vl_diagrams", VL_DOCS / "make_diagrams.py")
kit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kit)
PAL, INK, SUB = kit.PAL, kit.INK, kit.SUB
T, box, arrow, line, shape, op, panel, header, light = kit.T, kit.box, kit.arrow, kit.line, kit.shape, kit.op, kit.panel, kit.header, kit.light
canvas, fmt, fmt2, residual_panel, swiglu_panel, layer_strip, param_bar = kit.canvas, kit.fmt, kit.fmt2, kit.residual_panel, kit.swiglu_panel, kit.layer_strip, kit.param_bar

BANDWIDTH = 273e9  # Jetson AGX Thor LPDDR5X, bytes/s (spec)
LOOP = PAL["decoder"][1]

# The worked example, produced with the Qwen2.5 tokenizer and chat template (identical for 1.5B and 3B).
EXAMPLE_PROMPT = "What is 2+2?"
EXAMPLE_IDS = [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]
EXAMPLE_PIECES = ["<|im_start|>", "system", "\\n", "You", " are", " Q", "wen", ",", " created", " by", " Alibaba", " Cloud", ".", " You", " are", " a", " helpful", " assistant", ".", "<|im_end|>", "\\n", "<|im_start|>", "user", "\\n", "What", " is", " ", "2", "+", "2", "?", "<|im_end|>", "\\n", "<|im_start|>", "assistant", "\\n"]
ANSWER_IDS = [17, 488, 220, 17, 16819, 220, 19, 13, 151645]
ANSWER_PIECES = ["2", " +", " ", "2", " equals", " ", "4", ".", "<|im_end|>"]
L_EX = len(EXAMPLE_IDS)  # 36

# Weight formats for the bytes-per-decode-step figure: (label, bytes per decoder-layer weight, bytes per LM-head weight).
# 4-bit: 0.5 B + one 16-bit scale per group of 128 weights (AWQ / GPTQ style); 8-bit per-channel scales are negligible.
INT4 = 0.5 + 2 / 128
FORMATS = [
    ("16-bit (bf16 / fp16)", 2, 2),
    ("8-bit, LM head 16-bit", 1, 2),
    ("8-bit, LM head 8-bit", 1, 1),
    ("4-bit, LM head 16-bit", INT4, 2),
    ("4-bit, LM head 4-bit", INT4, INT4),
]
SERIES = [("decoder layers", "#b0304f"), ("LM head (= embedding matrix)", "#34539c"), ("KV cache, 1,000 tokens (16-bit)", "#bf6a00")]  # validated: CVD ΔE 12.7, normal 16.0
KV_TOKENS = 1000  # about one GSM8K request (752 prompt tokens + the answer)


# ----------------------------------------------------------------------------------------------- model numbers


def load_model(size, path=None):
    cfg = json.load(open(path or HERE / "configs" / f"Qwen2.5-{size}-Instruct.json"))
    m = SimpleNamespace(
        size=size,
        name=f"Qwen2.5-{size}-Instruct",
        d=cfg["hidden_size"],
        inter=cfg["intermediate_size"],
        layers=cfg["num_hidden_layers"],
        heads=cfg["num_attention_heads"],
        kv=cfg["num_key_value_heads"],
        vocab=cfg["vocab_size"],
        tied=cfg["tie_word_embeddings"],
        theta=cfg["rope_theta"],
        eps=cfg["rms_norm_eps"],
        max_pos=cfg["max_position_embeddings"],
    )
    m.hd = m.d // m.heads
    m.kvd = m.kv * m.hd
    m.rep = m.heads // m.kv
    # parameters, from the layer shapes (nn.Linear weight + bias where the layer has one)
    m.p_embed = m.vocab * m.d
    m.p_attn = (m.d * m.d + m.d) + 2 * (m.kvd * m.d + m.kvd) + m.d * m.d
    m.p_mlp = 3 * m.d * m.inter
    m.p_layer = 2 * m.d + m.p_attn + m.p_mlp
    m.p_layers = m.layers * m.p_layer + m.d  # + final norm
    m.p_head = 0 if m.tied else m.vocab * m.d
    m.p_total = m.p_embed + m.p_layers + m.p_head
    m.kv_bytes = m.layers * 2 * m.kv * m.hd * 2  # 16-bit K and V per token
    m.gflops_tok = 2 * (m.layers * (m.p_attn + m.p_mlp) + m.vocab * m.d) / 1e9
    # one decode step reads every layer weight and the whole LM head, but only one embedding row
    m.step_bytes = 2 * (m.p_layers + m.vocab * m.d)
    m.head_share = m.vocab * m.d / (m.p_layers + m.vocab * m.d)
    return m


def check_weights(m):
    """Compare the derived parameter count with the safetensors headers (skipped if the weights are absent)."""
    files = glob.glob(f"{HF_CACHE}/models--Qwen--{m.name}/snapshots/*/*.safetensors")
    if not files:
        print(f"{m.name}: weights not in {HF_CACHE}, parameter check skipped")
        return
    counts = {"embed": 0, "lm_head": 0, "all": 0}
    for f in files:
        with open(f, "rb") as fh:
            header_ = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for name, t in header_.items():
            if name == "__metadata__":
                continue
            n = int(np.prod(t["shape"]))
            counts["all"] += n
            counts["embed"] += n if name == "model.embed_tokens.weight" else 0
            counts["lm_head"] += n if name == "lm_head.weight" else 0
    derived = {"embed": m.p_embed, "lm_head": m.p_head, "all": m.p_total}
    assert counts == derived, f"{m.name}: weights {counts} != derived {derived}"
    print(f"{m.name}: parameter counts match the weights ({m.p_total:,})")


def footer(ax, W, H, extra=""):
    T(ax, W - 2, H - 1.4, f"{extra}Sources: config.json of the checkpoint · transformers 4.57 modeling_qwen2.py · Qwen2.5 tokenizer · generated by make_diagrams.py", 6.5, ha="right", color="#8a8a8a")


def save(fig, name, pdf):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=170 if ext == "png" else None, facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def chips(ax, x, y, w, h, pieces, kinds, fs=5.6, index0=0, every=5):
    """A row of token chips (one per token, piece text rotated), with the position under every `every`-th chip."""
    gap = 0.25
    cw = (w - gap * (len(pieces) - 1)) / len(pieces)
    for i, (p, k) in enumerate(zip(pieces, kinds)):
        cx0 = x + i * (cw + gap)
        fc, ec = PAL[k]
        ax.add_patch(FancyBboxPatch((cx0, y), cw, h, boxstyle="round,pad=0,rounding_size=0.4", fc=fc, ec=ec, lw=0.7, zorder=3))
        T(ax, cx0 + cw / 2, y + h / 2, p.replace(" ", "␣"), fs, mono=True, rot=90, color=INK)
        if (index0 + i) % every == 0 or i == len(pieces) - 1:
            T(ax, cx0 + cw / 2, y + h + 1.3, str(index0 + i), 5.8, color=SUB)
    return cw + gap


def piece_kind(p):
    return "io" if p.startswith("<|") else "text"


# ----------------------------------------------------------------------------------------------- figure 1: overview


def fig_overview(m, pdf):
    W, H = 200, 118
    fig, ax = canvas(W, H)
    header(
        ax,
        f"{m.name} — end-to-end architecture",
        f"{fmt(m.p_total)} parameters = token embedding {fmt(m.p_embed)} (shared with the LM head) + {m.layers} decoder layers {fmt(m.p_layers)}.  Text in, text out: no vision part.",
        W,
    )
    box(
        ax,
        140,
        1.5,
        58,
        10,
        lines=[
            "Shapes  [rows, features]:  L = prompt length in tokens,",
            f"d = {m.d} (model width),  V = {m.vocab:,} (vocab rows)",
            f"Example in all figures: “{EXAMPLE_PROMPT}”  →  L = {L_EX}",
            f"answer “2 + 2 equals 4.” = {len(ANSWER_IDS)} tokens incl. <|im_end|>",
        ],
        kind="note",
        lfs=7.4,
        align="left",
    )

    # prompt -> tokens
    panel(ax, 2, 14, 155, 48, "PROMPT → TOKEN IDS  (CPU, before the model)", "text")
    y, h = 19.5, 20
    pr = box(ax, 4, y, 16, h, "Prompt", ["one user", "message:", f"“{EXAMPLE_PROMPT}”"], kind="io", fs=10, lfs=7.7)
    tpl = box(
        ax,
        26,
        y,
        48,
        h,
        "Chat template",
        [("<|im_start|>system",), ("You are Qwen, created by Alibaba Cloud.",), ("You are a helpful assistant.<|im_end|>",), ("<|im_start|>user",), (f"{EXAMPLE_PROMPT}<|im_end|>",), ("<|im_start|>assistant",)],
        kind="text",
        fs=9.5,
        lfs=6.6,
    )
    tok = box(ax, 80, y, 30, h, "Tokenizer", ["byte-level BPE", "151,643 tokens", "+ 22 special (<|im_start|> …)", "= 151,665 ids"], kind="text", fs=9.5, lfs=7.7)
    emb = box(ax, 116, y, 38, h, "Token embedding", ["lookup table V × d", f"= {m.vocab:,} × {m.d}", "one row per token id", "rows ≥ 151,665: unused padding"], kind="text", fs=9.5, lfs=7.7)
    for a, b, s in ((pr, tpl, ""), (tpl, tok, "text"), (tok, emb, f"[{L_EX}] ids")):
        arrow(ax, a.r, b.l)
        if s:
            shape(ax, (a.r[0] + b.l[0]) / 2, a.r[1] - 2.2, s, fs=6.6)
    T(ax, 4, 43.2, f"The {L_EX} prompt tokens of the example (␣ = space, \\n = newline); position under every 5th:", 7.6, ha="left", color=SUB)
    chips(ax, 4, 45, 151, 11, EXAMPLE_PIECES, [piece_kind(p) for p in EXAMPLE_PIECES])

    # language model column
    panel(ax, 160, 14, 38, 100, "LANGUAGE MODEL", "decoder")
    x0, w0 = 163, 32
    pos = box(ax, x0, 19.5, w0, 12, "Positions (RoPE)", ["position ids 0 … L−1", "→ cos / sin, used to", "rotate q, k in every layer"], kind="norm", fs=8.8, lfs=7.2, dashed=True)
    dec = box(
        ax,
        x0,
        37,
        w0,
        27,
        f"Decoder layer ×{m.layers}",
        ["pre-RMSNorm + residual", "self-attention (GQA, causal)", f"{m.heads} query heads, {m.kv} KV heads", f"head dim {m.hd}, RoPE on q, k", f"SwiGLU MLP {m.d}→{m.inter}→{m.d}", "KV cache for decoding"],
        kind="decoder",
        fs=9.5,
        lfs=7.7,
    )
    nrm = box(ax, x0, 69, w0, 6, "Final RMSNorm", kind="norm", fs=8.8)
    head = box(ax, x0, 80, w0, 12, "LM head", [f"Linear {m.d} → {m.vocab:,}", "no bias", "same matrix as the embedding" if m.tied else "own weights (not tied)"], kind="decoder", fs=9.5, lfs=7.7)
    nxt = box(ax, x0, 97, w0, 13, "Next token", ["argmax (greedy, as in", "the benchmark) → “2”", "only the last position"], kind="io", fs=9.5, lfs=7.7)
    arrow(ax, pos.b, dec.t, dashed=True)
    line(ax, emb.r, (158.5, emb.r[1]), (158.5, dec.cy))
    arrow(ax, (158.5, dec.cy), dec.l)
    shape(ax, 158.5, emb.r[1] + 5, f"[{L_EX},{m.d}]", fs=6.2)
    arrow(ax, dec.b, nrm.t)
    shape(ax, dec.cx + 1, 66.5, f"[{L_EX}, {m.d}]", ha="left", fs=6.8)
    arrow(ax, nrm.b, head.t)
    shape(ax, nrm.cx + 1, 77.5, f"[1, {m.d}]  last pos.", ha="left", fs=6.8)
    arrow(ax, head.b, nxt.t)
    shape(ax, head.cx + 1, 94.5, f"[1, {m.vocab:,}] logits", ha="left", fs=6.8)

    # decode loop: the answer, token by token
    panel(ax, 2, 65, 155, 23, "DECODE LOOP  — one token per pass, earlier keys / values come from the KV cache", "decoder", f=0.8)
    T(ax, 4, 71, "pass", 7.2, ha="left", color=SUB)
    T(ax, 4, 74.5, "token out", 7.2, ha="left", color=SUB)
    T(ax, 4, 82.8, "KV cache", 7.2, ha="left", color=SUB)
    xs, cw = 19, 14.6
    for i, p in enumerate(ANSWER_PIECES):
        cx = xs + i * (cw + 0.5) + cw / 2
        T(ax, cx, 71, "prefill" if i == 0 else f"decode {i}", 6.8, color=SUB)
        fc, ec = PAL[piece_kind(p)]
        ax.add_patch(FancyBboxPatch((cx - cw / 2 + 0.6, 72.6), cw - 1.2, 4.6, boxstyle="round,pad=0,rounding_size=0.5", fc=fc, ec=ec, lw=0.8, zorder=3))
        T(ax, cx, 74.9, p.replace(" ", "␣"), 7.2, mono=True)
        T(ax, cx, 79.4, f"{L_EX} in" if i == 0 else "1 in", 6.6, color=SUB)
        T(ax, cx, 82.8, f"{L_EX + i} → {L_EX + i + 1}" if i else f"0 → {L_EX}", 6.6, mono=True, color=SUB)
    T(ax, 4, 86.2, f"The pass that reads the {L_EX}-token prompt also produces the first token (time to first token). <|im_end|> ends the turn: {len(ANSWER_IDS)} passes in all.", 7.2, ha="left", color=SUB)
    line(ax, nxt.l, (157.5, nxt.cy), (157.5, 90), color=LOOP, lw=1.4)
    line(ax, (157.5, 90), (80, 90), color=LOOP, lw=1.4)
    arrow(ax, (80, 90), (80, 88), color=LOOP, lw=1.4)

    # size cheat-sheet and parameter split
    box(
        ax,
        4,
        92,
        58,
        22,
        "Sizes of this model (from config.json)",
        [
            (f"width d = {m.d}, {m.layers} layers",),
            (f"{m.heads} q-heads / {m.kv} KV-heads × {m.hd}",),
            (f"MLP {m.d}→{m.inter}→{m.d}",),
            (f"vocab rows {m.vocab:,}, tied head: {'yes' if m.tied else 'no'}",),
            (f"RoPE θ = {m.theta:,.0f}, RMSNorm ε = {m.eps:g}",),
        ],
        kind="note",
        fs=10,
        lfs=8.4,
        align="left",
    )
    box(ax, 66, 92, 91, 22, kind="note")
    T(ax, 68, 95, "Where the parameters are", 9.5, ha="left", bold=True)
    parts = [("token embedding = LM head" if m.tied else "token embedding", m.p_embed, "text"), (f"attention ×{m.layers}", m.layers * (m.p_attn), "attn"), (f"MLP ×{m.layers}", m.layers * m.p_mlp, "mlp"), ("RMSNorms", m.p_layers - m.layers * (m.p_attn + m.p_mlp), "norm")]
    param_bar(ax, 68, 97.5, 87, parts, m.p_total)
    T(ax, 68, 112, f"bf16 weights: {m.p_total * 2 / 1e9:.2f} GB  ·  KV cache: {m.kv_bytes // 1024} KiB per token  ·  ≈ {m.gflops_tok:.1f} GFLOPs per token (linear layers)", 7.4, ha="left", color=SUB)
    footer(ax, W, H)
    save(fig, f"01_overview_{m.size.lower()}", pdf)


# ----------------------------------------------------------------------------------------------- figure 2: decoder


def fig_decoder(m, pdf):
    W, H = 200, 122
    fig, ax = canvas(W, H)
    header(
        ax,
        f"{m.name} — decoder in detail",
        f"Qwen2 decoder: width {m.d}, {m.layers} layers, {m.heads} query heads / {m.kv} KV heads × {m.hd}, vocab rows {m.vocab:,}" + (", embedding tied with the LM head" if m.tied else "") + f".  Shapes for the example: L = {L_EX}.",
        W,
    )
    y, h = 15, 24
    inp = box(ax, 3, y, 22, h, "inputs_embeds", [f"[{L_EX}, {m.d}]", "one embedding row", "per prompt token"], kind="io", fs=10, lfs=8.4)
    st = layer_strip(ax, 30, y, 80, h, m.layers, f"{m.layers} decoder layers (same shape, own weights)", ["decoder"] * m.layers, {0, m.layers - 1}, ["every layer: causal self-attention over the full context", "(sliding window disabled: use_sliding_window = false)"])
    nrm = box(ax, 115, y, 18, h, "Final RMSNorm", [f"{m.d} weights", f"ε = {m.eps:g}"], kind="norm", fs=10, lfs=8.4)
    head = box(ax, 138, y, 31, h, "LM head", [f"Linear {m.d} → {m.vocab:,}", "no bias", "same matrix as the" if m.tied else "own matrix", "token embedding (tied)" if m.tied else ""], kind="decoder", fs=10, lfs=8.4)
    out = box(ax, 174, y, 23, h, "Next token", [f"logits [1, {m.vocab:,}]", "argmax → “2”", "→ append, repeat"], kind="io", fs=10, lfs=8.4)
    for a, b, s in ((inp, st, f"[{L_EX},{m.d}]"), (st, nrm, f"[{L_EX},{m.d}]"), (nrm, head, f"[1,{m.d}]"), (head, out, "")):
        arrow(ax, a.r, b.l)
        if s:
            shape(ax, (a.r[0] + b.l[0]) / 2, a.r[1] - 2.2, s, fs=6.2)
    r1 = box(ax, 30, 43, 39, 11, "RoPE cos / sin (computed once)", [f"position ids 0…{L_EX - 1} → cos, sin [{L_EX}, {m.hd}]", f"θ = {m.theta:,.0f}; 64 frequency pairs per head"], kind="norm", fs=8.5, lfs=7.2, dashed=True)
    r2 = box(ax, 71, 43, 39, 11, "KV cache (one per layer)", [f"K and V: {m.kv} heads × {m.hd} per token", f"{m.layers} layers × 2 × {m.kv} × {m.hd} × 2 B = {m.kv_bytes // 1024} KiB / token"], kind="norm", fs=8.5, lfs=7.2, dashed=True)
    arrow(ax, (r1.cx, r1.y), (r1.cx, st.y + h), dashed=True)
    arrow(ax, (r2.cx, r2.y), (r2.cx, st.y + h), dashed=True)
    box(
        ax,
        115,
        43,
        82,
        11,
        lines=[
            f"Prefill: all {L_EX} prompt tokens go through the layers in one pass and fill the KV cache.",
            "Decode: one new token per pass; its query attends to every cached key / value.",
            "Only the last position needs the final norm + LM head (the next-token prediction).",
        ],
        kind="note",
        lfs=7.5,
        align="left",
    )

    y0 = 58
    residual_panel(
        ax,
        2,
        y0,
        44,
        60,
        f"① Decoder layer (×{m.layers})",
        "decoder",
        f"x  [L, {m.d}]",
        "RMSNorm input_layernorm",
        ("Self-attention (GQA)", "causal · see ②"),
        "RMSNorm post_attention_ln",
        ("SwiGLU MLP", "see ③"),
        f"[L, {m.d}] → next layer",
        f"{fmt2(m.p_layer)} params / layer (×{m.layers} = {fmt(m.layers * m.p_layer)})",
    )
    x, y, w = 48, y0, 62
    panel(ax, x, y, w, 60, "② Grouped-query self-attention", "attn")
    cx = x + w / 2
    shape(ax, cx, y + 5.6, f"h = input_layernorm(x)   [L, {m.d}]", fs=7)
    cols = {"q": x + 11, "k": x + 31, "v": x + 51}
    line(ax, (cx, y + 6.8), (cx, y + 7.8))
    line(ax, (cols["q"], y + 7.8), (cols["v"], y + 7.8))
    out_dim = {"q": m.d, "k": m.kvd, "v": m.kvd}
    heads = {"q": m.heads, "k": m.kv, "v": m.kv}
    for name, c in cols.items():
        arrow(ax, (c, y + 7.8), (c, y + 9.3))
        box(ax, c - 9, y + 9.3, 18, 5.5, f"{name}_proj", [f"{m.d}→{out_dim[name]} + bias"], kind="attn", fs=8, lfs=6.8)
        shape(ax, c, y + 16.6, f"{name} [L,{heads[name]},{m.hd}]", fs=6.6)
    box(ax, x + 3, y + 18.6, 37, 4.4, "RoPE: rotate q, k by position", kind="norm", fs=7.6)
    box(ax, x + 22, y + 25.2, 38, 4.4, "KV cache: append k, v of the new tokens", kind="norm", fs=7.2, dashed=True)
    arrow(ax, (cols["q"], y + 17.6), (cols["q"], y + 18.6))
    arrow(ax, (cols["k"], y + 17.6), (cols["k"], y + 18.6))
    arrow(ax, (cols["q"], y + 23), (cols["q"], y + 32))
    arrow(ax, (cols["k"], y + 23), (cols["k"], y + 25.2))
    arrow(ax, (cols["v"], y + 17.6), (cols["v"], y + 25.2))
    arrow(ax, (cols["k"], y + 29.6), (cols["k"], y + 32))
    arrow(ax, (cols["v"], y + 29.6), (cols["v"], y + 32))
    T(ax, x + 33.5, y + 31, "all past k, v", 6.5, ha="left", color=SUB, italic=True)
    box(
        ax,
        x + 3,
        y + 32,
        w - 6,
        11,
        f"Attention: {m.heads} heads, each KV head shared by {m.rep}",
        [f"softmax( q · kᵀ / √{m.hd} + causal mask ) · v", "token i attends to tokens 0 … i only", f"example: “?” (pos. 30) sees positions 0 … 30"],
        kind="attn",
        fs=8.3,
        lfs=7.1,
    )
    arrow(ax, (cx, y + 43), (cx, y + 47))
    shape(ax, cx, y + 45, f"concat {m.heads} × {m.hd} → [L, {m.d}]", fs=6.8)
    box(ax, x + 9, y + 47, w - 18, 5, f"o_proj: Linear {m.d} → {m.d}  (no bias)", kind="attn", fs=8.2)
    arrow(ax, (cx, y + 52), (cx, y + 54.3))
    shape(ax, cx, y + 55.6, f"[L, {m.d}]")
    T(ax, cx, y + 58.4, f"{fmt2(m.p_attn)} params / layer", 7.2, color=SUB)

    swiglu_panel(ax, 112, y0, 44, 60, "③ MLP (SwiGLU)", "decoder", "L", m.d, m.inter, False, f"{fmt2(m.p_mlp)} params / layer")

    # GQA illustration
    x, y, w = 158, y0, 40
    panel(ax, x, y, w, 60, "④ Who shares what (GQA)", "attn")
    cw = (w - 4) / m.kv
    s, g = 2.2, 0.5
    per_row = max(1, int((cw - 1 + g) // (s + g)))
    group_cols = ["#e8833a", "#3b7dd8", "#3fa34d", "#9b59b6"]
    T(ax, x + w / 2, y + 6.3, f"{m.heads} query heads", 7.8, bold=True)
    for gi in range(m.kv):
        gx0 = x + 2 + gi * cw
        width = min(m.rep, per_row) * (s + g) - g
        sx0 = gx0 + (cw - width) / 2
        kvx = gx0 + cw / 2
        for qi in range(m.rep):
            r, c = divmod(qi, per_row)
            qx, qy = sx0 + c * (s + g), y + 8.5 + r * (s + g)
            ax.add_patch(Rectangle((qx, qy), s, s, fc=light(group_cols[gi], 0.35), ec=group_cols[gi], lw=0.7, zorder=3))
            line(ax, (qx + s / 2, qy + s), (kvx, y + 26), color=light(group_cols[gi], 0.2), lw=0.6, z=1)
        ax.add_patch(Rectangle((kvx - 2.2, y + 26), 4.4, 3.2, fc=group_cols[gi], ec="none", zorder=3))
        T(ax, kvx, y + 27.6, f"KV{gi}", 6.4, color="white", bold=True, z=5)
    T(ax, x + w / 2, y + 31.6, f"{m.kv} key/value heads", 7.8, bold=True)
    lines = [
        f"Each group of {m.rep} query heads reads",
        "the same key and value head.",
        "",
        "K / V cache per token and layer:",
        f"{m.kv} heads × {m.hd} × 2 (K, V) × 2 bytes",
        f"= {m.kv * m.hd * 4:,} B  → × {m.layers} layers",
        f"= {m.kv_bytes // 1024} KiB per token",
        f"({m.rep}× smaller than with {m.heads} KV heads)",
    ]
    for i, s_ in enumerate(lines):
        T(ax, x + w / 2, y + 36 + i * 2.6, s_, 7.3, color=SUB)
    footer(ax, W, H)
    save(fig, f"02_decoder_{m.size.lower()}", pdf)


# ----------------------------------------------------------------------------------------------- figure 3: prefill vs decode


def fig_prefill_decode(a, b, pdf):
    W, H = 200, 100
    fig, ax = canvas(W, H)
    header(ax, "Prefill and decode — why decoding is limited by memory bandwidth", f"One pass reads every weight once. Prefill uses each weight for all L prompt tokens; a decode step uses it for one token. Thor's memory: {BANDWIDTH / 1e9:.0f} GB/s.", W)

    # left: the two kinds of pass, for the example
    panel(ax, 2, 14, 96, 82, f"The example on {b.name} (16-bit weights)", "decoder", f=0.8)
    rows = [
        ("Prefill (1 pass)", f"{L_EX} tokens in", f"{L_EX} × {b.gflops_tok:.1f} = {L_EX * b.gflops_tok:.0f} GFLOPs", f"{b.step_bytes / 1e9:.2f} GB weights", f"≈ {L_EX * b.gflops_tok / (b.step_bytes / 1e9):.0f} FLOPs / byte"),
        ("Decode (1 pass each)", "1 token in", f"{b.gflops_tok:.1f} GFLOPs", f"{b.step_bytes / 1e9:.2f} GB weights + KV", "≈ 1 FLOP / byte"),
    ]
    heads = ["", "input", "compute", "memory read", "intensity"]
    xs = [4, 28, 44, 66, 84]
    for j, hd in enumerate(heads):
        T(ax, xs[j], 20.5, hd, 7.6, ha="left", bold=True)
    for i, r in enumerate(rows):
        yy = 25 + i * 6.5
        for j, c in enumerate(r):
            T(ax, xs[j], yy, c, 7.4, ha="left", bold=j == 0, color=INK if j == 0 else SUB)
    lines = [
        "Prefill does a lot of work per byte read, so it is limited by compute (matrix × matrix).",
        "A decode step reads the same weights to produce one token (matrix × vector): it is",
        "limited by how fast the weights stream from memory. A lower bound per token:",
        "",
        "      decode ms/token  ≥  bytes read per step / memory bandwidth",
        "",
        f"bytes per step = all layer weights + the whole LM head (V × d) + the KV cache so far;",
        "the embedding lookup reads only one row. Smaller weights (8-bit, 4-bit) mean fewer bytes",
        "per step, which is why quantization speeds up decoding more than prefill.",
    ]
    for i, s in enumerate(lines):
        T(ax, 4, 40 + i * 3.1, s, 7.6, ha="left", color=INK if "≥" in s else SUB, mono="≥" in s)
    tbl = [("", a.size, b.size), ("bytes per decode step (16-bit)", f"{a.step_bytes / 1e9:.2f} GB", f"{b.step_bytes / 1e9:.2f} GB"), (f"floor at {BANDWIDTH / 1e9:.0f} GB/s", f"{1e3 * a.step_bytes / BANDWIDTH:.1f} ms", f"{1e3 * b.step_bytes / BANDWIDTH:.1f} ms"), ("LM head share of those bytes", f"{100 * a.head_share:.0f} %", f"{100 * b.head_share:.0f} %"), ("KV cache at 4,096 tokens", f"{a.kv_bytes * 4096 / 2**20:.0f} MiB", f"{b.kv_bytes * 4096 / 2**20:.0f} MiB")]
    for i, (k, va, vb) in enumerate(tbl):
        yy = 71 + i * 4.2
        T(ax, 6, yy, k, 7.6, ha="left", bold=i == 0)
        T(ax, 66, yy, va, 7.6, mono=True, bold=i == 0)
        T(ax, 84, yy, vb, 7.6, mono=True, bold=i == 0)

    # right: KV cache growth during the example
    panel(ax, 101, 14, 97, 82, "The KV cache grows by one row per token, in every layer", "attn", f=0.8)
    n_tok = L_EX + len(ANSWER_IDS) - 1  # the final <|im_end|> is not written back
    x0, y0, cw, chh = 104, 22, 1.95, 3.2
    for t in range(n_tok):
        prompt = t < L_EX
        fc, ec = (PAL["text"] if prompt else PAL["decoder"])
        ax.add_patch(Rectangle((x0 + t * cw, y0), cw - 0.25, chh, fc=fc, ec=ec, lw=0.5, zorder=3))
    T(ax, x0, y0 + 5.3, f"positions 0 … {L_EX - 1}: written by prefill", 7.2, ha="left", color=PAL["text"][1])
    T(ax, x0 + n_tok * cw, y0 + 5.3, f"{L_EX} … {n_tok - 1}: one per decode step", 7.2, ha="right", color=PAL["decoder"][1])
    lines = [
        f"After the example the cache holds {n_tok} positions, in every layer:",
        f"  {b.size}: {n_tok} × {b.kv_bytes // 1024} KiB = {n_tok * b.kv_bytes / 2**20:.2f} MiB     {a.size}: {n_tok} × {a.kv_bytes // 1024} KiB = {n_tok * a.kv_bytes / 2**20:.2f} MiB",
        "",
        "Each decode step reads the whole cache once (its query attends to every",
        "cached key and value). With GQA the cache is small next to the weights:",
        f"  {b.size}, 1,000 tokens: {1000 * b.kv_bytes / 1e9:.3f} GB vs {b.step_bytes / 1e9:.2f} GB of weights",
        "",
        "No reuse across questions: the benchmark turns prefix caching off, so",
        "every question starts with an empty cache and a full prefill.",
        "",
        "Engines reserve the cache up front for the longest request:",
        f"  context 4,096 → {b.size} {b.kv_bytes * 4096 / 2**20:.0f} MiB, {a.size} {a.kv_bytes * 4096 / 2**20:.0f} MiB (16-bit; half with fp8 KV)",
    ]
    for i, s in enumerate(lines):
        T(ax, 104, 36 + i * 3.6, s, 7.6, ha="left", color=SUB, mono=s.startswith("  "))
    footer(ax, W, H)
    save(fig, "03_prefill_decode", pdf)


# ----------------------------------------------------------------------------------------------- figure 4: bytes per decode step


def fig_bytes(a, b, pdf):
    fig = plt.figure(figsize=(12, 6.6))
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.955, "Bytes read per decode step, by weight format", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.02, 0.915, f"Theoretical: layer weights + whole LM head + KV cache at {KV_TOKENS:,} tokens. Label = lower bound on decode ms/token at {BANDWIDTH / 1e9:.0f} GB/s. 4-bit includes one 16-bit scale per 128 weights.", fontsize=9, color=SUB)
    axes = fig.subplots(1, 2, sharex=True, gridspec_kw=dict(left=0.17, right=0.97, top=0.79, bottom=0.12, wspace=0.08))
    xmax = max(m.step_bytes + KV_TOKENS * m.kv_bytes for m in (a, b)) / 1e9 * 1.32
    for ax, m in zip(axes, (a, b)):
        ys = np.arange(len(FORMATS))[::-1]
        for y, (label, wb, hb) in zip(ys, FORMATS):
            segs = [m.p_layers * wb, m.vocab * m.d * hb, KV_TOKENS * m.kv_bytes]
            left = 0.0
            for (name, col), v in zip(SERIES, segs):
                ax.barh(y, v / 1e9, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.5, label=name if y == ys[0] else None)
                left += v / 1e9
            ax.text(left + xmax * 0.012, y + 0.12, f"{left:.2f} GB · ≥ {1e3 * left * 1e9 / BANDWIDTH:.1f} ms", va="center", fontsize=8.5, color=INK)
            ax.text(left + xmax * 0.012, y - 0.2, f"LM head {100 * segs[1] / sum(segs):.0f} %", va="center", fontsize=7.5, color=SUB)
        ax.set_yticks(ys)
        ax.set_yticklabels([f[0] for f in FORMATS] if m is a else [], fontsize=9)
        ax.set_title(m.name, fontsize=11, fontweight="bold", color=INK, loc="left")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("GB read per generated token", fontsize=9, color=SUB)
        ax.grid(axis="x", color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.tick_params(colors=SUB, labelsize=8.5, length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.17, 0.9), ncol=3, frameon=False, fontsize=9)
    fig.text(0.02, 0.025, "LM head % = its share of the bytes per step. Real kernels reach only part of the bandwidth, so measured decode times are higher; see jetson/results/thor/FINDINGS_qwen2_5.md.", fontsize=8, color=SUB)
    fig.text(0.98, 0.025, "generated by make_diagrams.py", fontsize=6.5, color="#8a8a8a", ha="right")
    save(fig, "04_bytes_per_token", pdf)


# ----------------------------------------------------------------------------------------------- figure 5: 1.5B vs 3B


def fig_compare(a, b, vl, pdf):
    W, H = 200, 88
    fig, ax = canvas(W, H)
    header(ax, "Qwen2.5-1.5B vs 3B — what changes", "Same recipe; 3B is wider and deeper. The language model inside Qwen2.5-VL-3B has exactly the 3B shape. Highlighted rows differ between 1.5B and 3B.", W)
    rows = [
        ("Parameters", fmt(a.p_total), fmt(b.p_total), fmt(vl.p_total), "VL-3B: language model only (+ 0.67 B vision)"),
        ("bf16 weights", f"{a.p_total * 2 / 1e9:.2f} GB", f"{b.p_total * 2 / 1e9:.2f} GB", f"{vl.p_total * 2 / 1e9:.2f} GB", "2 bytes per parameter"),
        ("hidden size d", str(a.d), str(b.d), str(vl.d), ""),
        ("decoder layers", str(a.layers), str(b.layers), str(vl.layers), ""),
        ("query heads / KV heads", f"{a.heads} / {a.kv}", f"{b.heads} / {b.kv}", f"{vl.heads} / {vl.kv}", f"query heads per KV head: {a.rep} vs {b.rep}"),
        ("head dim", str(a.hd), str(b.hd), str(vl.hd), ""),
        ("MLP intermediate size", f"{a.inter:,}", f"{b.inter:,}", f"{vl.inter:,}", f"{a.inter / a.d:.2f} × d  vs  {b.inter / b.d:.2f} × d"),
        ("vocab rows (embedding = LM head)", f"{a.vocab:,}", f"{b.vocab:,}", f"{vl.vocab:,}", "151,665 real tokens, the rest padding; tied in all three"),
        ("embedding share of the parameters", f"{100 * a.p_embed / a.p_total:.0f} %", f"{100 * b.p_embed / b.p_total:.0f} %", f"{100 * vl.p_embed / vl.p_total:.0f} %", "the vocabulary is a big part of a small model"),
        ("RoPE θ · max positions", f"{a.theta:,.0f} · {a.max_pos:,}", f"{b.theta:,.0f} · {b.max_pos:,}", f"{vl.theta:,.0f} · {vl.max_pos:,}", "VL: M-RoPE (equals 1D RoPE for text)"),
        ("KV cache per token (16-bit)", f"{a.kv_bytes // 1024} KiB", f"{b.kv_bytes // 1024} KiB", f"{vl.kv_bytes // 1024} KiB", "layers × 2 × KV heads × 128 × 2 B"),
        ("GFLOPs per token", f"{a.gflops_tok:.1f}", f"{b.gflops_tok:.1f}", f"{vl.gflops_tok:.1f}", "2 × (layer + LM head weights), w/o attention"),
        ("bytes per decode step (16-bit)", f"{a.step_bytes / 1e9:.2f} GB", f"{b.step_bytes / 1e9:.2f} GB", f"{vl.step_bytes / 1e9:.2f} GB", f"floor at {BANDWIDTH / 1e9:.0f} GB/s: {1e3 * a.step_bytes / BANDWIDTH:.1f} vs {1e3 * b.step_bytes / BANDWIDTH:.1f} ms/token"),
        ("LM head share of those bytes", f"{100 * a.head_share:.0f} %", f"{100 * b.head_share:.0f} %", f"{100 * vl.head_share:.0f} %", "matters when the layers are quantized and the head is not"),
    ]
    xs = [4, 66, 90, 114, 136]
    y = 15.5
    for j, s in enumerate(("1.5B", "3B", "VL-3B LLM")):
        T(ax, xs[j + 1] + 11, y, s, 10, bold=True)
    T(ax, xs[4], y, "note", 10, bold=True, ha="left")
    y += 4
    for r in rows:
        differs = r[1] != r[2]
        if differs:
            ax.add_patch(Rectangle((2, y - 1.75), 196, 3.5, fc="#fff4cc", ec="none", zorder=0.5))
        T(ax, xs[0] + 2, y, r[0], 8.4, ha="left")
        for j in range(3):
            T(ax, xs[j + 1] + 11, y, r[j + 1], 8.4, mono=True, bold=differs and j < 2, color=INK if j < 2 else SUB)
        T(ax, xs[4], y, r[4], 7.8, ha="left", color=SUB)
        y += 4.6
    footer(ax, W, H)
    save(fig, "05_1.5b_vs_3b", pdf)


def main():
    OUT.mkdir(exist_ok=True)
    a, b = load_model("1.5B"), load_model("3B")
    vl = load_model("3B", VL_DOCS / "configs" / "Qwen2.5-VL-3B-Instruct.json")
    vl.name = "Qwen2.5-VL-3B (language model)"
    for m in (a, b):
        check_weights(m)
    for m in (a, b, vl):
        print(f"{m.name}: KV {m.kv_bytes // 1024} KiB/token, {m.gflops_tok:.2f} GFLOPs/token, {m.step_bytes / 1e9:.3f} GB per decode step (16-bit), LM head {100 * m.head_share:.1f} % of it")
    with PdfPages(HERE / "qwen2_5_architecture.pdf") as pdf:
        fig_overview(a, pdf)
        fig_overview(b, pdf)
        fig_decoder(a, pdf)
        fig_decoder(b, pdf)
        fig_prefill_decode(a, b, pdf)
        fig_bytes(a, b, pdf)
        fig_compare(a, b, vl, pdf)
    print("wrote", HERE / "qwen2_5_architecture.pdf")


if __name__ == "__main__":
    main()
