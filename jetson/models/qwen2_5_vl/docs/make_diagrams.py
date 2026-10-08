#!/usr/bin/env python3
"""Teaching diagrams of the Qwen2.5-VL-3B / 7B-Instruct architecture -> figures/*.png, figures/*.svg and one PDF.

Every size in the figures is read from the checkpoints' config.json (copies in configs/). Parameter counts are
derived from the layer shapes and, when the weights are in the HF cache, checked against the safetensors headers.
The structure follows transformers 4.57 (models/qwen2_5_vl/modeling_qwen2_5_vl.py) and the Qwen2-VL image processor.
The worked example (a 448x336 image + "What is in the image?" -> 219 prompt tokens, M-RoPE positions) was produced
with the real processor and get_rope_index; see README.md.

Usage: python3 jetson/models/qwen2_5_vl/docs/make_diagrams.py      (needs only matplotlib + numpy)
"""

import glob
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
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
HF_CACHE = os.environ.get("HF_HUB_CACHE", "/opt/hf-cache/hub")

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path", "pdf.fonttype": 42})
MONO = "DejaVu Sans Mono"
INK, SUB, ARROW, SHAPE = "#1d1d1f", "#3f3f46", "#3a3a3a", "#0b5394"

# (fill, edge) per kind of block; the colours follow the original slides (vision green, projector purple,
# decoder pink, MLP blue) and add attention orange, text/tokens blue-grey and data/IO yellow.
PAL = {
    "io": ("#fff4cc", "#a07c00"),
    "text": ("#e2e9f8", "#34539c"),
    "vision": ("#d3efd3", "#2e7d32"),
    "merger": ("#eadbf2", "#7b3f8c"),
    "decoder": ("#fadbe2", "#b0304f"),
    "attn": ("#ffe6c7", "#bf6a00"),
    "mlp": ("#d3e9f6", "#2b6c99"),
    "norm": ("#eeeeee", "#666666"),
    "note": ("#ffffff", "#a0a0a0"),
    "full": ("#ffcf99", "#bf6a00"),
}
T_COL, H_COL, W_COL = "#e8833a", "#3b7dd8", "#3fa34d"  # M-RoPE time / height / width


# ----------------------------------------------------------------------------------------------- model numbers


def load_model(size):
    cfg = json.load(open(HERE / "configs" / f"Qwen2.5-VL-{size}-Instruct.json"))
    v = cfg["vision_config"]
    m = SimpleNamespace(
        size=size,
        name=f"Qwen2.5-VL-{size}-Instruct",
        d=cfg["hidden_size"],
        inter=cfg["intermediate_size"],
        layers=cfg["num_hidden_layers"],
        heads=cfg["num_attention_heads"],
        kv=cfg["num_key_value_heads"],
        vocab=cfg["vocab_size"],
        tied=cfg["tie_word_embeddings"],
        theta=cfg["rope_theta"],
        mrope=cfg["rope_scaling"]["mrope_section"],
        eps=cfg["rms_norm_eps"],
        vd=v["hidden_size"],
        vinter=v["intermediate_size"],
        depth=v["depth"],
        vheads=v["num_heads"],
        patch=v["patch_size"],
        tps=v["temporal_patch_size"],
        merge=v["spatial_merge_size"],
        window=v["window_size"],
        full=v["fullatt_block_indexes"],
        out=v["out_hidden_size"],
        chans=v["in_chans"],
    )
    m.hd = m.d // m.heads
    m.vhd = m.vd // m.vheads
    m.kvd = m.kv * m.hd
    m.rep = m.heads // m.kv
    m.pdim = m.chans * m.tps * m.patch**2  # 1176 values per patch
    m.mdim = m.vd * m.merge**2  # 5120 = 4 patches concatenated
    m.win_patches = m.window // m.patch  # 8 patches per window side
    # parameters, from the layer shapes (nn.Linear weight + bias where the layer has one)
    m.p_patch = m.vd * m.pdim
    m.p_vattn = (3 * m.vd * m.vd + 3 * m.vd) + (m.vd * m.vd + m.vd)
    m.p_vmlp = 2 * (m.vinter * m.vd + m.vinter) + (m.vd * m.vinter + m.vd)
    m.p_vblock = 2 * m.vd + m.p_vattn + m.p_vmlp
    m.p_vit = m.p_patch + m.depth * m.p_vblock
    m.p_merger = m.vd + (m.mdim * m.mdim + m.mdim) + (m.out * m.mdim + m.out)
    m.p_embed = m.vocab * m.d
    m.p_attn = (m.d * m.d + m.d) + 2 * (m.kvd * m.d + m.kvd) + m.d * m.d
    m.p_mlp = 3 * m.d * m.inter
    m.p_layer = 2 * m.d + m.p_attn + m.p_mlp
    m.p_layers = m.layers * m.p_layer + m.d  # + final norm
    m.p_head = 0 if m.tied else m.vocab * m.d
    m.p_lm = m.p_embed + m.p_layers + m.p_head
    m.p_total = m.p_vit + m.p_merger + m.p_lm
    m.kv_bytes = m.layers * 2 * m.kv * m.hd * 2  # bf16 K and V per token
    m.gflops_tok = 2 * (m.layers * (m.p_attn + m.p_mlp) + m.vocab * m.d) / 1e9
    return m


def check_weights(m):
    """Compare the derived parameter counts with the safetensors headers (skipped if the weights are absent)."""
    files = glob.glob(f"{HF_CACHE}/models--Qwen--{m.name}/snapshots/*/*.safetensors")
    if not files:
        print(f"{m.name}: weights not in {HF_CACHE}, parameter check skipped")
        return
    counts = {"visual.merger": 0, "visual": 0, "all": 0}
    for f in files:
        with open(f, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for name, t in header.items():
            if name == "__metadata__":
                continue
            n = int(np.prod(t["shape"]))
            counts["all"] += n
            counts["visual"] += n if name.startswith("visual.") else 0
            counts["visual.merger"] += n if name.startswith("visual.merger") else 0
    derived = {"visual.merger": m.p_merger, "visual": m.p_vit + m.p_merger, "all": m.p_total}
    assert counts == derived, f"{m.name}: weights {counts} != derived {derived}"
    print(f"{m.name}: parameter counts match the weights ({m.p_total:,})")


def fmt(n):
    return f"{n / 1e9:.2f} B" if n >= 1e9 else f"{n / 1e6:.1f} M"


def fmt2(n):
    return f"{n / 1e6:.2f} M"


# ----------------------------------------------------------------------------------------------- drawing kit
# Canvas units: 1 unit = 0.1 inch, y grows downwards.


def canvas(w, h):
    fig = plt.figure(figsize=(w / 10, h / 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def light(hexc, f=0.55):
    c = np.array(matplotlib.colors.to_rgb(hexc))
    return tuple(c + (1 - c) * f)


def T(ax, x, y, s, fs=9, ha="center", va="center", bold=False, color=INK, mono=False, italic=False, z=4, bg=None, rot=0):
    kw = dict(fontsize=fs, ha=ha, va=va, color=color, zorder=z, rotation=rot)
    kw["fontweight"] = "bold" if bold else "normal"
    if mono:
        kw["family"] = MONO
    if italic:
        kw["fontstyle"] = "italic"
    if bg:
        kw["bbox"] = dict(boxstyle="round,pad=0.12", fc=bg, ec="none")
    return ax.text(x, y, s, **kw)


class B:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.cx, self.cy = x + w / 2, y + h / 2
        self.l, self.r = (x, self.cy), (x + w, self.cy)
        self.t, self.b = (self.cx, y), (self.cx, y + h)


def box(ax, x, y, w, h, title=None, lines=(), kind="norm", fs=9.5, lfs=None, align="center", dashed=False, lw=1.2, fill=None, tcolor=INK):
    fc, ec = PAL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7", fc=fill or fc, ec=ec, lw=lw, ls=(0, (4, 2)) if dashed else "-", zorder=2))
    lfs = lfs or fs - 1.6
    th = fs * 0.19 if title else 0
    lh = lfs * 0.185
    yy = y + h / 2 - (th + len(lines) * lh) / 2
    tx, ha = (x + w / 2, "center") if align == "center" else (x + 1.2, "left")
    if title:
        T(ax, tx, yy + th / 2, title, fs, ha=ha, bold=True, color=tcolor)
        yy += th
    for ln in lines:
        mono = isinstance(ln, tuple)
        T(ax, tx, yy + lh / 2, ln[0] if mono else ln, lfs, ha=ha, mono=mono, color=SUB)
        yy += lh
    return B(x, y, w, h)


def arrow(ax, *pts, color=ARROW, lw=1.3, dashed=False, z=1.5):
    ls = (0, (4, 2.5)) if dashed else "-"
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=z, arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=14))


def line(ax, *pts, color=ARROW, lw=1.3, z=1.5):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, zorder=z, solid_capstyle="butt")


def shape(ax, x, y, s, ha="center", fs=7.5, color=SHAPE):
    return T(ax, x, y, s, fs, ha=ha, mono=True, color=color, bg="white", z=5)


def op(ax, x, y, sym, r=1.4):
    ax.add_patch(Circle((x, y), r, fc="white", ec=ARROW, lw=1.2, zorder=3))
    T(ax, x, y + 0.05, sym, 10.5, bold=True, z=4)


def panel(ax, x, y, w, h, title, kind, fs=10, f=0.6):
    fc, ec = PAL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=light(fc, f), ec=ec, lw=1.0, zorder=0.5))
    T(ax, x + 1.5, y + 2.4, title, fs, ha="left", bold=True, color=ec)


def header(ax, title, sub, W):
    T(ax, 4, 5, title, 20, ha="left", bold=True)
    T(ax, 4, 9.7, sub, 10.5, ha="left", color=SUB)


def footer(ax, W, H, extra=""):
    T(ax, W - 2, H - 1.4, f"{extra}Sources: config.json of the checkpoint · transformers 4.57 modeling_qwen2_5_vl.py · generated by make_diagrams.py", 6.5, ha="right", color="#8a8a8a")


def save(fig, name, pdf):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=170 if ext == "png" else None, facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ----------------------------------------------------------------------------------------------- shared panels


def residual_panel(ax, x, y, w, h, title, kind, in_label, n1, mid, n2, mlp, out_label, foot):
    """Pre-norm transformer block: x + Attn(Norm(x)), then + MLP(Norm(.))."""
    panel(ax, x, y, w, h, title, kind)
    cx, bw, rx = x + w * 0.58, w * 0.62, x + 4
    shape(ax, cx, y + 6.3, in_label)
    arrow(ax, (cx, y + 7.6), (cx, y + 10.5))
    box(ax, cx - bw / 2, y + 10.5, bw, 5, n1, kind="norm", fs=8.5)
    arrow(ax, (cx, y + 15.5), (cx, y + 18))
    box(ax, cx - bw / 2, y + 18, bw, 7, mid[0], mid[1:], kind="attn", fs=8.5, lfs=7.3)
    arrow(ax, (cx, y + 25), (cx, y + 27.6))
    op(ax, cx, y + 29, "+")
    arrow(ax, (cx, y + 30.4), (cx, y + 32.5))
    box(ax, cx - bw / 2, y + 32.5, bw, 5, n2, kind="norm", fs=8.5)
    arrow(ax, (cx, y + 37.5), (cx, y + 40))
    box(ax, cx - bw / 2, y + 40, bw, 7, mlp[0], mlp[1:], kind="mlp", fs=8.5, lfs=7.3)
    arrow(ax, (cx, y + 47), (cx, y + 49.6))
    op(ax, cx, y + 51, "+")
    arrow(ax, (cx, y + 52.4), (cx, y + 54.4))
    shape(ax, cx, y + 55.6, out_label)
    for top, bottom in ((y + 8.9, y + 29), (y + 31.3, y + 51)):
        line(ax, (cx, top), (rx, top), (rx, bottom))
        arrow(ax, (rx, bottom), (cx - 1.4, bottom))
    T(ax, rx + 0.8, y + 20, "residual", 7, ha="left", color="#777", rot=90, italic=True)
    T(ax, rx + 0.8, y + 42, "residual", 7, ha="left", color="#777", rot=90, italic=True)
    T(ax, x + w / 2, y + h - 1.6, foot, 7.2, color=SUB)


def swiglu_panel(ax, x, y, w, h, title, kind, n, din, dint, bias, foot):
    panel(ax, x, y, w, h, title, kind)
    cx, gx, ux, bw = x + w / 2, x + w * 0.27, x + w * 0.73, w * 0.42
    b = " + bias" if bias else ", no bias"
    shape(ax, cx, y + 6.3, f"h = norm2(x)   [{n}, {din}]" if kind == "vision" else f"h = post_attention_layernorm(x)  [{n}, {din}]", fs=7)
    line(ax, (cx, y + 7.6), (cx, y + 8.8))
    line(ax, (gx, y + 8.8), (ux, y + 8.8))
    arrow(ax, (gx, y + 8.8), (gx, y + 11))
    arrow(ax, (ux, y + 8.8), (ux, y + 11))
    box(ax, gx - bw / 2, y + 11, bw, 6, "gate_proj", [f"Linear {din}→{dint}{b}"], kind="mlp", fs=8.5, lfs=7.2)
    box(ax, ux - bw / 2, y + 11, bw, 6, "up_proj", [f"Linear {din}→{dint}{b}"], kind="mlp", fs=8.5, lfs=7.2)
    arrow(ax, (gx, y + 17), (gx, y + 19.5))
    box(ax, gx - bw * 0.35, y + 19.5, bw * 0.7, 4, "SiLU", kind="norm", fs=8.5)
    line(ax, (gx, y + 23.5), (gx, y + 28.5))
    arrow(ax, (gx, y + 28.5), (cx - 1.4, y + 28.5))
    line(ax, (ux, y + 17), (ux, y + 28.5))
    arrow(ax, (ux, y + 28.5), (cx + 1.4, y + 28.5))
    op(ax, cx, y + 28.5, "×")
    shape(ax, ux + 1, y + 23, f"[{n}, {dint}]", ha="left", fs=6.8)
    shape(ax, gx + 1, y + 26.2, f"[{n}, {dint}]", ha="left", fs=6.8)
    arrow(ax, (cx, y + 29.9), (cx, y + 33))
    box(ax, cx - w * 0.3, y + 33, w * 0.6, 6, "down_proj", [f"Linear {dint}→{din}{b}"], kind="mlp", fs=8.5, lfs=7.2)
    arrow(ax, (cx, y + 39), (cx, y + 42))
    shape(ax, cx, y + 43.3, f"[{n}, {din}]")
    T(ax, cx, y + 47.6, "MLP(h) = down( SiLU(gate(h)) ⊙ up(h) )", 8, mono=True)
    T(ax, cx, y + 51, "⊙ = element-wise product: the SiLU branch", 7.3, color=SUB)
    T(ax, cx, y + 53.6, f"decides how much of each of the {dint} features passes", 7.3, color=SUB)
    T(ax, cx, y + h - 1.6, foot, 7.2, color=SUB)


def layer_strip(ax, x, y, w, h, n, title, kinds, labels, sub):
    box(ax, x, y, w, h, kind="note", fill="white")
    T(ax, x + w / 2, y + 2.5, title, 10, bold=True)
    gap = 0.25
    cw = (w - 3 - gap * (n - 1)) / n
    for i in range(n):
        cx0 = x + 1.5 + i * (cw + gap)
        fc, ec = PAL[kinds[i]]
        ax.add_patch(Rectangle((cx0, y + 5), cw, 7, fc=fc, ec=ec, lw=0.8, zorder=3))
        if i in labels:
            T(ax, cx0 + cw / 2, y + 13.6, str(i), 6.8, color=SUB)
    for j, s in enumerate(sub):
        T(ax, x + w / 2, y + 16.2 + j * 1.9, s, 7.6, color=SUB)
    return B(x, y, w, h)


def param_bar(ax, x, y, w, parts, total):
    """Stacked bar of parameter shares with a legend underneath."""
    xx = x
    for name, n, kind in parts:
        fc, ec = PAL[kind]
        ww = w * n / total
        ax.add_patch(Rectangle((xx, y), ww, 5, fc=ec, ec="white", lw=0.8, alpha=0.85, zorder=3))
        xx += ww
    for i, (name, n, kind) in enumerate(parts):
        lx, ly = x + (i % 2) * w / 2, y + 8 + (i // 2) * 2.8
        ax.add_patch(Rectangle((lx, ly - 0.9), 1.8, 1.8, fc=PAL[kind][1], ec="none", alpha=0.85, zorder=3))
        T(ax, lx + 2.6, ly, f"{name}: {fmt(n)}  ({100 * n / total:.1f} %)", 8, ha="left")


# ----------------------------------------------------------------------------------------------- figure 1: overview


def fig_overview(m, pdf):
    W, H = 200, 122
    fig, ax = canvas(W, H)
    header(
        ax,
        f"{m.name} — end-to-end architecture",
        f"{fmt(m.p_total)} parameters = vision encoder {fmt(m.p_vit)} + patch merger {fmt(m.p_merger)} + language model {fmt(m.p_lm)}" + ("  (token embedding shared with the LM head)" if m.tied else ""),
        W,
    )
    box(
        ax,
        132,
        1.5,
        66,
        10,
        lines=[
            "Shapes  [rows, features]:   P = image patches of 14×14 px,   N = P / 4 visual tokens,",
            f"L = prompt length in tokens (text + N),   d = {m.d} (LLM width),   V = {m.vocab:,} (vocab rows)",
            "Example used in all figures: 448×336 px image + “What is in the image?”",
            "→  P = 32×24 = 768,   N = 16×12 = 192,   L = 219",
        ],
        kind="note",
        lfs=7.4,
        align="left",
    )

    # vision path
    panel(ax, 2, 14, 155, 31, "VISION PATH  — runs once per image", "vision")
    y, h = 19.5, 23
    img = box(ax, 4, y, 14, h, "Image", ["any size H × W", "RGB"], kind="io", fs=10)
    proc = box(
        ax,
        25,
        y,
        30,
        h,
        "Image processor (CPU)",
        ["resize: H, W → multiples of 28 px", "(aspect ratio kept, pixel budget", "min_pixels … max_pixels)", "normalize (CLIP mean / std)", "repeat the frame ×2 (temporal patch)", "cut into 14×14 px patches"],
        kind="io",
        fs=9.5,
        lfs=7.7,
    )
    pe = box(ax, 62, y, 21, h, "Patch embedding", [f"Conv3D {m.chans}→{m.vd}", "kernel = stride", "= (2, 14, 14)", f"≡ Linear {m.pdim}→{m.vd}", "no bias"], kind="vision", fs=9.5, lfs=7.7)
    vit = box(
        ax,
        90,
        y,
        30,
        h,
        f"Vision Transformer ×{m.depth}",
        [f"width {m.vd} · {m.vheads} heads × {m.vhd}", "2D RoPE on q, k (row, col)", f"window attention: {m.win_patches}×{m.win_patches} patches", f"full attention: blocks {', '.join(map(str, m.full))}", f"SwiGLU MLP {m.vd}→{m.vinter}→{m.vd}", "bidirectional (no causal mask)"],
        kind="vision",
        fs=9.5,
        lfs=7.7,
    )
    mg = box(ax, 127, y, 28, h, "Patch merger (projector)", ["2×2 neighbouring patches", "→ 1 token", f"RMSNorm, concat 4×{m.vd} = {m.mdim}", f"Linear {m.mdim}→{m.mdim} · GELU", f"Linear {m.mdim}→{m.out}"], kind="merger", fs=9.5, lfs=7.7)
    for a, b, s in ((img, proc, ""), (proc, pe, f"[P,{m.pdim}]"), (pe, vit, f"[P,{m.vd}]"), (vit, mg, f"[P,{m.vd}]")):
        arrow(ax, a.r, b.l)
        if s:
            shape(ax, (a.r[0] + b.l[0]) / 2, a.r[1] - 2.2, s, fs=6.6)

    # text path
    panel(ax, 2, 48, 155, 26, "TEXT PATH", "text")
    y, h = 53.5, 18
    pr = box(ax, 4, y, 14, h, "Prompt", ["question", "+ image"], kind="io", fs=10)
    tpl = box(
        ax,
        25,
        y,
        30,
        h,
        "Chat template",
        [("<|im_start|>system … <|im_end|>",), ("<|im_start|>user",), ("<|vision_start|><|image_pad|><|vision_end|>",), ("What is in the image?<|im_end|>",), ("<|im_start|>assistant",), "<|image_pad|> is repeated N times"],
        kind="text",
        fs=9.5,
        lfs=6.5,
    )
    tok = box(ax, 62, y, 21, h, "Tokenizer", ["byte-level BPE", "151,643 tokens", "+ 22 special", "= 151,665 ids"], kind="text", fs=9.5, lfs=7.7)
    emb = box(ax, 90, y, 30, h, "Token embedding", [f"lookup table V × d", f"= {m.vocab:,} × {m.d}", "rows ≥ 151,665: unused padding", "(text tokens → vectors)"], kind="text", fs=9.5, lfs=7.7)
    ins = box(ax, 127, y, 28, h, "Insert image features", ["the N rows at <|image_pad|>", "positions are replaced", "by the N merger outputs"], kind="merger", fs=9.5, lfs=7.7)
    for a, b, s in ((pr, tpl, ""), (tpl, tok, "text"), (tok, emb, "[L] ids"), (emb, ins, f"[L,{m.d}]")):
        arrow(ax, a.r, b.l)
        if s:
            shape(ax, (a.r[0] + b.l[0]) / 2, a.r[1] - 2.2, s, fs=6.6)
    arrow(ax, mg.b, ins.t)
    shape(ax, mg.cx + 1, 49.6, f"[N, {m.out}]", ha="left", fs=7)

    # language model column
    panel(ax, 160, 14, 38, 104, "LANGUAGE MODEL", "decoder")
    x0, w0 = 163, 32
    pos = box(ax, x0, 19.5, w0, 14, "3D position ids (M-RoPE)", ["text: t = h = w = position", "image: t shared, h = row, w = col", "from token ids + image grid", "→ cos / sin used in every layer"], kind="norm", fs=8.8, lfs=7.2, dashed=True)
    dec = box(
        ax,
        x0,
        39,
        w0,
        28,
        f"Decoder layer ×{m.layers}",
        ["pre-RMSNorm + residual", "self-attention (GQA, causal)", f"{m.heads} query heads, {m.kv} KV heads", f"head dim {m.hd}, M-RoPE on q, k", f"SwiGLU MLP {m.d}→{m.inter}→{m.d}", "KV cache for decoding"],
        kind="decoder",
        fs=9.5,
        lfs=7.7,
    )
    nrm = box(ax, x0, 72, w0, 6, "Final RMSNorm", kind="norm", fs=8.8)
    head = box(ax, x0, 83, w0, 12, "LM head", [f"Linear {m.d} → {m.vocab:,}", "no bias", "weights tied to token embedding" if m.tied else "own weights (not tied)"], kind="decoder", fs=9.5, lfs=7.7)
    nxt = box(ax, x0, 100, w0, 14, "Next token", ["softmax → greedy / sampling", "only the last position", "is needed"], kind="io", fs=9.5, lfs=7.7)
    arrow(ax, pos.b, dec.t, dashed=True)
    shape(ax, pos.cx + 1, 36.3, "[3, L]", ha="left", fs=6.8)
    arrow(ax, ins.r, (dec.x, ins.r[1]))
    shape(ax, 159, ins.r[1] - 2.2, f"[L,{m.d}]", fs=6.4)
    arrow(ax, dec.b, nrm.t)
    shape(ax, dec.cx + 1, 69.5, f"[L, {m.d}]", ha="left", fs=6.8)
    arrow(ax, nrm.b, head.t)
    shape(ax, nrm.cx + 1, 80.5, f"[1, {m.d}]  last pos.", ha="left", fs=6.8)
    arrow(ax, head.b, nxt.t)
    shape(ax, head.cx + 1, 97.5, f"[1, {m.vocab:,}] logits", ha="left", fs=6.8)
    # decode loop
    line(ax, nxt.l, (158.5, nxt.cy), (158.5, 79.5), (emb.cx, 79.5), color="#b0304f", lw=1.4)
    arrow(ax, (emb.cx, 79.5), emb.b, color="#b0304f", lw=1.4)
    T(ax, 132, 81.6, "decode loop: the new token is embedded and fed back,", 7.6, color="#b0304f", italic=True)
    T(ax, 132, 83.6, "one token per step; earlier keys / values come from the KV cache", 7.6, color="#b0304f", italic=True)

    # size cheat-sheet and parameter split
    box(
        ax,
        4,
        87,
        58,
        31,
        "Sizes of this model (from config.json)",
        [
            ("Vision encoder (same shape in 3B and 7B)",),
            (f"  width {m.vd}, {m.depth} blocks, {m.vheads} heads × {m.vhd}",),
            (f"  MLP {m.vd}→{m.vinter}→{m.vd}, window {m.window} px",),
            ("Language model",),
            (f"  width d = {m.d}, {m.layers} layers",),
            (f"  {m.heads} q-heads / {m.kv} KV-heads × {m.hd}",),
            (f"  MLP {m.d}→{m.inter}→{m.d}",),
            (f"  vocab rows {m.vocab:,}, tied head: {'yes' if m.tied else 'no'}",),
            (f"  RoPE θ = {m.theta:,.0f}, RMSNorm ε = {m.eps:g}",),
        ],
        kind="note",
        fs=10,
        lfs=8.4,
        align="left",
    )
    box(ax, 66, 87, 91, 31, kind="note")
    T(ax, 68, 90, "Where the parameters are", 9.5, ha="left", bold=True)
    parts = [("ViT (patch embed + 32 blocks)", m.p_vit, "vision"), ("patch merger", m.p_merger, "merger"), ("token embedding" + (" = LM head" if m.tied else ""), m.p_embed, "text"), (f"{m.layers} decoder layers + final norm", m.p_layers, "decoder")]
    if not m.tied:
        parts.append(("LM head", m.p_head, "attn"))
    param_bar(ax, 68, 93, 87, parts, m.p_total)
    T(ax, 68, 115, f"bf16 weights: {m.p_total * 2 / 1e9:.1f} GB  ·  KV cache: {m.kv_bytes // 1024} KiB per token  ·  decoder ≈ {m.gflops_tok:.1f} GFLOPs per generated token (linear layers)", 7.4, ha="left", color=SUB)
    footer(ax, W, H)
    save(fig, f"01_overview_{m.size.lower()}", pdf)


# ----------------------------------------------------------------------------------------------- figure 3: vision encoder


def fig_vision(m, pdf):
    W, H = 200, 122
    fig, ax = canvas(W, H)
    header(
        ax,
        f"{m.name} — vision encoder in detail",
        f"ViT {fmt(m.p_vit)} + patch merger {fmt(m.p_merger)}.  Identical layout in 3B and 7B except the merger's last Linear ({m.mdim}→{m.out}).  Example numbers: P = 768 patches, N = 192 tokens.",
        W,
    )
    y, h = 15, 24
    pv = box(ax, 3, y, 19, h, "pixel_values", ["[P, 1176]", "one row per patch:", "3 colours × 2 frames", "× 14 × 14 px", "(already normalized)"], kind="io", fs=10, lfs=8.4)
    pe = box(ax, 27, y, 23, h, "Patch embedding", [f"Conv3D {m.chans}→{m.vd}", "kernel = stride = (2,14,14)", f"= one Linear {m.pdim}→{m.vd}", "per patch, no bias", f"{fmt2(m.p_patch)} params"], kind="vision", fs=10, lfs=8.4)
    ro = box(ax, 55, y, 24, h, "Window reordering", ["permute the patches so the", f"{m.win_patches}×{m.win_patches} patches of each", f"{m.window}×{m.window} px window are", "contiguous in the sequence", "(no parameters)"], kind="norm", fs=10, lfs=8.4)
    kinds = ["full" if i in m.full else "vision" for i in range(m.depth)]
    st = layer_strip(ax, 84, y, 62, h, m.depth, f"{m.depth} vision blocks (same shape, own weights)", kinds, set(m.full) | {0}, ["green = window attention (each patch sees ≤ 64 patches)", f"orange = full attention (sees all P patches): {', '.join(map(str, m.full))}"])
    mg = box(ax, 151, y, 21, h, "Patch merger", ["groups 2×2 patches", f"[P, {m.vd}] →", f"[N, {m.out}]", "see panel ④"], kind="merger", fs=10, lfs=8.4)
    un = box(ax, 177, y, 20, h, "Undo reordering", ["tokens back in", "row-major order", "(12 rows × 16 cols", "in the example)"], kind="norm", fs=10, lfs=8.4)
    for a, b, s in ((pv, pe, "[P,1176]"), (pe, ro, f"[P,{m.vd}]"), (ro, st, f"[P,{m.vd}]"), (st, mg, f"[P,{m.vd}]"), (mg, un, f"[N,{m.out}]")):
        arrow(ax, a.r, b.l)
        shape(ax, (a.r[0] + b.l[0]) / 2, a.r[1] - 2.2, s, fs=6.2)
    arrow(ax, un.b, (un.cx, 45))
    shape(ax, un.cx, 46.6, f"→ LLM  [N, {m.out}]", fs=7)
    r1 = box(ax, 84, 43, 30, 11, "2D RoPE table (computed once)", ["from each patch's (row, col)", f"→ cos, sin  [P, {m.vhd}]  for all blocks"], kind="norm", fs=8.5, lfs=7.2, dashed=True)
    r2 = box(ax, 116, 43, 30, 11, "Attention spans (cu_seqlens)", ["window blocks: 12 spans of 64", "full blocks: 1 span of 768 / image"], kind="norm", fs=8.5, lfs=7.2, dashed=True)
    arrow(ax, r1.t, (r1.cx, st.y + h), dashed=True)
    arrow(ax, r2.t, (r2.cx, st.y + h), dashed=True)

    y0 = 58
    residual_panel(
        ax,
        2,
        y0,
        44,
        60,
        "① Vision block (×32)",
        "vision",
        f"x  [P, {m.vd}]",
        f"RMSNorm  norm1 ({m.vd})",
        ("Self-attention", "window or full · see ②"),
        f"RMSNorm  norm2 ({m.vd})",
        ("SwiGLU MLP", "see ③"),
        f"[P, {m.vd}] → next block",
        f"{fmt2(m.p_vblock)} params / block (×{m.depth} = {fmt(m.depth * m.p_vblock)})",
    )
    # attention panel
    x, y, w = 48, y0, 54
    panel(ax, x, y, w, 60, "② Vision self-attention", "attn")
    cx = x + w / 2
    shape(ax, cx, y + 6.3, f"h = norm1(x)   [P, {m.vd}]")
    arrow(ax, (cx, y + 7.6), (cx, y + 9))
    box(ax, x + 7, y + 9, w - 14, 5, f"qkv: Linear {m.vd} → 3×{m.vd}  (+ bias)", kind="attn", fs=8.3)
    cols = {"q": x + 11, "k": x + 25, "v": x + 43}
    for name, c in cols.items():
        arrow(ax, (c, y + 14), (c, y + 16.5))
        box(ax, c - 6, y + 16.5, 12, 4.5, name, [f"[P,{m.vheads},{m.vhd}]"], kind="attn", fs=8, lfs=6.6)
    rope = box(ax, x + 4, y + 23.5, 28, 4.6, "2D RoPE: rotate q, k by (row, col)", kind="norm", fs=7.5)
    for c in (cols["q"], cols["k"]):
        arrow(ax, (c, y + 21), (c, y + 23.5))
        arrow(ax, (c, y + 28.1), (c, y + 31))
    arrow(ax, (cols["v"], y + 21), (cols["v"], y + 31))
    box(ax, x + 3, y + 31, w - 6, 11, "Scaled dot-product attention, per head", [f"softmax( q · kᵀ / √{m.vhd} ) · v", "only inside each span: its window (≤ 64 patches)", "or, in blocks 7/15/23/31, the whole image", "no causal mask: a patch sees its whole span"], kind="attn", fs=8.3, lfs=7.1)
    arrow(ax, (cx, y + 42), (cx, y + 46.5))
    shape(ax, cx, y + 44.3, f"concat {m.vheads} heads → [P, {m.vd}]", fs=6.8)
    box(ax, x + 7, y + 46.5, w - 14, 5, f"proj: Linear {m.vd} → {m.vd}  (+ bias)", kind="attn", fs=8.3)
    arrow(ax, (cx, y + 51.5), (cx, y + 54.3))
    shape(ax, cx, y + 55.6, f"[P, {m.vd}]")
    T(ax, cx, y + 58.4, f"{fmt2(m.p_vattn)} params / block", 7.2, color=SUB)

    swiglu_panel(ax, 104, y0, 46, 60, "③ Vision MLP (SwiGLU)", "vision", "P", m.vd, m.vinter, True, f"{fmt2(m.p_vmlp)} params / block")

    # merger panel
    x, y, w = 152, y0, 46
    panel(ax, x, y, w, 60, "④ Patch merger", "merger")
    cx = x + w / 2
    shape(ax, cx, y + 6.3, f"[P, {m.vd}]  (window order)")
    arrow(ax, (cx, y + 7.6), (cx, y + 9))
    box(ax, x + 5, y + 9, w - 10, 4.6, f"RMSNorm ln_q  (each patch, {m.vd})", kind="norm", fs=7.8)
    arrow(ax, (cx, y + 13.6), (cx, y + 16))
    box(ax, x + 3, y + 16, w - 6, 11.5, kind="merger")
    T(ax, cx, y + 18.3, "Concatenate 4 consecutive rows", 8.3, bold=True)
    colors = ["#c9a3d9", "#b07fc6", "#9a62b5", "#7f4a9e"]
    for i in range(4):
        ax.add_patch(Rectangle((cx - 16 + i * 8, y + 20.6), 7.6, 2.4, fc=colors[i], ec="white", lw=0.6, zorder=3))
        T(ax, cx - 12.2 + i * 8, y + 21.8, str(m.vd), 6.5, color="white", bold=True, z=5)
    T(ax, cx, y + 25.4, f"the 2×2 patches of one token → [N, {m.mdim}]", 7.2, color=SUB)
    arrow(ax, (cx, y + 27.5), (cx, y + 29.5))
    box(ax, x + 5, y + 29.5, w - 10, 4.6, f"Linear {m.mdim} → {m.mdim}  (+ bias)", kind="merger", fs=8)
    arrow(ax, (cx, y + 34.1), (cx, y + 36))
    box(ax, x + 12, y + 36, w - 24, 4.2, "GELU (exact)", kind="norm", fs=8)
    arrow(ax, (cx, y + 40.2), (cx, y + 42))
    box(ax, x + 5, y + 42, w - 10, 4.6, f"Linear {m.mdim} → {m.out}  (+ bias)", kind="merger", fs=8)
    arrow(ax, (cx, y + 46.6), (cx, y + 49.3))
    shape(ax, cx, y + 50.6, f"[N, {m.out}]   {m.out} = LLM width d")
    T(ax, cx, y + 54, "then reordered back to row-major", 7.2, color=SUB)
    T(ax, cx, y + 58.4, f"{fmt2(m.p_merger)} params", 7.2, color=SUB)
    footer(ax, W, H)
    save(fig, f"03_vision_encoder_{m.size.lower()}", pdf)


# ----------------------------------------------------------------------------------------------- figure 5: language model


def fig_decoder(m, pdf):
    W, H = 200, 122
    fig, ax = canvas(W, H)
    header(
        ax,
        f"{m.name} — language model in detail",
        f"Qwen2.5 decoder: {fmt(m.p_lm)} parameters, width {m.d}, {m.layers} layers, {m.heads} query heads / {m.kv} KV heads × {m.hd}, vocab rows {m.vocab:,}" + (", embedding tied with the LM head" if m.tied else ", separate LM head"),
        W,
    )
    y, h = 15, 24
    inp = box(ax, 3, y, 22, h, "inputs_embeds", [f"[L, {m.d}]", "token embeddings, with", "the image features in", "place of <|image_pad|>"], kind="io", fs=10, lfs=8.4)
    st = layer_strip(ax, 30, y, 80, h, m.layers, f"{m.layers} decoder layers (same shape, own weights)", ["decoder"] * m.layers, {0, m.layers - 1}, ["every layer: causal self-attention over the full context", "(sliding window disabled: use_sliding_window = false)"])
    nrm = box(ax, 115, y, 18, h, "Final RMSNorm", [f"{m.d} weights", f"ε = {m.eps:g}"], kind="norm", fs=10, lfs=8.4)
    head = box(ax, 138, y, 31, h, "LM head", [f"Linear {m.d} → {m.vocab:,}", "no bias", "same matrix as the" if m.tied else f"own {fmt(m.p_head)} matrix", "token embedding (tied)" if m.tied else "(not tied in 7B)"], kind="decoder", fs=10, lfs=8.4)
    out = box(ax, 174, y, 23, h, "Next token", [f"logits [1, {m.vocab:,}]", "softmax → argmax", "or sampling", "→ append, repeat"], kind="io", fs=10, lfs=8.4)
    for a, b, s in ((inp, st, f"[L,{m.d}]"), (st, nrm, f"[L,{m.d}]"), (nrm, head, f"[1,{m.d}]"), (head, out, "")):
        arrow(ax, a.r, b.l)
        if s:
            shape(ax, (a.r[0] + b.l[0]) / 2, a.r[1] - 2.2, s, fs=6.2)
    r1 = box(ax, 30, 43, 39, 11, "M-RoPE cos / sin (computed once)", ["position ids [3, L] → cos, sin [L, 128]", f"θ = {m.theta:,.0f};  t / h / w sections {'/'.join(map(str, m.mrope))}"], kind="norm", fs=8.5, lfs=7.2, dashed=True)
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
            "Prefill: all L prompt tokens go through the layers in one pass and fill the KV cache.",
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
        f"RMSNorm input_layernorm",
        ("Self-attention (GQA)", "causal · see ②"),
        f"RMSNorm post_attention_ln",
        ("SwiGLU MLP", "see ③"),
        f"[L, {m.d}] → next layer",
        f"{fmt2(m.p_layer)} params / layer (×{m.layers} = {fmt(m.layers * m.p_layer)})",
    )
    # GQA attention
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
    rope = box(ax, x + 3, y + 18.6, 37, 4.4, "M-RoPE: rotate q, k by (t, h, w)", kind="norm", fs=7.6)
    cache = box(ax, x + 22, y + 25.2, 38, 4.4, "KV cache: append k, v of the new tokens", kind="norm", fs=7.2, dashed=True)
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
        [f"softmax( q · kᵀ / √{m.hd} + causal mask ) · v", "token i attends to tokens 0 … i only", "(image tokens too: no bidirectional image block)"],
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
        rows = (m.rep + per_row - 1) // per_row
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
    save(fig, f"05_language_model_{m.size.lower()}", pdf)


# ----------------------------------------------------------------------------------------------- figure 2: pixels → tokens


def synthetic_photo(h, w):
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    img = np.zeros((h, w, 3))
    sky = np.array([0.55, 0.75, 0.95]) * (1 - yy[..., None] * 0.5) + np.array([0.95, 0.85, 0.75]) * yy[..., None] * 0.5
    img[:] = sky
    sun = (xx - 0.78) ** 2 + ((yy - 0.25) * h / w) ** 2 < 0.07**2
    img[sun] = [1.0, 0.85, 0.3]
    hill1 = yy > 0.62 + 0.08 * np.sin(xx * 7)
    img[hill1] = [0.42, 0.68, 0.35]
    hill2 = yy > 0.78 + 0.05 * np.sin(xx * 11 + 1)
    img[hill2] = [0.30, 0.55, 0.28]
    house = (np.abs(xx - 0.3) < 0.08) & (yy > 0.5) & (yy < 0.72)
    img[house] = [0.85, 0.45, 0.35]
    roof = (yy > 0.38) & (yy <= 0.5) & (np.abs(xx - 0.3) < (yy - 0.38) * 0.9)
    img[roof] = [0.55, 0.25, 0.2]
    return img


def fig_tokens(m, pdf):
    W, H = 200, 108
    fig, ax = canvas(W, H)
    header(ax, "From pixels to visual tokens", "What the image processor and the ViT do with a picture (identical in 3B and 7B).  Example: a 450×340 photo.", W)

    # panel 1: resize and grids
    panel(ax, 2, 14, 66, 78, "1 · Resize, cut into patches, group, window", "io")
    T(ax, 35, 19.2, "450×340 → 448×336 px (nearest multiples of 28)", 8, color=SUB)
    x0, y0, pw = 7, 22, 56 / 32  # 32 × 24 patches
    ax.imshow(synthetic_photo(336, 448), extent=(x0, x0 + 32 * pw, y0 + 24 * pw, y0), zorder=1, interpolation="bilinear")
    for i in range(33):
        lw, c = (2.2, "#bf6a00") if i % 8 == 0 else (0.9, "#222222") if i % 2 == 0 else (0.3, "#ffffff")
        ax.plot([x0 + i * pw] * 2, [y0, y0 + 24 * pw], color=c, lw=lw, zorder=2 + (i % 8 == 0), alpha=0.9)
    for j in range(25):
        lw, c = (2.2, "#bf6a00") if j % 8 == 0 else (0.9, "#222222") if j % 2 == 0 else (0.3, "#ffffff")
        ax.plot([x0, x0 + 32 * pw], [y0 + j * pw] * 2, color=c, lw=lw, zorder=2 + (j % 8 == 0), alpha=0.9)
    ax.add_patch(Rectangle((x0, y0), 8 * pw, 8 * pw, fc="none", ec="#d62728", lw=2.4, zorder=4))
    T(ax, x0 + 8 * pw + 0.6, y0 + 8 * pw + 1.3, "window 0 → panel 2", 7, ha="left", color="#d62728", bold=True, bg="white")
    legend = [
        (0.4, "#999999", "thin: 14×14 px patch", "32 × 24 = 768 patches (P)"),
        (1.2, "#222222", "medium: 28×28 px = 2×2 patches", "= 1 LLM token: 16 × 12 = 192 (N)"),
        (2.4, "#bf6a00", "thick: 112×112 px = 8×8 patches", "= 1 attention window: 4 × 3 = 12"),
    ]
    for i, (lw, c, a, b) in enumerate(legend):
        yy = 67 + i * 7.6
        ax.plot([6, 12], [yy, yy], color=c, lw=lw + 0.6)
        T(ax, 14, yy - 1, a, 8.2, ha="left", bold=True)
        T(ax, 14, yy + 1.6, b, 7.8, ha="left", color=SUB)
    T(ax, 35, 89, "Windows at the right / bottom edge can be smaller.", 7.2, color=SUB, italic=True)

    # panel 2: order inside one window
    panel(ax, 70, 14, 62, 78, "2 · Sequence order inside one window", "vision")
    cs, gx0, gy0 = 4.3, 70 + (62 - 8 * 4.3) / 2, 19
    shades = ["#d3efd3", "#b7e3b7"]
    for r in range(8):
        for c in range(8):
            unit_r, unit_c, i, j = r // 2, c // 2, r % 2, c % 2
            idx = (unit_r * 4 + unit_c) * 4 + i * 2 + j
            ax.add_patch(Rectangle((gx0 + c * cs, gy0 + r * cs), cs, cs, fc=shades[(unit_r + unit_c) % 2], ec="white", lw=0.6, zorder=2))
            T(ax, gx0 + (c + 0.5) * cs, gy0 + (r + 0.5) * cs, str(idx), 7, z=4)
    for k in range(0, 9, 2):
        ax.plot([gx0 + k * cs] * 2, [gy0, gy0 + 8 * cs], color="#2e7d32", lw=1.6, zorder=3)
        ax.plot([gx0, gx0 + 8 * cs], [gy0 + k * cs] * 2, color="#2e7d32", lw=1.6, zorder=3)
    ax.add_patch(Rectangle((gx0, gy0), 8 * cs, 8 * cs, fc="none", ec="#d62728", lw=2.4, zorder=4))
    notes = [
        "• numbers = position in the ViT sequence (window 0)",
        "• the 4 patches of a 2×2 unit (green outline) are",
        "   consecutive — the merger later joins rows 4k … 4k+3",
        "• units go row by row inside a window; windows go",
        "   row by row over the image (window 1 starts at 64)",
        "• window attention: a patch attends to the ≤ 64",
        "   patches of its own window (28 of 32 blocks)",
        f"• full attention (blocks {', '.join(map(str, m.full))}): to all 768",
        "• after the merger, the 192 tokens are put back",
        "   in plain row-major order (12 rows × 16 columns)",
    ]
    for i, s in enumerate(notes):
        T(ax, 73, 57.5 + i * 3.25, s, 7.6, ha="left", color=SUB)

    # panel 3: one patch -> vector, 4 patches -> one token
    panel(ax, 134, 14, 64, 78, "3 · One patch → one vector → one token", "merger")
    for k, off in enumerate((1.6, 0)):
        ax.add_patch(Rectangle((139 + off, 20 + off), 10, 10, fc=["#9ec5e8", "#7fb0dd"][k], ec="#2b6c99", lw=1, zorder=3 + k))
    T(ax, 144, 25, "14×14", 7.5, z=6, bold=True)
    T(ax, 153.5, 21.5, "14 × 14 px, RGB, the image", 7.6, ha="left")
    T(ax, 153.5, 24.1, "copied into 2 frames", 7.6, ha="left")
    T(ax, 153.5, 26.7, "(videos: 2 real frames)", 7.6, ha="left", color=SUB)
    T(ax, 153.5, 29.6, "3 · 2 · 14 · 14 = 1176 numbers", 7.8, ha="left", bold=True)
    arrow(ax, (166, 32.5), (166, 35))
    box(ax, 140, 35, 52, 4.6, f"Patch embedding: Linear 1176 → {m.vd}", kind="vision", fs=8)
    arrow(ax, (166, 39.6), (166, 42))
    ax.add_patch(Rectangle((150, 42), 32, 2.4, fc="#2e7d32", ec="none", alpha=0.8, zorder=3))
    T(ax, 166, 43.2, f"patch vector, {m.vd} numbers", 6.8, color="white", bold=True, z=5)
    T(ax, 166, 47.6, f"… {m.depth} vision blocks (shape stays [P, {m.vd}]) …", 7.6, color=SUB, italic=True)
    colors = ["#c9a3d9", "#b07fc6", "#9a62b5", "#7f4a9e"]
    for i in range(4):
        ax.add_patch(Rectangle((139 + i * 13.2, 50.5), 12.6, 2.4, fc=colors[i], ec="none", zorder=3))
        T(ax, 145.3 + i * 13.2, 51.7, f"patch {i + 1}", 6.6, color="white", bold=True, z=5)
    T(ax, 166, 55.4, f"concat → {m.mdim} numbers (one 2×2 unit)", 7.6, bold=True)
    arrow(ax, (166, 56.7), (166, 58.5))
    box(ax, 140, 58.5, 52, 4.6, f"Merger MLP: {m.mdim} → {m.mdim} → d", kind="merger", fs=8)
    arrow(ax, (166, 63.1), (166, 65.5))
    ax.add_patch(Rectangle((146, 65.5), 40, 2.6, fc="#b0304f", ec="none", alpha=0.85, zorder=3))
    T(ax, 166, 66.8, "1 visual token, d numbers", 6.8, color="white", bold=True, z=5)
    T(ax, 166, 71, "d = 2048 (3B)  ·  d = 3584 (7B)", 8, bold=True)
    for i, s in enumerate(["It replaces one <|image_pad|> embedding,", "so the LLM reads it like any other token.", "One token covers 28 × 28 px of the image."]):
        T(ax, 166, 76 + i * 3, s, 7.6, color=SUB)

    box(
        ax,
        2,
        94,
        196,
        11,
        lines=[
            "Visual tokens  N = (H / 28) × (W / 28)   e.g.  (336 / 28) × (448 / 28) = 12 × 16 = 192          Patches  P = 4 N = 768",
            "Default pixel budget: min_pixels 3,136 … max_pixels 12,845,056  →  4 … 16,384 tokens per image.   This harness: 256 … 2,048 tokens (jetson/models/qwen2_5_vl/model.sh).",
        ],
        kind="note",
        lfs=8.6,
    )
    footer(ax, W, H, "Image processor: Qwen2VLImageProcessor (smart_resize, patch flattening).  ")
    save(fig, "02_pixels_to_tokens", pdf)


# ----------------------------------------------------------------------------------------------- figure 4: positions


def fig_positions(m, pdf):
    W, H = 200, 112
    fig, ax = canvas(W, H)
    header(ax, "Where is each token?  Rotary position encodings", f"LLM: 3D multimodal RoPE (M-RoPE, head dim {m.hd}).  ViT: 2D RoPE (head dim {m.vhd}).  Same in 3B and 7B.  No learned position embeddings anywhere.", W)

    panel(ax, 2, 14, 196, 54, "A · Position ids of the example prompt  (L = 219 tokens; image = 12 × 16 tokens) — values from Qwen2_5_VLModel.get_rope_index", "text", fs=9.5)
    rows = [("tokens", 24), ("sequence index", 34.5), ("t  (time)", 38.5), ("h  (height)", 42.5), ("w  (width)", 46.5)]
    for name, yy in rows:
        T(ax, 4, yy, name, 8, ha="left", bold=True, color=[INK, INK, T_COL, H_COL, W_COL][rows.index((name, yy))])
    segs = [
        (22, 38, "text", [("<|im_start|>system",), ("You are a helpful",), ("assistant.<|im_end|>",), ("<|im_start|>user",)], "0 … 13", "0 … 13", "0 … 13", "0 … 13"),
        (62, 15, "text", [("<|vision_",), ("start|>",)], "14", "14", "14", "14"),
        (79, 52, "merger", [], "15 … 206", "15  (all 192)", "15 + row = 15 … 26", "15 + col = 15 … 30"),
        (133, 15, "text", [("<|vision_",), ("end|>",)], "207", "31", "31", "31"),
        (150, 30, "text", [("What is in the",), ("image?<|im_end|>",), ("<|im_start|>",), ("assistant",)], "208 … 218", "32 … 42", "32 … 42", "32 … 42"),
        (182, 15, "io", ["1st new", "token"], "219", "43", "43", "43"),
    ]
    for x, w, kind, toks, idx, t, hh, ww in segs:
        box(ax, x, 18.5, w, 11, lines=toks, kind=kind, lfs=7.4, dashed=kind == "io")
        for (name, yy), val, col in zip(rows[1:], (idx, t, hh, ww), (INK, T_COL, H_COL, W_COL)):
            T(ax, x + w / 2, yy, val, 8, color=col, bold=name != "sequence index")
    # image token grid inside segment 3
    gx, gy, c = 81, 19.3, 0.6
    for r in range(12):
        for cc in range(16):
            ax.add_patch(Rectangle((gx + cc * c, gy + r * c), c * 0.9, c * 0.9, fc=matplotlib.cm.Blues(0.3 + r / 20), ec="none", zorder=3))
    T(ax, gx + 16 * c + 1.5, 21.5, "192 × <|image_pad|>", 7, ha="left", mono=True)
    T(ax, gx + 16 * c + 1.5, 24, "12 rows (h) × 16 columns (w)", 7, ha="left", color=SUB)
    T(ax, gx + 16 * c + 1.5, 26.5, "row-major, as the merger outputs them", 7, ha="left", color=SUB)
    notes = [
        "• Text tokens get t = h = w = their running position, so for pure text M-RoPE is exactly the 1D RoPE of Qwen2.5.",
        "• All image tokens share t (one frame) and take h, w from their row / column in the 12 × 16 grid, offset by the start position 15.",
        "• Text after the image resumes at max + 1 = 31: the 192 image tokens advance the position by only 16 (max(rows, cols)), not by 192.",
        "• Decoding: every new token gets t = h = w = sequence index − 176 (the stored rope delta = 43 − 219), e.g. index 219 → 43, 220 → 44.",
        "• Videos: t = frame-group index × seconds per group × 2 (tokens_per_second), so t follows real time.",
    ]
    for i, s in enumerate(notes):
        T(ax, 4, 51.5 + i * 3.1, s, 7.8, ha="left", color=SUB)

    # panel B: LLM head-dim allocation
    panel(ax, 2, 70, 97, 39, f"B · LLM: which of the {m.hd} dims of a head see t, h or w", "decoder", fs=9.5)
    secs = m.mrope + m.mrope
    names = ["t", "h", "w"] * 2
    cols = [T_COL, H_COL, W_COL] * 2
    bx, bw = 6, 89
    unit = bw / m.hd
    xx, start = bx, 0
    for n, s, col in zip(names, secs, cols):
        ax.add_patch(Rectangle((xx, 78), s * unit, 6, fc=col, ec="white", lw=1, zorder=3))
        T(ax, xx + s * unit / 2, 81, f"{n}  {s}", 8, color="white", bold=True, z=5)
        T(ax, xx, 86, str(start), 6.8, color=SUB)
        xx += s * unit
        start += s
    T(ax, xx, 86, str(start), 6.8, color=SUB)
    T(ax, bx + 32 * unit * 2 / 2 + 0.0, 76, "first half (dims 0 … 63)", 7.2, color=SUB)
    T(ax, bx + 96 * unit, 76, "second half (dims 64 … 127)", 7.2, color=SUB)
    for i, s in enumerate(
        [
            "RoPE rotates dim i together with dim i + 64, at frequency θ^(−2i/128), θ = 10⁶.",
            f"Of the 64 rotation pairs, the {m.mrope[0]} fastest use t, the next {m.mrope[1]} use h,",
            f"the slowest {m.mrope[2]} use w   (rope_scaling.mrope_section = {m.mrope}).",
            "Applied to q and k in every layer; v is not rotated.",
        ]
    ):
        T(ax, 5, 91 + i * 3.4, s, 7.8, ha="left", color=SUB)

    # panel C: ViT head-dim allocation
    panel(ax, 101, 70, 97, 39, f"C · ViT: 2D RoPE inside the {m.vhd} dims of a vision head", "vision", fs=9.5)
    q = m.vhd // 4
    bx, unit = 105, 89 / m.vhd
    xx = bx
    for k, (n, col) in enumerate([("row", H_COL), ("col", W_COL), ("row", H_COL), ("col", W_COL)]):
        ax.add_patch(Rectangle((xx, 78), q * unit, 6, fc=col, ec="white", lw=1, zorder=3))
        T(ax, xx + q * unit / 2, 81, f"{n}  {q}", 8, color="white", bold=True, z=5)
        T(ax, xx, 86, str(k * q), 6.8, color=SUB)
        xx += q * unit
    T(ax, xx, 86, str(m.vhd), 6.8, color=SUB)
    for i, s in enumerate(
        [
            f"Dim j rotates with dim j + {m.vhd // 2}. Row and column each use the same {q}",
            f"frequencies 10000^(−2k/{m.vhd // 2}), k = 0 … {q - 1}  (θ = 10⁴).",
            "Positions = the patch's (row, col) in the 14-px grid of its own image.",
            f"cos / sin are computed once and reused by all {m.depth} blocks.",
        ]
    ):
        T(ax, 104, 91 + i * 3.4, s, 7.8, ha="left", color=SUB)
    footer(ax, W, H)
    save(fig, "04_positions_mrope", pdf)


# ----------------------------------------------------------------------------------------------- figure 6: 3B vs 7B


def fig_compare(a, b, pdf):
    W, H = 200, 93
    fig, ax = canvas(W, H)
    header(ax, "Qwen2.5-VL-3B vs 7B — what changes", "Same recipe; the vision encoder keeps its shape, the language model is wider (7B) or deeper (3B). Highlighted rows differ.", W)
    rows = [
        ("Total parameters", fmt(a.p_total), fmt(b.p_total), ""),
        ("bf16 weights", f"{a.p_total * 2 / 1e9:.1f} GB", f"{b.p_total * 2 / 1e9:.1f} GB", "2 bytes per parameter"),
        ("VISION ENCODER",),
        ("width · blocks · heads × dim", f"{a.vd} · {a.depth} · {a.vheads}×{a.vhd}", f"{b.vd} · {b.depth} · {b.vheads}×{b.vhd}", "same shape, separately trained weights"),
        ("MLP (SwiGLU, with bias)", f"{a.vd}→{a.vinter}→{a.vd}", f"{b.vd}→{b.vinter}→{b.vd}", ""),
        ("full-attention blocks", ", ".join(map(str, a.full)), ", ".join(map(str, b.full)), f"the others use {a.window}×{a.window} px windows"),
        ("ViT parameters", fmt(a.p_vit), fmt(b.p_vit), ""),
        ("patch merger", f"{a.mdim}→{a.mdim}→{a.out}", f"{b.mdim}→{b.mdim}→{b.out}", "output = LLM width"),
        ("merger parameters", fmt(a.p_merger), fmt(b.p_merger), ""),
        ("LANGUAGE MODEL",),
        ("hidden size d", str(a.d), str(b.d), ""),
        ("decoder layers", str(a.layers), str(b.layers), "3B is deeper but much narrower"),
        ("query heads / KV heads", f"{a.heads} / {a.kv}", f"{b.heads} / {b.kv}", f"query heads per KV head: {a.rep} vs {b.rep}"),
        ("head dim", str(a.hd), str(b.hd), ""),
        ("MLP intermediate size", f"{a.inter:,}", f"{b.inter:,}", f"{a.inter / a.d:.2f} × d  vs  {b.inter / b.d:.2f} × d"),
        ("vocab rows (embedding / LM head)", f"{a.vocab:,}", f"{b.vocab:,}", "151,665 real tokens, the rest padding"),
        ("LM head", "tied to embedding" if a.tied else "separate", "tied to embedding" if b.tied else f"separate (+{fmt(b.p_head)})", ""),
        ("RoPE θ · M-RoPE sections", f"{a.theta:,.0f} · {a.mrope}", f"{b.theta:,.0f} · {b.mrope}", ""),
        ("language-model parameters", fmt(a.p_lm), fmt(b.p_lm), "incl. embedding (and LM head)"),
        ("KV cache per token (bf16)", f"{a.kv_bytes // 1024} KiB", f"{b.kv_bytes // 1024} KiB", "layers × 2 × KV heads × 128 × 2 B"),
        ("decoder GFLOPs per token", f"{a.gflops_tok:.1f}", f"{b.gflops_tok:.1f}", "2 × (layer + LM head weights), w/o attention"),
    ]
    xs = [4, 72, 104, 136]
    y = 15
    T(ax, xs[1] + 14, y, "3B", 10, bold=True)
    T(ax, xs[2] + 14, y, "7B", 10, bold=True)
    T(ax, xs[3], y, "note", 10, bold=True, ha="left")
    y += 3.6
    for r in rows:
        if len(r) == 1:
            T(ax, xs[0], y + 0.4, r[0], 8.5, ha="left", bold=True, color=PAL["decoder"][1] if "VISION" not in r[0] else PAL["vision"][1])
            y += 3.6
            continue
        differs = r[1] != r[2]
        if differs:
            ax.add_patch(Rectangle((2, y - 1.65), 196, 3.3, fc="#fff4cc", ec="none", zorder=0.5))
        T(ax, xs[0] + 2, y, r[0], 8.4, ha="left")
        T(ax, xs[1] + 14, y, r[1], 8.4, mono=True, bold=differs)
        T(ax, xs[2] + 14, y, r[2], 8.4, mono=True, bold=differs)
        T(ax, xs[3], y, r[3], 7.8, ha="left", color=SUB)
        y += 3.4
    footer(ax, W, H)
    save(fig, "06_3b_vs_7b", pdf)


def main():
    OUT.mkdir(exist_ok=True)
    m3, m7 = load_model("3B"), load_model("7B")
    for m in (m3, m7):
        check_weights(m)
    with PdfPages(HERE / "qwen2_5_vl_architecture.pdf") as pdf:
        fig_overview(m3, pdf)
        fig_overview(m7, pdf)
        fig_tokens(m3, pdf)
        fig_vision(m3, pdf)
        fig_vision(m7, pdf)
        fig_positions(m3, pdf)
        fig_decoder(m3, pdf)
        fig_decoder(m7, pdf)
        fig_compare(m3, m7, pdf)
    print("wrote", HERE / "qwen2_5_vl_architecture.pdf")


if __name__ == "__main__":
    main()
