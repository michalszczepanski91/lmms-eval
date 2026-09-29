"""Write a copy of a Qwen2.5-VL checkpoint whose vision-encoder residual stream is scaled down by a constant.

Edge-LLM exports the Qwen2.5-VL vision encoder in fp16 only (``tensorrt-edgellm-export --dtype`` is ignored for it),
and its residual stream exceeds the fp16 range (65504) on some photos: fp32 reaches ~76k on TextVQA val images of
1024x768. The engine then produces NaN embeddings, the LLM NaN logits, and the answer is empty (token 151935 until
max length). Every block reads the residual stream through an RMSNorm, which is invariant to the scale of its input,
so dividing the patch embedding and every block's output projections (attn.proj, mlp.down_proj) by ``scale`` divides
the residual stream by ``scale`` and leaves every norm output, and so the encoder output, unchanged (up to the norm
epsilon). A power of two is exact in fp16/bf16.

Usage: python scale_vision.py <checkpoint dir> <output dir> [scale=8]
Only the shards holding vision tensors are rewritten; the other files are symlinked to their resolved paths.
"""

import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

SCALED = re.compile(r"^visual\.(patch_embed\.proj\.weight|blocks\.\d+\.(attn\.proj|mlp\.down_proj)\.(weight|bias))$")


def main(src: str, out: str, scale: float) -> None:
    os.makedirs(out, exist_ok=True)
    weight_map = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    names = [name for name in weight_map if SCALED.match(name)]
    shards = sorted({weight_map[name] for name in names})
    assert names, f"no vision tensors to scale in {src}"
    for entry in os.listdir(src):
        target = os.path.join(out, entry)
        if entry in shards or os.path.lexists(target):
            continue
        os.symlink(os.path.realpath(os.path.join(src, entry)), target)
    for shard in shards:
        tensors = load_file(os.path.join(src, shard))
        for name in tensors:
            if SCALED.match(name):
                tensors[name] = (tensors[name].to(torch.float32) / scale).to(tensors[name].dtype)
        save_file(tensors, os.path.join(out, shard), metadata={"format": "pt"})
    print(f"scaled {len(names)} vision tensors by 1/{scale:g} in {len(shards)} shard(s) -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 8.0)
