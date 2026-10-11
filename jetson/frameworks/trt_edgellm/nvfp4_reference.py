#!/usr/bin/env python3
"""Is a bad NVFP4 Edge-LLM engine the quantizer's fault or the engine's? Run the quantized checkpoint in PyTorch.

Reads the checkpoint tensorrt-edgellm-quantize writes (ModelOpt NVFP4: uint8 weights holding two E2M1 values each,
low nibble first; an E4M3 scale per 16 weights; an FP32 per-tensor scale), dequantizes it into the base model and
answers a few GSM8K questions greedily, in three ways:
  bf16       the base checkpoint (reference)
  weights    NVFP4 weights dequantized, activations in bf16: tests the quantized weights
  w4a4       also fake-quantizes each linear layer's input to NVFP4 (blocks of 16, E4M3 block scales, the calibrated
             input_scale as global scale): the arithmetic the engine is meant to do
If "weights" / "w4a4" answer like bf16 while the engine produces garbage, the engine (kernels / plugins) is at fault.

Usage (eval container, GPU): python jetson/frameworks/trt_edgellm/nvfp4_reference.py <base model dir> <quantized dir> [-n 8]
Get the quantized dir with: KEEP_QUANTIZED=1 MODEL=qwen2_5 jetson/frameworks/trt_edgellm/export.sh 1.5b nvfp4 <workspace>
"""

import argparse
import glob
import json
import os

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequantize(packed, block_scale, global_scale):
    """[out, in/2] uint8 -> [out, in] float32: value = E2M1 * E4M3 block scale (per 16) * FP32 global scale."""
    low, high = (packed & 0x0F).long(), (packed >> 4).long()
    values = torch.stack([E2M1[low], E2M1[high]], dim=-1).reshape(packed.shape[0], -1)
    scales = block_scale.float().repeat_interleave(16, dim=1)
    return values * scales * global_scale.float()


def round_e2m1(x):
    grid = E2M1[:8].to(x.device)
    idx = (x.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    return grid[idx] * x.sign()


def fake_quant_nvfp4(x, global_scale):
    """NVFP4 activation quantization along the last dim: E4M3 block scale per 16 values relative to global_scale."""
    shape = x.shape
    blocks = x.float().reshape(-1, shape[-1] // 16, 16)
    amax = blocks.abs().amax(-1, keepdim=True)
    scale = (amax / 6.0 / global_scale).to(torch.float8_e4m3fn).float() * global_scale
    q = round_e2m1(blocks / scale.clamp(min=1e-12)) * scale
    return q.reshape(shape).to(x.dtype)


def load(base, quantized, mode):
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).cuda().eval()
    if mode == "bf16":
        return model
    tensors = {}
    for path in glob.glob(os.path.join(quantized, "*.safetensors")):
        tensors.update(load_file(path))
    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or f"{name}.weight_scale" not in tensors:
            continue
        w = dequantize(tensors[f"{name}.weight"], tensors[f"{name}.weight_scale"], tensors[f"{name}.weight_scale_2"])
        module.weight.data.copy_(w.to(module.weight.dtype))
        if f"{name}.bias" in tensors:
            module.bias.data.copy_(tensors[f"{name}.bias"].to(module.bias.dtype))
        if mode == "w4a4":
            g = tensors[f"{name}.input_scale"].float().cuda()
            module.register_forward_pre_hook(lambda m, args, g=g: (fake_quant_nvfp4(args[0], g),))
        replaced += 1
    print(f"{mode}: {replaced} linear layers from {quantized}")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base")
    p.add_argument("quantized")
    p.add_argument("-n", type=int, default=8)
    a = p.parse_args()
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(a.base)
    docs = list(load_dataset("gsm8k", "main", split="test").select(range(a.n)))
    for mode in ("bf16", "weights", "w4a4"):
        model = load(a.base, a.quantized, mode)
        correct = 0
        for doc in docs:
            ids = tok.apply_chat_template([{"role": "user", "content": f"Question: {doc['question']}\nAnswer:"}], add_generation_prompt=True, return_tensors="pt").cuda()
            out = model.generate(ids, max_new_tokens=256, do_sample=False, repetition_penalty=1.0)
            text = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
            ref = doc["answer"].split("####")[-1].strip().replace(",", "")
            nums = [t.strip(".$").replace(",", "") for t in text.split() if any(c.isdigit() for c in t)]
            correct += bool(nums) and nums[-1] == ref
        print(json.dumps({"mode": mode, "correct": f"{correct}/{len(docs)}", "last_answer": text[-160:]}))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
