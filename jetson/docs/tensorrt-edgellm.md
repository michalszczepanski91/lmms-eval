# TensorRT Edge-LLM

[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) runs Qwen2.5-VL and the text-only Qwen2.5-Instruct as TensorRT engines through a C++ runner. It takes three steps: export to ONNX, build engines on the device, then evaluate.

| | Orin (JetPack 6.2) | Thor (JetPack 7.1, tested) |
|---|---|---|
| Edge-LLM release | v0.6.0 | v0.10.1 |
| Export | x86 GPU host only (compute capability 8.0+, about 8–16 GB VRAM for 3B, 20–48 GB for 7B) | on the device |
| Precisions | fp16, int4_awq (no FP8 / NVFP4 on SM87) | fp16, int4_awq, fp8, nvfp4 |

## Thor workflow

```bash
export TRT_WORKSPACE=/opt/models/trt-edgellm      # ONNX and engines: <workspace>/<model>-<precision>/{onnx,engines}

# 1. Runtime image, on top of lmms-eval-jetson (L4T_RELEASE / L4T_SOC: the rXX.Y part and the SoC directory
#    in /etc/apt/sources.list.d/nvidia-l4t-apt-source.list)
docker build --build-arg BASE_IMAGE=lmms-eval-jetson:latest --build-arg EDGELLM_REF=v0.10.1 \
  --build-arg EMBEDDED_TARGET=jetson-thor --build-arg CUDA_CTK_VERSION=13.0 --build-arg ENABLE_CUTE_DSL=ALL \
  --build-arg L4T_RELEASE=r38.4 --build-arg L4T_SOC=som \
  -f jetson/frameworks/trt_edgellm/Dockerfile -t lmms-eval-trt-edgellm:latest jetson/frameworks/trt_edgellm

# 2. Engines: export (quantize if needed) + build, only the missing ones; about 2-12 min per precision
jetson/scripts/download_assets.sh hf 3b
jetson/frameworks/trt_edgellm/ensure_engines.sh 3b fp16 fp8 int4_awq
#    or step by step: export.sh 3b fp8 (-> onnx/, builds the export image on first use), then build_engines.sh 3b fp8

# 3. Evaluate like any other framework
jetson/scripts/run_eval.sh trt_edgellm 3b-fp8 mme 8
jetson/scripts/run_eval.sh trt_edgellm 3b-fp8 mme
```

The script headers list the other options: `EDGELLM_REF`, `BASE_IMAGE`, `GPU_FLAGS` (x86: `--gpus all`), `KEEP_QUANTIZED`, `AWQ_SOURCE`, `VISION_SCALE`, `REMOVE_ONNX`.

## How it works

- **Export** (`export.sh`, in an image built from `Dockerfile.export` on `nvcr.io/nvidia/pytorch:26.05-py3`):
  - `tensorrt-edgellm-quantize llm` quantizes for fp8 / nvfp4 / int4_awq (default text calibration; the LM head and the vision encoder stay unquantized); then `tensorrt-edgellm-export` exports.
  - The LLM and the vision encoder are exported separately. The vision encoder always comes from the base checkpoint, in fp16, so every precision uses the same one.
  - Before export, `scale_vision.py` scales the encoder down by `VISION_SCALE` (default 8) to keep it in fp16 range (see "Known issues").
- **Build** (`build_engines.sh`): `llm_build` (batch 1, input ≤ 2,560 tokens, KV cache 3,072) and `visual_build` with the model's image-token range (256–2,048 for Qwen2.5-VL).
- **Evaluate** (lmms-eval `trt_edgellm` backend):
  - The backend writes every image as a PNG (about 230 ms each, counted in wall time) and puts all requests into one input JSON. `llm_inference` then runs once and saves its profile as `trt_profile.json`.
  - The runner's per-stage GPU time distributions (vision encoder, prefill, decode) provide medians, but TTFT is available only as an average, not per sample.
  - Failed requests are scored as empty answers and counted in the profile as `failed_requests`.
  - The encoder cache is off (`encoder_cache_budget_bytes=0`), as it is for the other frameworks.

## Text-only models (Qwen2.5-Instruct)

The same scripts handle a text-only model when its `model.sh` sets `MODEL_MODALITY=text` (`MODEL=qwen2_5`):

- **Export**: the LLM only (no vision encoder, no `scale_vision.py`).
- **Build**: the LLM engine only, with max input and KV cache = `MAX_MODEL_LEN` (4,096) instead of the VL limits.
- **Evaluate**: `engine_dir=<engines>/llm`, no multimodal engine, no image options. The engine stores the model's chat template and default system prompt (`processed_chat_template.json`), so the backend's own default system prompt is turned off (`system_prompt=none`).

```bash
MODEL=qwen2_5 jetson/frameworks/trt_edgellm/ensure_engines.sh 3b fp16 fp8 int4_awq
MODEL=qwen2_5 jetson/scripts/run_eval.sh trt_edgellm 3b-fp8 gsm8k_jetson 8
```

**Weight configurations** (`frameworks/trt_edgellm/recipes.sh`; the precision name is the run's `<precision>`). Options of v0.10.1: `tensorrt-edgellm-quantize llm --quantization {fp8, int4_awq, nvfp4, mxfp8, int8_sq}`, `--lm_head_quantization {fp8, int4_awq, nvfp4, mxfp8}`, `--kv_cache_quantization fp8`, `--text_dataset {cnn_dailymail, wikitext, zh_en_mixed}`; no w4a8. The exporter also imports pre-quantized AWQ and GPTQ checkpoints.

| Precision | How | Status (Qwen2.5-1.5B / 3B, Thor) |
|---|---|---|
| `fp16` | base checkpoint as is | works |
| `fp8`, `int4_awq` | quantizer, default calibration (cnn_dailymail, 512 texts × ≤ 512 tokens), LM head 16-bit | work |
| `int8_sq`, `nvfp4`, `mxfp8` | quantizer | Phase 2, see [FINDINGS_qwen2_5.md](../results/thor/FINDINGS_qwen2_5.md) |
| `fp8_kv` | fp8 weights + `--kv_cache_quantization fp8` | Phase 2 |
| `fp8_lmhead`, `int4_awq_lmhead` | LM head quantized too (`--lm_head_quantization`) | Phase 2 |
| `int4_awq_taskcal` | int4_awq calibrated on 512 texts in the benchmark's format (GSM8K train + MMLU validation, `calibration/gsm8k_mmlu.jsonl`), passed through the documented local override of a built-in dataset (`EDGELLM_QUANT_DATASET_WIKITEXT`) | Phase 2 |
| `int4_awq_qwen` | Qwen's AWQ checkpoint exported directly | builds and runs, **garbage text** ("100000…", 3B) |
| `int4_awq_qwen_v1` | the same with the legacy int4 GEMM plugin (`--int4-gemm-plugin-version 1`) | Phase 2 (diagnostic) |
| `int4_gptq_qwen` | Qwen's GPTQ-Int4 checkpoint exported directly | Phase 2 |
| Qwen GPTQ-Int8 | not possible: the exporter maps every `quant_method: gptq` checkpoint to int4 and does not read `bits` | – |

`ensure_engines.sh` writes `build_info.txt` in each engine directory: quantize / export / build seconds, peak system memory per step, engine size.

## Known issues (Thor, Edge-LLM v0.10.1)

- **The fp16 vision encoder overflows (worked around).**
  - Qwen2.5-VL's ViT residual stream exceeds fp16's range (65,504) on some photos; fp32 reaches about 76,000 on 1024×768 TextVQA images. The engine then produces NaN, and the answer comes back empty. This hit about 20% of TextVQA, in every precision.
  - Edge-LLM exports this encoder only in fp16: `--dtype bfloat16` is ignored.
  - The workaround is `scale_vision.py`. It divides the patch embedding and each block's `attn.proj` / `mlp.down_proj` by `VISION_SCALE`. Each block reads the residual stream through RMSNorm, so the output doesn't change.
  - Against fp32, the scaled fp16 encoder is within 2–4.5%, against 11–27% for bf16. Speed is unchanged.
- **One NaN request can break the next ones.** After a request produces NaN, later requests in the same runner process can fail too. With the workaround above no NaNs occur, so this no longer matters here.
- **Images with a side above 4096 px are rejected** by the runner ("exceeds the GPU-resize budget"), for example 64 of MME's landmark photos. `trt_edgellm.sh` sets `max_image_side=4096`, which downscales them first (`TRT_MAX_IMAGE_SIDE` to change). Without it, MME perception drops by about 75 points.
- **int4_awq from Qwen's AWQ checkpoint** (`AWQ_SOURCE=qwen`, the checkpoint vLLM `awq` uses) answers text prompts, but every image prompt returns an empty answer (with the text-only Qwen2.5-3B-Instruct-AWQ, text answers are garbage too). The default therefore quantizes the base checkpoint with Edge-LLM's own AWQ, so Edge-LLM int4_awq and vLLM awq are **different** 4-bit models. The Edge-LLM version, calibrated on text only, loses 5.7 points on TextVQA for the 3B and 1.1 for the 7B.
- **nvfp4** (3B, quantized here) gives garbage output even for text-only prompts, while fp8 from the same pipeline is fine. NVIDIA's pre-quantized `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` would show whether the kernels or the quantization are at fault.
- **Speed varies between builds.** Two builds of the same vision engine, made on different days, differed by about 8% in speed.
- **Build-image quirks**, handled in the Dockerfiles:
  - The Thor base image ships stray TensorRT 10.13.2 headers in `/usr/include`, so the include dir is set from the apt package.
  - The base image exports `CUDAARCHS=110`; it is unset, since the FP4 kernels need `110a`.
  - The plugin path must be set (`EDGELLM_PLUGIN_PATH`).
