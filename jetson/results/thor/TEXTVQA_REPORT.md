# TextVQA on Jetson AGX Thor: Qwen2.5-VL-3B/7B across HF, vLLM, llama.cpp and TensorRT Edge-LLM

TextVQA val: 5,000 questions about text in 3,166 photos, scored by VQA accuracy against 10 human answers. Board: Jetson AGX Thor at **MAXN**, batch size 1, 256–2048 visual tokens per image. Runs from 2026-09-28 22:02 to 2026-09-29 23:33. Per-run numbers come from [SUMMARY.md](SUMMARY.md); setup and log are in [NOTES.md](NOTES.md) under "TextVQA".

## Answer first

- **Best accuracy per latency and per joule: TensorRT Edge-LLM fp8 (3B).** It scores 79.6 (HF bf16: 78.9) with about 130 ms of GPU time to the first token, against 361 ms TTFT for the fastest PyTorch stack (vLLM bf16). It uses 23 J per sample, against 37 J for vLLM bf16 and 39 J for HF. Edge-LLM fp16 comes second (79.5, ~157 ms, 29 J). Edge-LLM engines exist for 3B only in this study.
- **For 7B, use vLLM bf16.** It scores 83.8 with a TTFT p50 of 456 ms and p99 of 645 ms, and uses 56 J per sample. vLLM AWQ is *slower* to the first token (637 ms), because its 4-bit kernels help decode, not the ~1,000-token prefill that dominates here. It only pays off for longer answers (20 vs 63 ms per token). The 7B is worth +4.5 points over the 3B, for about +100 ms and +19 J per sample in vLLM.
- **Quantization cost on text reading depends on the method, not the bit width:**
  - fp8 (Edge-LLM): free, +0.1 ± 0.2 against fp16.
  - Qwen's official AWQ (vLLM): −0.3 ± 0.3 (3B) and −0.7 ± 0.3 (7B) against bf16.
  - llama.cpp Q4_K_M against Q8_0: −1.1 ± 0.3 (3B) and −0.2 ± 0.2 (7B).
  - **Edge-LLM's own int4 AWQ: −5.7 ± 0.4** against fp16. It is calibrated on text-only news data, which is enough for GQA (−2.5) but not for OCR. It is also slower to the first token than fp16 (232 vs 157 ms).
- **llama.cpp is the slowest by far** on this workload: TTFT about 1.1 s (3B) and 1.4–1.5 s (7B), and about 72 J / 100 J per sample. Its vision encoder alone takes 640–700 ms, against 87 ms in Edge-LLM and 276 ms in HF.
- **Bug found and fixed:** Qwen2.5-VL's vision encoder overflows fp16. In Edge-LLM, whose exporter produces only fp16 vision encoders, about 1,000 of 5,000 TextVQA answers were empty for every precision. `export.sh` now rescales the encoder (details under "Failures, deviations and fixes"), and the Edge-LLM runs were repeated. **The shared engines in `/opt/models/trt-edgellm` still have the bug; see "Needs your action".**

## Results per model

TTFT is the time from the model call to the first token: vision encoder + LLM prefill + one decode step. "Answer" is the whole model call.
- **Preprocess** is client-side work before the call: chat template and image transport.
- **Stage times** are medians. HF uses CUDA events and llama.cpp its server log. Edge-LLM reports the runner's GPU time per stage and only averages for TTFT. vLLM doesn't separate the stages.
- **Energy** is average module power × wall time / samples.
- **Wall time** covers the whole lmms-eval run, including model load and image preparation.

### Qwen2.5-VL-3B-Instruct

| Framework / precision | TextVQA acc | Δ vs HF (paired SE) | TTFT p50 / p90 / p99 (ms) | Answer p50 (ms) | Preprocess / vision / prefill (ms) | Decode (ms/token) | Throughput (samples/s) | Energy (J/sample) | Wall (min) | Peak RAM (GB) | Avg GPU / module (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HF bf16 (reference) | 78.86 | – | 397 / 497 / 562 | 515 | 29 / 276 / 119 | 52 | 1.72 | 38.7 | 48.5 | 29.6 | 38.9 / 66.5 |
| vLLM bf16 | 79.39 | +0.53 ± 0.22 | 361 / 446 / 509 | 426 | 240 / – / – | 27 | 1.43 | 36.6 | 58.3 | 31.1 | 27.2 / 52.3 |
| vLLM AWQ (Qwen checkpoint) | 79.10 | +0.24 ± 0.32 | 453 / 568 / 647 | 478 | 240 / – / – | 11 | 1.34 | 39.8 | 62.4 | 30.4 | 30.3 / 53.2 |
| llama.cpp Q8_0 | 79.86 | +1.00 ± 0.35 | 1091 / 1565 / 1868 | 1144 | 231 / 642 / 319 | 18 | 0.70 | 72.4 | 119.4 | 28.5 | 28.8 / 50.5 |
| llama.cpp Q4_K_M | 78.73 | −0.13 ± 0.37 | 1144 / 1626 / 1922 | 1187 | 231 / 665 / 352 | 16 | 0.68 | 72.4 | 122.3 | 28.6 | 28.6 / 49.3 |
| Edge-LLM fp16 | 79.45 | +0.59 ± 0.24 | ~157 (avg) | – | 230* / 87 / 74 | 20 | 2.04 | 29.1 | 40.9 | 8.5 GPU** | 35.8 / 59.2 |
| **Edge-LLM fp8** | **79.59** | +0.73 ± 0.26 | **~130 (avg)** | – | 230* / 87 / 44 | 12 | **2.32** | **23.2** | **35.9** | 5.8 GPU** | 31.9 / 53.9 |
| Edge-LLM int4_awq (Edge-LLM's AWQ) | 73.75 | **−5.11 ± 0.45** | ~232 (avg) | – | 230* / 87 / 152 | 8 | 1.92 | 30.9 | 43.3 | 4.5 GPU** | 39.3 / 59.3 |

\* Edge-LLM: the backend writes all 5,000 images as PNG before it starts the runner. That took 19.4 min per run (about 230 ms per sample, in the wall time but not in TTFT). The runner itself took 19.7 min (fp16), 14.8 min (fp8) and 22.1 min (int4_awq): 236, 178 and 265 ms per sample, including image decoding and resizing.
\** Edge-LLM: the runner's own peak GPU allocation (host peak: 12 GB). The tegrastats RAM of these three runs is not comparable. The board still held about 35 GB from the engine builds and probes that ran just before the fp16 run (base RAM 59 GB, falling to 33 GB by int4_awq).

### Qwen2.5-VL-7B-Instruct

| Framework / precision | TextVQA acc | Δ vs HF (paired SE) | TTFT p50 / p90 / p99 (ms) | Answer p50 (ms) | Preprocess / vision / prefill (ms) | Decode (ms/token) | Throughput (samples/s) | Energy (J/sample) | Wall (min) | Peak RAM (GB) | Avg GPU / module (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HF bf16 (reference) | 82.96 | – | 475 / 600 / 708 | 666 | 45 / 276 / 202 | 80 | 1.30 | 56.3 | 64.3 | 29.3 | 42.4 / 72.9 |
| **vLLM bf16** | **83.84** | +0.88 ± 0.20 | **456 / 547 / 645** | 596 | 240 / – / – | 63 | 1.12 | **55.7** | 74.6 | 30.9 | 32.4 / 62.3 |
| vLLM AWQ (Qwen checkpoint) | 83.13 | +0.17 ± 0.27 | 637 / 782 / 878 | 676 | 241 / – / – | 20 | 1.05 | 62.4 | 79.6 | 31.9 | 38.9 / 65.3 |
| llama.cpp Q8_0 | 82.39 | −0.57 ± 0.32 | 1434 / 1982 / 2332 | 1543 | 231 / 700 / 620 | 36 | 0.55 | 99.8 | 152.5 | 28.6 | 31.8 / 54.6 |
| llama.cpp Q4_K_M | 82.15 | −0.81 ± 0.33 | 1530 / 2118 / 2462 | 1622 | 231 / 706 / 699 | 29 | 0.53 | 100.4 | 157.8 | 28.6 | 31.7 / 53.0 |

There are no Edge-LLM 7B engines in this experiment; the prompt's matrix covers the 3B only.

Answers average 4 tokens, so "decode ms/token" is (answer − TTFT) / 3, a noisy number that also absorbs per-request overhead. Use a captioning task to compare decode speed.

## Accuracy: quantization and published numbers

The paired SE is the standard error of the per-question score difference over the same 5,000 questions. A delta smaller than about two SEs is noise.

| Model | Quantization step | Δ accuracy |
|---|---|---:|
| 3B | Edge-LLM fp16 → fp8 | +0.14 ± 0.19 |
| 3B | Edge-LLM fp16 → int4_awq (Edge-LLM AWQ, text calibration) | **−5.70 ± 0.44** |
| 3B | vLLM bf16 → AWQ (Qwen's checkpoint) | −0.30 ± 0.29 |
| 7B | vLLM bf16 → AWQ (Qwen's checkpoint) | −0.71 ± 0.26 |
| 3B | llama.cpp Q8_0 → Q4_K_M | −1.13 ± 0.26 |
| 7B | llama.cpp Q8_0 → Q4_K_M | −0.24 ± 0.19 |
| 3B | HF bf16 → Edge-LLM fp16 | +0.59 ± 0.24 |

**Against the published numbers** (Qwen reports 79.3 for 3B and 84.9 for 7B): HF bf16 gives 78.9 (−0.4) and 83.0 (−1.9); vLLM bf16 gives 79.4 and 83.8. The 256–2048 visual-token limit is not the cause: the largest TextVQA image uses 2,028 tokens, and most use 999 (1024×768). The rest of the 7B gap is probably Qwen's own prompt and harness; this experiment didn't investigate it further.

**Frameworks differ by up to ±1 point on the same weights, and that isn't a quality difference.** On TextVQA, Qwen often has to decide whether to stop after the first word ("Stop" vs "Stop Global Warming", "Steve" vs "Steve Jobs"). The logits are close, and small numeric differences between kernels tip that decision either way. In the 7B, HF and vLLM bf16 score differently on 190 questions. On 130 of them vLLM wins, almost always with the shorter answer, which matches more human answers. HF gives the longer answer more often, which is why every other framework scores slightly above it.

## Where the time to first token goes

| Model | Framework | Vision encoder | LLM prefill | Vision share | Vision TFLOP/s | Prefill TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| 3B | HF bf16 | 276 ms | 119 ms | 70% | 19 | 49 |
| 3B | llama.cpp Q8_0 | 642 ms | 319 ms | 67% | 8 | 18 |
| 3B | Edge-LLM fp16 | 87 ms | 74 ms | 54% | 62 | 78 |
| 3B | Edge-LLM fp8 | 87 ms | 44 ms | 66% | 62 | 132 |
| 3B | Edge-LLM int4_awq | 87 ms | 152 ms | 36% | 62 | 38 |
| 7B | HF bf16 | 276 ms | 202 ms | 58% | 19 | 67 |
| 7B | llama.cpp Q8_0 | 700 ms | 620 ms | 53% | 8 | 22 |

(TFLOP/s = analytical FLOPs of the stage / median stage time. Vision is 5.36 TFLOP per sample for both sizes, since the 3B and 7B share the 32-layer ViT; prefill is 5.79 TFLOP for the 3B and 13.5 TFLOP for the 7B.)

What this means for optimization:
- **The vision encoder is the bottleneck, not the LLM.** At about 1,000 image tokens, the ViT costs as many FLOPs as the whole 3B prefill, and it runs at a third of the prefill's rate in HF and at half of it in llama.cpp. Qwen2.5-VL's window attention runs many small attention calls, which suits neither framework. Edge-LLM's fused ViT (CuTe DSL attention kernels) is 3.2× faster than HF and 7× faster than llama.cpp. Next steps: a faster ViT (TensorRT or `torch.compile` in the PyTorch stacks), or fewer image tokens. Edge-LLM's `--visualPrune` and a lower `max_pixels` are both options, but both cost OCR accuracy and would have to be measured.
- **Weight-only 4-bit quantization makes TTFT worse here.** Prefill at about 1,000 tokens is compute-bound, and W4A16 kernels dequantize on the fly: Edge-LLM int4_awq prefill takes 152 ms against 74 ms for fp16, and vLLM AWQ TTFT is 453 ms against 361 ms (3B). fp8 (W8A8) is the precision that speeds up prefill (44 ms).
- **Image transport costs as much as the model in the server-style stacks.** vLLM (default message path) and llama.cpp spend 230–240 ms per sample encoding each ~1024 px photo as PNG/base64 before the request. That is 65% of vLLM's TTFT, and it's why vLLM's wall time and throughput are worse than HF's (1.43 vs 1.72 samples/s for 3B) even though its TTFT is lower. The Edge-LLM backend spends the same 230 ms writing PNGs. Passing PIL images (vLLM `pass_pil_images=True`, the `+pil` variant) or JPEG paths would remove most of it.
- **llama.cpp's gap is its vision path** (CLIP-style mmproj kernels at 8 TFLOP/s), plus about 130 ms of HTTP, base64 decoding and image preprocessing on the server (TTFT 1,091 ms against 961 ms of encoder + prefill). Its LLM prefill is also 2.7–3× slower than HF's.

## Compute (FLOPs per sample)

Every framework processes the same work: 977 image tokens, 39 text tokens and 4 output tokens on average. FLOPs match across frameworks to within 0.03%.

| Model | Vision (GFLOP) | Prefill (GFLOP) | Decode (GFLOP) | Total (GFLOP) |
|---|---:|---:|---:|---:|
| 3B | 5,360 | 5,794 | 19 | 11,173 |
| 7B | 5,376 | 13,471 | 43 | 18,890 |

Achieved TFLOP/s, as (vision + prefill FLOPs) / mean TTFT and total FLOPs / mean answer time (a lower bound):

| Framework | 3B to 1st token | 3B answer | 7B to 1st token | 7B answer |
|---|---:|---:|---:|---:|
| HF bf16 | 28.1 | 20.4 | 39.4 | 26.2 |
| vLLM bf16 | 30.6 | 25.2 | 41.6 | 29.7 |
| vLLM AWQ | 24.4 | 22.9 | 29.8 | 27.2 |
| llama.cpp Q8_0 | 9.8 | 9.3 | 12.7 | 11.9 |
| llama.cpp Q4_K_M | 9.4 | 9.1 | 12.0 | 11.4 |
| Edge-LLM fp16 | 71.0 | 51.9 | – | – |
| Edge-LLM fp8 | 86.0 | 68.3 | – | – |
| Edge-LLM int4_awq | 48.2 | 43.7 | – | – |

Edge-LLM's rates use GPU stage times (no host overhead) and are therefore flattered compared with the others' wall-clock TTFT.

## TextVQA vs MME and GQA

The older Thor MME/GQA runs were at **120 W**, and TextVQA ran at **MAXN**, so compare rankings, not absolute times. TextVQA images are also larger: about 977 image tokens, against 358 for GQA and 868 for MME.

- **The latency ranking holds.** 3B TTFT on GQA: Edge-LLM fp8 59 < fp16 74 < int4_awq 103 < vLLM bf16 185 < HF 199 < vLLM AWQ 217 ms. On TextVQA the order is the same: 130 < 157 < 232 < 361 < 397 < 453 ms. vLLM AWQ is slower than bf16 to the first token on all three tasks. llama.cpp has no full MME/GQA runs on Thor to compare (the earlier hang).
- **Quantization hurts more on OCR, but only for the weaker methods:**

| Quantization (3B unless noted) | GQA Δ | MME Δ | TextVQA Δ |
|---|---:|---:|---:|
| Edge-LLM fp8 vs fp16 | +0.04 | perception +1.7, cognition 0 (no-downscale runs) | +0.14 |
| Edge-LLM int4_awq vs fp16 | −2.5 | cognition −62 (no-downscale runs) | **−5.7** |
| vLLM AWQ vs bf16 | −0.5 | perception +3, cognition −11 | −0.3 |
| vLLM AWQ vs bf16 (7B) | −0.3 | perception −0.5, cognition −10 | −0.7 |

  Qwen's official AWQ checkpoint loses under one point everywhere. Edge-LLM's own AWQ, calibrated on text-only news data, loses 2.5 points on GQA and more than twice that on TextVQA. Reading small text is where its weight error shows. fp8 costs nothing on any task.
- The fp16 vision-encoder overflow never showed on GQA (small images, 0 empty answers). On MME it was masked by the landmark-image failures of the no-downscale runs. TextVQA's 1024 px text-heavy photos are what exposed it.

## Failures, deviations and fixes

1. **Edge-LLM empty answers (fixed; runs repeated).** The first fp16, fp8 and int4_awq runs had 1,006, 1,011 and 1,005 empty answers (accuracy 63.5 for fp16), on the same questions.
   - **Diagnosis:** calling `llm_inference` directly returned `finish_reason: max-length` with empty text. The logprobs were NaN from the first token: every top-3 entry was −FLT_MAX at token 151935, the last in the vocabulary, which decodes to nothing.
   - **Cause:** in HF with an fp16 vision tower, the residual stream reaches 46,000–60,000 on these images and overflows at block 31 on some; fp32 reaches 76,000. Edge-LLM's exporter always produces an fp16 vision encoder: `--dtype bfloat16` is accepted but the ONNX stays FLOAT16. Its TensorRT kernels tip over on more images than PyTorch fp16.
   - **Fix:** `jetson/frameworks/trt_edgellm/scale_vision.py`, called by `export.sh` (`VISION_SCALE`, default 8), divides the patch embedding and each block's `attn.proj` / `mlp.down_proj` by 8 before export. The blocks read the residual stream only through RMSNorm, so the encoder output is unchanged. Against fp32, the scaled fp16 encoder's output is within 2–4.5%, the same as unscaled fp16 on images that don't overflow; bf16 is within 11–27%. The failing samples now answer correctly in all three precisions.
   - **Speed:** the fix costs nothing. Fresh builds of the unscaled encoder and of encoders scaled by 2, 4 or 8 all take 75.5–76.7 ms on the same 40 images; the shared engine built on 2026-09-28 takes 68–70 ms. So Edge-LLM's vision-encoder time varies by about 8% between builds.
   - **Side finding (Edge-LLM bug, not fixed here):** after a NaN request, later requests in the same runner process can also fail. Doc 0 answers alone but not after docs 14–101. A single bad image therefore corrupts a whole batch file.
   - **What was kept:** the broken runs are marked `INVALID` and kept (`trt_edgellm-*/20260929-135231`, `-152905`, `-163808`). The repeated runs are `-212748`, `-220950` and `-224646`.
2. **`check_run.py`** now marks runs with 2% or more empty answers `INVALID`, not only `????` runs (`test/tools/test_jetson_check_run.py`). Healthy short-answer runs have none.
3. **Engines outside the shared workspace:** the repeated runs used engines in `~/trt-edgellm-vscale8`. Replacing `/opt/models/trt-edgellm` was blocked by the session's permission rules, since it's a shared directory.
4. **Other job:** a `dsvlm` job (`~/dev/distilation-edge-ai`, `scripts/run.sh --gpu`) started at 22:13 during the Edge-LLM fp8 run. Its container is listed in that run's `run_info.txt`. It waited for an idle GPU the whole time (`board busy (other GPU work); waiting`) and used no GPU, and the fp8 stage times match the fp16 and int4_awq runs. No other GPU process ran and no foreign process pinned a CPU core.
5. **Lost time:** the wait loop around the engine rebuild matched its own command line with `pgrep`, so the re-runs started about 3 h later than they could have. No measurement was affected.
6. **Deviation:** Edge-LLM int4_awq is Edge-LLM's own AWQ of the base checkpoint, not Qwen's AWQ checkpoint (as in the MME/GQA runs; Qwen's checkpoint gives empty answers for every image prompt with Edge-LLM 0.10.1). So "AWQ" means different weights in vLLM and Edge-LLM.

## Needs your action

- **Rebuild the shared Edge-LLM engines** (fp16, fp8 and int4_awq for 3B in `/opt/models/trt-edgellm`) with the fixed `export.sh`. As they are, they give NaN on large images. Either run `mv` on the three `Qwen2.5-VL-3B-Instruct-*` dirs and then `jetson/frameworks/trt_edgellm/ensure_engines.sh 3b fp16 fp8 int4_awq`, or copy the fixed sets from `~/trt-edgellm-vscale8` (17 GB).
- **Delete my leftover test directory** `/opt/models/trt-edgellm/visual-bf16-test` (about 2.7 GB), from the bf16 attempt.
- Re-run the Edge-LLM MME/GQA runs with the fixed engines if you want those tables to match. GQA had no empty answers, so its scores should barely move.
