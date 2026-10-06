# Jetson Thor: detailed results

Qwen2.5-VL-3B and 7B on a Jetson AGX Thor, across HF transformers, vLLM, llama.cpp and TensorRT Edge-LLM, on TextVQA, MME and GQA. This file holds the curated numbers and what they mean.

- The conclusions in plain words: [FINDINGS.md](FINDINGS.md).
- Every run, including excluded ones: [SUMMARY.md](SUMMARY.md).
- Setup, problems and fixes: [NOTES.md](NOTES.md).
- How each metric is measured: [methodology](../../docs/methodology.md).

**Power mode:** TextVQA and the Edge-LLM MME/GQA runs ran at **MAXN**; the HF and vLLM MME/GQA runs ran at **120 W**. Across those two groups, compare rankings, not absolute times.

**Measurement notes, for all tables:**
- TTFT = model call to first token (vision encoder + LLM prefill + one decode step).
- Edge-LLM reports only an average TTFT, measured as GPU time without host overhead.

## 1. TextVQA

5,000 questions about text in 3,166 photos, scored against 10 human answers. On average, a question has 977 image tokens, 39 text tokens and a 4-token answer.

### 3B

| Framework / precision | Accuracy | Δ vs HF | TTFT p50 / p90 / p99 (ms) | Preprocess / vision / prefill (ms) | Decode (ms/token) | Samples/s | J/sample | Avg GPU / module (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HF bf16 (reference) | 78.86 | – | 397 / 497 / 562 | 29 / 276 / 119 | 52 | 1.72 | 38.7 | 38.9 / 66.5 |
| vLLM bf16 | 79.39 | +0.53 ± 0.22 | 361 / 446 / 509 | 240 / – / – | 27 | 1.43 | 36.6 | 27.2 / 52.3 |
| vLLM bf16, stage timing (2026-10-06)‡ | 79.39 | +0.53 ± 0.22 | 301 / 381 / 440 | 231 / 166 / 138 | 26 | 1.60 | 36.6 | 30.0 / 58.7 |
| vLLM bf16, PIL images (2026-10-06)‡ | 79.39 | +0.53 ± 0.22 | 281 / 354 / 412 | 7 / – / – | 26 | 2.65 | 31.9 | 47.4 / 84.6 |
| vLLM AWQ (Qwen's checkpoint) | 79.10 | +0.24 ± 0.32 | 453 / 568 / 647 | 240 / – / – | 11 | 1.34 | 39.8 | 30.3 / 53.2 |
| vLLM fp8 W8A8 (RedHatAI; torch `_scaled_mm`)†‡ | 79.48 | +0.62 ± 0.24 | 318 / 383 / 437 | 231 / 165 / 152 | 16 | 1.67 | 33.1 | 27.9 / 55.3 |
| vLLM int8 W8A8 (RedHatAI; Triton)† | 79.27 | +0.41 ± 0.28 | 359 / 449 / 513 | 231 / – / – | 16 | 1.54 | 37.0 | 29.0 / 57.1 |
| vLLM fp8 weights, 16-bit activations (RedHatAI; Marlin)† | 79.46 | +0.60 ± 0.22 | 434 / 545 / 615 | 232 / – / – | 15 | 1.38 | 43.7 | 31.8 / 60.2 |
| llama.cpp f16 | 79.84 | +0.98 ± 0.35 | 1059 / 1576 / 1848 | 231 / 640 / 289 | 28 | 0.69 | 77.3 | 28.1 / 53.1 |
| llama.cpp Q8_0 | 79.86 | +1.00 ± 0.35 | 1091 / 1565 / 1868 | 231 / 642 / 319 | 18 | 0.70 | 72.4 | 28.8 / 50.5 |
| llama.cpp Q4_K_M | 78.73 | −0.13 ± 0.37 | 1144 / 1626 / 1922 | 231 / 665 / 352 | 16 | 0.68 | 72.4 | 28.6 / 49.3 |
| Edge-LLM fp16 | 79.45 | +0.59 ± 0.24 | ~157 (avg) | 230* / 87 / 74 | 27 | 2.04 | 29.1 | 35.8 / 59.2 |
| **Edge-LLM fp8** | **79.59** | +0.73 ± 0.26 | **~130 (avg)** | 230* / 87 / 44 | 16 | **2.32** | **23.2** | 31.9 / 53.9 |
| Edge-LLM int4_awq (Edge-LLM's AWQ) | 73.75 | **−5.11 ± 0.45** | ~232 (avg) | 230* / 87 / 152 | 11 | 1.92 | 30.9 | 39.3 / 59.3 |

### 7B

| Framework / precision | Accuracy | Δ vs HF | TTFT p50 / p90 / p99 (ms) | Preprocess / vision / prefill (ms) | Decode (ms/token) | Samples/s | J/sample | Avg GPU / module (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HF bf16 (reference) | 82.96 | – | 475 / 600 / 708 | 45 / 276 / 202 | 80 | 1.30 | 56.3 | 42.4 / 72.9 |
| vLLM bf16 | 83.84 | +0.88 ± 0.20 | 456 / 547 / 645 | 240 / – / – | 63 | 1.12 | 55.7 | 32.4 / 62.3 |
| vLLM bf16, stage timing (2026-10-06)‡ | 83.80 | +0.84 ± 0.19 | 394 / 482 / 578 | 231 / 166 / 230 | 61 | 1.22 | 56.3 | 35.3 / 68.9 |
| vLLM bf16, PIL images (2026-10-06)‡ | 83.84 | +0.88 ± 0.20 | 389 / 468 / 568 | 7 / – / – | 61 | 1.71 | 51.9 | 48.2 / 88.8 |
| vLLM AWQ (Qwen's checkpoint) | 83.13 | +0.17 ± 0.27 | 637 / 782 / 878 | 241 / – / – | 20 | 1.05 | 62.4 | 38.9 / 65.3 |
| vLLM fp8 W8A8 (RedHatAI; torch `_scaled_mm`)†‡ | 83.81 | +0.84 ± 0.21 | 361 / 443 / 516 | 231 / 166 / 196 | 33 | 1.43 | 44.1 | 32.4 / 63.2 |
| vLLM int8 W8A8 (RedHatAI; Triton)† | 83.65 | +0.69 ± 0.24 | 509 / 617 / 712 | 232 / – / – | 33 | 1.17 | 56.7 | 33.3 / 66.3 |
| vLLM fp8 weights, 16-bit activations (RedHatAI; Marlin)† | 83.74 | +0.78 ± 0.20 | 681 / 840 / 951 | 231 / – / – | 32 | 0.98 | 73.8 | 38.7 / 72.5 |
| llama.cpp f16 | 82.58 | −0.38 ± 0.31 | 1449 / 2002 / 2273 | 231 / 741 / 594 | 59 | 0.52 | 113.9 | 30.7 / 59.4 |
| llama.cpp Q8_0 | 82.39 | −0.57 ± 0.32 | 1434 / 1982 / 2332 | 231 / 700 / 620 | 35 | 0.55 | 99.8 | 31.8 / 54.6 |
| llama.cpp Q4_K_M | 82.15 | −0.81 ± 0.33 | 1530 / 2118 / 2462 | 231 / 706 / 699 | 29 | 0.53 | 100.4 | 31.7 / 53.0 |
| Edge-LLM fp16 | **83.98** | +1.02 ± 0.21 | ~246 (avg) | 230* / 87 / 160 | 59 | 1.48 | 47.3 | 42.6 / 69.9 |
| **Edge-LLM fp8** | 83.88 | +0.92 ± 0.23 | **~180 (avg)** | 230* / 89 / 91 | 32 | **1.88** | **33.8** | 38.9 / 63.6 |
| Edge-LLM int4_awq (Edge-LLM's AWQ) | 82.92 | −0.05 ± 0.31 | ~397 (avg) | 230* / 88 / 323 | 20 | 1.40 | 51.8 | 51.1 / 72.4 |

- **Δ vs HF:** the mean per-question score difference over the same 5,000 questions, ± its standard error. A difference smaller than about 2 SE is noise.
- **Preprocess** (client side, before the model call) is mostly converting the image to PNG/base64 for vLLM and llama.cpp. \*Edge-LLM's backend writes all images as PNG before starting the runner (19.4 min per run). That time counts in samples/s and J/sample, but not in TTFT.
- **Stage times** (vision / prefill) are medians: CUDA events for HF and for vLLM with `stage_timing=True`, the server log for llama.cpp, runner GPU time for Edge-LLM. Other vLLM runs don't separate them.
- **‡ 2026-10-05/06 runs:** HF and vLLM ran 10–15% faster than on 2026-09-28/29 (see "Run-to-run variation" below), so compare vLLM settings within the same dates: the stage-timing bf16 row is the same-day reference for the PIL and fp8 rows. The fp8 rows are the stage-timing runs (`vllm-fp8+torch-stages`); they replace the 2026-10-05 runs without stage timing (identical answers, TTFT within 2%).
- **Decode** (ms per token after the first) barely affects these numbers: answers are only about 4 tokens long. HF, vLLM and llama.cpp: client side, (answer − TTFT) / (tokens − 1); Edge-LLM: median GPU time of the runner's decode steps. llama-server's own "eval time" per token is about 1.5× too low, because it counts the first token, which comes from the prefill.
- **† vLLM 8-bit:** vLLM 0.19.0's default kernels don't run on Thor (int8 refused on SM110; the fp8 Cutlass kernel supports sm100f only and crashes; NOTES.md, problem 17). These runs use vLLM's alternative kernels, chosen with `EVAL_ENV`: `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel` (fp8 → `ChannelWiseTorchFP8ScaledMMLinearKernel`), `VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel` (int8 → `TritonInt8ScaledMMLinearKernel`), `VLLM_TEST_FORCE_FP8_MARLIN=1` (fp8 weights only → `MarlinFP8ScaledMMLinearKernel`). Result dirs `vllm-fp8+torch`, `vllm-w8a8+triton`, `vllm-fp8+marlin`; each run.log names the selected kernel.
- **Memory:** Edge-LLM's runner peak GPU allocation is 8.5 / 5.8 / 4.5 GB (3B) and 16.3 / 10.2 / 7.3 GB (7B) for fp16 / fp8 / int4_awq. The other frameworks peak at 28.5–31.9 GB of whole-board memory.

### Run-to-run variation (3B TextVQA, MAXN)

| Setting | First run → repeat | Answers that differ | TTFT p50 | Decode (ms/token) | Questions/s | J/question |
|---|---|---:|---:|---:|---:|---:|
| HF bf16 | 2026-09-28 → 10-06 | 0 | 397 → 387 (−3%) | 52 → 41 | 1.72 → 1.88 (+9%) | 38.7 → 39.9 (+3%) |
| vLLM bf16 | 2026-09-28 → 10-06 | 0 | 361 → 308 (−15%) | 27 → 26 | 1.43 → 1.58 (+10%) | 36.6 → 36.9 (+1%) |
| Edge-LLM fp16 | 2026-09-29 → 10-06 | 0 | ~157 → ~159 (+1%) | 27 → 26 | 2.04 → 2.04 (0%) | 29.1 → 31.0 (+7%) |
| vLLM fp8 (`_scaled_mm`) | 2026-10-05 → 10-06 (with stage timing) | 0 | 325 → 318 (−2%) | 15 → 16 | 1.64 → 1.67 (+2%) | 33.1 → 33.1 (0%) |
| vLLM bf16, 7B | 2026-09-29 → 10-06 (with stage timing) | 86 (Δ −0.04 ± 0.09) | 456 → 394 (−14%) | 63 → 61 | 1.12 → 1.22 (+9%) | 55.7 → 56.3 (+1%) |

- **Accuracy reproduces exactly** within a framework on the same compiled code. The 7B difference (86 answers, no accuracy change) came with a fresh `torch.compile` after the stale cache was moved aside (NOTES.md, problem 19).
- **Edge-LLM's times reproduce** (TTFT +1%; energy +7% from a slightly higher module power). **HF and vLLM ran 10–15% faster on 2026-10-05/06 than on 2026-09-28/29**, with the same image, board and power mode and no other GPU process or container recorded (CPU load is not recorded); the cause is not known. Runs a day apart agree within 2% (vLLM fp8).

## 2. MME and GQA

Current runs only; SUMMARY.md also lists the older ones. GQA has 12,578 questions (358 image tokens on average); MME has 2,374 yes/no questions (868 image tokens).

| Model | Framework / precision | MME perception / cognition | GQA accuracy | TTFT MME / GQA (ms) | J/sample MME / GQA | Power |
|---|---|---:|---:|---:|---:|---|
| 3B | HF bf16 | 1548.3 / 623.9 | 60.09 | 241 / 199 | 31.1 / 16.4 | 120 W |
| 3B | vLLM bf16 | 1540.6 / 613.6 | 59.96 | 241 / 185 | 37.2 / 15.6 | 120 W |
| 3B | vLLM AWQ | 1544.0 / 602.9 | 59.45 | 282 / 217 | 39.0 / 15.5 | 120 W |
| 3B | Edge-LLM fp16 | 1526.5 / 614.3 | 60.00 | ~146 / ~69 (avg) | 27.8 / 11.7 | MAXN |
| 3B | **Edge-LLM fp8** | 1522.4 / 622.5 | **60.18** | **~121 / ~56** (avg) | **23.8 / 8.7** | MAXN |
| 3B | Edge-LLM int4_awq | 1529.7 / 565.7 | 57.55 | ~211 / ~97 (avg) | 31.4 / 11.6 | MAXN |
| 7B | HF bf16 | 1680.4 / 616.4 | 60.88 | 283 / not valid† | 43.1 / – | 120 W |
| 7B | vLLM bf16 | 1687.1 / 621.1 | 60.98 | 367 / 223 | 52.1 / 22.5 | 120 W |
| 7B | vLLM AWQ | 1686.6 / 610.7 | 60.70 | 396 / 300 | 57.4 / 23.2 | 120 W |

- **TTFT:** HF and vLLM are p50; Edge-LLM is a mean, and MME's large images make its mean higher than a median would be.
- **† HF 7B GQA:** the timing is not valid (a foreign process loaded the CPU during the run); its accuracy is.
- **llama.cpp** has no full MME or GQA runs on Thor.
- **The 7B scores better than the 3B on MME perception** (about +140), but not on GQA (+0.8).
- **MME is noisy:** Edge-LLM's perception is about 20 below HF, spread over 1–3 questions in each of a few categories. vLLM bf16 is also 8 below HF.

## 3. Accuracy

### Quantization cost (3B unless noted)

| Step | TextVQA | GQA | MME perception / cognition |
|---|---:|---:|---:|
| Edge-LLM fp16 → fp8 | +0.14 ± 0.19 | +0.18 | −4.1 / +8.2 |
| Edge-LLM fp16 → int4_awq (Edge-LLM's AWQ, text-only calibration) | **−5.70 ± 0.44** | **−2.45** | +3.2 / **−48.6** |
| Edge-LLM fp16 → fp8, 7B | −0.10 ± 0.16 | – | – |
| Edge-LLM fp16 → int4_awq, 7B | −1.06 ± 0.27 | – | – |
| vLLM bf16 → AWQ (Qwen's checkpoint) | −0.30 ± 0.29 | −0.51 | +3.4 / −10.7 |
| vLLM bf16 → AWQ, 7B | −0.71 ± 0.26 | −0.28 | −0.5 / −10.4 |
| vLLM bf16 → fp8 W8A8 (torch `_scaled_mm`) | +0.09 ± 0.19 | – | – |
| vLLM bf16 → fp8 W8A8, 7B | −0.03 ± 0.15 | – | – |
| vLLM bf16 → int8 W8A8 (Triton) | −0.12 ± 0.23 | – | – |
| vLLM bf16 → int8 W8A8, 7B | −0.19 ± 0.20 | – | – |
| vLLM bf16 → fp8 weights only (Marlin) | +0.07 ± 0.17 | – | – |
| vLLM bf16 → fp8 weights only (Marlin), 7B | −0.10 ± 0.13 | – | – |
| llama.cpp f16 → Q8_0 | −0.02 ± 0.11 | – | – |
| llama.cpp f16 → Q8_0, 7B | −0.19 ± 0.11 | – | – |
| llama.cpp Q8_0 → Q4_K_M | −1.13 ± 0.26 | – | – |
| llama.cpp Q8_0 → Q4_K_M, 7B | −0.24 ± 0.19 | – | – |

- **8-bit costs nothing:** fp8 on any task, and vLLM's int8 W8A8 on TextVQA (RedHatAI's int8 is GPTQ-calibrated on 512 Flickr30k image–caption pairs; its fp8 needs no calibration).
- **Qwen's AWQ checkpoint** loses under 1 point on TextVQA and GQA.
- **Edge-LLM's own AWQ** loses the most where the model reads text or reasons over it. MME cognition falls 49 points: code reasoning −7.5, numerical calculation −5 and commonsense reasoning −5 percentage points. Its calibration data is plain news text. The 7B loses much less with the same quantizer (−1.1 vs −5.7 on TextVQA).

### Frameworks agree

With full-size weights, frameworks differ by up to ±1 point, and that isn't a quality difference. On TextVQA, Qwen often has to decide whether to stop after the first word ("Stop" vs "Stop Global Warming"), and small numeric differences between kernels tip it either way. HF and vLLM bf16 (7B) disagree on 190 questions. vLLM wins 130 of them, almost always with the shorter answer, which matches more human answers.

### Against the published numbers

Qwen reports 79.3 (3B) and 84.9 (7B) on TextVQA. We measure 78.9 and 83.0 with HF, 79.4 and 83.8 with vLLM, and 79.5 and 84.0 with Edge-LLM fp16. Our image-token limit is not the cause: the largest TextVQA image needs 2,028 tokens, just under the 2,048 cap. The rest of the 7B gap was not investigated; Qwen's own prompt and evaluation harness are the likely difference.

## 4. Where the time to the first token goes

TextVQA medians:

| Model | Framework | Vision encoder | LLM prefill | Vision share | Vision TFLOP/s | Prefill TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| 3B | HF bf16 | 276 ms | 119 ms | 70% | 19 | 49 |
| 3B | vLLM bf16 (stage timing) | 166 ms | 138 ms | 55% | 32 | 42 |
| 3B | vLLM fp8 (`_scaled_mm`, stage timing) | 165 ms | 152 ms | 52% | 32 | 38 |
| 3B | llama.cpp f16 | 640 ms | 289 ms | 69% | 8 | 20 |
| 3B | llama.cpp Q8_0 | 642 ms | 319 ms | 67% | 8 | 18 |
| 3B | Edge-LLM fp16 | 87 ms | 74 ms | 54% | 62 | 78 |
| 3B | Edge-LLM fp8 | 87 ms | 44 ms | 66% | 62 | 132 |
| 3B | Edge-LLM int4_awq | 87 ms | 152 ms | 36% | 62 | 38 |
| 7B | HF bf16 | 276 ms | 202 ms | 58% | 19 | 67 |
| 7B | vLLM bf16 (stage timing) | 166 ms | 230 ms | 42% | 32 | 59 |
| 7B | vLLM fp8 (`_scaled_mm`, stage timing) | 166 ms | 196 ms | 46% | 32 | 69 |
| 7B | llama.cpp f16 | 741 ms | 594 ms | 56% | 7 | 23 |
| 7B | llama.cpp Q8_0 | 700 ms | 620 ms | 53% | 8 | 22 |
| 7B | Edge-LLM fp16 | 87 ms | 160 ms | 35% | 62 | 84 |
| 7B | Edge-LLM fp8 | 89 ms | 91 ms | 49% | 60 | 148 |
| 7B | Edge-LLM int4_awq | 88 ms | 323 ms | 21% | 61 | 42 |

- **The vision encoder is the bottleneck.** At about 1,000 image tokens, the ViT needs as many FLOPs as the whole 3B prefill (5.4 vs 5.8 TFLOP per question), but HF runs it at a third of its prefill rate. Qwen2.5-VL's window attention makes many small attention calls. vLLM runs the same model code 1.7× faster (compiled, with its FlashAttention backend); Edge-LLM's fused ViT is 1.9× faster than vLLM's, 3.2× faster than HF's and 7× faster than llama.cpp's. The levers are a faster ViT (TensorRT, or `torch.compile` in the PyTorch stacks) or fewer image tokens. Edge-LLM's `--visualPrune` and a lower `max_pixels` both reduce tokens, but both may cost OCR accuracy.
- **Weight-only 4-bit slows the first token.** A ~1,000-token prefill is compute-bound, and W4A16 kernels dequantize on the fly. Edge-LLM int4_awq prefill takes 152 ms against 74 ms for fp16 (7B: 323 vs 160 ms), and vLLM AWQ TTFT is 453 ms against 361 ms. Only fp8 (W8A8) speeds prefill up: 44 ms (7B: 91 ms).
- **Image transport costs as much as the model.** Encoding a ~1024 px photo as PNG/base64 takes 230–240 ms per question in vLLM's default path, in llama.cpp and in the Edge-LLM backend; HF passes PIL images and spends 29 ms. That is why vLLM processes fewer questions per second than HF (1.43 vs 1.72 for 3B) despite a lower TTFT. Passing PIL images (vLLM `pass_pil_images=True`) or JPEG paths would remove most of it.
- **llama.cpp loses on both stages, whatever the weight format.** Its vision encoder runs at 7–8 TFLOP/s and its prefill is 2.4–3× slower than HF's. f16 weights don't change that: the vision encoder is the same, and f16 prefill is only 4–9% faster than Q8_0's (no unpacking). Q8_0's 8-bit weights speed up only decode, 1.5–1.7×. HTTP, base64 decoding and server-side image preparation add another ~130 ms (TTFT 1,091 ms vs 961 ms for encoder + prefill).

## 5. Compute (FLOPs per question)

The analytical FLOPs match across frameworks to within 0.03%, so every framework did the same work.

| Task | Model | Vision (GFLOP) | Prefill (GFLOP) | Decode (GFLOP) | Total (GFLOP) |
|---|---|---:|---:|---:|---:|
| TextVQA | 3B | 5,360 | 5,794 | 19 | 11,173 |
| TextVQA | 7B | 5,376 | 13,471 | 43 | 18,890 |
| MME | 3B / 7B | 4,861 / 4,874 | 5,239 / 12,139 | 6 / 15 | 10,107 / 17,028 |
| GQA | 3B / 7B | 1,892 / 1,898 | 2,240 / 5,243 | 8 / 18 | 4,140 / 7,160 |

Achieved TFLOP/s on TextVQA, as (vision + prefill FLOPs) / mean TTFT:

| Model | HF bf16 | vLLM bf16 | vLLM AWQ | llama.cpp f16 / Q8_0 / Q4_K_M | Edge-LLM fp16 / fp8 / int4_awq |
|---|---:|---:|---:|---:|---:|
| 3B | 28.1 | 30.6 | 24.4 | 9.8 / 9.8 / 9.4 | 71.0 / 86.0 / 48.2 |
| 7B | 39.4 | 41.6 | 29.8 | 12.6 / 12.7 / 12.0 | 76.7 / 104.4 / 47.5 |

vLLM 8-bit (alternative kernels): fp8 `_scaled_mm` / int8 Triton / fp8 Marlin reach 36.2 / 31.3 / 25.9 TFLOP/s (3B) and 52.8 / 37.6 / 28.0 (7B), against 36.9 and 48.3 for bf16 measured on the same days (30.6 and 41.6 in the 2026-09-28/29 runs).

Edge-LLM's rates use GPU time only, so they are higher than a wall-clock rate would be.
