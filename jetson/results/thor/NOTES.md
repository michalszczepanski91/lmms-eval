# Jetson Thor: setup and lab notes

How the Thor benchmark was run, what went wrong, and how each problem was fixed.
- The results: [FINDINGS.md](FINDINGS.md) (plain words), [REPORT.md](REPORT.md) (numbers), [SUMMARY.md](SUMMARY.md) (every run).
- How the measurements work: [methodology](../../docs/methodology.md).

## Setup

**Board**
- Jetson AGX Thor: L4T R38.4 (JetPack 7.1), CUDA 13.0, TensorRT 10.13.3; 14 CPU cores, 125.8 GB unified memory.
- **Power mode:** 120 W (the board default) for the runs of 2026-09-23 to 09-25, because `sudo` wasn't available to change it; **MAXN** from 2026-09-28 on (TextVQA and the Edge-LLM re-runs). `jetson_clocks` was never used.

**Software**
- `lmms-eval-jetson` image, built on `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor`: torch 2.10.0, transformers 4.57.3, vLLM 0.19.0, flash-attn 2.8.4. HF uses `flash_attention_2`.
- `llamacpp-jetson:b11135`: llama.cpp b11135 built for `CUDA_ARCH=110`. It accumulates in bf16, because fp16 overflows (problem 10).
- **TensorRT Edge-LLM v0.10.1:**
  - runtime image `lmms-eval-trt-edgellm`;
  - export image `lmms-eval-trt-export:v0.10.1` (NGC PyTorch 26.05, transformers 5.14.1, modelopt 0.45.0).

**Edge-LLM 3B and 7B engines** in `/opt/models/trt-edgellm`
- Precisions:
  - fp16;
  - fp8, quantized with Edge-LLM's quantizer;
  - int4_awq: Edge-LLM's own AWQ of the base checkpoint, **not** the Qwen AWQ checkpoint that vLLM uses.
- Limits: LLM input 2,560 tokens, KV cache 3,072, vision encoder 256–2,048 image tokens.
- Vision encoder: fp16, scaled by `VISION_SCALE=8` (problem 7).
- Sizes: 3B fp16 7.7 GB, fp8 5.1 GB, int4_awq 3.8 GB; 7B fp16 16 GB, fp8 9.5 GB, int4_awq 6.5 GB. The 7B set took 53 min to build (fp16 6 min, fp8 7 min, int4_awq 40 min, mostly AWQ calibration).

**Paths**
- HF cache: `/opt/hf-cache`.
- Engines:
  - `/opt/models/trt-edgellm` for the current ones;
  - `/opt/models/trt-edgellm/old-unscaled-vision-20260928` for the broken ones;
  - `~/trt-edgellm-vscale8`, an identical copy used by the TextVQA re-runs.

**Git**
- The branch lives only on your fork. `origin` fetches over HTTPS and pushes over SSH, using the deploy key `~/.ssh/id_ed25519_github`, which has write access to this repository only.

## Problems and fixes

Each problem is described once. How a mechanism works is in the linked docs; this section records what happened on this board.

### TensorRT Edge-LLM

Details of each issue: [tensorrt-edgellm.md, "Known issues"](../../docs/tensorrt-edgellm.md#known-issues-thor-edge-llm-v0101).

1. **Building v0.10.1 on Thor needed several fixes:**
   - Dockerfile quirks.
   - The new export CLI (`tensorrt-edgellm-quantize` / `tensorrt-edgellm-export`).
   - `USER` and a readable `HF_TOKEN_PATH` for the quantizer.
   - Exporting the vision encoder from the base checkpoint, because Qwen's AWQ repos lack vision config fields.
2. **An engine built from Qwen's AWQ checkpoint** gave empty answers to every image prompt. Edge-LLM's own AWQ is now the default; its runs are the `int4_awq` ones.
3. **nvfp4** produced garbage (open).
4. **The runner exits 1 when any request fails.** The backend now scores failed requests as empty answers instead of aborting (`test/models/test_trt_edgellm.py`).
5. **Images above 4096 px were rejected:** 128 MME landmark questions failed, and perception was about 1,450 instead of about 1,526. Fixed with `max_image_side`; the old runs are marked `SUPERSEDED`.
6. **Result files were written to the run dir root** instead of `lmms_eval/`. Fixed in `run_eval.sh`.
7. **The fp16 vision encoder overflowed.**
   - About 1,000 of 5,000 TextVQA answers were empty for fp16, fp8 and int4_awq, on the same questions. Found by checking the runner's logprobs: NaN from the first token, decoded as "".
   - Fixed with `scale_vision.py`. The three broken runs are marked `INVALID`, and all Edge-LLM runs were repeated.
   - A second Edge-LLM bug showed up at the same time: after one NaN request, later requests can fail too (open).
8. **Two builds of the same vision engine differ by about 8% in speed:** 68–70 ms vs 75.5–76.7 ms on the same 40 images. It's not caused by the fix: unscaled builds and builds scaled by 2, 4 or 8 all give 75.5–76.7 ms.

### llama.cpp

9. **The 3B MME run hung** at sample 935 of 2,374 for about 27 h, with llama-server stuck on one request. That led to the stall watchdog. The run is marked `STALLED`. llama.cpp was left out of MME and GQA on Thor.
10. **fp16 overflow** (found on Orin): answers become `????`, fixed with bf16 accumulation ([methodology](../../docs/methodology.md#implementation-notes)). This may also explain problem 9.

### Measurements and infrastructure

11. **Other GPU jobs:** another user's HF job on 2026-09-23, and your own `dsvlm` experiments on 09-29 and 09-30. `run_eval.sh` waits for an idle GPU. The 2026-09-30 re-runs also waited until the GPU had been idle for 10 minutes; no run overlapped with other GPU work.
12. **A foreign CPU load.** During the 2026-09-25 GQA runs, something outside the benchmark pinned one CPU core at 100%. HF 7B GQA ran 3× slower with the GPU mostly idle. It is marked `TIMING_INVALID`; its score is still valid. The load was gone by 2026-09-28.
13. **Memory left over from builds.** Engine builds and tests just before the Edge-LLM TextVQA re-runs left about 35 GB of board memory in use, so those runs' RAM figures aren't comparable. REPORT.md uses the runner's peak GPU memory instead.
14. **Clock.** In the first session the system clock was 7 h 13 min slow (no sudo to fix it). Run IDs from 2026-09-23 to 09-25 are therefore about 7 h early, and `docker build` needed `APT_OPTS="-o Acquire::Check-Date=false"`. Result file names inside the container carry a timestamp 6 h ahead of local time.
15. **Tooling added along the way:**
    - `summarize.py` reads Thor's power rails.
    - `slim_run.py` cut the results from 218 MB to 16 MB.
    - `check_run.py` and the run markers catch silent failures ([methodology](../../docs/methodology.md#result-files)).
16. **Decode per token was under-reported for llama.cpp and Edge-LLM** (found 2026-10-02 and 10-05). Both servers' own per-token averages divide by every generated token, including the first one, which comes from the prefill ([methodology](../../docs/methodology.md#implementation-notes)).
    - llama.cpp, via `plot_stages.py` and FINDINGS: 12 / 11 ms for Q8_0 / Q4_K_M, really 18 / 16 ms. The giveaway was 3B f16 at "19 ms" per token, which would read ~6.2 GB of weights at ~320 GB/s, above Thor's 273 GB/s. Per request, dividing by N − 1 gives a flat ~59.5 ms for 7B f16, whatever the answer length, and matches the client-side figure.
    - Edge-LLM, via `summarize.py` and SUMMARY.md: 20 / 12 / 8 ms for fp16 / fp8 / int4_awq, really 27 / 16 / 11 ms (the runner's `llm_generation` stage). FINDINGS already used the correct values.
    - Both scripts are fixed and the figures regenerated. Accuracy, TTFT and stage times were not affected.
17. **vLLM 8-bit (fp8, w8a8) doesn't run on Thor** (2026-10-05, vLLM 0.19.0+cu130, torch 2.10.0, CUDA 13.0). RedHatAI's W8A8 checkpoints (`*-FP8-dynamic` / `-FP8-Dynamic`, `*-quantized.w8a8`), 3B and 7B, all fail at engine start-up in the 8-sample smoke test:
    - **w8a8:** vLLM selects `CutlassInt8ScaledMMLinearKernel` for `CompressedTensorsW8A8Int8`, then `RuntimeError: Int8 not supported on SM110. Use FP8 quantization instead, or run on older arch (SM < 100).`
    - **fp8:** vLLM selects `CutlassFP8ScaledMMLinearKernel` for `CompressedTensorsW8A8Fp8`; the kernel prints "This kernel only supports sm100f." (2,272 times for 3B, 3,616 for 7B) and the engine dies with `torch.AcceleratorError: CUDA error: unspecified launch failure` during memory profiling.
    - The failed smoke runs are kept (`vllm-fp8/`, `vllm-w8a8/` `_limit8` dirs, no results). Edge-LLM's own fp8 kernels do run on SM110.
    - **Workaround (on request, same day):** vLLM's own kernel switches, passed with `EVAL_ENV`, as separately tagged variants. `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel` → `ChannelWiseTorchFP8ScaledMMLinearKernel` (fp8 GEMM through `torch._scaled_mm` / cuBLASLt, scales applied afterwards), `RUN_TAG=torch`. `VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel` → `TritonInt8ScaledMMLinearKernel`, `RUN_TAG=triton`. `VLLM_TEST_FORCE_FP8_MARLIN=1` → `MarlinFP8ScaledMMLinearKernel` (fp8 weights, 16-bit activations), `RUN_TAG=marlin`. All six full TextVQA runs (3B, 7B) are valid; each run.log names the selected kernel. Results: REPORT.md section 1, FINDINGS.md section 2.
18. **Agent pitfall.** A wait loop using `pgrep -f <name>` matched its own command line and never ended, which cost about 3 h. Wait on log contents or on a PID instead.
19. **A stale vLLM compile cache crashed 7B bf16** (2026-10-06). Both 7B bf16 runs of that day (stage timing, PIL) failed at engine start-up with `CUDA error: an illegal instruction was encountered`, right after "Directly load AOT compilation from …/torch_aot_compile/fcdb4fb9f86d…" (an entry written 2026-09-28 and loaded successfully on 09-29). All other runs that day worked, 7B fp8 included. An 8-question run reproduced it; the same run with `VLLM_DISABLE_COMPILE_CACHE=1` worked. That entry and its backbone cache (`549a44b603`) were moved to `jetson/.cache/vllm/stale-20261006/`; the re-runs compiled fresh and are valid. Why the entry went stale is not known (same image, driver and L4T). Diagnostic runs: `vllm-bf16+diag`, `vllm-bf16+diag-nocache` (`_limit8`).
20. **Run-to-run variation** (2026-10-06 repeats, REPORT.md "Run-to-run variation"). Answers reproduce exactly. Edge-LLM's times repeat within ±1–4%; HF and vLLM ran 10–15% faster than on 2026-09-28/29, cause not known (CPU load is not recorded). The 7B bf16 stage-timing run, compiled fresh after problem 19, differs from the 09-29 run in 86 answers (accuracy −0.04 ± 0.09 points). The older vLLM bf16 runs are kept (accuracy agrees, TTFT does not); the 2026-10-05 fp8 `+torch` runs are marked SUPERSEDED by the `+torch-stages` runs (identical answers, TTFT within 2%).

## Timeline

| Date | What |
|---|---|
| 2026-09-23 | Setup, images and Edge-LLM 3B engines built; smoke tests; full MME for HF, vLLM and Edge-LLM (120 W). |
| 2026-09-24 | llama.cpp 3B MME hangs (problem 9). |
| 2026-09-25 | Matrix stopped; analytical FLOPs added; full GQA for HF, vLLM and Edge-LLM (120 W). |
| 2026-09-28 | Results slimmed; `jetson/` reorganized; Orin fixes merged. MAXN from now on. Engines rebuilt; TextVQA smoke test 21:46–22:02; full matrix started 22:02. |
| 2026-09-29 | Matrix done at 17:45 (13 runs × 5,000). Edge-LLM empty answers diagnosed and fixed (problem 7). Engines rebuilt 17:58–18:18; Edge-LLM TextVQA re-runs 21:27–23:31. |
| 2026-09-30 | Shared engines replaced by the fixed ones (sha256-identical). Edge-LLM MME and GQA re-run with them, 16:14–19:45, after the dsvlm jobs had finished. Branch pushed to the fork. |
| 2026-10-01 | llama.cpp f16 TextVQA (16-bit weights, like the other frameworks), MAXN: smoke test 12:00, 3B 12:01–14:03, 7B 14:03–16:44. Both valid (no `????`, no empty answers). |
| 2026-10-02 | f16 results added to REPORT.md / FINDINGS.md. llama.cpp decode per token found under-reported (problem 16). |
| 2026-10-05 | Same decode error found in `summarize.py` for Edge-LLM; SUMMARY.md regenerated. Edge-LLM 7B engines (fp16, fp8, int4_awq, `VISION_SCALE=8`) built 10:14–11:07; smoke tests passed; full TextVQA 11:10–13:55, all three valid (no empty or `????` answers, so no `VISION_SCALE=16` rebuild was needed). Logs `logs/edgellm_7b_engines.log`, `logs/edgellm_7b_textvqa.log`. |
| 2026-10-05 | vLLM `fp8` / `w8a8` precisions added (RedHatAI checkpoints, 3B and 7B downloaded); all four smoke tests fail at engine start-up on SM110 (problem 17), so no full runs. |
| 2026-10-05 | vLLM 8-bit with alternative kernels (problem 17): smoke tests passed; full TextVQA 15:21–21:48 (fp8 `+torch`, int8 `+triton`, fp8 `+marlin`; 3B and 7B), all valid. Log `logs/vllm_8bit_workarounds.log`. |
| 2026-10-06 | vLLM `stage_timing=True` added and validated (50 questions: TTFT −1%). Full TextVQA 14:08–19:47: vLLM bf16 and fp8 with stage timing, bf16 with PIL images, repeats of HF / vLLM / Edge-LLM 3B; the two 7B bf16 runs failed (problem 19) and were re-run 19:53–21:52. Logs `logs/measurement_gaps.log`, `logs/measurement_gaps_7b_retry.log`. |

## Open items

- **Not run on Thor:** llama.cpp MME and GQA; Edge-LLM 7B MME and GQA; HF 7B GQA timing (problem 12).
- **vLLM 8-bit on Thor:** runs only with alternative kernels (problem 17); retry the default Cutlass kernels with a vLLM release that supports SM110.
- **Not solved:** nvfp4 (problem 3); Edge-LLM's NaN state bug (problem 7); an engine built from Qwen's AWQ checkpoint (problem 2).
- **Not explained:** the 7B TextVQA gap to Qwen's published number (83.0 vs 84.9).
- **Optional disk cleanup (17 GB each):** `/opt/models/trt-edgellm/old-unscaled-vision-20260928` and `~/trt-edgellm-vscale8`.
