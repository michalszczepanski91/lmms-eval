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
- `llamacpp-jetson:b11135`: llama.cpp b11135 built for `CUDA_ARCH=110`. It accumulates in bf16, because fp16 overflows (see problem 12).
- **TensorRT Edge-LLM v0.10.1:**
  - runtime image `lmms-eval-trt-edgellm`;
  - export image `lmms-eval-trt-export:v0.10.1` (NGC PyTorch 26.05, transformers 5.14.1, modelopt 0.45.0).

**Edge-LLM 3B engines** in `/opt/models/trt-edgellm`
- Precisions:
  - fp16;
  - fp8, quantized with Edge-LLM's quantizer;
  - int4_awq: Edge-LLM's own AWQ of the base checkpoint, **not** the Qwen AWQ checkpoint that vLLM uses.
- Limits: LLM input 2,560 tokens, KV cache 3,072, vision encoder 256–2,048 image tokens.
- Vision encoder: fp16, scaled by `VISION_SCALE=8` (problem 9).
- Sizes: fp16 7.7 GB, fp8 5.1 GB, int4_awq 3.8 GB.

**Paths**
- HF cache: `/opt/hf-cache`.
- Engines:
  - `/opt/models/trt-edgellm` for the current ones;
  - `/opt/models/trt-edgellm/old-unscaled-vision-20260928` for the broken ones;
  - `~/trt-edgellm-vscale8`, an identical copy used by the TextVQA re-runs.

**Shared machine**
- Other people's GPU jobs run here: another user's HF job on 2026-09-23, and your own `dsvlm` experiments on 09-29 and 09-30.
- `run_eval.sh` waits until no other GPU process is running and records any other containers in `run_info.txt`.

**Git**
- The branch lives only on your fork. `origin` fetches over HTTPS and pushes over SSH, using the deploy key `~/.ssh/id_ed25519_github`, which has write access to this repository only.

## Problems and fixes

Each problem is described once, grouped by component.

### TensorRT Edge-LLM

1. **Building v0.10.1 on Thor.**
   - The old export commands were removed in Edge-LLM 0.8, so `export.sh` now uses `tensorrt-edgellm-quantize` + `tensorrt-edgellm-export`.
   - The base image's stray TensorRT 10.13.2 headers broke CUDA compilation; fixed by passing `-DTensorRT_INCLUDE_DIR`.
   - `CUDAARCHS=110` broke the FP4 kernels; unset for the build.
   - `EDGELLM_PLUGIN_PATH` must be set.
   - `Dockerfile.export` keeps NGC's torch.
   - The quantizer needs `USER` set and a readable `HF_TOKEN_PATH`.
   - Typical times: export 2 min (fp16) or 9 min (with quantization), engine build 2 min.
2. **Qwen's AWQ checkpoints can't be exported directly.**
   - Their `vision_config` lacks default fields, so the visual export fails. The vision encoder is now always exported from the base checkpoint; its weights are bit-identical anyway.
   - Even so, an engine built from Qwen's AWQ LLM answered every image prompt with an empty answer. The default is therefore Edge-LLM's own AWQ; `AWQ_SOURCE=qwen` restores the old path.
3. **nvfp4 is broken (open).** The 3B quantized with Edge-LLM's nvfp4 produces garbage even for text-only prompts; fp8 from the same pipeline is fine. Not investigated further.
4. **The runner exits 1 if any request fails** and writes the error message as the answer. The backend now turns failed requests into empty answers and counts them (`test/models/test_trt_edgellm.py`).
5. **Cross-request encoder cache.** Edge-LLM 0.10 caches vision-encoder outputs by default (256 MiB). The backend passes `encoder_cache_budget_bytes=0`, matching vLLM and llama.cpp with caching off.
6. **Images above 4096 px are rejected** ("exceeds the GPU-resize budget"). This hit 128 MME landmark questions, so perception scored about 1,450 instead of about 1,526. The backend option `max_image_side=4096` now downscales such images first. The old runs are marked `SUPERSEDED`.
7. **Result files were written to the wrong place** (the run dir root instead of `lmms_eval/`). Fixed in `run_eval.sh`.
8. **An 8% speed difference between builds.** Rebuilding the same vision engine on different days gave 68–70 ms and 75.5–76.7 ms on the same 40 images. Edge-LLM stage times are therefore good to about ±8%.
9. **The fp16 vision encoder overflows, causing empty answers.**
   - **Symptom:** about 1,000 of 5,000 TextVQA answers were empty for fp16, fp8 and int4_awq alike, on the same questions. GQA and MME weren't affected.
   - **Cause:** the logits are NaN from the first token, and the NaN token (id 151935) decodes to "". Qwen2.5-VL's ViT residual stream goes beyond fp16's range of 65,504: fp32 reaches about 76,000 on 1024×768 photos. Edge-LLM builds this encoder only in fp16; `--dtype bfloat16` is ignored.
   - **Fix:** `scale_vision.py`, called by `export.sh` (`VISION_SCALE`, default 8), divides the patch embedding and each block's `attn.proj` / `mlp.down_proj` by 8. Each block reads the residual stream through RMSNorm, so the output is unchanged.
   - **Accuracy:** compared with fp32, the scaled fp16 encoder is off by 2–4.5%, against 11–27% for bf16.
   - **Speed:** unchanged (see problem 8).
   - **Old runs:** the three broken runs are marked `INVALID`.
10. **After one NaN request, later requests in the same runner process can fail too** (open, an Edge-LLM bug). With problem 9 fixed there are no NaNs left, so it no longer matters here.

### llama.cpp

11. **The 3B MME run hung** at sample 935 of 2,374 for about 27 h, with llama-server stuck on one request. Since then, `run_matrix.sh` has a 30-minute stall watchdog. The run is marked `STALLED`. llama.cpp was left out of MME and GQA on Thor.
12. **fp16 overflow gives `????` answers** (found on Orin): one MME image overflows the 3B, and every later answer becomes `????`. llama.cpp now accumulates in bf16 (`GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16`), and `check_run.py` marks such runs `INVALID`. This may also explain problem 11.

### Measurements and infrastructure

13. **Other GPU jobs.** `run_eval.sh` waits for an idle GPU (`WAIT_GPU_IDLE`). The 2026-09-30 re-runs also waited until the GPU had been idle for 10 minutes.
14. **A foreign CPU load.** During the 2026-09-25 GQA runs, something outside the benchmark pinned one CPU core at 100%. HF 7B GQA ran 3× slower with the GPU mostly idle. It is marked `TIMING_INVALID`; its score is still valid. The load was gone by 2026-09-28.
15. **Memory left over from builds.** Engine builds and tests just before the Edge-LLM TextVQA re-runs left about 35 GB of board memory in use, so those runs' tegrastats RAM figures aren't comparable. REPORT.md uses the runner's peak GPU memory instead.
16. **Power rails.** Thor reports `VDD_GPU` (GPU only), where Orin reports `VDD_GPU_SOC`. `summarize.py` reads both and also reports a "module" total that is comparable across boards.
17. **Clock.** In the first session the system clock was 7 h 13 min slow (no sudo to fix it). Run IDs from 2026-09-23 to 09-25 are therefore about 7 h early, and `docker build` needed `APT_OPTS="-o Acquire::Check-Date=false"`. Result file names inside the container carry a timestamp 6 h ahead of local time.
18. **Result size.** `slim_run.py` strips progress bars and gzips logs and per-sample files, which shrank the Thor results from 218 MB to 16 MB.
19. **Catching silent failures.**
    - `check_run.py` marks a run `INVALID` when at least 1% of its answers are `????` or at least 2% are empty.
    - In `summarize.py`, the markers `SUPERSEDED` (replaced by a later run) and `TIMING_INVALID` (keep the score, drop the timings) leave such runs out of the main tables and list them with their reason.
20. **Agent pitfall.** A wait loop using `pgrep -f <name>` matched its own command line and never ended, which cost about 3 h. Wait on log contents or on a PID instead.

## Timeline

| Date | What |
|---|---|
| 2026-09-23 | Setup, images and Edge-LLM 3B engines built; smoke tests; full MME for HF, vLLM and Edge-LLM (120 W). |
| 2026-09-24 | llama.cpp 3B MME hangs (problem 11). |
| 2026-09-25 | Matrix stopped; analytical FLOPs added; full GQA for HF, vLLM and Edge-LLM (120 W). |
| 2026-09-28 | Results slimmed; `jetson/` reorganized; Orin fixes merged. MAXN from now on. Engines rebuilt; TextVQA smoke test 21:46–22:02; full matrix started 22:02. |
| 2026-09-29 | Matrix done at 17:45 (13 runs × 5,000). Edge-LLM empty answers diagnosed and fixed (problem 9). Engines rebuilt 17:58–18:18; Edge-LLM TextVQA re-runs 21:27–23:31. |
| 2026-09-30 | Shared engines replaced by the fixed ones (sha256-identical). Edge-LLM MME and GQA re-run with them, 16:14–19:45, after the dsvlm jobs had finished. Branch pushed to the fork. |

## Open items

- **Not run on Thor:** llama.cpp MME and GQA; Edge-LLM 7B; HF 7B GQA timing (problem 14).
- **Not solved:** nvfp4 (problem 3); Edge-LLM's NaN state bug (problem 10); an engine built from Qwen's AWQ checkpoint (problem 2).
- **Not explained:** the 7B TextVQA gap to Qwen's published number (83.0 vs 84.9).
- **Optional disk cleanup (17 GB each):** `/opt/models/trt-edgellm/old-unscaled-vision-20260928` and `~/trt-edgellm-vscale8`.
