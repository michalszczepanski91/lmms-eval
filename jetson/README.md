# lmms-eval on Jetson: inference-framework benchmarks

This folder benchmarks vision-language models on NVIDIA Jetson with lmms-eval. The same model and tasks run through four inference frameworks, and accuracy, latency, memory, power and FLOPs are measured the same way for each. So far: Qwen2.5-VL-3B/7B-Instruct on a Jetson AGX Orin 32GB and a Jetson AGX Thor.

| Framework | `<framework>` | Precisions (`<size>-<precision>`, first = default) | How it runs |
|---|---|---|---|
| Hugging Face transformers | `hf` | `bf16` | in process, lmms-eval `qwen2_5_vl` backend; the reference |
| vLLM | `vllm` | `bf16`, `awq` (official AWQ 4-bit checkpoint), `fp8`, `w8a8` (RedHatAI 8-bit W8A8 checkpoints) | in process, lmms-eval `vllm` backend |
| llama.cpp | `llamacpp` | `q4_k_m`, `q8_0`, `f16` (GGUF + f16 vision projector) | `llama-server` container + lmms-eval `openai` backend, streaming |
| TensorRT Edge-LLM | `trt_edgellm` | `fp16`, `int4_awq`, `fp8`, `nvfp4` (fp8/nvfp4: Thor only) | C++ runner via lmms-eval `trt_edgellm` backend; see [docs/tensorrt-edgellm.md](docs/tensorrt-edgellm.md) |

## Results

| Board | Results | Notes |
|---|---|---|
| Jetson AGX Thor | [FINDINGS.md](results/thor/FINDINGS.md) (start here), [REPORT.md](results/thor/REPORT.md) (numbers), [SUMMARY.md](results/thor/SUMMARY.md) (all runs) | [NOTES.md](results/thor/NOTES.md): setup, problems and fixes |
| Jetson AGX Orin 32GB (MAXN) | [SUMMARY.md](results/orin/SUMMARY.md): full MME for HF 3B/7B, vLLM bf16/AWQ 3B, vLLM AWQ 7B and llama.cpp Q8_0/Q4_K_M 7B | llama.cpp 3B runs are invalid (fp16 overflow; see [methodology](docs/methodology.md#implementation-notes)) |

How everything is measured, and which settings keep the frameworks comparable: [docs/methodology.md](docs/methodology.md).

## Quick start

All commands run from the repo root. Results go to `jetson/results/<board>` (`orin` or `thor`, detected from the board; override with `RESULTS_DIR`). On Thor, follow [docs/setup-thor.md](docs/setup-thor.md) for the image build arguments.

```bash
# 1. Build the images once
docker build -f jetson/docker/Dockerfile -t lmms-eval-jetson:latest .
docker build -f jetson/frameworks/llamacpp/Dockerfile -t llamacpp-jetson:b11135 jetson/frameworks/llamacpp   # llama.cpp only

# 2. Download the model for a framework (+ MME) into the shared HF cache (/opt/hf-cache); runs are offline afterwards
jetson/scripts/download_assets.sh vllm 3b-awq
jetson/scripts/download_assets.sh dataset:lmms-lab-encoder/GQA:testdev_balanced_instructions   # other datasets: dataset:<repo>[:<config>]

# 3. One run: <framework> <size>[-<precision>] [tasks=mme] [limit]
jetson/scripts/run_eval.sh vllm 3b-awq mme 8      # smoke test on 8 samples first
jetson/scripts/run_eval.sh vllm 3b-awq mme        # full MME

# 4. Many runs in a row (SUMMARY.md is refreshed at the end), or a whole experiment
jetson/scripts/run_matrix.sh mme -- hf 3b vllm 3b-awq llamacpp 3b-q8_0
jetson/experiments/qwen2_5_vl/gqa.sh              # SMOKE=1 for 8 samples per run
python3 jetson/scripts/summarize.py               # rebuild SUMMARY.md at any time
```

Useful options (environment variables of `run_eval.sh`; its header lists them all):

```bash
ATTN=sdpa jetson/scripts/run_eval.sh hf 7b mme                                   # HF without flash-attn
VLLM_GPU_MEM=0.3 jetson/scripts/run_eval.sh vllm 7b mme                          # vLLM memory fraction
EXTRA_MODEL_ARGS=max_pixels=802816 jetson/scripts/run_eval.sh hf 7b mme          # extra --model_args
RUN_TAG=pil EXTRA_MODEL_ARGS=pass_pil_images=True jetson/scripts/run_eval.sh vllm 3b mme   # variant -> <framework>-<precision>+pil
OFFLINE=0 jetson/scripts/run_eval.sh hf 3b mme                                   # allow Hub downloads during the run
WAIT_GPU_IDLE=0 jetson/scripts/run_eval.sh hf 3b mme                             # don't wait for other GPU jobs to finish
```

## Experiments

One script per question, under [experiments/qwen2_5_vl/](experiments/qwen2_5_vl/):

| Script | Question |
|---|---|
| `mme.sh` | Main comparison: MME (2,374 yes/no questions) for every framework and precision, 3B and 7B |
| `gqa.sh` | Short-answer VQA: GQA testdev-balanced (12,578 questions), same matrix |
| `textvqa.sh` | Reading text in images: TextVQA val (5,000 questions), same matrix. Builds missing Edge-LLM 3B and 7B engines first (`SKIP_ENGINES=1` to skip) |
| `followups.sh` | Image transport overhead (PIL vs PNG/base64), decode-heavy captioning (COCO), image-resolution sweep |

A full matrix takes 12–16 h. Start it detached, so it survives a closed SSH session. It runs unattended: stalled runs are stopped and bad runs are marked (see [methodology](docs/methodology.md#result-files)).

```bash
export HF_CACHE=/opt/hf-cache TRT_WORKSPACE=/opt/models/trt-edgellm SHARED_GROUP=mlusers
nohup setsid jetson/experiments/qwen2_5_vl/textvqa.sh > jetson/results/thor/logs/textvqa.log 2>&1 &
```

## Layout

```
jetson/
  README.md
  docs/
    methodology.md          what is measured and how; fairness settings; FLOPs; result files
    setup-thor.md           building and running on Jetson Thor (JetPack 7)
    tensorrt-edgellm.md     TensorRT Edge-LLM: export, engine build, evaluation, known issues
  docker/                   lmms-eval image (Dockerfile, requirements-jetson.txt)
  scripts/
    run_eval.sh             one run: start servers, eval in the container, tegrastats, FLOPs, slim the run dir
    run_matrix.sh           many runs in a row + summary; stall watchdog and answer check per run
    check_run.py            marks runs with "????" or >= 2% empty answers INVALID
    download_assets.sh      models / datasets into the shared HF cache (offline-ready)
    summarize.py            results/<board>/SUMMARY.md
    compute_flops.sh        analytical FLOPs per sample (runs models/<model>/flops.py in the container)
    slim_run.py             shrink a run dir for git (strip progress bars, gzip per-sample files)
    plot_stages.py          SVG of time per stage (vision encoder / prefill / decode) for chosen runs; --title, --xmax (shared scale), --by-size
    stages_table.py         Markdown table of time per stage per question for chosen runs
  datasets/
    run.sh                  dataset numbers + examples (image, prompt, reference, model answer), in the container
    common.py               shared code; textvqa.py, mme.py, gqa.py: one per dataset
    questions.py            one row per question: prompt, image size, image tokens, model answer, token counts
  frameworks/
    common.sh               argument parsing, board / model resolution, Hub cache helpers
    hf.sh, vllm.sh, llamacpp.sh, trt_edgellm.sh   per framework: precisions, assets, backend args, server start/stop
    llamacpp/Dockerfile     llama.cpp server image
    trt_edgellm/            runtime + export images, export.sh, build_engines.sh, ensure_engines.sh, scale_vision.py
  models/
    qwen2_5_vl/
      model.sh              sizes -> Hub repos (base, AWQ, GGUF), image-token range
      flops.py              FLOPs model of Qwen2.5-VL (vision encoder, LLM prefill, decode)
  experiments/qwen2_5_vl/   mme.sh, gqa.sh, textvqa.sh, followups.sh
  results/<board>/          SUMMARY.md, notes and reports, logs/, <model>/<task>/<framework>-<precision>[+tag]/<run id>/
```

## Extending

- **Framework**: add `frameworks/<name>.sh` defining `FW_PRECISIONS`, `fw_assets`, `fw_setup` and, for servers, `fw_start` / `fw_stop`. Use the model variables (`HF_REPO`, `IMAGE_MIN_PIXELS`, ...) rather than repo names.
- **Model**: add `models/<model>/model.sh` (same variables as `models/qwen2_5_vl/model.sh`) and run with `MODEL=<model>`. `flops.py` is architecture-specific, so a new model family needs its own; without one, set `FLOPS=0`.
- **Task**: any lmms-eval task works. Download its dataset first with `download_assets.sh dataset:<repo>[:<config>]`, since runs are offline.
