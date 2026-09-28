# Task: TextVQA benchmark of Qwen2.5-VL on this Jetson Thor, unattended

You are on the Jetson AGX Thor in `~/dev/lmms-eval`, branch `jetson-benchmark`. Run the TextVQA experiment for Qwen2.5-VL-3B/7B across HF, vLLM, llama.cpp and TensorRT Edge-LLM, then write the report. The user is away. Don't wait for answers: make reasonable decisions, record them in `jetson/results/thor/NOTES.md`, and keep going. Stop only if something blocks every remaining step.

First read `jetson/README.md`, `jetson/docs/methodology.md` and `jetson/results/thor/NOTES.md` (the "TextVQA" section says what has already been done). This prompt adds the rules and the order.

## Status (2026-09-28 22:05)

Done: preparation (dataset, Edge-LLM 3B engines), smoke test (all 13 combinations OK, committed). **Running**: the full
matrix, started 22:02 as a detached process (`pgrep -f run_matrix.sh`), log `jetson/results/thor/logs/textvqa.log`.
Don't start it a second time: wait for it (the log ends with `summary:` and possibly `failed:` lines), then go on
with the re-runs of phase 3 and with phase 4. If it is not running and the log has no `summary:` line, it died:
re-run only the missing items with `run_matrix.sh`.

## What the user wants

- **Quality**: TextVQA val accuracy (5,000 questions, VQA accuracy against 10 human answers) for every framework and precision. Qwen reports ~79.3 (3B) and ~84.9 (7B); a large gap for the HF bf16 reference means a prompt or image-size problem, which you should find before running the matrix.
- **Speed**: latency of the whole network to the first token (TTFT: vision encoder + LLM prefill + first token), split into preprocessing / vision encoder / LLM prefill where the framework allows it, p50/p90/p99, answer latency, decode ms/token, throughput, energy per sample, memory, power, FLOPs and achieved TFLOP/s. `jetson/scripts/summarize.py` produces all of it in `SUMMARY.md` (tables "Latency breakdown and efficiency" and "Compute (FLOPs)").

## Ground rules

- Every shell: `export HF_CACHE=/opt/hf-cache TRT_WORKSPACE=/opt/models/trt-edgellm SHARED_GROUP=mlusers`.
- **Git: commit locally only; do not push** (the user reviews and pushes). Commit only the files you created or changed plus results; leave the user's uncommitted `CLAUDE.md` edit alone. Conventional commit messages. The root `.gitignore` ignores `scripts/` and `logs/`: add files under `jetson/scripts/` and `jetson/results/thor/logs/` with `git add -f`. Don't commit the dir of a run that is still in progress.
- **Clean measurements**: no other GPU process, no two benchmarks at once, no benchmark next to a Docker build or an Edge-LLM export. `run_eval.sh` waits for an idle GPU on its own. Before a long phase, also check `top` for a foreign process pinning a CPU core (on 2026-09-25 one made HF 7B GQA 3x slower); record it in NOTES.md if found. Power mode is MAXN now (earlier Thor runs were at 120 W): record `nvpmodel -q` at the start, and compare timings with the old MME/GQA runs only with that caveat.
- **Don't disturb running scripts**: never edit `run_eval.sh` or `run_matrix.sh` in place while a matrix is running (write a copy and `mv` it over). No `git stash` / `checkout` / reset while anything runs.
- **Long jobs**: start them detached (`nohup setsid ... > jetson/results/thor/logs/<name>.log 2>&1 &`) and wait with a sleep loop that checks for completion every few minutes; don't poll every few seconds. A full TextVQA run takes roughly 25–90 min; the whole matrix roughly 12–16 h.
- **Hangs**: `run_matrix.sh` has a watchdog (`STALL_MINUTES`, default 30) that stops a run whose `run.log` stopped growing, writes `STALLED` into its run dir and continues. Re-run a stalled item once at the end; if it stalls again, log it as failed.
- **Answer checks**: `run_matrix.sh` runs `jetson/scripts/check_run.py` after every run. It marks runs whose answers are `????` as `INVALID`. Still look at a few answers per run, and at the empty / failed-request counts it prints.
- **Fixes**: when something fails, read `run.log` / `server.log.gz`, fix the cause in the scripts or backends (general fixes, no Thor-only hacks), and re-run. At most 3 attempts per item, then log it as failed and move on. Lint Python you change with `uvx ruff check` / `uvx ruff format --line-length 240` (if `uv` is missing, a standalone copy installs with `curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$HOME/.local/uv UV_NO_MODIFY_PATH=1 sh`), and run `test/tools/test_jetson_stage_timing.py`, `test/eval/test_token_counts.py` and `test/models/test_trt_edgellm.py` in the eval image with pytest on `PYTHONPATH` (e.g. `uv pip install --target <dir> pytest`, mounted into the container; pass `-e USER=$(id -un)`).

## Phases

Skip whatever NOTES.md says is done.

**1. Preparation**
1. `git status`, `docker ps`, `nvidia-smi`, `nvpmodel -q`, `df -h /`.
2. Dataset: `$HF_CACHE/hub/datasets--lmms-lab-encoder--textvqa` must exist; otherwise `jetson/scripts/download_assets.sh dataset:lmms-lab-encoder/textvqa`.
3. Edge-LLM 3B engines: `jetson/frameworks/trt_edgellm/ensure_engines.sh 3b fp16 fp8 int4_awq` (skips existing ones; about 15 min per missing precision; log `logs/textvqa_engines.log`). int4_awq is Edge-LLM's own AWQ of the base checkpoint (see `docs/tensorrt-edgellm.md`).

**2. Smoke test**: `SMOKE=1 SKIP_ENGINES=1 jetson/experiments/qwen2_5_vl/textvqa.sh` (8 samples per combination, ~1 h because of model loads). For every combination check:
- the run finished and the answers are sensible short strings (`lmms_eval/*_samples_textvqa_val.jsonl.gz`, field `filtered_resps`), the accuracy is not ~0;
- latency: `token_counts` has `time_to_first_token_seconds`; HF samples also have `vision_seconds`; llama.cpp `server.log.gz` has `encoding mtmd batch` / `decoding image batch` lines; Edge-LLM `trt_profile.json` has `stages`;
- `flops.json` exists;
- `python3 jetson/scripts/summarize.py --include-smoke` fills the vision encoder / prefill columns for HF, llama.cpp and Edge-LLM.
Fix what fails. Commit (results + fixes).

**3. Full matrix**: `SKIP_ENGINES=1 jetson/experiments/qwen2_5_vl/textvqa.sh` detached, log `jetson/results/thor/logs/textvqa.log`. When it ends, re-run failed or stalled items (`jetson/scripts/run_matrix.sh textvqa_val "" -- <framework> <size-precision> ...`). Commit after the matrix and after the re-runs.

**4. Report**
1. `python3 jetson/scripts/summarize.py` → `jetson/results/thor/SUMMARY.md`.
2. Write `jetson/results/thor/TEXTVQA_REPORT.md`. Lead with the answer (which framework/precision gives the best accuracy per latency and per joule, and how much accuracy each quantization costs on text reading), then:
   - a table per model (3B, 7B): framework/precision → TextVQA accuracy, TTFT p50/p90/p99, answer p50, preprocess / vision encoder / LLM prefill (ms), decode ms/token, throughput, energy per sample, wall time, peak RAM, average GPU and module power;
   - accuracy vs HF bf16 for each quantization (bf16 → AWQ / Q8_0 / Q4_K_M / fp8 / int4_awq), and vs the published Qwen numbers;
   - where the time to first token goes (vision encoder vs LLM prefill) per framework, and what that means for optimization;
   - FLOPs per sample (vision / prefill / decode) and achieved TFLOP/s per framework;
   - TextVQA vs MME / GQA: does the framework ranking hold, and does quantization hurt more on OCR? Remember that the older MME/GQA runs were at 120 W, not MAXN;
   - failures, deviations and every fix made.
3. Add a "TextVQA (date)" log section to `NOTES.md` with the setup (power mode, versions, engines), deviations and fixes.
4. Commit locally. End with a short message: the report path, the headline findings, anything that needs the user's action, and a reminder that the commits are not pushed.
