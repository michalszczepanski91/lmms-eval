# Methodology

How the Jetson benchmarks measure accuracy, speed, memory, power and compute, and which settings keep the frameworks comparable.

## Keeping the frameworks comparable

| Setting | Why |
|---|---|
| Same lmms-eval task code for every framework | Scores differ only because of the model runtime, not the prompt or the metric |
| Same image resolution range: 256–2048 visual tokens (Qwen2.5-VL: 200,704–1,605,632 pixels) | Image size drives the vision encoder and prefill cost. HF/vLLM pass the limits to the processor, llama.cpp `--image-min/max-tokens`, Edge-LLM `visual_build --min/maxImageTokens`; all read them from `models/<model>/model.sh` |
| Batch size 1, one request at a time | Edge latency, not server throughput |
| Cross-request caching off (vLLM prefix + multimodal cache, llama.cpp prompt cache, Edge-LLM encoder cache) | MME asks two questions per image, so a cache would skip the vision encoder and most of the prefill for every second sample, which the HF reference cannot do |
| Offline runs from a shared, pre-downloaded HF cache | No download time in the measurements; identical checkpoints |
| No other GPU process during a run | `run_eval.sh` waits for an idle GPU and records other containers and GPU processes in `run_info.txt` |
| Power mode recorded per run, `jetson_clocks` not used | Orin ran at MAXN; Thor at 120 W (2026-09-23 to 09-25) and MAXN (from 09-28). SUMMARY.md shows the mode per run; compare like with like |

Remaining differences are properties of the frameworks and are reported, not removed: quantization (AWQ, GGUF Q4/Q8, FP8), image transport (PNG/base64 over HTTP for llama.cpp and vLLM's default path; `+pil` / `+png1` variants measure that overhead), and Edge-LLM downscaling images with a side above 4096 px before its runner (`max_image_side`; the other frameworks shrink such images to ≤ 2048 visual tokens anyway).

## Metrics

| Metric | Source |
|---|---|
| Scores | lmms-eval task metrics (MME perception / cognition, GQA exact match, TextVQA VQA accuracy (reported as `exact_match`: min(1, matching human answers / 3), averaged over the 10 leave-one-out subsets), COCO CIDEr, ...) |
| TTFT per sample | From the model call to the first generated token: vision encoder + prefill + first decode step. HF: generation streamer. vLLM: engine request metrics. llama.cpp: first streamed chunk (includes HTTP). Edge-LLM: runner averages only (vision encoder + prefill) |
| Answer time per sample | Whole model call, all generated tokens |
| Decode ms/token | (answer − TTFT) / (output tokens − 1). Edge-LLM: median GPU time of the runner's `llm_generation` stage (one entry per decode step) |
| Preprocess | Prompt and image preparation before the model call, client side (HF: chat template + processor; vLLM: message building incl. PNG/base64 encoding, or none with `pass_pil_images=True`; llama.cpp: message building incl. PNG/base64 encoding) |
| Vision encoder / LLM prefill | The two parts of TTFT, p50. HF: CUDA events around the vision encoder (`stage_timing=True`, read after `generate`, so no extra syncs), prefill = TTFT − encoder. llama.cpp: server log at verbosity 4, encoder = `encoding mtmd batch` → first `decoding image batch`, prefill = `prompt eval time` − encoder (server side, no HTTP). Edge-LLM: median GPU time of the runner's `vision_encoder` and `llm_prefill` stages. vLLM: with `stage_timing=True` (`EXTRA_MODEL_ARGS=stage_timing=True`), CUDA events around the model's `visual` module, as for HF: the engine runs in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), the hooks are registered through `LLM.apply_model` and read after each request, prefill = TTFT − encoder; no measurable overhead (TTFT −1% on 50 questions). Without it, vLLM reports TTFT only (the encoder runs inside its prefill step) |
| TTFT / answer p99 | Tail latency over all samples |
| Throughput | Samples / eval wall time (lmms-eval `total_evaluation_time_seconds`, which includes model load and vLLM compilation: significant only for smoke runs) |
| Energy per sample | Average module power × eval wall time / samples: includes preprocessing and gaps between requests |
| Wall time | Whole lmms-eval run, including image loading and preprocessing |
| RAM | Whole-board unified memory from `tegrastats` at 1 Hz: base (before model load) and peak; other processes included |
| Power | `tegrastats` rails averaged over the run. GPU rail: `VDD_GPU_SOC` on Orin (GPU + SoC), `VDD_GPU` on Thor (GPU only). Module: GPU + CPU/SoC + 5 V system rails, comparable across boards |
| FLOPs per sample | Analytical, see below |
| Achieved TFLOP/s | (vision + prefill FLOPs) / mean TTFT, and total FLOPs / mean answer time. A lower bound on GPU throughput, since TTFT includes host overhead (and HTTP for llama.cpp); Edge-LLM's TTFT is GPU time only, so its rate is not a lower bound |

Per-sample numbers are in each run's `lmms_eval/*_samples_<task>.jsonl.gz` under `token_counts`: `input_tokens`, `output_tokens`, `preprocess_seconds`, `time_to_first_token_seconds`, `generation_seconds`. MME, GQA and TextVQA answers are one word or a short phrase: 2.0 / 2.3 / 4.1 generated tokens on average, including the end token. Decode is then < 0.2% of the FLOPs but 16–26% of HF's model time; use a longer-answer task such as COCO captioning to compare decode speed.

## Text-only models

The text-only study (Qwen2.5-1.5B / 3B-Instruct, `models/qwen2_5/`, `MODEL_MODALITY=text`) uses the same harness without the vision parts, plus these settings. Plan, decisions and decision rule: [results/thor/NOTES_qwen2_5.md](../results/thor/NOTES_qwen2_5.md).

| Setting | Why |
|---|---|
| Tasks through lmms-eval's chat path: the few-shot examples and the question in one user turn, the model's own chat template | Every framework receives the same messages. HF, vLLM and llama.cpp apply the template; the Edge-LLM engine stores the same template and default system prompt ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."; the backend's own default is turned off with `system_prompt=none`). HF, vLLM and llama.cpp record identical prompt token counts per question |
| Fixed few-shot examples (`fewshot_config: sampler: first_n`): the first 5 train (GSM8K) / dev (MMLU) examples | lmms-eval creates a few-shot sampler only with a `fewshot_config`; the random default would draw different examples per framework |
| Generation stops at the end-of-turn token, with an explicit cap per task (GSM8K 512, MMLU 32, IFEval 1,280, sweep 256 tokens) | No backend applies a task's `until` strings, and the default caps differ (HF 4,096, vLLM max(task, 1,024), llama.cpp 1,024, Edge-LLM 128). `vllm.sh` lowers vLLM's own default for text models so the task's cap applies |
| Context 4,096 tokens everywhere (vLLM / llama.cpp context, Edge-LLM max input and KV cache) | The longest request is MMLU's ~3,170 tokens. Set once in `model.sh` (`MAX_MODEL_LEN`) |
| Task variants under `jetson/tasks/` (loaded with `--include_path`), nothing changed under `lmms_eval/tasks` | GSM8K 5-shot as lmms-eval's prompt; MMLU 5-shot generative with "Answer the last question with the letter of the correct option only." (in one user turn the chat model otherwise answers all six questions); IFEval unchanged |

**Metrics that differ from the VL runs.**

- **TTFT is the prefill**: there is no vision encoder, so the time to first token is the whole prefill. HF records it with its generation streamer (generic `huggingface` chat backend), vLLM from its request metrics, llama.cpp from the first streamed chunk. Edge-LLM reports no per-request TTFT; its median `llm_prefill` GPU time from the runner profile stands in (GPU time only, so slightly favourable to Edge-LLM).
- **Output tokens**: the backends' counts; Edge-LLM records none, so `flops.py` recounts the answer text with the tokenizer (+1 for the end token). Answer text was not saved for filtered tasks (GSM8K, MMLU) before 2026-10-08 15:50 (NOTES_qwen2_5.md problem 7).
- **Energy per output token** = energy per question / mean output tokens. It includes the prefill, so it is only comparable within a task.
- **Accuracy differences** are paired: every framework answers the same questions, so the difference to HF bf16 and its standard error come from the per-question score differences (`experiments/qwen2_5/analyze.py`). GSM8K: flexible-extract exact match; MMLU: first A–D letter, all 14,042 questions (weighted by subject size); IFEval: prompt-level strict accuracy.
- **Time per question** = eval wall time / questions, as "throughput" above, so it includes each framework's own answer length.

**Controlled-length sweep** (`tasks/length_sweep`, `experiments/qwen2_5/length_sweep.sh`). The benchmark tasks mix prompt and answer lengths; the sweep fixes both. 16 prompts each of exactly 128, 512 and 2,048 tokens (chat template included, checked with the tokenizer; `datasets/length_sweep_data.py`), each asking the model to count to 1,000, one number per line. That answer is far longer than 256 tokens, so every framework stops at the same cap of 256 generated tokens without an ignore-EOS option (the Edge-LLM runner has none); the run's mean output tokens confirm it. 16- and 8-bit weights in every framework, both sizes.

**Edge-LLM build cost** (`ensure_engines.sh`): quantize, export and build seconds, peak system memory during each step (shared CPU/GPU memory, sampled every 2 s) and engine size are written to `build_info.txt` in each engine directory.

## FLOPs

`models/qwen2_5_vl/flops.py` (text-only: `models/qwen2_5/flops.py`, the same LLM terms without the vision encoder) computes FLOPs per sample from the model's `config.json`, each sample's image grid and its prompt / output token counts. `scripts/compute_flops.sh` runs it in the eval container after every run (`FLOPS=0` to skip) and writes `flops.json`.

- **Counted**: matrix multiplies at 2 FLOPs per multiply-accumulate: every linear layer (patch embedding, attention projections, MLPs, patch merger, LM head) plus attention scores and attention-weighted values. Norms, activations, rotary embeddings and softmax are left out (< 1%).
- **Vision encoder**: ViT with window attention over 8×8-patch windows except the full-attention blocks, then the 2×2 patch merger into the LLM width.
- **Prefill**: the LLM over the whole prompt (text + image tokens) plus the LM head for the first output token. Causal attention counts L(L+1)/2 query–key pairs, the work FlashAttention-style kernels do; an implementation that computes the full L×L matrix and masks it does up to 2× the attention FLOPs.
- **Decode**: one LLM step per further output token against the KV cache, plus the LM head.
- **Token counts**: those recorded by the backend. Edge-LLM records none, so they are recounted with the tokenizer (≈ 0.1% difference in prefill).
- **Same for every framework and precision**: quantization changes the cost per FLOP, not the count. A FLOPs difference between frameworks on the same task means different images or prompts.
- **Validated** against PyTorch's `FlopCounterMode` on a scaled-down random Qwen2.5-VL built from the real transformers classes (eager attention, 4 image shapes): vision encoder exact, LLM exact up to the causal convention above.

## Result files

Each run writes `results/<board>/<model>/<tasks>/<framework>-<precision>[+tag]/<run id>[_limitN]/`:

| File | Content |
|---|---|
| `run_info.txt` | board, L4T, power mode, git commit, image IDs, other containers / GPU processes, exact lmms-eval command, versions |
| `run.log` | console output, without per-request progress bars (see `slim_run.py`) |
| `server.log.gz` / `trt_profile.json` | llama-server log (verbosity 4) / Edge-LLM runner profile |
| `INVALID` / `STALLED` / `SUPERSEDED` | run left out of the summary tables and listed under "Excluded runs" with the file's first line as the reason. `INVALID`: `check_run.py` found ≥ 1% `????` or ≥ 2% empty answers. `STALLED`: `run_matrix.sh` stopped the run because `run.log` had not grown for `STALL_MINUTES` (default 30). `SUPERSEDED`: replaced by a later run (written by hand) |
| `TIMING_INVALID` | scores kept, timing / memory / power dropped from the summary (e.g. a foreign CPU load during the run) |
| `tegrastats.log.gz` | 1 Hz RAM, CPU, GPU, temperatures, power rails |
| `flops.json`, `flops_samples.jsonl.gz`, `flops.log` | FLOPs summary per task / per sample / log |
| `lmms_eval/*_results.json` | scores and the full lmms-eval config |
| `lmms_eval/*_samples_<task>.jsonl.gz` | per-sample input, answer, target, score and `token_counts` |

Run IDs with `_limitN` are smoke tests; `summarize.py` leaves them out unless `--include-smoke`.

## Implementation notes

- Containers run as the calling user, plus group `mlusers` if it exists (`SHARED_GROUP`), so files in the shared cache stay group-writable.
- `download_assets.sh` also prepares the `datasets` Arrow cache, which works offline and avoids the HF login that task configs with `token: True` (MME, GQA) would otherwise require.
- Models load from local snapshot paths, because transformers 4.57.3 queries the Hub for repo IDs even in offline mode.
- lmms-eval exits 0 when evaluation fails, so `run_eval.sh` checks the log for `Error during evaluation`.
- The repo is mounted at its host path: `lmms_eval/llm_judge/factory.py` fails when it is mounted at a shallow path such as `/workspace`.
- llama.cpp runs with `GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16`: with fp16 accumulation one MME image overflows Qwen2.5-VL-3B and every later answer is `????` until the server restarts.
- TensorRT Edge-LLM specifics (fp16 vision-encoder overflow, image-size limit, AWQ differences): [tensorrt-edgellm.md](tensorrt-edgellm.md#known-issues-thor-edge-llm-v0101).
- vLLM keeps its `torch.compile` / AOT caches in `jetson/.cache/vllm` (`VLLM_CACHE_ROOT`) between runs. A stale entry can crash the engine at start-up (`CUDA error: an illegal instruction`, results/thor/NOTES.md problem 19): move the entry named in run.log ("Directly load AOT compilation from …") aside, or run once with `EVAL_ENV=VLLM_DISABLE_COMPILE_CACHE=1`.
- Out of memory on Jetson shows up as `NVML_SUCCESS == r INTERNAL ASSERT FAILED` in PyTorch's CUDA allocator.
- `docker build` has no GPU driver, so the Dockerfiles do not run CUDA binaries.
- Don't use the frameworks' own per-token decode averages. llama-server's `eval time = X ms / N tokens` and Edge-LLM's `generation.average_time_per_token_ms` both divide by all N generated tokens, but the first token comes from the prefill, so there are only N − 1 decode steps. With 2–4-token answers, that makes decode look 1.3–1.5× faster than it is. `summarize.py` and `plot_stages.py` divide by N − 1, or use Edge-LLM's per-step `llm_generation` stage.
