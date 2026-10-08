# Text-only Qwen2.5-Instruct (1.5B, 3B) on Jetson Thor: notes

Plan, decisions, problems and fixes, timeline, open items. Template: the Qwen2.5-VL study ([NOTES.md](NOTES.md), [FINDINGS.md](FINDINGS.md)).

## Plan (written 2026-10-08, before building)

**Questions.** Q1: is TensorRT Edge-LLM also the best framework for a text-only LLM? Q2: which Edge-LLM weight configuration is best for 1.5B and 3B?

**Q1 decision rule** (fixed before any Phase 1 result):
1. A configuration counts only if its accuracy is within noise of HF bf16 on that task and size: the paired difference is within 2 standard errors, or it is above HF.
2. Among the counting configurations, per task (GSM8K, IFEval, MMLU) and size (1.5B, 3B), compare four metrics: energy per question, time per question (wall time / questions), TTFT (median) and decode ms/token (median). Each framework enters with its best 16-bit, best 8-bit and best 4-bit configuration; the comparison is made within each bit width.
3. A framework is "best" on a metric if it has the lowest value, and "tied" if it is within 5% of the lowest (the measured repeat variation for Edge-LLM; vLLM/HF varied up to 15% between days, so values from different days get a 15% band).
4. Edge-LLM wins Q1 if it is best or tied on at least 3 of the 4 metrics in at least 2 of the 3 tasks, for both sizes, and never fails rule 1 where another framework passes it. Otherwise the verdict is split or negative, and is reported per task and metric (for example "wins on prefill-heavy MMLU, ties on decode-heavy GSM8K").

**Scope** (user decision, 2026-10-08): the full plan was ~6.5–7 days of GPU time, over the 4-day limit. Cut: full splits everywhere, but no vLLM w8a8 (Triton int8 was slower than bf16 in the VL study); Edge-LLM int4 from Qwen's AWQ checkpoint only if its smoke test works; the Phase 2 extra configurations (fp8 KV, calibration, GPTQ import, nvfp4, LM head) on GSM8K and MMLU only, not IFEval.

**Phases.**
- **Phase 0 (harness, CPU + smoke tests):** model folder `jetson/models/qwen2_5/` (model.sh with `MODEL_MODALITY=text`, flops.py); text-only paths in hf.sh / vllm.sh / llamacpp.sh / trt_edgellm.sh and the Edge-LLM export / build / ensure scripts; per-sample timing in the generic HF chat backend; summarize / plot_stages / stages_table without a vision stage; Jetson-local task variants (`jetson/tasks/`, loaded with `--include_path`, nothing changed under `lmms_eval/tasks`); dataset reports; VL regression smoke tests and the CPU test suite.
- **Phase 1 (Q1):** 3B first (links to the VL data), then 1.5B. Configurations per size: HF bf16; vLLM bf16, fp8 (`_scaled_mm`), Qwen AWQ; llama.cpp f16, Q8_0, Q4_K_M; Edge-LLM fp16, fp8, int4_awq (own), int4_awq from Qwen's AWQ (if it works) = 10–11 configurations × 3 tasks × 2 sizes. Plus a controlled-length sweep (16-bit and fp8, prompts 128 / 512 / 2,048 tokens, 256 output tokens) and repeats on a different day.
- **Phase 2 (Q2):** Edge-LLM fp16, fp8, fp8 + fp8 KV, int4_awq (default and a better-matched calibration), int8_sq, nvfp4, mxfp8, LM-head variants, imported Qwen AWQ / GPTQ if the exporter can; both sizes; GSM8K + MMLU for the extras, all three tasks for the main ones.

**Runtime estimate** (from the VL per-token times; refined after the smoke tests): per 3B run, MMLU 5-shot ~1 h, GSM8K ~2 h (HF ~4 h), IFEval ~1.3 h (HF ~3 h); 1.5B ~0.6×. Phase 1 ~80 h, length sweep and repeats ~15 h, Phase 2 ~35–45 h, engine builds ~5 h: **~4–4.5 days**.

**Runtime estimate, revised 2026-10-08 from the 3B smoke tests** (s/question incl. model load: HF 0.87, vLLM 0.68, Edge-LLM 0.57, llama.cpp 0.52, mostly MMLU): MMLU 0.3–0.55 s/question, GSM8K 2–4 s at 16-bit. Phase 1 ~68 h (3B ~41 h, 1.5B ~27 h), Phase 2 extras ~29 h, sweep / repeats / builds ~11 h: **~4.5 days**. User decision: go ahead with full splits; re-check before Phase 2.

## Decisions

1. Tasks are loaded through the chat path: for text-only tasks lmms-eval puts the full few-shot context into one user turn, so every backend gets the same messages and the same chat template.
2. GGUF weights from `Qwen/<tag>-GGUF` (official; `ggml-org` has no Qwen2.5 text GGUFs). File names are lowercase (`qwen2.5-3b-instruct-q8_0.gguf`); the 3B fp16 is split in two files.
3. Edge-LLM v0.10.1 quantizer (`tensorrt-edgellm-quantize llm --help`): `--quantization {fp8,int4_awq,nvfp4,mxfp8,int8_sq}`, `--lm_head_quantization {fp8,int4_awq,nvfp4,mxfp8}`, `--kv_cache_quantization {fp8}`, `--text_dataset {cnn_dailymail,wikitext,zh_en_mixed}` (custom datasets need its customization guide). No w4a8.

4. **Generation length is set by the task, the same in every framework.** No backend applies a task's `until` stop strings, and their default caps differ (HF 4,096, vLLM max(task, 1,024), llama.cpp 1,024, Edge-LLM 128). The Jetson task variants therefore generate until the end-of-turn token with an explicit cap (GSM8K 512, MMLU 32, IFEval 1,280 as in the original), and `vllm.sh` lowers vLLM's own default for text models so the task's cap applies.
5. **Fixed few-shot examples** (`first_n`: the first 5 training / dev examples before every question). lmms-eval only creates a few-shot sampler when a task has a `fewshot_config`, and the random default would draw different examples per framework, since vLLM builds its prompts in a thread pool. For MMLU, `first_n` from the dev split is the standard 5-shot setup; for GSM8K it differs from lm-eval's random 5-shot.
6. **MMLU instruction.** With the five examples and the question in one user turn, the chat model worked through all six questions ("Let's address each question one by one"). Each subject description now ends with "Answer the last question with the letter of the correct option only." The answer is the first A-D letter in the output.
7. **Same system prompt everywhere.** Qwen2.5's template adds "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." when there is no system message; HF, vLLM and llama.cpp apply the template, and the Edge-LLM engine stores the same default (`default_system_prompt` in `processed_chat_template.json`). The Edge-LLM backend's own default ("You are a helpful assistant.", right for Qwen2.5-VL) is turned off for text models (`system_prompt=none`).
8. **Context 4,096 tokens** for every framework and the Edge-LLM engine (max input 4,096, KV 4,096). Measured prompt lengths (chat template, Qwen2.5 tokenizer): GSM8K mean 752 / max 879, IFEval 75 / 390, MMLU 725 / 3,125 (+13 with the instruction); with the caps, the longest request is MMLU's ~3,170 tokens.
9. **IFEval dependencies** (langdetect, immutabledict, NLTK punkt) are installed into `jetson/.cache` by `jetson/scripts/install_text_deps.sh` and added to the eval container only for text models, so the images and the VL runs stay unchanged.
10. **Controlled-length sweep** (`jetson/tasks/length_sweep`, `experiments/qwen2_5/length_sweep.sh`): 16 prompts each of exactly 128, 512 and 2,048 tokens (chat template included, checked with the tokenizer) and 256 generated tokens. The Edge-LLM runner has no ignore-EOS option (it stops at the tokenizer's EOS), so instead of a backend option the prompt asks the model to count to 1,000, one number per line, which takes far more than 256 tokens; every framework stops at the same cap. The prompt is synthetic shipping notes, cut to length and padded with one-token words (`jetson/datasets/length_sweep_data.py`). Configurations: 16-bit and 8-bit per framework (llama.cpp Q8_0 as its 8-bit), both sizes.
11. **Phase 2 configurations** (checked in the Edge-LLM v0.10.1 export image before building anything). The exporter imports pre-quantized AWQ and GPTQ checkpoints, but `config.py` maps every `quant_method: gptq` checkpoint to `int4_gptq` and never reads `bits`: **Qwen's GPTQ-Int8 cannot be imported** (it would be read as int4); it appears in the tables with that reason. Qwen's GPTQ-Int4 is tried (`int4_gptq_qwen`). For the garbage int4 import of Qwen's AWQ (problem 6) one diagnostic is cheap: the same import with the legacy int4 GEMM plugin (`--int4-gemm-plugin-version 1`, AWQ-swizzled weights) instead of the default cuteDSL V2 (`int4_awq_qwen_v1`).
12. **Better-matched calibration** (`int4_awq_taskcal`). The quantizer's built-in text sets are cnn_dailymail (default), wikitext and zh_en_mixed; a custom set goes through its documented local override of a built-in (`EDGELLM_QUANT_DATASET_<NAME>` = a JSONL with a `text` field). `jetson/datasets/edgellm_calibration.py` writes 512 texts in the benchmark's own format (question as the Jetson task asks it, then the reference answer, in the Qwen2.5 chat template): 256 from GSM8K **train**, 256 from MMLU **validation** (57 subjects); no test split and nothing from IFEval (it has only its evaluation set). 192 tokens on average (the quantizer cuts at 512); without the 5 examples, so shorter than the benchmark prompts.
13. **Engine build cost** is recorded by `ensure_engines.sh` in `build_info.txt` per engine: quantize, export and build seconds, peak system memory during each step (Thor's memory is shared), engine bytes. The 3B Phase 1 engines were built before this existed (fp16: 2 min 44 s build).

## Problems and fixes

1. **The generic HF chat backend loaded the bare decoder.** `lmms_eval/models/chat/huggingface.py` picked the model class with `config.model_type in AutoModelForCausalLM._model_mapping.keys()`, but those keys are config classes, so the check never matched and every model fell back to `AutoModel` (`'Qwen2Model' object has no attribute 'generate'`). Fixed: `model_class_for(config)` checks the config class (test in `test/models/test_huggingface_chat_text.py`).
2. **Text-model chat templates reject list content.** Qwen2.5-Instruct's template concatenates the message content as a string (`can only concatenate str (not "list") to str`); the backend passed the multimodal parts list. Fixed for text-only requests (they also go through the tokenizer, not the multimodal processor).
3. **Few-shot text tasks had no sampler in the chat path** (`'ConfigurableMessagesTask' object has no attribute 'sampler'`): fixed in the task variant with `fewshot_config: {sampler: first_n}` (decision 5).
4. **MMLU answers ran to the cap** (the model answered all six questions): decision 6.
6. **Edge-LLM int4 from Qwen's AWQ checkpoint produces garbage** (3B smoke test, 2026-10-08): GSM8K and IFEval answers such as "100000000000…" (0 / 8 correct), while Edge-LLM's own int4_awq of the base checkpoint answers normally (6 / 8 GSM8K). The engine builds and runs, so the weights are imported wrongly (packing or zero-point convention); with Qwen2.5-VL it gave empty answers on every image prompt. Following the plan, `int4_awq_qwen` is not in the Phase 1 matrix; it appears in the tables as "produces garbage". Smoke runs kept (`trt_edgellm-int4_awq_qwen/*_limit8`).
5. **The HF backend recorded no per-question timing**: added the generation-streamer first-token timer the VL backend uses (now shared in `lmms_eval/models/model_utils/stage_timing.py`).

## Timeline

| Date | What |
|---|---|
| 2026-10-08 | Plan written; Hub repos checked (all Qwen / RedHatAI repos exist; GGUF from Qwen). Harness changes (Phase 0), task variants, downloads (11:57–13:00). First Edge-LLM text engine (3B fp16) built in 2 min 44 s. HF 3B smoke test: fixed problems 1–4. 3B smoke tests (8 questions per task) pass in HF, vLLM, llama.cpp f16 and Edge-LLM fp16: GSM8K 0.50–0.63, IFEval 0.50–0.63, MMLU 0.64–0.65. |

## Open items

- Edge-LLM system prompt verified with a direct runner probe: `<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>`, as in HF. HF, vLLM and llama.cpp record identical prompt token counts per question (smoke runs).
- **Next:** start Phase 1 with `nohup setsid jetson/experiments/qwen2_5/frameworks.sh > jetson/results/thor/logs/qwen2_5_phase1.log 2>&1 &`.
- Still to do: length sweep task, dataset reports, Phase 2, model docs, FINDINGS / REPORT, README / methodology / tensorrt-edgellm updates.

