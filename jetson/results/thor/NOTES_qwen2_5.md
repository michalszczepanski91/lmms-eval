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

## Decisions

1. Tasks are loaded through the chat path: for text-only tasks lmms-eval puts the full few-shot context into one user turn, so every backend gets the same messages and the same chat template.
2. GGUF weights from `Qwen/<tag>-GGUF` (official; `ggml-org` has no Qwen2.5 text GGUFs). File names are lowercase (`qwen2.5-3b-instruct-q8_0.gguf`); the 3B fp16 is split in two files.
3. Edge-LLM v0.10.1 quantizer (`tensorrt-edgellm-quantize llm --help`): `--quantization {fp8,int4_awq,nvfp4,mxfp8,int8_sq}`, `--lm_head_quantization {fp8,int4_awq,nvfp4,mxfp8}`, `--kv_cache_quantization {fp8}`, `--text_dataset {cnn_dailymail,wikitext,zh_en_mixed}` (custom datasets need its customization guide). No w4a8.

## Problems and fixes

(none yet)

## Timeline

| Date | What |
|---|---|
| 2026-10-08 | Plan written; Hub repos checked (all Qwen / RedHatAI repos exist; GGUF from Qwen). |

## Open items

- Choose the GSM8K variant and the MMLU answer extraction after end-to-end checks.
