# ifeval

Tokenizer and chat template: `Qwen/Qwen2.5-3B-Instruct`.

| | |
|---|---|
| Questions | 541 |
| Prompt tokens (chat template + few-shot context), avg (min–max) | 75 (42–390) |
| Question tokens (the question alone), avg (min–max) | 46 (13–361) |
| Instructions per prompt, avg (min–max) | 1.5 (1–3) |
| Generated tokens (end token included), hf-bf16 | 329.8 (2–1,280) |
| Generated tokens (end token included), vllm-bf16 | 326.2 (2–1,280) |
| Generated tokens (end token included), llamacpp-f16 | 319.0 (2–1,280) |
| Generated tokens (end token included), trt_edgellm-fp16 | 323.1 (2–1,280) |
| Generated tokens (end token included), vllm-fp8 | 348.2 (2–1,280) |
| Generated tokens (end token included), llamacpp-q8_0 | 311.6 (2–1,280) |
| Generated tokens (end token included), trt_edgellm-fp8 | 342.8 (2–1,280) |
| Generated tokens (end token included), vllm-awq | 320.5 (2–1,280) |
| Generated tokens (end token included), llamacpp-q4_k_m | 352.5 (2–1,280) |
| Generated tokens (end token included), trt_edgellm-int4_awq | 354.5 (2–1,280) |
| Scoring | fraction of prompts whose verifiable instructions are all followed (prompt-level, strict) |

## Example (first question)

```
Write a 300+ word summary of the wikipedia page "https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.
```

Reference: punctuation:no_comma, detectable_format:number_highlighted_sections, length_constraints:number_words
