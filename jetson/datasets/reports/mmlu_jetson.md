# mmlu_jetson

Tokenizer and chat template: `Qwen/Qwen2.5-3B-Instruct`.

| | |
|---|---|
| Questions | 14,042 |
| Prompt tokens (chat template + few-shot context), avg (min–max) | 738 (282–3,138) |
| Question tokens (the question alone), avg (min–max) | 59 (2–951) |
| Reference answer words / tokens, avg (min–max) | 1.0 (1–1) / 1.0 (1–1) |
| Generated tokens (end token included), hf-bf16 | 13.1 (2–32) |
| Generated tokens (end token included), vllm-bf16 | 13.1 (2–32) |
| Generated tokens (end token included), llamacpp-f16 | 7.3 (2–32) |
| Generated tokens (end token included), trt_edgellm-fp16 | 12.8 (2–32) |
| Generated tokens (end token included), vllm-fp8 | 13.8 (2–32) |
| Generated tokens (end token included), llamacpp-q8_0 | 6.7 (2–32) |
| Generated tokens (end token included), trt_edgellm-fp8 | 16.5 (2–32) |
| Generated tokens (end token included), vllm-awq | 4.5 (2–32) |
| Generated tokens (end token included), llamacpp-q4_k_m | 10.0 (2–32) |
| Generated tokens (end token included), trt_edgellm-int4_awq | 17.3 (2–32) |
| Scoring | the first A-D letter of the answer; accuracy weighted by subject size |

## Example (first question)

```
The following are multiple choice questions (with answers) about miscellaneous. Answer the last question with the letter of the correct option only.
How many axles does a standard automobile have?
A. one
B. two
C. four
D. eight
Answer: B

What place is named in the title of the 1979 live album by rock legends Cheap Trick?
A. Budapest
B. Budokan
C. Bhutan
D. Britain
Answer: B

Who is the shortest man to ever win an NBA slam dunk competition?
A. Anthony 'Spud' Webb
B. Michael 'Air' Jordan
C. Tyrone 'Muggsy' Bogues
D. Julius 'Dr J' Erving
Answer: A

What is produced during photosynthesis?
A. hydrogen
B. nylon
C. oxygen
D. light
Answer: C

Which of these songs was a Top 10 hit for the rock band The Police?
A. 'Radio Ga-Ga'
B. 'Ob-la-di Ob-la-da'
C. 'De Do Do Do De Da Da Da'
D. 'In-a-Gadda-Da-Vida'
Answer: C

An important source of information on the credit rating of retail businesses is
A. the Retail Merchants Association
B. the local chamber of commerce
C. Dun & Bradstreet, Inc.
D. the United States Retail Credit Association
Answer:
```

Reference: C
