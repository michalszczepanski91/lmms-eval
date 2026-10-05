# GQA

Task `gqa`, tokenizer `Qwen/Qwen2.5-VL-3B-Instruct`.

| | |
|---|---|
| Questions (distinct images) | 12,578 (398) |
| Image size, MP, avg (min–max) | 0.28 (0.10–0.41), avg 580×482 px |
| Image tokens, avg (min–max) | 358.5 (260–529) |
| Question tokens (prompt text), avg (min–max) | 19.7 (14–36) |
| Reference answer words, avg (min–max) | 1.06 (1–3) |
| Reference answer tokens, avg (min–max) | 1.31 (1–4) |
| Reference answers of one word / with a digit | 94% / 0% |
| Scoring | exact match |

## Examples

### doc 0

![gqa doc 0](gqa_0.jpg)

| | |
|---|---|
| Image | 500×280 px → resized 616×336 px → 264 image tokens |
| Prompt (15 tokens) | Is it overcast?<br>Answer the question using a single word or phrase. |
| Reference | "no" |
| Reference tokens | `no` |

### doc 12334

![gqa doc 12334](gqa_12334.jpg)

| | |
|---|---|
| Image | 640×480 px → resized 644×476 px → 391 image tokens |
| Prompt (16 tokens) | Who rides on the skateboard?<br>Answer the question using a single word or phrase. |
| Reference | "skateboarder" |
| Reference tokens | `sk` `ate` `board` `er` |
