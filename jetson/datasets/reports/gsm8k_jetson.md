# gsm8k_jetson

Tokenizer and chat template: `Qwen/Qwen2.5-3B-Instruct`.

| | |
|---|---|
| Questions | 1,319 |
| Prompt tokens (chat template + few-shot context), avg (min–max) | 752 (714–879) |
| Question tokens (the question alone), avg (min–max) | 61 (23–188) |
| Reference answer words / tokens, avg (min–max) | 1.0 (1–1) / 2.3 (1–9) |
| Generated tokens (end token included), hf-bf16 | 200.8 (48–512) |
| Generated tokens (end token included), vllm-bf16 | 201.0 (46–512) |
| Generated tokens (end token included), llamacpp-f16 | 196.6 (46–512) |
| Generated tokens (end token included), trt_edgellm-fp16 | 201.4 (48–512) |
| Generated tokens (end token included), vllm-fp8 | 215.9 (54–512) |
| Generated tokens (end token included), llamacpp-q8_0 | 193.5 (50–512) |
| Generated tokens (end token included), trt_edgellm-fp8 | 199.5 (58–512) |
| Generated tokens (end token included), vllm-awq | 192.9 (51–512) |
| Generated tokens (end token included), llamacpp-q4_k_m | 212.2 (39–512) |
| Generated tokens (end token included), trt_edgellm-int4_awq | 211.2 (48–512) |
| Scoring | exact match of the last number in the answer (flexible-extract) |

## Example (first question)

```
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
#### 5

Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Answer: Maila read 12 x 2 = <<12*2=24>>24 pages today.
So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.
There are 120 - 36 = <<120-36=84>>84 pages left to be read.
Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.
#### 42

Question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
Answer: He writes each friend 3*2=<<3*2=6>>6 pages a week
So he writes 6*2=<<6*2=12>>12 pages every week
That means he writes 12*52=<<12*52=624>>624 pages a year
#### 624

Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer:
```

Reference: 18
