# MME: questions 0–49

Task `mme`, tokenizer `Qwen/Qwen2.5-VL-3B-Instruct`, answers from `jetson/results/thor/Qwen2.5-VL-3B-Instruct/mme/hf-bf16/20260923-215339`.

- **Image tokens:** the image is resized to multiples of 28 px within 256–2,048 tokens; each 28×28 px block is one token (W × H = tokens across × down).
- **Prompt tokens:** the prompt text only; the chat template adds 21 tokens. The model reads prompt + image tokens.
- **Answer tokens:** the model's answer tokenized; **generated tokens:** as logged by the run (end token included; empty for Edge-LLM).
- **Reference:** up to 3 of the most common reference answers.

| Doc | Prompt text | Prompt tokens | Image size (px) | Resized (px) | Image tokens W × H | Image tokens | Model answer | Answer tokens | Generated tokens | Reference | Score |
|---:|---|---:|---|---|---|---:|---|---:|---:|---|---:|
| 0 | Is a python code shown in the picture?<br>Answer the question using a single word or phrase. | 19 | 416×114 | 868×252 | 31 × 9 | 279 | Yes | 1 | 2 | Yes | 1 |
| 1 | Is a c++ code shown in the picture?<br>Answer the question using a single word or phrase. | 20 | 416×114 | 868×252 | 31 × 9 | 279 | No | 1 | 2 | No | 1 |
| 2 | The image shows a python code. Is the output of the code '7'?<br>Answer the question using a single word or phrase. | 27 | 612×206 | 784×280 | 28 × 10 | 280 | Yes | 1 | 2 | Yes | 1 |
| 3 | The image shows a python code. Is the output of the code '1'?<br>Answer the question using a single word or phrase. | 27 | 612×206 | 784×280 | 28 × 10 | 280 | No | 1 | 2 | No | 1 |
| 4 | The image shows a python code. Is the output of the code 'a cat'?<br>Answer the question using a single word or phrase. | 28 | 686×196 | 840×252 | 30 × 9 | 270 | No | 1 | 2 | Yes | 0 |
| 5 | The image shows a python code. Is the output of the code 'a dog'?<br>Answer the question using a single word or phrase. | 28 | 686×196 | 840×252 | 30 × 9 | 270 | No | 1 | 2 | No | 1 |
| 6 | The image shows a python code. Is the output of the code '36'?<br>Answer the question using a single word or phrase. | 28 | 286×176 | 588×364 | 21 × 13 | 273 | Yes | 1 | 2 | Yes | 1 |
| 7 | The image shows a python code. Is the output of the code '6'?<br>Answer the question using a single word or phrase. | 27 | 286×176 | 588×364 | 21 × 13 | 273 | No | 1 | 2 | No | 1 |
| 8 | The image shows a python code. Is the output of the code '12'?<br>Answer the question using a single word or phrase. | 28 | 276×80 | 840×252 | 30 × 9 | 270 | No | 1 | 2 | Yes | 0 |
| 9 | The image shows a python code. Is the output of the code '5'?<br>Answer the question using a single word or phrase. | 27 | 276×80 | 840×252 | 30 × 9 | 270 | No | 1 | 2 | No | 1 |
| 10 | Is a c++ code shown in the picture?<br>Answer the question using a single word or phrase. | 20 | 456×294 | 560×364 | 20 × 13 | 260 | Yes | 1 | 2 | Yes | 1 |
| 11 | Is a python code shown in the picture?<br>Answer the question using a single word or phrase. | 19 | 456×294 | 560×364 | 20 × 13 | 260 | No | 1 | 2 | No | 1 |
| 12 | The image shows a python code. Is the output of the code '11'?<br>Answer the question using a single word or phrase. | 28 | 478×192 | 728×308 | 26 × 11 | 286 | Yes | 1 | 2 | Yes | 1 |
| 13 | The image shows a python code. Is the output of the code '9'?<br>Answer the question using a single word or phrase. | 27 | 478×192 | 728×308 | 26 × 11 | 286 | No | 1 | 2 | No | 1 |
| 14 | The image shows a python code. Is the output of the code 'the list has more than 2 numbers'?<br>Answer the question using a single word or phrase. | 34 | 1080×318 | 1092×308 | 39 × 11 | 429 | Yes | 1 | 2 | Yes | 1 |
| 15 | The image shows a python code. Is the output of the code 'the list has less than 2 numbers'?<br>Answer the question using a single word or phrase. | 34 | 1080×318 | 1092×308 | 39 × 11 | 429 | No | 1 | 2 | No | 1 |
| 16 | The image shows a python code. Will the number 3 appear in the output of the code?<br>Answer the question using a single word or phrase. | 30 | 462×156 | 784×280 | 28 × 10 | 280 | Yes | 1 | 2 | Yes | 1 |
| 17 | The image shows a python code. Will the number 6 appear in the output of the code?<br>Answer the question using a single word or phrase. | 30 | 462×156 | 784×280 | 28 × 10 | 280 | No | 1 | 2 | No | 1 |
| 18 | The image shows a python code. Is the output of the code 'working hard'?<br>Answer the question using a single word or phrase. | 28 | 454×212 | 672×308 | 24 × 11 | 264 | Yes | 1 | 2 | Yes | 1 |
| 19 | The image shows a python code. Is the output of the code 'playing hard'?<br>Answer the question using a single word or phrase. | 28 | 454×212 | 672×308 | 24 × 11 | 264 | Yes | 1 | 2 | No | 0 |
| 20 | The image shows a python code. Is the output of the code 'a cat'?<br>Answer the question using a single word or phrase. | 28 | 576×188 | 812×280 | 29 × 10 | 290 | Yes | 1 | 2 | Yes | 1 |
| 21 | The image shows a python code. Is the output of the code 'a dog'?<br>Answer the question using a single word or phrase. | 28 | 576×188 | 812×280 | 29 × 10 | 290 | No | 1 | 2 | No | 1 |
| 22 | The image shows a python code. Is the output of the code '12'?<br>Answer the question using a single word or phrase. | 28 | 294×222 | 532×392 | 19 × 14 | 266 | Yes | 1 | 2 | Yes | 1 |
| 23 | The image shows a python code. Is the output of the code '5'?<br>Answer the question using a single word or phrase. | 27 | 294×222 | 532×392 | 19 × 14 | 266 | No | 1 | 2 | No | 1 |
| 24 | The image shows a python code. Is the output of the code '0'?<br>Answer the question using a single word or phrase. | 27 | 400×120 | 840×252 | 30 × 9 | 270 | Yes | 1 | 2 | Yes | 1 |
| 25 | The image shows a python code. Is the output of the code '1'?<br>Answer the question using a single word or phrase. | 27 | 400×120 | 840×252 | 30 × 9 | 270 | No | 1 | 2 | No | 1 |
| 26 | The image shows a python code. Is the output of the code 'Hello'?<br>Answer the question using a single word or phrase. | 27 | 404×220 | 616×336 | 22 × 12 | 264 | Yes | 1 | 2 | Yes | 1 |
| 27 | The image shows a python code. Is the output of the code 'World'?<br>Answer the question using a single word or phrase. | 27 | 404×220 | 616×336 | 22 × 12 | 264 | Yes | 1 | 2 | No | 0 |
| 28 | The image shows a python code. Is the output of the code 'x is smaller than 10'?<br>Answer the question using a single word or phrase. | 33 | 604×212 | 784×280 | 28 × 10 | 280 | Yes | 1 | 2 | Yes | 1 |
| 29 | The image shows a python code. Is the output of the code 'x is larger than 10'?<br>Answer the question using a single word or phrase. | 33 | 604×212 | 784×280 | 28 × 10 | 280 | No | 1 | 2 | No | 1 |
| 30 | The image shows a python code. Is the output of the code '0'?<br>Answer the question using a single word or phrase. | 27 | 436×196 | 672×308 | 24 × 11 | 264 | Yes | 1 | 2 | Yes | 1 |
| 31 | The image shows a python code. Is the output of the code '1'?<br>Answer the question using a single word or phrase. | 27 | 436×196 | 672×308 | 24 × 11 | 264 | No | 1 | 2 | No | 1 |
| 32 | The image shows a python code. Is the output of the code '3'?<br>Answer the question using a single word or phrase. | 27 | 554×240 | 700×308 | 25 × 11 | 275 | Yes | 1 | 2 | Yes | 1 |
| 33 | The image shows a python code. Is the output of the code '2'?<br>Answer the question using a single word or phrase. | 27 | 554×240 | 700×308 | 25 × 11 | 275 | No | 1 | 2 | No | 1 |
| 34 | The image shows a python code. Is the output of the code '1234'?<br>Answer the question using a single word or phrase. | 30 | 352×186 | 644×336 | 23 × 12 | 276 | No | 1 | 2 | Yes | 0 |
| 35 | The image shows a python code. Is the output of the code '12345'?<br>Answer the question using a single word or phrase. | 31 | 352×186 | 644×336 | 23 × 12 | 276 | No | 1 | 2 | No | 1 |
| 36 | The image shows a python code. Is the output of the code '11'?<br>Answer the question using a single word or phrase. | 28 | 480×200 | 700×308 | 25 × 11 | 275 | Yes | 1 | 2 | Yes | 1 |
| 37 | The image shows a python code. Is the output of the code '12'?<br>Answer the question using a single word or phrase. | 28 | 480×200 | 700×308 | 25 × 11 | 275 | No | 1 | 2 | No | 1 |
| 38 | The image shows a python code. Is the output of the code '1'?<br>Answer the question using a single word or phrase. | 27 | 360×194 | 616×336 | 22 × 12 | 264 | Yes | 1 | 2 | Yes | 1 |
| 39 | The image shows a python code. Is the output of the code '5'?<br>Answer the question using a single word or phrase. | 27 | 360×194 | 616×336 | 22 × 12 | 264 | No | 1 | 2 | No | 1 |
| 40 | Does this artwork belong to the type of religious?<br>Answer the question using a single word or phrase. | 20 | 850×1003 | 840×1008 | 30 × 36 | 1080 | Yes | 1 | 2 | Yes | – |
| 41 | Does this artwork belong to the type of historical?<br>Answer the question using a single word or phrase. | 20 | 850×1003 | 840×1008 | 30 × 36 | 1080 | No | 1 | 2 | No | – |
| 42 | Is this artwork titled virgin and child with sts catherine, cecilia, barbara, and ursula?<br>Answer the question using a single word or phrase. | 34 | 1001×1200 | 1008×1204 | 36 × 43 | 1548 | Yes | 1 | 2 | Yes | – |
| 43 | Is this artwork titled sorrow?<br>Answer the question using a single word or phrase. | 16 | 1001×1200 | 1008×1204 | 36 × 43 | 1548 | No | 1 | 2 | No | – |
| 44 | Is this artwork titled doge ziani receiving the benediction of pope alexander iii?<br>Answer the question using a single word or phrase. | 29 | 1349×1100 | 1344×1092 | 48 × 39 | 1872 | Yes | 1 | 2 | Yes | – |
| 45 | Is this artwork titled the adoration of the shepherds?<br>Answer the question using a single word or phrase. | 23 | 1349×1100 | 1344×1092 | 48 × 39 | 1872 | No | 1 | 2 | No | – |
| 46 | Is this artwork displayed in musée toulouse-lautrec, albi?<br>Answer the question using a single word or phrase. | 26 | 1000×1505 | 1008×1512 | 36 × 54 | 1944 | Yes | 1 | 2 | Yes | – |
| 47 | Is this artwork displayed in kupferstichkabinett, gotha?<br>Answer the question using a single word or phrase. | 26 | 1000×1505 | 1008×1512 | 36 × 54 | 1944 | Yes | 1 | 2 | No | – |
| 48 | Does this artwork exist in the form of painting?<br>Answer the question using a single word or phrase. | 20 | 1600×1259 | 1428×1120 | 51 × 40 | 2040 | Yes | 1 | 2 | Yes | – |
| 49 | Does this artwork exist in the form of stained-glass?<br>Answer the question using a single word or phrase. | 22 | 1600×1259 | 1428×1120 | 51 × 40 | 2040 | Yes | 1 | 2 | No | – |
