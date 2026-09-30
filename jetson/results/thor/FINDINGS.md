# Jetson Thor: what we did and what we found

## What we did

We ran the vision-language model **Qwen2.5-VL** (a small 3B and a larger 7B version) on an **NVIDIA Jetson AGX Thor** board. We ran it through four inference programs ("frameworks"):

- **HF** (Hugging Face transformers): the plain reference.
- **vLLM**: a fast server engine.
- **llama.cpp**: a lightweight engine popular for small devices.
- **TensorRT Edge-LLM**: NVIDIA's engine built for Jetson.

We also compared **precisions**. A model can store its numbers in full size (bf16/fp16) or compressed to 8 or 4 bits. Compressed models are smaller and can be faster, but may lose accuracy.

We tested on three benchmarks:

- **TextVQA**: 5,000 questions about text in photos (signs, labels, screens).
- **MME**: 2,374 yes/no questions.
- **GQA**: 12,578 short questions about objects in scenes.

For every run we measured accuracy, speed, energy, memory and compute. All frameworks got the same images and the same questions, and ran one question at a time with nothing else on the GPU.

## What we found

### 1. Best choices

| Model | Use | Accuracy (TextVQA) | Time to first word | Energy per question |
|---|---|---:|---:|---:|
| 3B | **Edge-LLM fp8** | 79.6% | ~0.13 s | 23 J |
| 7B | **vLLM bf16** | 83.8% | ~0.46 s | 56 J |
| (3B, for comparison) | HF bf16 | 78.9% | ~0.40 s | 39 J |
| (3B, for comparison) | llama.cpp Q8_0 | 79.9% | ~1.1 s | 72 J |

- **Edge-LLM is about 3× faster** than HF and vLLM and uses about 40% less energy, with the same accuracy.
- **The 7B is about 4.5 points more accurate** than the 3B, for about 0.1 s more per question.
- **llama.cpp is the slowest:** 3–8× slower than the others and about twice the energy.

### 2. All frameworks give the same accuracy

With full-size weights, every framework scores within about 1 point of the others. The small differences come from close calls about when to stop answering ("Stop" vs "Stop Global Warming"), not from one framework being better. Our 3B score matches Qwen's published number (78.9% vs 79.3%); our 7B is 1.9 points below it (83.0% vs 84.9%).

### 3. Compression: the method matters more than the number of bits

| Compression | Accuracy change |
|---|---|
| **8-bit fp8** (Edge-LLM) | none, on every benchmark |
| **4-bit AWQ from Qwen** (vLLM) | small: under 1 point |
| **llama.cpp 4-bit vs 8-bit** | small: 0.2–1.1 points |
| **Edge-LLM's own 4-bit AWQ** | **large: −5.7 on TextVQA, −2.5 on GQA, −49 on MME reasoning** |

Edge-LLM's own 4-bit version was tuned on plain news text only, so it gets worse at reading text in images and at reasoning. **fp8 is free**; if you want 4-bit, use Qwen's official AWQ model.

### 4. Where the time goes

- **Most of the waiting is the image, not the answer.** Before answering, the model turns the photo into about 1,000 "image tokens" with its vision encoder. That step takes 55–70% of the time to the first word. The answers are only 2–4 words, so writing them is quick.
- **Edge-LLM's vision encoder is 3× faster than HF's and 7× faster than llama.cpp's.** That is where its lead comes from.
- **4-bit makes the first word slower, not faster.** 4-bit only speeds up writing the answer word by word; reading a long input (the image) gets slower. **Only fp8 speeds up reading the input.**
- **Hidden cost: preparing the image.** vLLM, llama.cpp and our Edge-LLM setup spend about 0.24 s per question converting each photo to PNG before the model sees it, about as long as the model itself. HF passes the image directly (0.03 s).

### 5. A bug we found and fixed

In Edge-LLM, about **1 in 5 TextVQA answers came back empty**.

- **Cause:** for some larger photos, numbers inside Qwen's vision encoder grow beyond what 16-bit numbers can hold (over 65,504). They turn into "not a number", and the model then outputs nothing. Edge-LLM can only build this part in 16-bit.
- **Fix:** before building the engine, we scale some of the encoder's weights down by 8. This keeps the numbers in range without changing the result, because the model normalizes them again right after. It costs no accuracy and no speed.
- **Result:** no empty answers anymore. We rebuilt the shared engines and re-ran all Edge-LLM tests. As a side effect, MME is now measured correctly too (1,526 instead of 1,448).
- **Prevention:** our result check now rejects any run with 2% or more empty answers, so this kind of silent failure is caught automatically.

## Things to keep in mind

- **Power mode:** the older HF/vLLM MME and GQA runs used the 120 W power mode; TextVQA and the latest Edge-LLM runs used MAXN (full power). Compare which is faster, not exact times, across the two.
- **Coverage:** Edge-LLM was tested only with the 3B model, and llama.cpp only on TextVQA.
- **Build variation:** Edge-LLM speed can vary by about 8% between two builds of the same engine.

## Where to find more

- [REPORT.md](REPORT.md): all the numbers behind these findings, per task and per model.
- [SUMMARY.md](SUMMARY.md): every run, generated automatically; replaced and invalid runs are listed separately, with the reason.
- [NOTES.md](NOTES.md): setup, every problem and its fix, timeline and open items.
