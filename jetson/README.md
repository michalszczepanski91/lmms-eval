# lmms-eval on Jetson Orin

Scripts for running lmms-eval benchmarks on a Jetson AGX Orin 32GB (JetPack 6, L4T R36.4, CUDA 12.6, MAXN).

## Execution stack

Evaluation runs with **PyTorch + Hugging Face transformers** on the Orin GPU, in bf16 with flash-attention 2 (`--model qwen2_5_vl`). This is lmms-eval's reference backend for Qwen2.5-VL, so scores are comparable with published numbers. It does **not** use llama.cpp or TensorRT, because lmms-eval has no backend for either.

Other options, if you ever need them:
- **vLLM** (`--model vllm`) is installed in the same image. It is faster, but it reserves most of the unified memory up front.
- **llama.cpp or TensorRT-LLM** would need an OpenAI-compatible server with a vision projector, called through `--model openai`. The quantized weights then change the scores, so those results measure the deployment, not the model.

The container `lmms-eval-jetson:latest` ([Dockerfile](Dockerfile)) extends `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`, which provides Tegra-built torch 2.10, transformers 4.57.3, flash-attn and vLLM. It adds only lmms-eval's pure-Python dependencies ([requirements-jetson.txt](requirements-jetson.txt)). The repo is bind-mounted at run time, so code edits don't need a rebuild.

## Usage

```bash
docker build -f jetson/Dockerfile -t lmms-eval-jetson:latest .   # once (and after dependency changes)
jetson/download_assets.sh                                        # once: models + MME into /opt/hf-cache
jetson/run_eval.sh 3b mme 8                                      # smoke test, 8 samples
jetson/run_eval.sh 3b mme                                        # full MME
jetson/run_eval.sh 7b mme
```

Runs are offline by default (`OFFLINE=0` to allow downloads). The shared HF cache `/opt/hf-cache` is reused by other projects on the board. Files are written as the calling user and are group-writable for `mlusers`.

## Results layout

```
jetson/results/<model>/<task>/<timestamp>[_limitN]/
  run_info.txt          board, L4T, power mode, git commit, image id, exact command, versions
  run.log               full console output
  tegrastats.log        1 Hz RAM / GPU / power sampling (peak memory → "does it fit")
  <model_dir>/*_results.json          lmms-eval scores + config
  <model_dir>/*_samples_<task>.jsonl  per-sample prompts, outputs, scores
```

Summary of completed full runs: [results/SUMMARY.md](results/SUMMARY.md).
