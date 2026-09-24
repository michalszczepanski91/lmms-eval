# Running the benchmarks on Jetson Thor

How to repeat the Orin experiments (HF, vLLM and llama.cpp on MME, the follow-ups, and TensorRT Edge-LLM) on a Jetson AGX Thor. None of this has been run on Thor yet. Wherever Thor differs from Orin, the step below says what to check.

**What differs from Orin**

| | Orin (tested) | Thor |
|---|---|---|
| JetPack / L4T | 6.2 / R36.4 | 7.x / R38+ |
| CUDA / GPU arch | 12.6 / sm_87 | 13.x / sm_110 |
| Memory | 32 GB unified | 128 GB unified |
| Base image | `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin` | `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` |
| TensorRT Edge-LLM | runtime only (export needs x86) | export **and** runtime on the device; FP8 and NVFP4 available |

## 1. Prepare the board

```bash
sudo nvpmodel -m 0 && nvpmodel -q          # MAXN, same as the Orin runs (jetson_clocks was NOT used on Orin)
docker info | grep -i runtime               # needs the nvidia runtime; add yourself to the docker group if needed
git clone -b jetson-benchmark https://github.com/michalszczepanski91/lmms-eval.git && cd lmms-eval
```

Pick a cache location with about 100 GB free for models and datasets, and export it for every command below. If you have a shared group like `mlusers` on Orin, set `SHARED_GROUP`; if the group doesn't exist, the scripts simply skip it.

```bash
export HF_CACHE=/opt/hf-cache               # any writable dir; created files stay yours
export RESULTS_DIR=jetson/results-thor      # keeps Thor results separate from Orin's jetson/results
export SHARED_GROUP=mlusers                 # optional
```

## 2. Build the images for Thor

```bash
docker build --build-arg BASE_IMAGE=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
  -f jetson/Dockerfile -t lmms-eval-jetson:latest .
docker build --build-arg BASE_IMAGE=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor --build-arg CUDA_ARCH=110 \
  -f jetson/frameworks/llamacpp/Dockerfile -t llamacpp-jetson:b11135 jetson/frameworks/llamacpp
```

Check what the Thor base image provides, since its versions will differ from Orin's (torch 2.10, transformers 4.57.3, vLLM 0.19):

```bash
docker run --rm --runtime nvidia lmms-eval-jetson:latest python -c "
import torch, transformers, vllm; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(), transformers.__version__, vllm.__version__)
import flash_attn; print('flash_attn', flash_attn.__version__)"
```

- If `flash_attn` fails to import, run HF with `ATTN=sdpa` (for example `ATTN=sdpa jetson/run_eval.sh hf 3b mme`) and note it when comparing with Orin.
- If the `pip install` step of the lmms-eval build fails on a version pin, the Thor image ships different versions. Loosen the conflicting line in `jetson/requirements-jetson.txt`, as was done for `av` on Orin.
- If llama.cpp fails to compile for `110`, check `nvcc --list-gpu-arch` inside the base image and pass the architecture it reports.

## 3. Download models and data (same commands as Orin)

```bash
for spec in "hf 3b" "hf 7b" "vllm 3b-awq" "vllm 7b-awq" "llamacpp 3b-q8_0" "llamacpp 3b-q4_k_m" "llamacpp 7b-q8_0" "llamacpp 7b-q4_k_m"; do
  jetson/download_assets.sh $spec
done
jetson/download_assets.sh dataset:lmms-lab-encoder/LMMs-Eval-Lite:coco2017_cap_val
```

## 4. Smoke test, then the main comparison

```bash
jetson/run_matrix.sh mme 8        # every framework/model on 8 samples; all should finish and look like Orin's smoke runs
jetson/run_matrix.sh mme          # full MME: hf, vllm, llamacpp x 3B/7B x precisions (~6-10 h on Orin; Thor should be faster)
```

`run_matrix.sh` writes `jetson/results-thor/SUMMARY.md` when it finishes. Rebuild the summary at any time with `python3 jetson/summarize.py`, keeping `RESULTS_DIR` exported.

Thor has 4× Orin's memory. The vLLM memory fractions in `jetson/frameworks/vllm.sh` (0.45, or 0.7 for 7B bf16) then reserve about 58–77 GB, which is more than needed but harmless. To keep the KV-cache budget similar to Orin, set `VLLM_GPU_MEM=0.15`.

## 5. Follow-up experiments

```bash
jetson/experiments/followups_20260923.sh
```

This runs the same image-transport runs, COCO captioning and resolution sweep as on Orin, logging to `$RESULTS_DIR/followups_20260923.log`. Its results go to `$RESULTS_DIR` like any other run.

## 6. TensorRT Edge-LLM (Thor can do everything on the device)

1. **Export to ONNX** on Thor, using the Edge-LLM release and PyTorch container that NVIDIA lists for your JetPack 7 version (see the [Edge-LLM installation guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html); Jetson AI Lab currently uses `nvcr.io/nvidia/pytorch:26.05-py3`):
   ```bash
   export TRT_WORKSPACE=/opt/models/trt-edgellm
   for p in fp16 int4_awq fp8 nvfp4; do
     GPU_FLAGS="--runtime nvidia" IMAGE=nvcr.io/nvidia/pytorch:26.05-py3 EDGELLM_REF=<release for JetPack 7> \
       jetson/frameworks/trt_edgellm/export.sh 3b $p $TRT_WORKSPACE
   done
   ```
   `export.sh` writes to `$TRT_WORKSPACE/<model>-<precision>/onnx`, which is where `build_engines.sh` looks, so nothing needs copying. FP8 export needs a lot of CPU RAM (up to about 20× the model size), so the 128 GB on Thor is enough for 3B and 7B.
2. **Build the runtime image.** Take `L4T_RELEASE` and `L4T_SOC` from `/etc/apt/sources.list.d/nvidia-l4t-apt-source.list` (the `rXX.Y` part and the SoC directory):
   ```bash
   docker build --build-arg BASE_IMAGE=lmms-eval-jetson:latest --build-arg EMBEDDED_TARGET=jetson-thor \
     --build-arg EDGELLM_REF=<same release> --build-arg L4T_RELEASE=<rXX.Y> --build-arg L4T_SOC=<soc> \
     -f jetson/frameworks/trt_edgellm/Dockerfile -t lmms-eval-trt-edgellm:latest jetson/frameworks/trt_edgellm
   ```
   If `apt-get install libnvinfer-dev` fails, check the host's package names with `dpkg -l | grep -i nvinfer` and adjust the Dockerfile.
3. **Build the engines and evaluate:**
   ```bash
   for p in fp16 int4_awq fp8 nvfp4; do
     jetson/frameworks/trt_edgellm/build_engines.sh 3b $p
     jetson/run_eval.sh trt_edgellm 3b-$p mme 8 && jetson/run_eval.sh trt_edgellm 3b-$p mme
   done
   ```
   TensorRT latency is reported as averages from the runner profile (`trt_profile.json`), not per sample.

## 7. Bring the results back

```bash
python3 jetson/summarize.py
git add jetson/results-thor && git commit -m "results: Jetson Thor" && git push
```

Each run's `run_info.txt` records the board, L4T, power mode, image IDs and exact command, so Orin and Thor runs can be compared line by line. Compare like with like: the same framework, precision and image transport (`+pil` / `+png1`), and the same power mode.
