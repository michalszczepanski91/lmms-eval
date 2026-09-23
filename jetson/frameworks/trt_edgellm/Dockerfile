# lmms-eval + TensorRT Edge-LLM C++ runtime (llm_build, visual_build, llm_inference) for Jetson.
# UNTESTED end to end: engines need ONNX exported on an x86 GPU host or Jetson Thor (see export.sh).
#
# Orin (JetPack 6.2, L4T r36.4) - defaults:
#   docker build -f jetson/frameworks/trt_edgellm/Dockerfile -t lmms-eval-trt-edgellm:latest jetson/frameworks/trt_edgellm
# Thor (JetPack 7): build jetson/Dockerfile on a Thor base first, then e.g.
#   --build-arg BASE_IMAGE=lmms-eval-jetson-thor:latest --build-arg EMBEDDED_TARGET=jetson-thor \
#   --build-arg L4T_RELEASE=<r38.x from /etc/apt/sources.list.d/nvidia-l4t-apt-source.list> --build-arg L4T_SOC=<soc from the same file>
ARG BASE_IMAGE=lmms-eval-jetson:latest
FROM ${BASE_IMAGE}

ARG EDGELLM_REF=v0.6.0
ARG EMBEDDED_TARGET=jetson-orin
ARG L4T_RELEASE=r36.4
ARG L4T_SOC=t234

# TensorRT headers/libs from NVIDIA's L4T apt repo (same versions JetPack installs on the host).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | gpg --dearmor -o /usr/share/keyrings/nvidia-l4t.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-l4t.gpg] https://repo.download.nvidia.com/jetson/common ${L4T_RELEASE} main" > /etc/apt/sources.list.d/nvidia-l4t.list \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-l4t.gpg] https://repo.download.nvidia.com/jetson/${L4T_SOC} ${L4T_RELEASE} main" >> /etc/apt/sources.list.d/nvidia-l4t.list \
    && apt-get update && apt-get install -y --no-install-recommends libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${EDGELLM_REF} https://github.com/NVIDIA/TensorRT-Edge-LLM.git /opt/TensorRT-Edge-LLM \
    && cd /opt/TensorRT-Edge-LLM && git submodule update --init --recursive --depth 1 \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR=/usr \
        -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
        -DEMBEDDED_TARGET=${EMBEDDED_TARGET} \
    && cmake --build build -j"$(nproc)"

ENV PATH=/opt/TensorRT-Edge-LLM/build/examples/llm:/opt/TensorRT-Edge-LLM/build/examples/multimodal:$PATH
