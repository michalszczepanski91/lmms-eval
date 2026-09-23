# llama.cpp server for Jetson, built from a pinned release (default: Orin, sm_87).
# Uses the same Jetson base as lmms-eval-jetson (CUDA toolkit included), so no extra base image is pulled.
#   docker build -f jetson/frameworks/llamacpp/Dockerfile -t llamacpp-jetson:b11135 jetson/frameworks/llamacpp
# Thor: --build-arg BASE_IMAGE=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor --build-arg CUDA_ARCH=110
ARG BASE_IMAGE=ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin
FROM ${BASE_IMAGE}

ARG LLAMA_CPP_REF=b11135
ARG CUDA_ARCH=87
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build /opt/llama.cpp/build -j"$(nproc)" --target llama-server llama-mtmd-cli llama-bench
# (No `llama-server --version` check here: the CUDA driver is only mounted at `docker run --runtime nvidia`.)

ENV PATH=/opt/llama.cpp/build/bin:$PATH \
    LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:$LD_LIBRARY_PATH
ENTRYPOINT []
CMD ["llama-server", "--help"]
