# llama.cpp server for Jetson Orin (sm_87), built from a pinned release.
# Uses the same Jetson base as lmms-eval-jetson (CUDA toolkit included), so no extra base image is pulled.
#   docker build -f jetson/frameworks/llamacpp/Dockerfile -t llamacpp-jetson:b11135 jetson/frameworks/llamacpp
FROM ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin

ARG LLAMA_CPP_REF=b11135
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=87 \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build /opt/llama.cpp/build -j"$(nproc)" --target llama-server llama-mtmd-cli llama-bench
# (No `llama-server --version` check here: the CUDA driver is only mounted at `docker run --runtime nvidia`.)

ENV PATH=/opt/llama.cpp/build/bin:$PATH \
    LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:$LD_LIBRARY_PATH
ENTRYPOINT []
CMD ["llama-server", "--help"]
