# Reproducible cloud-GPU image: builds all three implementations (sbsom, ssom, somoclu) + the
# eval harness into one container, multi-arch so a single image runs on any modern NVIDIA GPU.
#
# Built from a STAGED context (cloud/build.sh copies the three first-party repos in), so no
# GitHub auth is needed for the private repos and the exact local source is pinned. somoclu is
# public and cloned at build time.
#
#   cloud/build.sh                       # stage source + docker build -> sparsesom-repro:latest
#   docker run --gpus all -v $PWD/data:/data sparsesom-repro   # mounts the transferred corpus
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Multi-arch SASS + a PTX fallback → one binary covers T4(75) V100(70) A100(80) A10/A40(86)
# L4/RTX4090(89) H100(90), and JITs forward via the 90-virtual PTX. Override at build with
# --build-arg CUDA_ARCHES=... to trim for a known card (faster build, smaller binary).
ARG CUDA_ARCHES="70-real;75-real;80-real;86-real;89-real;90-real;90-virtual"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ninja-build git ca-certificates \
        gcc-12 g++-12 cmake python3 python3-pip curl \
 && pip install --no-cache-dir numpy scipy \
 && rm -rf /var/lib/apt/lists/*
# gcc-12 host compiler: nvcc 12.8 + gcc-13 trips a _Float32 incompatibility.
ENV CC=gcc-12 CXX=g++-12

WORKDIR /opt
COPY SparseSOM   /opt/SparseSOM
COPY StandardSparseSOM /opt/StandardSparseSOM
COPY SparseBinEval     /opt/SparseBinEval

# sbsom (bespoke fused feature-major SOM: binary + float paths)
RUN cmake -S /opt/SparseSOM -B /opt/SparseSOM/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHES}" \
 && cmake --build /opt/SparseSOM/build -j --target sparsesom

# ssom (cuSPARSE baseline: feature-major + node-major)
RUN cmake -S /opt/StandardSparseSOM -B /opt/StandardSparseSOM/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHES}" \
 && cmake --build /opt/StandardSparseSOM/build -j --target standardsparsesom

# shared cosine evaluator + somoclu (public, CPU-only sparse kernel)
RUN g++ -O3 -fopenmp /opt/SparseBinEval/tools/metrics.cpp -o /opt/SparseBinEval/tools/metrics
RUN /opt/SparseBinEval/tools/build_somoclu.sh /opt/somoclu

ENV SBSOM_BIN=/opt/SparseSOM/build/sparsesom \
    SSOM_BIN=/opt/StandardSparseSOM/build/standardsparsesom \
    SOMOCLU_BIN=/opt/somoclu/somoclu_cpu \
    METRICS_BIN=/opt/SparseBinEval/tools/metrics

WORKDIR /opt/SparseBinEval
ENTRYPOINT ["bash", "/opt/SparseBinEval/cloud/run.sh"]
