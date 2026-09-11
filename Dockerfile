# syntax=docker/dockerfile:1

ARG CUDA_VERSION=13.3.0
ARG UBUNTU_VERSION=24.04
ARG NCCL_PACKAGE_VERSION=2.30.7-1+cuda13.3
ARG CUDA_BINDINGS_VERSION=13.3.1

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG NCCL_PACKAGE_VERSION
ARG CUDA_BINDINGS_VERSION
ARG NCCL_TESTS_REF=717b68318278e93f371d8ffb46b076069d7c7851

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        libnccl-dev="${NCCL_PACKAGE_VERSION}" \
        libopenmpi-dev \
        make \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
        --no-cache-dir \
        --no-compile \
        --target /opt/ncclstage/python \
        "cuda-bindings==${CUDA_BINDINGS_VERSION}"

RUN git clone https://github.com/NVIDIA/nccl-tests.git /src/nccl-tests \
    && git -C /src/nccl-tests checkout --detach "${NCCL_TESTS_REF}" \
    && test "$(git -C /src/nccl-tests rev-parse HEAD)" = "${NCCL_TESTS_REF}" \
    && mpi_home="/usr/lib/$(gcc -print-multiarch)/openmpi" \
    && make -C /src/nccl-tests \
        -j"$(nproc)" \
        MPI=1 \
        MPI_HOME="${mpi_home}" \
        MPI_INCLUDE="${mpi_home}/include" \
        NAME_SUFFIX=_mpi

COPY container/ncclstage/ncclstage.py \
     container/ncclstage/cuda_api.py \
     container/ncclstage/nccl_api.py \
     /opt/ncclstage/app/

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

ARG GIT_COMMIT_ID
ARG NCCL_PACKAGE_VERSION
ARG CUDA_BINDINGS_VERSION
ARG NCCL_TESTS_REF=717b68318278e93f371d8ffb46b076069d7c7851

LABEL org.opencontainers.image.title="MLPerf common utilities" \
      org.opencontainers.image.description="Small CUDA, NCCL, MPI, RDMA, and data-movement utility image" \
      org.opencontainers.image.revision="${GIT_COMMIT_ID}" \
      org.opencontainers.image.source="https://github.com/NVIDIA/mlperf-common" \
      com.nvidia.mlperf.nccl-package-version="${NCCL_PACKAGE_VERSION}" \
      com.nvidia.mlperf.cuda-bindings-version="${CUDA_BINDINGS_VERSION}" \
      com.nvidia.mlperf.nccl-tests-revision="${NCCL_TESTS_REF}"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        file \
        git \
        ibverbs-providers \
        ibverbs-utils \
        iproute2 \
        iputils-ping \
        jq \
        libibverbs1 \
        libnccl2="${NCCL_PACKAGE_VERSION}" \
        netcat-openbsd \
        numactl \
        openmpi-bin \
        openssh-client \
        pciutils \
        pigz \
        procps \
        python3 \
        rsync \
        tar \
        unzip \
        wget \
        xz-utils \
        zstd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/nccl-tests/build/*_perf_mpi /usr/local/bin/
COPY --from=builder /opt/ncclstage /opt/ncclstage

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/opt/ncclstage/app:/opt/ncclstage/python

RUN ln -s /opt/ncclstage/app/ncclstage.py /usr/local/bin/ncclstage \
    && chmod 0755 /opt/ncclstage/app/ncclstage.py \
    && /usr/local/bin/ncclstage --help >/dev/null \
    && python3 -c 'import cuda_api; from nccl_api import Nccl; print(f"NCCL {Nccl().version()}")' \
    && printf 'ncclstage_runtime_files=' \
    && find /opt/ncclstage -xdev -type f | wc -l \
    && printf 'ncclstage_runtime_entries=' \
    && find /opt/ncclstage -xdev -printf . | wc -c \
    && printf 'ncclstage_runtime_bytes=' \
    && du --bytes --summarize /opt/ncclstage | cut --fields=1

ENV NCCL_DEBUG=WARN
ENV GIT_COMMIT_ID=${GIT_COMMIT_ID}

CMD ["/bin/bash"]
