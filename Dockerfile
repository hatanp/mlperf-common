# syntax=docker/dockerfile:1

ARG CUDA_VERSION=13.3.0
ARG UBUNTU_VERSION=24.04
ARG NCCL_PACKAGE_VERSION=2.30.7-1+cuda13.3
ARG CUDA_BINDINGS_VERSION=13.3.1
ARG HPCX_VERSION=v2.24.1
ARG HPCX_SHA256_AARCH64=0bc5c26a4f0ca98fd6292aac9d51cc2a4ee3277d38d4011b64eafd133be633b4
ARG HPCX_SHA256_X86_64=b34ae9a65ef653be6590147b747091c3c3753210bc98ad74a2304e6d95d5795b

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS hpcx

ARG HPCX_VERSION
ARG HPCX_SHA256_AARCH64
ARG HPCX_SHA256_X86_64
ARG TARGETARCH

SHELL ["/bin/bash", "-eux", "-c"]

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        bzip2 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && case "${TARGETARCH}" in \
        arm64) hpcx_arch=aarch64; hpcx_sha256="${HPCX_SHA256_AARCH64}" ;; \
        amd64) hpcx_arch=x86_64; hpcx_sha256="${HPCX_SHA256_X86_64}" ;; \
        *) echo "Unsupported HPC-X architecture: ${TARGETARCH}" >&2; exit 1 ;; \
       esac \
    && hpcx_url="https://content.mellanox.com/hpc/hpc-x/${HPCX_VERSION}_cuda13/hpcx-${HPCX_VERSION}-gcc-doca_ofed-ubuntu24.04-cuda13-${hpcx_arch}.tbz" \
    && curl --fail --location "${hpcx_url}" --output /tmp/hpcx.tbz \
    && echo "${hpcx_sha256}  /tmp/hpcx.tbz" | sha256sum --check - \
    && mkdir -p /opt/hpcx \
    && tar -xf /tmp/hpcx.tbz --strip-components=1 --directory=/opt/hpcx \
    && rm /tmp/hpcx.tbz \
    && test "$(cat /opt/hpcx/VERSION | head -n 1)" = "HPC-X ${HPCX_VERSION}" \
    && test -x /opt/hpcx/ompi/bin/mpicc

FROM hpcx AS builder

ARG NCCL_PACKAGE_VERSION
ARG CUDA_BINDINGS_VERSION
ARG NCCL_TESTS_REF=717b68318278e93f371d8ffb46b076069d7c7851

SHELL ["/bin/bash", "-eux", "-c"]

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        libnccl2="${NCCL_PACKAGE_VERSION}" \
        libnccl-dev="${NCCL_PACKAGE_VERSION}" \
        make \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /opt/hpcx/ompi /usr/local/mpi

RUN python3 -m pip install \
        --no-cache-dir \
        --no-compile \
        --target /opt/ncclstage/python \
        "cuda-bindings==${CUDA_BINDINGS_VERSION}"

RUN git clone https://github.com/NVIDIA/nccl-tests.git /src/nccl-tests \
    && git -C /src/nccl-tests checkout --detach "${NCCL_TESTS_REF}" \
    && test "$(git -C /src/nccl-tests rev-parse HEAD)" = "${NCCL_TESTS_REF}" \
    && make -C /src/nccl-tests \
        -j"$(nproc)" \
        MPI=1 \
        MPI_HOME=/usr/local/mpi \
        MPI_INCLUDE=/usr/local/mpi/include \
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
ARG HPCX_VERSION
ARG HPCX_SHA256_AARCH64
ARG HPCX_SHA256_X86_64
ARG TARGETARCH

LABEL org.opencontainers.image.title="MLPerf common utilities" \
      org.opencontainers.image.description="Small CUDA, NCCL, MPI, RDMA, and data-movement utility image" \
      org.opencontainers.image.revision="${GIT_COMMIT_ID}" \
      org.opencontainers.image.source="https://github.com/NVIDIA/mlperf-common" \
      com.nvidia.mlperf.nccl-package-version="${NCCL_PACKAGE_VERSION}" \
      com.nvidia.mlperf.cuda-bindings-version="${CUDA_BINDINGS_VERSION}" \
      com.nvidia.mlperf.nccl-tests-revision="${NCCL_TESTS_REF}" \
      com.nvidia.mlperf.hpcx-version="${HPCX_VERSION}" \
      com.nvidia.mlperf.hpcx-architecture="${TARGETARCH}" \
      com.nvidia.mlperf.hpcx-sha256-aarch64="${HPCX_SHA256_AARCH64}" \
      com.nvidia.mlperf.hpcx-sha256-x86-64="${HPCX_SHA256_X86_64}"

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

COPY --from=hpcx /opt/hpcx /opt/hpcx
COPY --from=builder /src/nccl-tests/build/*_perf_mpi /usr/local/bin/
COPY --from=builder /opt/ncclstage /opt/ncclstage

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/opt/ncclstage/app:/opt/ncclstage/python
ENV PATH=/usr/local/mpi/bin:${PATH}

RUN ln -s /opt/hpcx/ompi /usr/local/mpi \
    && ln -s /opt/ncclstage/app/ncclstage.py /usr/local/bin/ncclstage \
    && chmod 0755 /opt/ncclstage/app/ncclstage.py \
    && test "$(cat /opt/hpcx/VERSION | head -n 1)" = "HPC-X ${HPCX_VERSION}" \
    && test -x "$(command -v mpirun)" \
    && ompi_info --version \
    && for binary in /usr/local/bin/*_perf_mpi; do \
           echo "Dynamic dependencies for ${binary}"; \
           ldd "${binary}" > /tmp/nccl-test-ldd; \
           cat /tmp/nccl-test-ldd; \
           if grep -q 'not found' /tmp/nccl-test-ldd; then exit 1; fi; \
       done \
    && rm /tmp/nccl-test-ldd \
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
