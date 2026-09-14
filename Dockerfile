# syntax=docker/dockerfile:1

ARG CUDA_VERSION=13.3.0
ARG UBUNTU_VERSION=24.04
ARG NCCL_PACKAGE_VERSION=2.30.7-1+cuda13.3
ARG CUDA_BINDINGS_VERSION=13.3.1
ARG DOCA_VERSION=3.5.0
ARG OPENMPI_PACKAGE_VERSION=5.0.10rc2.2608140153-1.3c5e09e178
ARG UCX_PACKAGE_VERSION=1.22.0.2608140153-1.8a6b06fb8
ARG OPENMPI_SHA256_AARCH64=2678ac4e5e190b874411384481b48c0705d391cd4c61fc9025136dbad27a87ef
ARG OPENMPI_SHA256_X86_64=9466ab2f034261a0bfe533be63af3d74d435c1738f3cf7a0dbb1bd1db7f3139b
ARG UCX_SHA256_AARCH64=4fbbaeacb987ddd73052c4035b8932c7e205fbf99a327c9e1a9f6647501a5ed5
ARG UCX_SHA256_X86_64=3c000ccf8feeb1af8e58bc9d82f3ef0f4538e1f2fadd4b63294fca62d7bae74f

FROM ubuntu:${UBUNTU_VERSION} AS mpi-packages

ARG DOCA_VERSION
ARG OPENMPI_PACKAGE_VERSION
ARG UCX_PACKAGE_VERSION
ARG OPENMPI_SHA256_AARCH64
ARG OPENMPI_SHA256_X86_64
ARG UCX_SHA256_AARCH64
ARG UCX_SHA256_X86_64
ARG TARGETARCH

SHELL ["/bin/bash", "-eux", "-c"]

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && case "${TARGETARCH}" in \
        arm64) repo_arch=arm64-sbsa; deb_arch=arm64; openmpi_sha256="${OPENMPI_SHA256_AARCH64}"; ucx_sha256="${UCX_SHA256_AARCH64}" ;; \
        amd64) repo_arch=x86_64; deb_arch=amd64; openmpi_sha256="${OPENMPI_SHA256_X86_64}"; ucx_sha256="${UCX_SHA256_X86_64}" ;; \
        *) echo "Unsupported DOCA package architecture: ${TARGETARCH}" >&2; exit 1 ;; \
       esac \
    && repo_url="https://linux.mellanox.com/public/repo/doca/${DOCA_VERSION}/ubuntu24.04/${repo_arch}/pool" \
    && mkdir -p /packages \
    && curl --fail --location \
        "${repo_url}/openmpi_${OPENMPI_PACKAGE_VERSION}_${deb_arch}.deb" \
        --output /packages/openmpi.deb \
    && curl --fail --location \
        "${repo_url}/ucx_${UCX_PACKAGE_VERSION}_${deb_arch}.deb" \
        --output /packages/ucx.deb \
    && echo "${openmpi_sha256}  /packages/openmpi.deb" | sha256sum --check - \
    && echo "${ucx_sha256}  /packages/ucx.deb" | sha256sum --check -

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG NCCL_PACKAGE_VERSION
ARG CUDA_BINDINGS_VERSION
ARG NCCL_TESTS_REF=717b68318278e93f371d8ffb46b076069d7c7851
ARG OPENMPI_PACKAGE_VERSION
ARG UCX_PACKAGE_VERSION

SHELL ["/bin/bash", "-eux", "-c"]

COPY --from=mpi-packages /packages /tmp/mpi-packages

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        /tmp/mpi-packages/openmpi.deb \
        /tmp/mpi-packages/ucx.deb \
        ca-certificates \
        git \
        libnccl2="${NCCL_PACKAGE_VERSION}" \
        libnccl-dev="${NCCL_PACKAGE_VERSION}" \
        make \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* /tmp/mpi-packages \
    && test "$(dpkg-query -W -f='${Version}' openmpi)" = "${OPENMPI_PACKAGE_VERSION}" \
    && test "$(dpkg-query -W -f='${Version}' ucx)" = "${UCX_PACKAGE_VERSION}" \
    && mpi_home="$(dirname "$(dirname "$(readlink -f /usr/bin/mpicc)")")" \
    && ln -s "${mpi_home}" /usr/local/mpi \
    && test -x /usr/local/mpi/bin/mpicc

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
ARG DOCA_VERSION
ARG OPENMPI_PACKAGE_VERSION
ARG UCX_PACKAGE_VERSION
ARG OPENMPI_SHA256_AARCH64
ARG OPENMPI_SHA256_X86_64
ARG UCX_SHA256_AARCH64
ARG UCX_SHA256_X86_64
ARG TARGETARCH

LABEL org.opencontainers.image.title="MLPerf common utilities" \
      org.opencontainers.image.description="Small CUDA, NCCL, MPI, RDMA, and data-movement utility image" \
      org.opencontainers.image.revision="${GIT_COMMIT_ID}" \
      org.opencontainers.image.source="https://github.com/NVIDIA/mlperf-common" \
      com.nvidia.mlperf.nccl-package-version="${NCCL_PACKAGE_VERSION}" \
      com.nvidia.mlperf.cuda-bindings-version="${CUDA_BINDINGS_VERSION}" \
      com.nvidia.mlperf.nccl-tests-revision="${NCCL_TESTS_REF}" \
      com.nvidia.mlperf.doca-version="${DOCA_VERSION}" \
      com.nvidia.mlperf.openmpi-package-version="${OPENMPI_PACKAGE_VERSION}" \
      com.nvidia.mlperf.ucx-package-version="${UCX_PACKAGE_VERSION}" \
      com.nvidia.mlperf.openmpi-sha256-aarch64="${OPENMPI_SHA256_AARCH64}" \
      com.nvidia.mlperf.openmpi-sha256-x86-64="${OPENMPI_SHA256_X86_64}" \
      com.nvidia.mlperf.ucx-sha256-aarch64="${UCX_SHA256_AARCH64}" \
      com.nvidia.mlperf.ucx-sha256-x86-64="${UCX_SHA256_X86_64}" \
      com.nvidia.mlperf.mpi-architecture="${TARGETARCH}"

COPY --from=mpi-packages /packages /tmp/mpi-packages

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        /tmp/mpi-packages/openmpi.deb \
        /tmp/mpi-packages/ucx.deb \
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
    && rm -rf /var/lib/apt/lists/* /tmp/mpi-packages \
    && test "$(dpkg-query -W -f='${Version}' openmpi)" = "${OPENMPI_PACKAGE_VERSION}" \
    && test "$(dpkg-query -W -f='${Version}' ucx)" = "${UCX_PACKAGE_VERSION}" \
    && mpi_home="$(dirname "$(dirname "$(readlink -f /usr/bin/mpicc)")")" \
    && ln -s "${mpi_home}" /usr/local/mpi \
    && test -x /usr/local/mpi/bin/mpicc

COPY --from=builder /src/nccl-tests/build/*_perf_mpi /usr/local/bin/
COPY --from=builder /opt/ncclstage /opt/ncclstage

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/opt/ncclstage/app:/opt/ncclstage/python
ENV PATH=/usr/local/mpi/bin:${PATH}

RUN ln -s /opt/ncclstage/app/ncclstage.py /usr/local/bin/ncclstage \
    && chmod 0755 /opt/ncclstage/app/ncclstage.py \
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
