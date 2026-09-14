# MLPerf common utility container

This benchmark builds a small ARM64- or AMD64-capable utility image for work
that should not wait for a full training container. It contains:

- the CUDA runtime and NCCL;
- checksum-pinned NVIDIA DOCA Open MPI and UCX packages matching the framework
  communication stack, plus InfiniBand verbs userspace support;
- MPI-enabled NVIDIA `nccl-tests` binaries;
- the one-rank-per-GPU NCCL dataset-staging program at
  `/usr/local/bin/ncclstage`;
- bounded data movement, preparation, and inspection tools including `rsync`,
  `curl`, `wget`, Git, the OpenSSH client, Python 3, common archive tools,
  `jq`, `iproute2`, `numactl`, and `procps`.

The multi-stage build compiles `nccl-tests` against NVIDIA's Open MPI 5 package
and installs the pinned NVIDIA `cuda.bindings` wheel used by `ncclstage`. Exact
Open MPI and UCX packages are downloaded from the public, versioned DOCA
repository for the selected AMD64 or ARM64 build platform, verified against
architecture-specific SHA-256 digests, and installed normally through `apt`.
The package-selected MPI prefix is exposed through the conventional
`/usr/local/mpi` compatibility link used by NVIDIA framework builds.
Only those two DOCA packages are included; the image does not install a broad
DOCA profile, host drivers, optional NCCL plugins, SHARP, or the standalone
HPC-X bundle. The wheel and its runtime dependencies use a flat application
path; the final image does not
copy a virtual environment, pip, setuptools, activation scripts, or generated
Python bytecode. Full system Python is retained because the readable NCCL
binding uses the standard-library `ctypes` native module.

The final stage verifies exact installed Open MPI and UCX package versions and
MPI executables, rejects an unresolved nccl-tests dynamic dependency, executes
the CLI help path, imports the CUDA wrapper, loads NCCL and prints its version,
and reports regular files, total filesystem entries, and bytes below
`/opt/ncclstage`. These checks catch builder/runtime Python and MPI linkage
mismatches without requiring a GPU. GPU, multi-rank MPI, network, whole-image
inode-count, and Enroot creation-time validation remain cluster tests.
Compilers, development headers, Git source trees, and build intermediates are
not copied from the builder stage. The final image deliberately installs the
Git and OpenSSH clients as runtime utilities, but contains no credentials,
tokens, SSH keys, or registry authentication.

## Build

Build from the repository root. Pass the source revision into the image label
when provenance matters:

```bash
docker build \
  --build-arg GIT_COMMIT_ID="$(git rev-parse HEAD)" \
  --tag mlperf-common-utils:local \
  .
```

The Dockerfile supports both AMD64 and ARM64 CUDA base images. Select the
platform through the builder or build service rather than changing source.
Pinned dependency versions and architecture-specific DOCA package checksums can be
overridden deliberately with Docker build arguments; for example:

```bash
docker build \
  --build-arg NCCL_TESTS_REF=<full-commit-sha> \
  --tag mlperf-common-utils:local \
  .
```

Keep published tags or digests tied to the source revision that produced them.
Image publication is deployment-specific: some clusters can consume a registry
reference directly, while others require the image to be imported into a
shared SquashFS path before a job starts.

The source for the container-only staging application is documented under
[`ncclstage/`](ncclstage/). It is intentionally separate from the installable
`mlperf_common` Python package.

## Runtime scope

The image is a utility environment, not a replacement for a benchmark's exact
framework, SHARP, or NCCL-plugin stack. Its Open MPI and UCX package versions
are pinned independently and must be updated deliberately when the framework
communication stack changes. A successful NCCL test establishes the behavior
of the NCCL and MPI packages in this image. It does not prove that a larger
training image has identical communication behavior or performance.

`ncclstage` uses one rank per GPU, per-local-GPU NCCL communicator lanes, and a
small TCP bootstrap instead of MPI or PMIx. It runs through host Slurm/Pyxis
with explicit shared-source, node-local destination, and manifest mounts.

Prefer direct registry consumption through Pyxis/Enroot for initial validation.
Keep Slurm output in the selected experiment's permanent, job-qualified log
directory. Utility launchers must not remove unrelated Enroot containers.
