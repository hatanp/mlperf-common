# ncclstage

`ncclstage` copies a shared dataset to node-local storage with one Slurm rank
per GPU and no MPI dependency. Local GPU ordinal `g` joins a communicator with
ordinal `g` on every node. Each communicator owns a disjoint range of every
file, gathers the pieces read by its nodes, and writes that range into the one
node-local destination file.

The implementation uses a compact three-slot software pipeline. One reader
thread fills the next pinned-host segment, the main thread preserves collective
order while running the current H2D, NCCL AllGather, D2H, and stream
synchronization, and one writer thread drains the previous assembled window.
This overlaps shared-filesystem reads and node-local writes without introducing
multiple NCCL streams or changing collective order. Python owns manifest
parsing, TCP bootstrap, file I/O, and offsets; NVIDIA `cuda.bindings` owns CUDA
resources, and a small `ctypes` module exposes only the required NCCL calls.
The program is container-specific and is not installed as part of the
`mlperf_common` Python package.

The launcher must start exactly `gpus-per-node * node-count` tasks and provide
the normal Slurm rank variables. Rank zero distributes one NCCL unique ID per
local-GPU lane through the explicitly supplied bootstrap address and port. If
all GPUs are visible to every task, `SLURM_LOCALID` selects the device. If
Slurm masks every task to one GPU, local CUDA ordinal zero is selected.

An optional priority manifest lets one invocation and one set of communicators
stage two ordered groups of files. `--priority-manifest` is completed first;
after a collective file-boundary barrier, global rank zero atomically publishes
`--ready-file`. The process then continues with `--manifest` without repeating
Python imports, CUDA initialization, TCP bootstrap, or NCCL communicator
creation. The DSv3 launcher uses this mode to make the RAM-resident training
image available to Pyxis while the same staging process proceeds with the
dataset.

The default 256 MiB assembled window allocates roughly 256 MiB of receive
storage plus one node-sized segment in both pinned host and GPU memory per
rank and per pipeline slot. Three slots therefore use roughly 768 MiB of pinned
host receive storage per rank, while the GPU allocation remains one receive
window per rank. `--window` can reduce that bound for constrained systems.

After all manifest files complete, every rank emits one `NCCLSTAGE_TIMING`
record. It reports cumulative stage wall time and submission-to-completion
totals for every reader and writer operation. Those totals include executor
queueing as well as worker execution. The existing path counters retain the
active pinned-buffer zero-fill, `pread` and Python-to-pinned-buffer copy, and
pinned-buffer-to-Python materialization plus `pwrite` subphases. The combined
H2D/AllGather/D2H/synchronize block is labeled as a total. Main-thread future
waits are explicitly labeled `read_blocked_ms` and `write_blocked_ms`; they are
pipeline backpressure diagnostics, not operation durations. Timers use
monotonic wall time, add no CUDA synchronization, and do not log per round.
Cumulative operation totals may sum to more than stage wall time because the
pipeline deliberately overlaps them.

`NCCLSTAGE_PHASE` records expose process startup, CUDA-module import,
NCCL-wrapper import, CUDA device discovery and selection, NCCL library load,
TCP bootstrap, communicator initialization, each manifest payload, readiness,
and communicator close. These are per-rank wall boundaries; they make rank
skew visible without interpreting a blocked main-thread wait as device or I/O
execution time.

## Unit tests

The CPU-only logic and single-rank fake-CUDA path can be tested from the
repository root without building the image:

```bash
python3 -m unittest container/ncclstage/test_ncclstage.py
```

The TCP bootstrap test requires permission to open a loopback listener and is
skipped when the local sandbox does not allow one.
