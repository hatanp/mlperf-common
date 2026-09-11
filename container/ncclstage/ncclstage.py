#!/usr/bin/python3

"""Stage shared files to node-local storage through per-GPU NCCL lanes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import socket
import struct
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


UNIQUE_ID_BYTES = 128
REQUEST = struct.Struct("!4sII")
MAGIC = b"NCS1"
PIPELINE_SLOTS = 3


@dataclass(frozen=True)
class Topology:
    global_rank: int
    world_size: int
    local_rank: int
    node_rank: int
    nodes: int
    gpus_per_node: int

    @property
    def lane(self) -> int:
        return self.local_rank


@dataclass
class PipelineSlot:
    send_host: object
    receive_host: object
    read_future: Future[TimedIO] | None = None
    write_future: Future[TimedIO] | None = None


@dataclass(frozen=True)
class TimedIO:
    bytes: int
    io_ns: int
    zero_ns: int = 0
    total_ns: int = 0
    prepare_ns: int = 0
    lock_wait_ns: int = 0
    syscall_ns: int = 0


@dataclass
class StageMetrics:
    files: int = 0
    rounds: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    stage_ns: int = 0
    zero_ns: int = 0
    read_ns: int = 0
    read_total_ns: int = 0
    gpu_ns: int = 0
    write_ns: int = 0
    write_prepare_ns: int = 0
    write_lock_wait_ns: int = 0
    write_syscall_ns: int = 0
    write_total_ns: int = 0
    read_wait_ns: int = 0
    write_wait_ns: int = 0


def format_metrics(rank: int, metrics: StageMetrics) -> str:
    def milliseconds(value: int) -> str:
        return f"{value / 1_000_000:.3f}"

    return (
        f"NCCLSTAGE_TIMING rank={rank} files={metrics.files} "
        f"rounds={metrics.rounds} read_bytes={metrics.read_bytes} "
        f"write_bytes={metrics.write_bytes} stage_ms={milliseconds(metrics.stage_ns)} "
        f"zero_fill_ms={milliseconds(metrics.zero_ns)} "
        f"read_path_ms={milliseconds(metrics.read_ns)} "
        f"read_total_ms={milliseconds(metrics.read_total_ns)} "
        f"gpu_collective_total_ms={milliseconds(metrics.gpu_ns)} "
        f"write_path_ms={milliseconds(metrics.write_ns)} "
        f"write_prepare_ms={milliseconds(metrics.write_prepare_ns)} "
        f"write_lock_wait_ms={milliseconds(metrics.write_lock_wait_ns)} "
        f"write_syscall_ms={milliseconds(metrics.write_syscall_ns)} "
        f"write_total_ms={milliseconds(metrics.write_total_ns)} "
        f"read_blocked_ms={milliseconds(metrics.read_wait_ns)} "
        f"write_blocked_ms={milliseconds(metrics.write_wait_ns)}"
    )


def phase_marker(
    rank: int, phase: str, event: str, start_ns: int | None = None
) -> int:
    """Print an absolute boundary that can be aligned with launcher timers."""

    now_ns = time.perf_counter_ns()
    elapsed = "" if start_ns is None else f" elapsed_ms={(now_ns - start_ns) / 1e6:.3f}"
    print(
        f"NCCLSTAGE_PHASE rank={rank} phase={phase} event={event} "
        f"epoch_ms={time.time_ns() // 1_000_000}{elapsed}",
        flush=True,
    )
    return now_ns


def partition(size: int, parts: int, rank: int) -> tuple[int, int]:
    """Return a balanced, contiguous half-open range."""
    if size < 0 or parts < 1 or not 0 <= rank < parts:
        raise ValueError("invalid partition arguments")
    start = size * rank // parts
    end = size * (rank + 1) // parts
    return start, end


def parse_size(value: str) -> int:
    suffixes = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    text = value.strip().upper()
    suffix = text[-1] if text and text[-1] in suffixes else ""
    number = text[:-1] if suffix else text
    try:
        size = int(number) * suffixes[suffix]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value}") from error
    if size < 1:
        raise argparse.ArgumentTypeError("byte size must be positive")
    return size


def topology_from_environment(gpus_per_node: int, expect_nodes: int) -> Topology:
    def required(name: str) -> int:
        try:
            return int(os.environ[name])
        except (KeyError, ValueError) as error:
            raise RuntimeError(f"valid {name} is required") from error

    topology = Topology(
        global_rank=required("SLURM_PROCID"),
        world_size=required("SLURM_NTASKS"),
        local_rank=required("SLURM_LOCALID"),
        node_rank=required("SLURM_NODEID"),
        nodes=required("SLURM_NNODES"),
        gpus_per_node=gpus_per_node,
    )
    if topology.nodes != expect_nodes:
        raise RuntimeError(f"expected {expect_nodes} nodes, Slurm reports {topology.nodes}")
    if topology.world_size != topology.nodes * gpus_per_node:
        raise RuntimeError(
            f"expected {topology.nodes * gpus_per_node} tasks, Slurm reports "
            f"{topology.world_size}"
        )
    if not 0 <= topology.global_rank < topology.world_size:
        raise RuntimeError(f"invalid global rank {topology.global_rank}")
    if not 0 <= topology.local_rank < gpus_per_node:
        raise RuntimeError(f"invalid local rank {topology.local_rank}")
    if not 0 <= topology.node_rank < topology.nodes:
        raise RuntimeError(f"invalid node rank {topology.node_rank}")
    return topology


def select_cuda_device(visible_devices: int, topology: Topology) -> int:
    if visible_devices == topology.gpus_per_node:
        return topology.local_rank
    if visible_devices == 1:
        return 0
    raise RuntimeError(
        f"rank {topology.global_rank} sees {visible_devices} CUDA devices; expected "
        f"1 or {topology.gpus_per_node}"
    )


def receive_exact(connection: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = connection.recv(count - len(chunks))
        if not chunk:
            raise ConnectionError("bootstrap peer closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


class BootstrapServer:
    def __init__(
        self,
        host: str,
        port: int,
        identifiers: list[bytes],
        topology: Topology,
        timeout: float,
    ) -> None:
        self.identifiers = identifiers
        self.topology = topology
        self.error: BaseException | None = None
        self.seen = {(topology.lane, topology.node_rank)}
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(timeout)
            self.socket.bind((host, port))
            self.socket.listen(min(topology.world_size, 128))
        except BaseException:
            self.socket.close()
            raise
        self.thread = threading.Thread(
            target=self._serve, name="nccl-bootstrap", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def _serve(self) -> None:
        try:
            for _ in range(self.topology.world_size - 1):
                connection, _ = self.socket.accept()
                with connection:
                    magic, lane, node_rank = REQUEST.unpack(
                        receive_exact(connection, REQUEST.size)
                    )
                    key = (lane, node_rank)
                    if magic != MAGIC or not 0 <= lane < self.topology.gpus_per_node:
                        raise RuntimeError("invalid NCCL bootstrap request")
                    if not 0 <= node_rank < self.topology.nodes or key in self.seen:
                        raise RuntimeError(f"duplicate or invalid NCCL rank {key}")
                    self.seen.add(key)
                    connection.sendall(self.identifiers[lane])
        except BaseException as error:
            self.error = error
        finally:
            self.socket.close()

    def finish(self) -> None:
        self.thread.join()
        if self.error:
            raise RuntimeError("NCCL bootstrap server failed") from self.error


def fetch_identifier(
    host: str, port: int, topology: Topology, timeout: float
) -> bytes:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=min(timeout, 5.0)) as connection:
                connection.settimeout(timeout)
                connection.sendall(REQUEST.pack(MAGIC, topology.lane, topology.node_rank))
                return receive_exact(connection, UNIQUE_ID_BYTES)
        except OSError as error:
            last_error = error
            time.sleep(0.2)
    raise TimeoutError(f"could not reach NCCL bootstrap at {host}:{port}") from last_error


def load_manifest(path: str) -> list[tuple[str, str]]:
    entries = []
    with open(path, encoding="utf-8") as manifest:
        for line_number, raw_line in enumerate(manifest, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected source and destination")
            entries.append((fields[0], fields[1]))
    if not entries:
        raise ValueError(f"manifest is empty: {path}")
    return entries


def marker_path(destination: str, job_id: str, kind: str, rank: int | None = None) -> str:
    digest = hashlib.sha256(destination.encode()).hexdigest()[:16]
    suffix = f".{rank}" if rank is not None else ""
    return f"{destination}.ncclstage-{job_id}-{digest}.{kind}{suffix}"


def temporary_path(destination: str, job_id: str) -> str:
    digest = hashlib.sha256(destination.encode()).hexdigest()[:16]
    return f"{destination}.ncclstage-{job_id}-{digest}.tmp"


def wait_for(path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {path}")


def prepare_destination(destination: str, size: int, topology: Topology, timeout: float) -> int:
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    ready = marker_path(destination, job_id, "ready")
    temporary = temporary_path(destination, job_id)
    if topology.local_rank == 0:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        for rank in range(topology.gpus_per_node):
            Path(marker_path(destination, job_id, "done", rank)).unlink(missing_ok=True)
        Path(ready).unlink(missing_ok=True)
        Path(temporary).unlink(missing_ok=True)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        try:
            os.ftruncate(fd, size)
            # Every local GPU rank must be able to write its lane, but dataset
            # files do not need execute bits.
            os.fchmod(fd, 0o666)
        finally:
            os.close(fd)
        Path(ready).touch()
    else:
        wait_for(ready, timeout)
    return os.open(temporary, os.O_WRONLY)


def finish_destination(destination: str, topology: Topology, timeout: float) -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    done = marker_path(destination, job_id, "done", topology.local_rank)
    Path(done).touch()
    if topology.local_rank == 0:
        for rank in range(topology.gpus_per_node):
            wait_for(marker_path(destination, job_id, "done", rank), timeout)
        os.replace(temporary_path(destination, job_id), destination)
        parent_fd = os.open(str(Path(destination).parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        for rank in range(topology.gpus_per_node):
            Path(marker_path(destination, job_id, "done", rank)).unlink()
        Path(marker_path(destination, job_id, "ready")).unlink()


def agree_file_metadata(source: str, source_stat, topology: Topology, communicator, cuda) -> None:
    """Fail collectively when nodes do not see the same source file identity."""
    payload = struct.pack("!QQ", source_stat.st_size, source_stat.st_mtime_ns)
    send_host = cuda.HostBuffer(len(payload))
    receive_host = cuda.HostBuffer(len(payload) * topology.nodes)
    send_device = cuda.DeviceBuffer(len(payload))
    receive_device = cuda.DeviceBuffer(len(payload) * topology.nodes)
    stream = cuda.Stream()
    try:
        send_host.write(payload)
        send_device.copy_from_host(send_host, stream)
        communicator.all_gather(
            send_device.pointer, receive_device.pointer, len(payload), stream.pointer
        )
        receive_device.copy_to_host(receive_host, stream)
        stream.synchronize()
        communicator.check_async_error()
        observed = [
            receive_host.bytes(rank * len(payload), len(payload))
            for rank in range(topology.nodes)
        ]
        if any(value != payload for value in observed):
            decoded = [struct.unpack("!QQ", value) for value in observed]
            raise RuntimeError(f"source metadata differs across nodes for {source}: {decoded}")
    finally:
        stream.close()
        receive_device.close()
        send_device.close()
        receive_host.close()
        send_host.close()


def collective_barrier(topology: Topology, communicator, cuda) -> None:
    """Wait until every node in this lane reaches the same file boundary."""

    send_host = cuda.HostBuffer(1)
    receive_host = cuda.HostBuffer(topology.nodes)
    send_device = cuda.DeviceBuffer(1)
    receive_device = cuda.DeviceBuffer(topology.nodes)
    stream = cuda.Stream()
    try:
        send_host.write(b"\x01")
        send_device.copy_from_host(send_host, stream)
        communicator.all_gather(
            send_device.pointer, receive_device.pointer, 1, stream.pointer
        )
        receive_device.copy_to_host(receive_host, stream)
        stream.synchronize()
        communicator.check_async_error()
        if receive_host.bytes(0, topology.nodes) != b"\x01" * topology.nodes:
            raise RuntimeError("collective file-boundary barrier returned invalid data")
    finally:
        stream.close()
        receive_device.close()
        send_device.close()
        receive_host.close()
        send_host.close()


def stage_file(
    source: str,
    destination: str,
    topology: Topology,
    window: int,
    communicator,
    cuda,
    timeout: float,
    metrics: StageMetrics | None = None,
) -> int:
    stage_start_ns = time.perf_counter_ns()
    if metrics is None:
        metrics = StageMetrics()
    source_stat = os.stat(source)
    agree_file_metadata(source, source_stat, topology, communicator, cuda)
    file_size = source_stat.st_size
    lane_start, lane_end = partition(file_size, topology.gpus_per_node, topology.lane)
    shard_start, shard_end = partition(lane_end - lane_start, topology.nodes, topology.node_rank)
    shard_start += lane_start
    shard_end += lane_start

    segment = max(1, (window + topology.nodes - 1) // topology.nodes)
    rounds = (max(partition(lane_end - lane_start, topology.nodes, rank)[1]
                  - partition(lane_end - lane_start, topology.nodes, rank)[0]
                  for rank in range(topology.nodes)) + segment - 1) // segment

    def read_round(
        slot: PipelineSlot, round_index: int, submitted_ns: int
    ) -> TimedIO:
        zero_start_ns = time.perf_counter_ns()
        slot.send_host.clear()
        zero_ns = time.perf_counter_ns() - zero_start_ns
        read_offset = shard_start + round_index * segment
        read_count = max(0, min(segment, shard_end - read_offset))
        read_start_ns = time.perf_counter_ns()
        if read_count:
            slot.send_host.read_from(source_fd, read_count, read_offset)
        finished_ns = time.perf_counter_ns()
        return TimedIO(
            read_count,
            finished_ns - read_start_ns,
            zero_ns,
            finished_ns - submitted_ns,
        )

    def write_round(
        slot: PipelineSlot, round_index: int, submitted_ns: int
    ) -> TimedIO:
        write_start_ns = time.perf_counter_ns()
        write_bytes = 0
        prepare_ns = 0
        lock_wait_ns = 0
        syscall_ns = 0
        for node_rank in range(topology.nodes):
            node_start, node_end = partition(
                lane_end - lane_start, topology.nodes, node_rank
            )
            write_offset = lane_start + node_start + round_index * segment
            write_count = max(0, min(segment, lane_start + node_end - write_offset))
            if not write_count:
                continue
            prepare_start_ns = time.perf_counter_ns()
            data = slot.receive_host.view(node_rank * segment, write_count)
            prepare_ns += time.perf_counter_ns() - prepare_start_ns
            position = 0
            while position < write_count:
                lock_start_ns = time.perf_counter_ns()
                fcntl.flock(destination_fd, fcntl.LOCK_EX)
                lock_wait_ns += time.perf_counter_ns() - lock_start_ns
                try:
                    syscall_start_ns = time.perf_counter_ns()
                    written = os.pwrite(
                        destination_fd,
                        data[position:],
                        write_offset + position,
                    )
                    syscall_ns += time.perf_counter_ns() - syscall_start_ns
                finally:
                    fcntl.flock(destination_fd, fcntl.LOCK_UN)
                if written == 0:
                    raise OSError("short write")
                position += written
                write_bytes += written
        finished_ns = time.perf_counter_ns()
        return TimedIO(
            write_bytes,
            finished_ns - write_start_ns,
            total_ns=finished_ns - submitted_ns,
            prepare_ns=prepare_ns,
            lock_wait_ns=lock_wait_ns,
            syscall_ns=syscall_ns,
        )

    def collect_read(slot: PipelineSlot) -> None:
        assert slot.read_future is not None
        wait_start_ns = time.perf_counter_ns()
        result = slot.read_future.result()
        metrics.read_wait_ns += time.perf_counter_ns() - wait_start_ns
        metrics.read_bytes += result.bytes
        metrics.read_ns += result.io_ns
        metrics.zero_ns += result.zero_ns
        metrics.read_total_ns += result.total_ns
        slot.read_future = None

    def collect_write(slot: PipelineSlot) -> None:
        assert slot.write_future is not None
        wait_start_ns = time.perf_counter_ns()
        result = slot.write_future.result()
        metrics.write_wait_ns += time.perf_counter_ns() - wait_start_ns
        metrics.write_bytes += result.bytes
        metrics.write_ns += result.io_ns
        metrics.write_prepare_ns += result.prepare_ns
        metrics.write_lock_wait_ns += result.lock_wait_ns
        metrics.write_syscall_ns += result.syscall_ns
        metrics.write_total_ns += result.total_ns
        slot.write_future = None

    published = False
    try:
        with ExitStack() as resources:
            destination_fd = prepare_destination(
                destination, file_size, topology, timeout
            )
            resources.callback(os.close, destination_fd)
            source_fd = os.open(source, os.O_RDONLY)
            resources.callback(os.close, source_fd)

            slots = []
            for _ in range(min(PIPELINE_SLOTS, rounds)):
                send_host = cuda.HostBuffer(segment)
                resources.callback(send_host.close)
                receive_host = cuda.HostBuffer(segment * topology.nodes)
                resources.callback(receive_host.close)
                slots.append(PipelineSlot(send_host, receive_host))

            send_device = cuda.DeviceBuffer(segment)
            resources.callback(send_device.close)
            receive_device = cuda.DeviceBuffer(segment * topology.nodes)
            resources.callback(receive_device.close)
            stream = cuda.Stream()
            resources.callback(stream.close)

            with (
                ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="nccl-read"
                ) as reader,
                ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="nccl-write"
                ) as writer,
            ):
                for round_index, slot in enumerate(slots):
                    slot.read_future = reader.submit(
                        read_round, slot, round_index, time.perf_counter_ns()
                    )

                for round_index in range(rounds):
                    slot = slots[round_index % len(slots)]
                    collect_read(slot)
                    if slot.write_future is not None:
                        collect_write(slot)

                    gpu_start_ns = time.perf_counter_ns()
                    send_device.copy_from_host(slot.send_host, stream)
                    communicator.all_gather(
                        send_device.pointer,
                        receive_device.pointer,
                        segment,
                        stream.pointer,
                    )
                    receive_device.copy_to_host(slot.receive_host, stream)
                    stream.synchronize()
                    communicator.check_async_error()
                    metrics.gpu_ns += time.perf_counter_ns() - gpu_start_ns

                    slot.write_future = writer.submit(
                        write_round, slot, round_index, time.perf_counter_ns()
                    )
                    next_round = round_index + len(slots)
                    if next_round < rounds:
                        slot.read_future = reader.submit(
                            read_round, slot, next_round, time.perf_counter_ns()
                        )

                for slot in slots:
                    if slot.write_future is not None:
                        collect_write(slot)

            final_stat = os.fstat(source_fd)
            if (final_stat.st_size, final_stat.st_mtime_ns) != (
                source_stat.st_size,
                source_stat.st_mtime_ns,
            ):
                raise RuntimeError(f"source changed while staging: {source}")
            os.fsync(destination_fd)

        finish_destination(destination, topology, timeout)
        published = True
    finally:
        if not published:
            job_id = os.environ.get("SLURM_JOB_ID", "manual")
            Path(temporary_path(destination, job_id)).unlink(missing_ok=True)
            Path(marker_path(destination, job_id, "ready")).unlink(missing_ok=True)
            Path(
                marker_path(destination, job_id, "done", topology.local_rank)
            ).unlink(missing_ok=True)
    collective_barrier(topology, communicator, cuda)
    metrics.files += 1
    metrics.rounds += rounds
    metrics.stage_ns += time.perf_counter_ns() - stage_start_ns
    return file_size


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--priority-manifest")
    parser.add_argument("--priority-window", type=parse_size)
    parser.add_argument("--ready-file")
    parser.add_argument("--expect-nodes", required=True, type=int)
    parser.add_argument("--gpus-per-node", required=True, type=int)
    parser.add_argument("--window", default="256M", type=parse_size)
    parser.add_argument("--bootstrap-host", required=True)
    parser.add_argument("--bootstrap-port", required=True, type=int)
    parser.add_argument("--timeout", default=120.0, type=float)
    args = parser.parse_args()
    if bool(args.priority_manifest) != bool(args.ready_file):
        parser.error("--priority-manifest and --ready-file must be used together")
    if args.priority_window is not None and not args.priority_manifest:
        parser.error("--priority-window requires --priority-manifest")
    return args


def signal_ready(path: str, topology: Topology) -> None:
    """Publish readiness on rank zero after every rank finished priority files."""

    if topology.global_rank != 0:
        return
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        f"ready rank=0 node_rank=0 epoch_ms={time.time_ns() // 1_000_000}\n"
    )
    os.replace(temporary, destination)


def stage_manifest(
    path: str,
    phase: str,
    window: int,
    topology: Topology,
    args: argparse.Namespace,
    communicator,
    cuda,
    metrics: StageMetrics,
) -> int:
    """Stage one ordered manifest and report its collective wall boundary."""

    phase_start_ns = phase_marker(topology.global_rank, phase, "start")
    total = 0
    for source, destination in load_manifest(path):
        size = stage_file(
            source,
            destination,
            topology,
            window,
            communicator,
            cuda,
            args.timeout,
            metrics,
        )
        total += size
        print(
            f"NCCLSTAGE rank={topology.global_rank} staged={source} "
            f"destination={destination} bytes={size}",
            flush=True,
        )
    phase_marker(topology.global_rank, phase, "end", phase_start_ns)
    return total


def main() -> int:
    args = parse_arguments()
    topology = topology_from_environment(args.gpus_per_node, args.expect_nodes)
    process_start_ns = phase_marker(topology.global_rank, "process", "start")

    # CUDA imports are intentionally delayed so the pure topology/layout tests run on CPUs.
    runtime_start_ns = phase_marker(topology.global_rank, "runtime_init", "start")
    cuda_import_start_ns = phase_marker(topology.global_rank, "cuda_import", "start")
    import cuda_api as cuda

    phase_marker(topology.global_rank, "cuda_import", "end", cuda_import_start_ns)
    nccl_import_start_ns = phase_marker(topology.global_rank, "nccl_import", "start")
    from nccl_api import Nccl, UniqueId

    phase_marker(topology.global_rank, "nccl_import", "end", nccl_import_start_ns)

    device_count_start_ns = phase_marker(topology.global_rank, "cuda_device_count", "start")
    visible = cuda.device_count()
    phase_marker(
        topology.global_rank, "cuda_device_count", "end", device_count_start_ns
    )
    device = select_cuda_device(visible, topology)
    set_device_start_ns = phase_marker(topology.global_rank, "cuda_set_device", "start")
    cuda.select_device(device)
    phase_marker(topology.global_rank, "cuda_set_device", "end", set_device_start_ns)
    nccl_load_start_ns = phase_marker(topology.global_rank, "nccl_library_load", "start")
    nccl = Nccl()
    phase_marker(topology.global_rank, "nccl_library_load", "end", nccl_load_start_ns)
    phase_marker(topology.global_rank, "runtime_init", "end", runtime_start_ns)

    print(
        f"NCCLSTAGE rank={topology.global_rank}/{topology.world_size} "
        f"node_rank={topology.node_rank}/{topology.nodes} lane={topology.lane} "
        f"visible_gpus={visible} cuda_device={device} nccl_version={nccl.version()}",
        flush=True,
    )

    bootstrap_start_ns = phase_marker(topology.global_rank, "bootstrap", "start")
    server = None
    if topology.global_rank == 0:
        identifiers = [nccl.unique_id().to_bytes() for _ in range(topology.gpus_per_node)]
        server = BootstrapServer(
            "0.0.0.0", args.bootstrap_port, identifiers, topology, args.timeout
        )
        server.start()
        identifier = identifiers[topology.lane]
    else:
        identifier = fetch_identifier(
            args.bootstrap_host, args.bootstrap_port, topology, args.timeout
        )
    phase_marker(topology.global_rank, "bootstrap", "end", bootstrap_start_ns)

    communicator_start_ns = phase_marker(
        topology.global_rank, "communicator_init", "start"
    )
    communicator = nccl.communicator(
        topology.nodes, UniqueId.from_bytes(identifier), topology.node_rank
    )
    phase_marker(
        topology.global_rank, "communicator_init", "end", communicator_start_ns
    )
    try:
        if server:
            bootstrap_finish_start_ns = phase_marker(
                topology.global_rank, "bootstrap_finish", "start"
            )
            server.finish()
            phase_marker(
                topology.global_rank,
                "bootstrap_finish",
                "end",
                bootstrap_finish_start_ns,
            )
        total = 0
        metrics = StageMetrics()
        if args.priority_manifest:
            total += stage_manifest(
                args.priority_manifest,
                "priority_payload",
                args.priority_window or args.window,
                topology,
                args,
                communicator,
                cuda,
                metrics,
            )
            signal_ready(args.ready_file, topology)
            phase_marker(topology.global_rank, "priority_ready", "mark")
        total += stage_manifest(
            args.manifest,
            "payload",
            args.window,
            topology,
            args,
            communicator,
            cuda,
            metrics,
        )
        close_start_ns = phase_marker(topology.global_rank, "communicator_close", "start")
        communicator.close()
        phase_marker(
            topology.global_rank, "communicator_close", "end", close_start_ns
        )
        print(format_metrics(topology.global_rank, metrics), flush=True)
        print(f"NCCLSTAGE rank={topology.global_rank} complete bytes={total}", flush=True)
        phase_marker(topology.global_rank, "process", "end", process_start_ns)
    except BaseException:
        communicator.abort()
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: ncclstage: {error}", file=sys.stderr, flush=True)
        raise
