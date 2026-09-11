#!/usr/bin/env python3

import importlib.util
import os
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).with_name("ncclstage.py")
SPEC = importlib.util.spec_from_file_location("ncclstage", MODULE_PATH)
ncclstage = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ncclstage
SPEC.loader.exec_module(ncclstage)


class PartitionTests(unittest.TestCase):
    def test_partitions_cover_range_without_overlap(self):
        for size in (0, 1, 3, 64, 1001):
            ranges = [ncclstage.partition(size, 4, rank) for rank in range(4)]
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], size)
            self.assertTrue(all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])))

    def test_nested_lane_and_node_partitions_cover_file(self):
        size = 10007
        offsets = []
        for lane in range(4):
            lane_start, lane_end = ncclstage.partition(size, 4, lane)
            for node in range(7):
                start, end = ncclstage.partition(lane_end - lane_start, 7, node)
                offsets.append((lane_start + start, lane_start + end))
        self.assertEqual(sorted(offsets)[0][0], 0)
        self.assertEqual(sorted(offsets)[-1][1], size)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(sorted(offsets), sorted(offsets)[1:])))


class ConfigurationTests(unittest.TestCase):
    def test_size_suffixes_are_binary(self):
        self.assertEqual(ncclstage.parse_size("256M"), 256 << 20)
        self.assertEqual(ncclstage.parse_size("1g"), 1 << 30)

    def test_full_visibility_selects_local_rank(self):
        topology = ncclstage.Topology(2, 8, 2, 0, 2, 4)
        self.assertEqual(ncclstage.select_cuda_device(4, topology), 2)

    def test_masked_visibility_selects_zero(self):
        topology = ncclstage.Topology(2, 8, 2, 0, 2, 4)
        self.assertEqual(ncclstage.select_cuda_device(1, topology), 0)

    def test_topology_requires_one_rank_per_gpu(self):
        environment = {
            "SLURM_PROCID": "0",
            "SLURM_NTASKS": "7",
            "SLURM_LOCALID": "0",
            "SLURM_NODEID": "0",
            "SLURM_NNODES": "2",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "expected 8 tasks"):
                ncclstage.topology_from_environment(4, 2)

    def test_manifest_contains_source_destination_pairs(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as manifest:
            manifest.write("/shared/a /local/a\n/shared/b /local/b\n")
            manifest.flush()
            self.assertEqual(
                ncclstage.load_manifest(manifest.name),
                [("/shared/a", "/local/a"), ("/shared/b", "/local/b")],
            )


class BootstrapTests(unittest.TestCase):
    def test_server_distributes_one_identifier_per_lane(self):
        root = ncclstage.Topology(0, 4, 0, 0, 1, 4)
        identifiers = [bytes([lane]) * ncclstage.UNIQUE_ID_BYTES for lane in range(4)]
        try:
            server = ncclstage.BootstrapServer("127.0.0.1", 0, identifiers, root, 2.0)
        except PermissionError:
            self.skipTest("sandbox does not permit a loopback listener")
        port = server.socket.getsockname()[1]
        server.start()
        for lane in range(1, 4):
            peer = ncclstage.Topology(lane, 4, lane, 0, 1, 4)
            self.assertEqual(
                ncclstage.fetch_identifier("127.0.0.1", port, peer, 2.0),
                identifiers[lane],
            )
        server.finish()


class _FakeCuda:
    buffers = {}
    read_threads = set()

    class HostBuffer:
        def __init__(self, size):
            self.data = bytearray(size)
            self.size = size
            self.pointer = id(self)
            _FakeCuda.buffers[self.pointer] = self

        def clear(self):
            self.data[:] = bytes(self.size)

        def write(self, value):
            self.data[: len(value)] = value

        def read_from(self, fd, count, offset):
            _FakeCuda.read_threads.add(threading.current_thread().name)
            self.data[:count] = os.pread(fd, count, offset)

        def bytes(self, offset, count):
            return bytes(self.data[offset : offset + count])

        def view(self, offset, count):
            return memoryview(self.data)[offset : offset + count]

        def close(self):
            _FakeCuda.buffers.pop(self.pointer)

    class DeviceBuffer(HostBuffer):
        def copy_from_host(self, host, stream):
            self.data[:] = host.data

        def copy_to_host(self, host, stream):
            host.data[:] = self.data

    class Stream:
        pointer = 0

        def synchronize(self):
            pass

        def close(self):
            pass


class _FakeCommunicator:
    def __init__(self):
        self.threads = set()

    def all_gather(self, send, receive, count, stream):
        self.threads.add(threading.current_thread().name)
        _FakeCuda.buffers[receive].data[:count] = _FakeCuda.buffers[send].data[:count]

    def check_async_error(self):
        pass


class StagingTests(unittest.TestCase):
    def test_only_global_rank_zero_publishes_ready_file(self):
        with tempfile.TemporaryDirectory() as directory:
            ready_file = os.path.join(directory, "container.ready")
            non_root = ncclstage.Topology(1, 2, 0, 1, 2, 1)
            ncclstage.signal_ready(ready_file, non_root)
            self.assertFalse(os.path.exists(ready_file))

            root = ncclstage.Topology(0, 2, 0, 0, 2, 1)
            ncclstage.signal_ready(ready_file, root)
            self.assertRegex(
                pathlib.Path(ready_file).read_text(),
                r"^ready rank=0 node_rank=0 epoch_ms=\d+\n$",
            )
            self.assertEqual(list(pathlib.Path(directory).iterdir()), [pathlib.Path(ready_file)])

    def test_metrics_format_is_stable_and_uses_milliseconds(self):
        metrics = ncclstage.StageMetrics(
            files=2,
            rounds=3,
            read_bytes=4,
            write_bytes=5,
            stage_ns=1_500_000,
            zero_ns=2_500_000,
            read_ns=3_500_000,
            read_total_ns=8_500_000,
            gpu_ns=4_500_000,
            write_ns=5_500_000,
            write_prepare_ns=5_750_000,
            write_lock_wait_ns=5_812_000,
            write_syscall_ns=5_875_000,
            write_total_ns=9_500_000,
            read_wait_ns=6_500_000,
            write_wait_ns=7_500_000,
        )
        self.assertEqual(
            ncclstage.format_metrics(8, metrics),
            "NCCLSTAGE_TIMING rank=8 files=2 rounds=3 read_bytes=4 "
            "write_bytes=5 stage_ms=1.500 zero_fill_ms=2.500 "
            "read_path_ms=3.500 read_total_ms=8.500 "
            "gpu_collective_total_ms=4.500 write_path_ms=5.500 "
            "write_prepare_ms=5.750 write_lock_wait_ms=5.812 "
            "write_syscall_ms=5.875 "
            "write_total_ms=9.500 read_blocked_ms=6.500 "
            "write_blocked_ms=7.500",
        )

    def test_single_node_single_gpu_copy_is_byte_exact(self):
        payload = bytes(range(251)) * 41
        communicator = _FakeCommunicator()
        metrics = ncclstage.StageMetrics()
        write_threads = set()
        lock_operations = []
        original_pwrite = os.pwrite
        original_flock = ncclstage.fcntl.flock

        def tracked_pwrite(fd, data, offset):
            write_threads.add(threading.current_thread().name)
            return original_pwrite(fd, data, offset)

        def tracked_flock(fd, operation):
            lock_operations.append(operation)
            return original_flock(fd, operation)

        _FakeCuda.read_threads.clear()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source")
            destination = os.path.join(directory, "destination")
            with open(source, "wb") as output:
                output.write(payload)
            topology = ncclstage.Topology(0, 1, 0, 0, 1, 1)
            with (
                mock.patch.dict(os.environ, {"SLURM_JOB_ID": "test"}),
                mock.patch.object(ncclstage.os, "pwrite", side_effect=tracked_pwrite),
                mock.patch.object(
                    ncclstage.fcntl, "flock", side_effect=tracked_flock
                ),
            ):
                ncclstage.stage_file(
                    source,
                    destination,
                    topology,
                    1024,
                    communicator,
                    _FakeCuda,
                    2.0,
                    metrics,
                )
            with open(destination, "rb") as staged:
                self.assertEqual(staged.read(), payload)
            self.assertEqual(os.stat(destination).st_mode & 0o777, 0o666)
        self.assertTrue(_FakeCuda.read_threads)
        self.assertTrue(
            all(name.startswith("nccl-read") for name in _FakeCuda.read_threads)
        )
        self.assertTrue(write_threads)
        self.assertTrue(all(name.startswith("nccl-write") for name in write_threads))
        self.assertEqual(len(lock_operations), 2 * metrics.rounds)
        self.assertEqual(lock_operations[::2], [ncclstage.fcntl.LOCK_EX] * metrics.rounds)
        self.assertEqual(lock_operations[1::2], [ncclstage.fcntl.LOCK_UN] * metrics.rounds)
        self.assertEqual(communicator.threads, {threading.current_thread().name})
        self.assertEqual(metrics.files, 1)
        self.assertEqual(metrics.rounds, 11)
        self.assertEqual(metrics.read_bytes, len(payload))
        self.assertEqual(metrics.write_bytes, len(payload))
        self.assertGreaterEqual(metrics.stage_ns, metrics.gpu_ns)
        self.assertGreaterEqual(
            metrics.read_total_ns, metrics.read_ns + metrics.zero_ns
        )
        self.assertGreaterEqual(metrics.write_total_ns, metrics.write_ns)

    def test_source_change_closes_and_removes_temporary_destination(self):
        communicator = _FakeCommunicator()
        topology = ncclstage.Topology(0, 1, 0, 0, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source")
            destination = os.path.join(directory, "destination")
            with open(source, "wb") as output:
                output.write(b"source data")
            source_stat = os.stat(source)
            changed_stat = mock.Mock(
                st_size=source_stat.st_size,
                st_mtime_ns=source_stat.st_mtime_ns + 1,
            )
            with (
                mock.patch.dict(os.environ, {"SLURM_JOB_ID": "test"}),
                mock.patch.object(ncclstage.os, "fstat", return_value=changed_stat),
            ):
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    ncclstage.stage_file(
                        source,
                        destination,
                        topology,
                        1024,
                        communicator,
                        _FakeCuda,
                        2.0,
                    )
            self.assertFalse(os.path.exists(destination))
            self.assertFalse(
                os.path.exists(ncclstage.temporary_path(destination, "test"))
            )


if __name__ == "__main__":
    unittest.main()
