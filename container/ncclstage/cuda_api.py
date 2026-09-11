#!/usr/bin/env python3

"""Readable resource wrappers around NVIDIA's cuda.bindings runtime API."""

from __future__ import annotations

import ctypes
import os

from cuda.bindings import runtime as cudart
from cuda.bindings.utils import get_cuda_native_handle


class CudaError(RuntimeError):
    pass


def _result(value, operation: str):
    if isinstance(value, tuple):
        status, *outputs = value
    else:
        status, outputs = value, []
    if int(status) != int(cudart.cudaError_t.cudaSuccess):
        detail = cudart.cudaGetErrorString(status)
        if isinstance(detail, tuple):
            detail = detail[-1]
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        raise CudaError(f"{operation} failed: {detail}")
    if not outputs:
        return None
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def native_handle(value) -> int:
    if isinstance(value, int):
        return value
    return get_cuda_native_handle(value)


def device_count() -> int:
    return int(_result(cudart.cudaGetDeviceCount(), "cudaGetDeviceCount"))


def select_device(device: int) -> None:
    _result(cudart.cudaSetDevice(device), "cudaSetDevice")


class Stream:
    def __init__(self) -> None:
        self.value = _result(cudart.cudaStreamCreate(), "cudaStreamCreate")
        self.closed = False

    @property
    def pointer(self) -> int:
        return native_handle(self.value)

    def synchronize(self) -> None:
        _result(cudart.cudaStreamSynchronize(self.value), "cudaStreamSynchronize")

    def close(self) -> None:
        if not self.closed:
            _result(cudart.cudaStreamDestroy(self.value), "cudaStreamDestroy")
            self.closed = True


class DeviceBuffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.value = _result(cudart.cudaMalloc(size), "cudaMalloc")
        self.closed = False

    @property
    def pointer(self) -> int:
        return native_handle(self.value)

    def copy_from_host(self, host: "HostBuffer", stream: Stream) -> None:
        _result(
            cudart.cudaMemcpyAsync(
                self.value,
                host.pointer,
                host.size,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream.value,
            ),
            "cudaMemcpyAsync(host-to-device)",
        )

    def copy_to_host(self, host: "HostBuffer", stream: Stream) -> None:
        _result(
            cudart.cudaMemcpyAsync(
                host.pointer,
                self.value,
                host.size,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream.value,
            ),
            "cudaMemcpyAsync(device-to-host)",
        )

    def close(self) -> None:
        if not self.closed:
            _result(cudart.cudaFree(self.value), "cudaFree")
            self.closed = True


class HostBuffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.value = _result(cudart.cudaMallocHost(size), "cudaMallocHost")
        self.closed = False

    @property
    def pointer(self) -> int:
        return native_handle(self.value)

    def clear(self) -> None:
        ctypes.memset(self.pointer, 0, self.size)

    def write(self, value: bytes) -> None:
        if len(value) > self.size:
            raise ValueError("value does not fit in host buffer")
        ctypes.memmove(self.pointer, value, len(value))

    def read_from(self, fd: int, count: int, offset: int) -> None:
        position = 0
        while position < count:
            chunk = os.pread(fd, count - position, offset + position)
            if not chunk:
                raise EOFError(f"short read at offset {offset + position}")
            ctypes.memmove(self.pointer + position, chunk, len(chunk))
            position += len(chunk)

    def bytes(self, offset: int, count: int) -> bytes:
        return ctypes.string_at(self.pointer + offset, count)

    def view(self, offset: int, count: int) -> memoryview:
        """Return a zero-copy byte view over a range of pinned host memory."""
        if offset < 0 or count < 0 or offset + count > self.size:
            raise ValueError("host buffer view is out of bounds")
        array = (ctypes.c_ubyte * count).from_address(self.pointer + offset)
        return memoryview(array).cast("B")

    def close(self) -> None:
        if not self.closed:
            _result(cudart.cudaFreeHost(self.value), "cudaFreeHost")
            self.closed = True
