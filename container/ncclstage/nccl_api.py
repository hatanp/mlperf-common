#!/usr/bin/env python3

"""Small ctypes binding for the NCCL calls used by ncclstage."""

from __future__ import annotations

import ctypes


NCCL_UNIQUE_ID_BYTES = 128
NCCL_UINT8 = 1


class NcclError(RuntimeError):
    pass


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * NCCL_UNIQUE_ID_BYTES)]

    def to_bytes(self) -> bytes:
        return ctypes.string_at(ctypes.byref(self), NCCL_UNIQUE_ID_BYTES)

    @classmethod
    def from_bytes(cls, value: bytes) -> "UniqueId":
        if len(value) != NCCL_UNIQUE_ID_BYTES:
            raise ValueError(f"NCCL unique ID must be {NCCL_UNIQUE_ID_BYTES} bytes")
        result = cls()
        ctypes.memmove(ctypes.byref(result), value, len(value))
        return result


class Nccl:
    def __init__(self, library: str = "libnccl.so.2") -> None:
        self.lib = ctypes.CDLL(library)
        self._declare()

    def _declare(self) -> None:
        self.lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        self.lib.ncclGetErrorString.restype = ctypes.c_char_p

        self.lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.ncclGetVersion.restype = ctypes.c_int
        self.lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(UniqueId)]
        self.lib.ncclGetUniqueId.restype = ctypes.c_int
        self.lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            UniqueId,
            ctypes.c_int,
        ]
        self.lib.ncclCommInitRank.restype = ctypes.c_int
        self.lib.ncclAllGather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.ncclAllGather.restype = ctypes.c_int
        self.lib.ncclCommGetAsyncError.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ncclCommGetAsyncError.restype = ctypes.c_int
        self.lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        self.lib.ncclCommDestroy.restype = ctypes.c_int
        self.lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
        self.lib.ncclCommAbort.restype = ctypes.c_int

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        message = self.lib.ncclGetErrorString(result)
        detail = message.decode(errors="replace") if message else f"result {result}"
        raise NcclError(f"{operation} failed: {detail}")

    def version(self) -> int:
        value = ctypes.c_int()
        self._check(self.lib.ncclGetVersion(ctypes.byref(value)), "ncclGetVersion")
        return value.value

    def unique_id(self) -> UniqueId:
        value = UniqueId()
        self._check(self.lib.ncclGetUniqueId(ctypes.byref(value)), "ncclGetUniqueId")
        return value

    def communicator(self, size: int, unique_id: UniqueId, rank: int) -> "Communicator":
        handle = ctypes.c_void_p()
        self._check(
            self.lib.ncclCommInitRank(ctypes.byref(handle), size, unique_id, rank),
            "ncclCommInitRank",
        )
        return Communicator(self, handle)


class Communicator:
    def __init__(self, nccl: Nccl, handle: ctypes.c_void_p) -> None:
        self.nccl = nccl
        self.handle = handle
        self.closed = False

    def all_gather(self, send: int, receive: int, count: int, stream: int) -> None:
        self.nccl._check(
            self.nccl.lib.ncclAllGather(
                ctypes.c_void_p(send),
                ctypes.c_void_p(receive),
                count,
                NCCL_UINT8,
                self.handle,
                ctypes.c_void_p(stream),
            ),
            "ncclAllGather",
        )

    def check_async_error(self) -> None:
        result = ctypes.c_int()
        self.nccl._check(
            self.nccl.lib.ncclCommGetAsyncError(self.handle, ctypes.byref(result)),
            "ncclCommGetAsyncError",
        )
        self.nccl._check(result.value, "NCCL asynchronous operation")

    def close(self) -> None:
        if not self.closed:
            self.nccl._check(self.nccl.lib.ncclCommDestroy(self.handle), "ncclCommDestroy")
            self.closed = True

    def abort(self) -> None:
        if not self.closed:
            self.nccl.lib.ncclCommAbort(self.handle)
            self.closed = True
