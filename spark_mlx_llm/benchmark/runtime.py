"""MLX-free device locking, native macOS VM counters and process inspection.

One cooperative Metal lock is shared by all checkouts on this machine.

The lock file is never unlinked or replaced. Pass ``fileno()`` to GPU workers
with ``subprocess.Popen(..., pass_fds=(lock.fileno(),))``. Python otherwise
closes this descriptor on exec. An inherited descriptor refers to the same
open file description: closing the parent's copy preserves the flock until
the last worker closes its copy. Explicit LOCK_UN would release the worker's
lock too, so close() deliberately only closes its own descriptor.
"""

import ctypes
import fcntl
import json
import math
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "DeviceLock",
    "NativeVMMonitor",
    "device_lock",
    "print_process_snapshot",
    "process_snapshot",
]


class DeviceLock:
    """Exclusive GPU lease; metadata is descriptive, never proof of ownership.

    ``acquire(timeout=0)`` makes one nonblocking attempt, ``None`` waits until
    acquired, and a finite timeout raises TimeoutError on contention. A
    stop_requested callback may return True (raising InterruptedError) or
    raise the caller's own cancellation exception. Both release a pending FD.
    An acquired object can be used as a context manager without reacquiring.
    """

    def __init__(self, path=None):
        value = (
            os.environ.get(
                "MLX_SPARK_DEVICE_LOCK",
                str(Path.home() / ".cache/mlx-spark/locks/metal-default.lock"),
            )
            if path is None
            else path
        )
        self.path = Path(value).expanduser()
        if not str(value) or not self.path.is_absolute():
            raise ValueError(
                "Device lock path must be absolute; use a location shared by all candidates"
            )
        self.path = self.path.resolve()
        self._fd = None
        self.metadata = None

    def acquire(self, timeout=None, stop_requested=None):
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError(
                "Device lock timeout must be finite and nonnegative, or None"
            )
        if self._fd is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        started = time.monotonic()
        try:
            while True:
                if stop_requested is not None and stop_requested():
                    raise InterruptedError("Device lock wait stopped")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    elapsed = time.monotonic() - started
                    if timeout is not None and elapsed >= timeout:
                        raise TimeoutError(
                            f"Timed out waiting for device lock: {self.path}"
                        ) from None
                    time.sleep(
                        0.05 if timeout is None else min(0.05, timeout - elapsed)
                    )
            metadata = {
                "owner_pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "path": str(self.path),
                "cwd": str(Path.cwd()),
                "ownership": "flock; inherited GPU workers may outlive owner_pid",
            }
            encoded = (json.dumps(metadata, sort_keys=True) + "\n").encode()
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(descriptor), "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self.metadata = metadata
            self._fd = descriptor
            return self
        except BaseException:
            os.close(descriptor)
            raise

    def fileno(self):
        if self._fd is None:
            raise ValueError("Device lock is not acquired")
        return self._fd

    def close(self):
        if self._fd is not None:
            descriptor, self._fd = self._fd, None
            # Never LOCK_UN: a worker may still share this open description.
            os.close(descriptor)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.close()


# macOS vm_statistics64_data_t; returned words must include rev1 swap fields.
class VMStatistics64(ctypes.Structure):
    _fields_ = (
        [
            (name, ctypes.c_uint32)
            for name in ("free_count", "active_count", "inactive_count", "wire_count")
        ]
        + [
            (name, ctypes.c_uint64)
            for name in (
                "zero_fill_count",
                "reactivations",
                "pageins",
                "pageouts",
                "faults",
                "cow_faults",
                "lookups",
                "hits",
                "purges",
            )
        ]
        + [(name, ctypes.c_uint32) for name in ("purgeable_count", "speculative_count")]
        + [
            (name, ctypes.c_uint64)
            for name in ("decompressions", "compressions", "swapins", "swapouts")
        ]
        + [
            (name, ctypes.c_uint32)
            for name in (
                "compressor_page_count",
                "throttled_count",
                "external_page_count",
                "internal_page_count",
            )
        ]
        + [
            (name, ctypes.c_uint64)
            for name in (
                "total_uncompressed_pages_in_compressor",
                "swapped_count",
                "total_tag_storage_pages",
                "nontag_pageable_tag_storage_pages",
                "nontag_wired_tag_storage_pages",
                "free_tag_storage_pages",
                "tag_storing_tag_storage_pages",
                "total_tagged_pages",
                "resident_tagged_pages",
                "compressed_tagged_pages",
                "tagged_compressions",
                "tagged_decompressions",
                "compressed_tag_storage_bytes",
            )
        ]
    )


class NativeVMMonitor:
    """Reuse loaded libraries and the host port for every observation."""

    def __init__(self):
        if sys.platform != "darwin":
            raise RuntimeError("native VM monitoring requires macOS")
        self.library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        self.library.mach_host_self.argtypes = []
        self.library.mach_host_self.restype = ctypes.c_uint32
        self.library.host_statistics64.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.library.host_statistics64.restype = ctypes.c_int
        self.host = self.library.mach_host_self()
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        self.buffer = VMStatistics64()

    def sample(self):
        start = time.monotonic()
        count = ctypes.c_uint32(ctypes.sizeof(self.buffer) // 4)
        result = self.library.host_statistics64(
            self.host, 4, ctypes.byref(self.buffer), ctypes.byref(count)
        )
        if result:
            raise RuntimeError(f"host_statistics64 failed: {result}")
        needed = (VMStatistics64.swapouts.offset + 8) // 4
        if count.value < needed:
            raise RuntimeError(
                f"host_statistics64 returned {count.value} words; need {needed}"
            )
        return {
            "source": "host_statistics64/HOST_VM_INFO64",
            "swapins": self.buffer.swapins,
            "swapouts": self.buffer.swapouts,
            "page_size": self.page_size,
            "pageins": self.buffer.pageins,
            "pageouts": self.buffer.pageouts,
            "free_pages": self.buffer.free_count,
            "compressor_pages": self.buffer.compressor_page_count,
            "returned_words": count.value,
            "monotonic_start": start,
            "monotonic_end": time.monotonic(),
        }


def device_lock(timeout=0):
    return DeviceLock().acquire(timeout=timeout)


def process_snapshot():
    """Record competing processes; this alone does not certify an idle GPU."""
    return subprocess.check_output(["ps", "-axo", "pid,pcpu,comm"], text=True)


def print_process_snapshot():
    print(
        "Device lock held; process inspection is not GPU idle certification.",
        flush=True,
    )
    busy = [
        line
        for line in process_snapshot().splitlines()[1:]
        if float(line.split()[1]) >= 5
    ]
    if busy:
        print("Processes using >=5% CPU:\n" + "\n".join(busy[:10]), flush=True)
