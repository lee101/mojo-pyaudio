from __future__ import annotations

import ctypes
import operator
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.environ.get("MOJO_PYAUDIO_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyaudio.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "mpa_ring_write": ([I, I, I, I, I], I),
    "mpa_ring_read": ([I, I, I, I, I], I),
    "mpa_ring_transfer": ([I, I, I, I, I, I, I], I),
    "mpa_copy": ([I, I, I], I),
}

_library: ctypes.CDLL | None = None
_ring_write = None
_ring_read = None
_ring_transfer = None
_copy = None


def _i64(value, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < -(1 << 63) or result >= 1 << 63:
        raise OverflowError(f"{name} does not fit the Mojo C ABI Int type")
    return result


def build() -> str:
    if os.environ.get("MOJO_PYAUDIO_LIB") and os.path.exists(LIB):
        return LIB
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SOURCE):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    try:
        return ctypes.addressof(ctypes.c_uint8.from_buffer(array))
    except TypeError:
        return array.ctypes.data


def ring_write(
    ring_addr: int,
    capacity: int,
    position: int,
    source_addr: int,
    count: int,
) -> int:
    global _ring_write
    if _ring_write is None:
        _ring_write = lib().mpa_ring_write
    result = int(
        _ring_write(
            _i64(ring_addr, "ring_addr"),
            _i64(capacity, "capacity"),
            _i64(position, "position"),
            _i64(source_addr, "source_addr"),
            _i64(count, "count"),
        )
    )
    if result < 0:
        raise RuntimeError("mpa_ring_write rejected invalid ring metadata")
    return result


def ring_read(
    ring_addr: int,
    capacity: int,
    position: int,
    destination_addr: int,
    count: int,
) -> int:
    global _ring_read
    if _ring_read is None:
        _ring_read = lib().mpa_ring_read
    result = int(
        _ring_read(
            _i64(ring_addr, "ring_addr"),
            _i64(capacity, "capacity"),
            _i64(position, "position"),
            _i64(destination_addr, "destination_addr"),
            _i64(count, "count"),
        )
    )
    if result < 0:
        raise RuntimeError("mpa_ring_read rejected invalid ring metadata")
    return result


def ring_transfer(
    source_addr: int,
    source_capacity: int,
    source_position: int,
    destination_addr: int,
    destination_capacity: int,
    destination_position: int,
    count: int,
) -> int:
    global _ring_transfer
    if _ring_transfer is None:
        _ring_transfer = lib().mpa_ring_transfer
    result = int(
        _ring_transfer(
            _i64(source_addr, "source_addr"),
            _i64(source_capacity, "source_capacity"),
            _i64(source_position, "source_position"),
            _i64(destination_addr, "destination_addr"),
            _i64(destination_capacity, "destination_capacity"),
            _i64(destination_position, "destination_position"),
            _i64(count, "count"),
        )
    )
    if result < 0:
        raise RuntimeError("mpa_ring_transfer rejected invalid ring metadata")
    return result


def copy_bytes(source_addr: int, destination_addr: int, count: int) -> None:
    global _copy
    if _copy is None:
        _copy = lib().mpa_copy
    if int(
        _copy(
            _i64(source_addr, "source_addr"),
            _i64(destination_addr, "destination_addr"),
            _i64(count, "count"),
        )
    ) < 0:
        raise RuntimeError("mpa_copy rejected invalid copy metadata")
