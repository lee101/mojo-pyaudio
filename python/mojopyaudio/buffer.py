from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
import threading
from typing import Iterator
import weakref

import numpy as np

from ._lib import addr, copy_bytes, ring_read, ring_transfer, ring_write

_DIRECT_COPY_THRESHOLD = 64 * 1024
_PARALLEL_COPY_THRESHOLD = 2 * 1024 * 1024
_PARALLEL_COPY_WORKERS = 4
_COPY_PARTITION_ALIGNMENT = 64
_COPY_POOL = ThreadPoolExecutor(
    max_workers=_PARALLEL_COPY_WORKERS,
    thread_name_prefix="mojo-pyaudio-copy",
)


def _copy_addresses(source_addr: int, destination_addr: int, count: int) -> None:
    if count < _PARALLEL_COPY_THRESHOLD:
        copy_bytes(source_addr, destination_addr, count)
        return
    copy_bytes(0, 0, 0)
    chunk = (count + _PARALLEL_COPY_WORKERS - 1) // _PARALLEL_COPY_WORKERS
    chunk = (
        (chunk + _COPY_PARTITION_ALIGNMENT - 1) // _COPY_PARTITION_ALIGNMENT
    ) * _COPY_PARTITION_ALIGNMENT
    futures = []
    for start in range(0, count, chunk):
        size = min(chunk, count - start)
        futures.append(
            _COPY_POOL.submit(
                copy_bytes, source_addr + start, destination_addr + start, size
            )
        )
    # Keep the caller-owned NumPy buffers alive until every native call has
    # stopped using its address, even if one worker reports an error.
    wait(futures)
    for future in futures:
        future.result()


def _readable_bytes(data) -> np.ndarray:
    try:
        view = memoryview(data)
    except TypeError as exc:
        raise TypeError("frames must be a bytes-like object") from exc
    if not view.c_contiguous:
        view = memoryview(view.tobytes())
    return np.frombuffer(view, dtype=np.uint8)


def _writable_bytes(data) -> np.ndarray:
    if (
        isinstance(data, np.ndarray)
        and data.dtype == np.uint8
        and data.flags.c_contiguous
        and data.flags.writeable
    ):
        return data if data.ndim == 1 else data.reshape(-1)
    try:
        view = memoryview(data)
    except TypeError as exc:
        raise TypeError("destination must be a writable bytes-like object") from exc
    if view.readonly:
        raise TypeError("destination must be writable")
    if not view.c_contiguous:
        raise ValueError("destination must be C-contiguous")
    return np.frombuffer(view, dtype=np.uint8)


class PCMBuffer:
    """A thread-safe bounded FIFO of complete interleaved PCM frames."""

    def __init__(
        self,
        capacity_frames: int,
        channels: int = 1,
        format: int = 8,
        *,
        overflow: str = "raise",
    ):
        from . import get_sample_size

        if not isinstance(capacity_frames, int) or capacity_frames <= 0:
            raise ValueError("capacity_frames must be a positive integer")
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if overflow not in {"raise", "overwrite"}:
            raise ValueError("overflow must be 'raise' or 'overwrite'")
        self.capacity_frames = capacity_frames
        self.channels = channels
        self.format = format
        self.sample_size = get_sample_size(format)
        self.frame_size = channels * self.sample_size
        self._storage = np.empty(capacity_frames * self.frame_size, dtype=np.uint8)
        self._storage_view = memoryview(self._storage)
        self._storage_addr = addr(self._storage)
        self._source_ref = None
        self._source_addr = 0
        self._target_ref = None
        self._target_addr = 0
        self._read_position = 0
        self._write_position = 0
        self._size_frames = 0
        self._overflow = overflow
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return self._size_frames

    @property
    def read_available(self) -> int:
        with self._lock:
            return self._size_frames

    @property
    def write_available(self) -> int:
        with self._lock:
            return self.capacity_frames - self._size_frames

    def get_read_available(self) -> int:
        return self.read_available

    def get_write_available(self) -> int:
        return self.write_available

    def clear(self) -> None:
        with self._lock:
            self._read_position = 0
            self._write_position = 0
            self._size_frames = 0

    def _frame_count(self, byte_count: int, num_frames: int | None) -> int:
        if num_frames is None:
            return byte_count // self.frame_size
        if not isinstance(num_frames, int):
            raise TypeError("num_frames must be an integer or None")
        if num_frames < 0:
            raise ValueError("num_frames must be non-negative")
        if byte_count < num_frames * self.frame_size:
            raise ValueError("frames does not contain num_frames complete frames")
        return num_frames

    def _source_address(self, source: np.ndarray) -> int:
        if self._source_ref is not None and self._source_ref() is source:
            return self._source_addr
        self._source_addr = addr(source)
        self._source_ref = weakref.ref(source)
        return self._source_addr

    def _target_address(self, target: np.ndarray) -> int:
        if self._target_ref is not None and self._target_ref() is target:
            return self._target_addr
        self._target_addr = addr(target)
        self._target_ref = weakref.ref(target)
        return self._target_addr

    def write(
        self,
        frames,
        num_frames: int | None = None,
        exception_on_underflow: bool = False,
    ) -> None:
        del exception_on_underflow
        source = _readable_bytes(frames)
        count = self._frame_count(source.size, num_frames)
        self.write_frames(source, count)

    def write_frames(self, frames, num_frames: int | None = None) -> int:
        # Treat every buffer as opaque bytes. Casting an int16/float32 ndarray
        # with astype(uint8) would narrow sample values instead of preserving PCM.
        source = _readable_bytes(frames)
        count = self._frame_count(source.size, num_frames)
        if count == 0:
            return 0
        byte_count = count * self.frame_size
        source_offset = 0
        with self._lock:
            if count > self.capacity_frames:
                if self._overflow == "raise":
                    raise BufferError("PCM buffer overflow")
                source_offset = byte_count - self._storage.size
                count = self.capacity_frames
                byte_count = self._storage.size
                self.clear()
            missing = count - self.write_available
            if missing > 0:
                if self._overflow == "raise":
                    raise BufferError("PCM buffer overflow")
                self._advance_read(missing)
            if byte_count <= _DIRECT_COPY_THRESHOLD:
                source_view = memoryview(source)
                first = min(byte_count, self._storage.size - self._write_position)
                source_end = source_offset + first
                self._storage_view[
                    self._write_position:self._write_position + first
                ] = source_view[source_offset:source_end]
                rest = byte_count - first
                if rest:
                    self._storage_view[:rest] = source_view[
                        source_end:source_end + rest
                    ]
                self._write_position = (
                    self._write_position + byte_count
                ) % self._storage.size
            elif byte_count < _PARALLEL_COPY_THRESHOLD:
                self._write_position = ring_write(
                    self._storage_addr,
                    self._storage.size,
                    self._write_position,
                    self._source_address(source) + source_offset,
                    byte_count,
                )
            else:
                source_addr = self._source_address(source) + source_offset
                first = min(byte_count, self._storage.size - self._write_position)
                _copy_addresses(
                    source_addr, self._storage_addr + self._write_position, first
                )
                rest = byte_count - first
                if rest:
                    _copy_addresses(
                        source_addr + first, self._storage_addr, rest
                    )
                self._write_position = (
                    self._write_position + byte_count
                ) % self._storage.size
            self._size_frames += count
        return count

    def _advance_read(self, count: int) -> None:
        self._read_position = (
            self._read_position + count * self.frame_size
        ) % self._storage.size
        self._size_frames -= count

    def read(self, num_frames: int, exception_on_overflow: bool = True) -> bytes:
        if not isinstance(num_frames, int):
            raise TypeError("num_frames must be an integer")
        if num_frames < 0:
            raise ValueError("num_frames must be non-negative")
        with self._lock:
            if num_frames > self._size_frames:
                if exception_on_overflow:
                    raise BufferError("not enough complete PCM frames are buffered")
                num_frames = self._size_frames
            destination = bytearray(num_frames * self.frame_size)
            self.readinto(destination, num_frames)
            return bytes(destination)

    def readinto(self, destination, num_frames: int | None = None) -> int:
        target = _writable_bytes(destination)
        count = self._frame_count(target.size, num_frames)
        with self._lock:
            if count > self._size_frames:
                raise BufferError("not enough complete PCM frames are buffered")
            if count:
                byte_count = count * self.frame_size
                if byte_count <= _DIRECT_COPY_THRESHOLD:
                    target_view = memoryview(target)
                    first = min(
                        byte_count, self._storage.size - self._read_position
                    )
                    target_view[:first] = self._storage_view[
                        self._read_position:self._read_position + first
                    ]
                    rest = byte_count - first
                    if rest:
                        target_view[first:first + rest] = self._storage_view[:rest]
                    self._read_position = (
                        self._read_position + byte_count
                    ) % self._storage.size
                elif byte_count < _PARALLEL_COPY_THRESHOLD:
                    self._read_position = ring_read(
                        self._storage_addr,
                        self._storage.size,
                        self._read_position,
                        self._target_address(target),
                        byte_count,
                    )
                else:
                    target_addr = self._target_address(target)
                    first = min(
                        byte_count, self._storage.size - self._read_position
                    )
                    _copy_addresses(
                        self._storage_addr + self._read_position,
                        target_addr,
                        first,
                    )
                    rest = byte_count - first
                    if rest:
                        _copy_addresses(
                            self._storage_addr, target_addr + first, rest
                        )
                    self._read_position = (
                        self._read_position + byte_count
                    ) % self._storage.size
                self._size_frames -= count
        return count

    def peek(self, num_frames: int | None = None, *, offset: int = 0) -> bytes:
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        with self._lock:
            available = self._size_frames - offset
            if available < 0:
                raise BufferError("offset is beyond the buffered frames")
            count = available if num_frames is None else num_frames
            if not isinstance(count, int) or count < 0:
                raise ValueError("num_frames must be a non-negative integer or None")
            if count > available:
                raise BufferError("not enough complete PCM frames are buffered")
            destination = np.empty(count * self.frame_size, dtype=np.uint8)
            if count:
                position = (
                    self._read_position + offset * self.frame_size
                ) % self._storage.size
                ring_read(
                    self._storage_addr,
                    self._storage.size,
                    position,
                    addr(destination),
                    destination.size,
                )
            return destination.tobytes()

    def discard(self, num_frames: int) -> None:
        if not isinstance(num_frames, int) or num_frames < 0:
            raise ValueError("num_frames must be a non-negative integer")
        with self._lock:
            if num_frames > self._size_frames:
                raise BufferError("not enough complete PCM frames are buffered")
            self._advance_read(num_frames)

    def resize(self, capacity_frames: int) -> None:
        if not isinstance(capacity_frames, int) or capacity_frames <= 0:
            raise ValueError("capacity_frames must be a positive integer")
        with self._lock:
            if capacity_frames < self._size_frames:
                raise BufferError("new capacity is smaller than buffered data")
            replacement = np.empty(capacity_frames * self.frame_size, dtype=np.uint8)
            byte_count = self._size_frames * self.frame_size
            if byte_count:
                ring_read(
                    self._storage_addr,
                    self._storage.size,
                    self._read_position,
                    addr(replacement),
                    byte_count,
                )
            self._storage = replacement
            self._storage_view = memoryview(replacement)
            self._storage_addr = addr(replacement)
            self.capacity_frames = capacity_frames
            self._read_position = 0
            self._write_position = byte_count % replacement.size

    def transfer_to(self, destination: "PCMBuffer", num_frames: int | None = None) -> int:
        if not isinstance(destination, PCMBuffer):
            raise TypeError("destination must be a PCMBuffer")
        if destination is self:
            raise ValueError("source and destination must be different buffers")
        if destination.frame_size != self.frame_size:
            raise ValueError("source and destination frame sizes differ")
        first, second = sorted((self, destination), key=id)
        with first._lock, second._lock:
            count = self._size_frames if num_frames is None else num_frames
            if not isinstance(count, int) or count < 0:
                raise ValueError("num_frames must be a non-negative integer or None")
            if count > self._size_frames:
                raise BufferError("not enough complete PCM frames are buffered")
            if count > destination.capacity_frames:
                raise BufferError("destination capacity is smaller than the transfer")
            if count > destination.write_available:
                if destination._overflow == "raise":
                    raise BufferError("destination PCM buffer overflow")
                destination._advance_read(count - destination.write_available)
            byte_count = count * self.frame_size
            if byte_count:
                destination._write_position = ring_transfer(
                    self._storage_addr,
                    self._storage.size,
                    self._read_position,
                    destination._storage_addr,
                    destination._storage.size,
                    destination._write_position,
                    byte_count,
                )
                self._advance_read(count)
                destination._size_frames += count
            return count

    @contextmanager
    def locked(self) -> Iterator["PCMBuffer"]:
        with self._lock:
            yield self
