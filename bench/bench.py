"""PCM ring throughput against an equivalent pure-Python byte ring."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojopyaudio as mpa  # noqa: E402


class PythonRing:
    def __init__(self, capacity: int):
        self.storage = bytearray(capacity)
        self.capacity = capacity
        self.read_position = 0
        self.write_position = 0
        self.size = 0

    def write(self, source: memoryview) -> None:
        count = len(source)
        if count > self.capacity - self.size:
            raise BufferError
        first = min(count, self.capacity - self.write_position)
        self.storage[self.write_position:self.write_position + first] = source[:first]
        rest = count - first
        if rest:
            self.storage[:rest] = source[first:]
        self.write_position = (self.write_position + count) % self.capacity
        self.size += count

    def readinto(self, destination: memoryview) -> None:
        count = len(destination)
        if count > self.size:
            raise BufferError
        first = min(count, self.capacity - self.read_position)
        destination[:first] = self.storage[self.read_position:self.read_position + first]
        rest = count - first
        if rest:
            destination[first:] = self.storage[:rest]
        self.read_position = (self.read_position + count) % self.capacity
        self.size -= count

    def transfer_to(self, destination: "PythonRing", count: int) -> None:
        if count > self.size or count > destination.capacity - destination.size:
            raise BufferError
        remaining = count
        while remaining:
            chunk = min(
                remaining,
                self.capacity - self.read_position,
                destination.capacity - destination.write_position,
            )
            destination.storage[
                destination.write_position:destination.write_position + chunk
            ] = self.storage[self.read_position:self.read_position + chunk]
            remaining -= chunk
            self.read_position = (self.read_position + chunk) % self.capacity
            destination.write_position = (
                destination.write_position + chunk
            ) % destination.capacity
        self.size -= count
        destination.size += count


def best_time(function, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def fifo_case(block_frames: int, cycles: int):
    frame_size = 4
    block_bytes = block_frames * frame_size
    capacity_frames = block_frames * 3 + 17
    source = np.arange(block_bytes, dtype=np.uint8)
    destination = np.empty_like(source)
    source_view = memoryview(source)
    destination_view = memoryview(destination)

    def mojo_run():
        ring = mpa.PCMBuffer(capacity_frames, channels=2, format=mpa.paInt16)
        for _ in range(cycles):
            ring.write_frames(source, block_frames)
            ring.readinto(destination, block_frames)

    def python_run():
        ring = PythonRing(capacity_frames * frame_size)
        for _ in range(cycles):
            ring.write(source_view)
            ring.readinto(destination_view)

    moved = 2 * block_bytes * cycles
    return mojo_run, python_run, moved


def transfer_case(block_frames: int, cycles: int):
    frame_size = 4
    block_bytes = block_frames * frame_size
    capacity_frames = block_frames * 2 + 17
    payload = np.arange(block_bytes, dtype=np.uint8)

    def mojo_run():
        left = mpa.PCMBuffer(capacity_frames, channels=2, format=mpa.paInt16)
        right = mpa.PCMBuffer(capacity_frames, channels=2, format=mpa.paInt16)
        left.write_frames(payload, block_frames)
        for _ in range(cycles):
            left.transfer_to(right, block_frames)
            right.transfer_to(left, block_frames)

    def python_run():
        left = PythonRing(capacity_frames * frame_size)
        right = PythonRing(capacity_frames * frame_size)
        left.write(memoryview(payload))
        for _ in range(cycles):
            left.transfer_to(right, block_bytes)
            right.transfer_to(left, block_bytes)

    moved = 2 * block_bytes * cycles
    return mojo_run, python_run, moved


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as info:
            for line in info:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    cases = [
        ("FIFO write+readinto, 256 frames", *fifo_case(256, 50_000)),
        ("FIFO write+readinto, 4K frames", *fifo_case(4096, 8_000)),
        ("FIFO write+readinto, 256K frames", *fifo_case(262_144, 256)),
        ("ring-to-ring transfer, 64K frames", *transfer_case(65_536, 1_000)),
    ]

    rows = []
    for name, mojo_run, python_run, moved in cases:
        mojo_run()
        mojo_seconds = best_time(mojo_run)
        python_seconds = best_time(python_run)
        mojo_gib = moved / mojo_seconds / 2**30
        python_gib = moved / python_seconds / 2**30
        rows.append((name, mojo_gib, python_gib, python_seconds / mojo_seconds))

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| case | Mojo | pure Python | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_gib, python_gib, ratio in rows:
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_gib:.2f} GiB/s | {python_gib:.2f} GiB/s | "
            f"{ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
