# mojo-pyaudio

Mojo-backed PCM stream buffering with a Python API shaped like
[PyAudio 0.2.14](https://people.csail.mit.edu/hubert/pyaudio/docs/).

```python
import mojopyaudio as pyaudio

audio = pyaudio.PyAudio()
stream = audio.open(
    rate=48_000,
    channels=2,
    format=pyaudio.paInt16,
    input=True,
    output=True,
    frames_per_buffer=1024,
)

block = bytes(1024 * 2 * pyaudio.get_sample_size(pyaudio.paInt16))
stream.write(block)
assert stream.get_read_available() == 1024
assert stream.read(1024) == block

stream.close()
audio.terminate()
```

This is a virtual, in-memory stream, not a replacement PortAudio device
backend. It is useful for producer/consumer pipelines, network audio,
deterministic tests, and staging interleaved PCM between callbacks or
processes. For the covered subset, existing code can use
`import mojopyaudio as pyaudio` and keep the familiar names and signatures.

## What is covered

| area | implemented |
| --- | --- |
| Sample formats | PyAudio's `paFloat32`, `paInt32`, `paInt24`, `paInt16`, `paInt8`, `paUInt8` and related constants |
| Format helpers | `get_sample_size`, `get_format_from_width`, including upstream errors |
| Manager | `PyAudio`, `open`, `close`, `terminate`, context management |
| Stream lifecycle | `start_stream`, `stop_stream`, `is_active`, `is_stopped`, `get_time`, `get_cpu_load`, latency accessors |
| Blocking stream I/O | `write`, `read`, `get_read_available`, `get_write_available`; complete interleaved frames |
| Explicit FIFO API | `PCMBuffer`, `readinto`, `peek`, `discard`, `clear`, `resize`, overwrite mode, direct buffer-to-buffer transfer |

The format helpers, constants, public stream signatures, and error values are
tested directly against the real `pyaudio` 0.2.14 package. Ring behavior is
tested against a pure-Python FIFO model, including 800 randomized operations,
because PyAudio does not implement an in-memory ring: its stream calls go
straight to PortAudio and a physical or virtual audio device.

## What is not covered

- Audio device discovery, host API inspection beyond an empty inventory, and
  hardware format probing.
- Real microphone capture or speaker playback.
- PortAudio callback threads and host-specific stream information.
- PortAudio version reporting.
- DSP and sample conversion. Samples stay opaque; no clipping, byte-order
  conversion, mixing, or resampling occurs.

Hardware-specific options and callbacks fail with `NotImplementedError`
instead of being silently ignored. Reading an empty virtual stream cannot
block waiting for hardware: with the default `exception_on_overflow=True` it
raises `IOError`; with `False` it returns the requested number of frames padded
with format-correct zero bytes.

## Install

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi install` provides the pinned Mojo nightly, Python, NumPy, pytest, and
upstream PyAudio used by the parity tests. `pixi run build` produces
`dist/libmojo-pyaudio.so`. The Python wrapper also rebuilds a missing or stale
library on first use.

To load a prebuilt library elsewhere:

```bash
MOJO_PYAUDIO_LIB=/path/to/libmojo-pyaudio.so python your_program.py
```

## Bounded buffering

`PCMBuffer` exposes the buffering layer without the stream facade:

```python
import mojopyaudio as pyaudio

fifo = pyaudio.PCMBuffer(
    capacity_frames=4096,
    channels=2,
    format=pyaudio.paInt24,
    overflow="overwrite",
)

fifo.write(network_packet)
frames_ready = fifo.get_read_available()
destination = bytearray(frames_ready * fifo.frame_size)
fifo.readinto(destination)
```

The default overflow policy is atomic `"raise"`: either all complete frames
are accepted or the buffer is unchanged. `"overwrite"` discards the oldest
complete frames and retains the newest. `readinto` avoids constructing a
temporary `bytes` result, and `transfer_to` copies directly between two rings.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic. The reference is an equivalent `bytearray` ring whose
slice assignments execute in optimized CPython C code. Throughput counts both
directions of a write/read or transfer cycle.

| case | Mojo | pure Python | ratio |
| --- | ---: | ---: | ---: |
| FIFO write+readinto, 256 frames | 0.29 GiB/s | 0.77 GiB/s | 0.37x slower |
| FIFO write+readinto, 4K frames | 2.42 GiB/s | 4.23 GiB/s | 0.57x slower |
| FIFO write+readinto, 256K frames | 7.03 GiB/s | 6.21 GiB/s | 1.13x faster |
| ring-to-ring transfer, 64K frames | 8.23 GiB/s | 5.73 GiB/s | 1.43x faster |

The result has a clear crossover. For small callback-sized blocks, Python's
C-level slice copies are already excellent. Blocks up to 64 KiB therefore use
Python buffer-view copies and avoid ctypes call overhead. Larger blocks cross the
FFI with cached NumPy addresses and use the Mojo SIMD kernel. Contiguous copy
segments of at least 2 MiB are split across four host workers, with each
independent partition executed by that kernel. Mojo wins for large bulk blocks
and direct ring-to-ring movement. These are real single-machine measurements,
not projected results.

There is no GPU path. PCM buffering performs opaque byte copies with zero
floating-point operations per byte, so it is far below the arithmetic-intensity
threshold where device allocation, transfer, and launch overhead can pay off.

Benchmarking against PyAudio's `Stream.read` or `Stream.write` would measure
the selected PortAudio driver and audio clock, not buffer-copy throughput, so
the benchmark deliberately does not claim that comparison.

## How it works

```
Python bytes / writable buffer
             |
             | ctypes: address + byte count
             v
src/kernels.mojo
  wrap-aware, hardware-width SIMD byte copies
             |
             v
NumPy-owned uint8 ring storage
```

PCM is stored exactly as PyAudio receives it: packed, interleaved frames with
`channels * get_sample_size(format)` bytes per frame. The Mojo compilation
unit treats it as opaque bytes. Its byte-vector width is derived from the
host's `float64` SIMD lane count, and a scalar tail handles every remainder
without requiring aligned input. Each wrapped operation has at most two
contiguous segments. Python owns every allocation and tracks read position,
write position, frame count, locking, and stream lifecycle.

Buffers cross the C ABI as `Int` addresses and are reconstructed inside
non-parametric `@export(...) ... abi("C")` wrappers as
`UnsafePointer[UInt8, AnyOrigin[mut=True]]`. Empty operations return before
constructing a pointer, so a zero-length Python buffer never becomes an
invalid Mojo pointer.

## License

MIT
