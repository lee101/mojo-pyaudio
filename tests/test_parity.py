from __future__ import annotations

import inspect
import time

import numpy as np
import pyaudio as upstream
import pytest

import mojopyaudio as mpa


CONSTANTS = [
    "paFloat32", "paInt32", "paInt24", "paInt16", "paInt8", "paUInt8",
    "paCustomFormat", "paInDevelopment", "paDirectSound", "paMME", "paASIO",
    "paSoundManager", "paCoreAudio", "paOSS", "paALSA", "paAL", "paBeOS",
    "paWDMKS", "paJACK", "paWASAPI", "paNoDevice", "paNoError",
    "paNotInitialized", "paUnanticipatedHostError", "paInvalidChannelCount",
    "paInvalidSampleRate", "paInvalidDevice", "paInvalidFlag",
    "paSampleFormatNotSupported", "paBadIODeviceCombination",
    "paInsufficientMemory", "paBufferTooBig", "paBufferTooSmall",
    "paNullCallback", "paBadStreamPtr", "paTimedOut", "paInternalError",
    "paDeviceUnavailable", "paIncompatibleHostApiSpecificStreamInfo",
    "paStreamIsStopped", "paStreamIsNotStopped", "paInputOverflowed",
    "paOutputUnderflowed", "paHostApiNotFound", "paInvalidHostApi",
    "paCanNotReadFromACallbackStream", "paCanNotWriteToACallbackStream",
    "paCanNotReadFromAnOutputOnlyStream", "paCanNotWriteToAnInputOnlyStream",
    "paIncompatibleStreamHostApi", "paContinue", "paComplete", "paAbort",
    "paInputUnderflow", "paInputOverflow", "paOutputUnderflow",
    "paOutputOverflow", "paPrimingOutput", "paFramesPerBufferUnspecified",
]


def test_constants_match_pyaudio():
    assert {name: getattr(mpa, name) for name in CONSTANTS} == {
        name: getattr(upstream, name) for name in CONSTANTS
    }


@pytest.mark.parametrize("format", [
    mpa.paFloat32, mpa.paInt32, mpa.paInt24,
    mpa.paInt16, mpa.paInt8, mpa.paUInt8,
])
def test_get_sample_size_matches_pyaudio(format):
    assert mpa.get_sample_size(format) == upstream.get_sample_size(format)


@pytest.mark.parametrize("format", [0, 3, mpa.paCustomFormat, 999])
def test_get_sample_size_error_matches_pyaudio(format):
    with pytest.raises(ValueError) as ours:
        mpa.get_sample_size(format)
    with pytest.raises(ValueError) as theirs:
        upstream.get_sample_size(format)
    assert ours.value.args == theirs.value.args


@pytest.mark.parametrize("width,unsigned", [
    (1, True), (1, False), (2, True), (2, False),
    (3, True), (4, True), (1.0, True), (True, False),
])
def test_get_format_from_width_matches_pyaudio(width, unsigned):
    assert mpa.get_format_from_width(width, unsigned) == \
        upstream.get_format_from_width(width, unsigned)


@pytest.mark.parametrize("width", [0, 5, "2", None])
def test_get_format_from_width_error_matches_pyaudio(width):
    with pytest.raises(ValueError) as ours:
        mpa.get_format_from_width(width)
    with pytest.raises(ValueError) as theirs:
        upstream.get_format_from_width(width)
    assert ours.value.args == theirs.value.args


def test_public_stream_signatures_match_pyaudio():
    for name in (
        "__init__", "close", "get_input_latency", "get_output_latency",
        "get_time", "get_cpu_load", "start_stream", "stop_stream",
        "is_active", "is_stopped", "write", "read",
        "get_read_available", "get_write_available",
    ):
        assert inspect.signature(getattr(mpa.PyAudio.Stream, name)) == \
            inspect.signature(getattr(upstream.PyAudio.Stream, name))


def test_wraparound_preserves_interleaved_frames():
    buffer = mpa.PCMBuffer(5, channels=2, format=mpa.paInt16)
    frames = np.arange(16, dtype=np.int16).reshape(8, 2)
    buffer.write(frames[:5].tobytes())
    assert np.array_equal(np.frombuffer(buffer.read(3), np.int16), frames[:3].ravel())
    buffer.write(frames[5:].tobytes())
    assert np.array_equal(np.frombuffer(buffer.read(5), np.int16), frames[3:].ravel())


def test_random_operations_match_reference_fifo():
    rng = np.random.default_rng(7)
    capacity = 31
    buffer = mpa.PCMBuffer(capacity, channels=2, format=mpa.paInt24)
    reference: list[bytes] = []
    frame_size = 6
    serial = 0

    for _ in range(800):
        if rng.random() < 0.58:
            count = int(rng.integers(0, 12))
            block = []
            for _ in range(count):
                block.append(serial.to_bytes(frame_size, "little"))
                serial += 1
            payload = b"".join(block)
            if len(reference) + count > capacity:
                before = buffer.peek()
                with pytest.raises(BufferError):
                    buffer.write(payload)
                assert buffer.peek() == before
            else:
                buffer.write(payload)
                reference.extend(block)
        else:
            count = int(rng.integers(0, min(10, len(reference)) + 1))
            assert buffer.read(count) == b"".join(reference[:count])
            del reference[:count]
        assert len(buffer) == len(reference)
        assert buffer.get_read_available() == len(reference)
        assert buffer.get_write_available() == capacity - len(reference)
        assert buffer.peek() == b"".join(reference)


def test_overwrite_keeps_newest_complete_frames():
    buffer = mpa.PCMBuffer(4, format=mpa.paUInt8, overflow="overwrite")
    buffer.write(b"abc")
    buffer.write(b"def")
    assert buffer.read(4) == b"cdef"
    buffer.write(b"0123456789")
    assert buffer.read(4) == b"6789"


def test_inferred_count_ignores_incomplete_trailing_frame():
    buffer = mpa.PCMBuffer(4, channels=2, format=mpa.paInt16)
    buffer.write(b"1234567890")
    assert len(buffer) == 2
    assert buffer.read(2) == b"12345678"


def test_explicit_count_validates_source_size_atomically():
    buffer = mpa.PCMBuffer(4, format=mpa.paInt16)
    buffer.write(b"abcd")
    with pytest.raises(ValueError):
        buffer.write(b"x", 1)
    assert buffer.peek() == b"abcd"


def test_invalid_explicit_stream_write_does_not_grow_buffer():
    manager = mpa.PyAudio()
    stream = manager.open(
        rate=8000, channels=1, format=mpa.paInt16,
        output=True, frames_per_buffer=2,
    )
    with pytest.raises(ValueError):
        stream.write(b"x", 100)
    assert stream.buffer.capacity_frames == 2


def test_readinto_writes_caller_buffer_without_allocation():
    buffer = mpa.PCMBuffer(8, channels=2, format=mpa.paInt16)
    source = np.arange(12, dtype=np.int16)
    buffer.write(memoryview(source))
    destination = np.empty(8, dtype=np.int16)
    assert buffer.readinto(destination, 2) == 2
    assert np.array_equal(destination[:4], source[:4])
    assert len(buffer) == 4


def test_write_frames_preserves_ndarray_bytes_without_dtype_narrowing():
    source = np.array([0x1234, -2, 0x5678, -32768], dtype=np.int16)
    buffer = mpa.PCMBuffer(4, format=mpa.paInt16)
    assert buffer.write_frames(source) == 4
    assert buffer.read(4) == source.tobytes()


def test_stream_infers_frames_from_ndarray_nbytes_not_element_count():
    source = np.arange(12, dtype=np.int16).reshape(3, 4)
    manager = mpa.PyAudio()
    stream = manager.open(
        rate=8000, channels=2, format=mpa.paInt16, input=True, output=True,
    )
    stream.write(source)
    assert stream.get_read_available() == 6
    assert stream.read(6) == source.tobytes()


def test_noncontiguous_source_is_copied_in_logical_order():
    source = np.arange(20, dtype=np.int16).reshape(4, 5)[:, ::2]
    buffer = mpa.PCMBuffer(12, format=mpa.paInt16)
    buffer.write_frames(source)
    assert buffer.read(12) == source.tobytes()


def test_readinto_rejects_readonly_and_noncontiguous_buffers():
    buffer = mpa.PCMBuffer(4, format=mpa.paUInt8)
    buffer.write(b"abcd")
    with pytest.raises(TypeError):
        buffer.readinto(bytes(2))
    target = np.empty(4, dtype=np.uint8)[::2]
    with pytest.raises(ValueError):
        buffer.readinto(target)
    assert buffer.peek() == b"abcd"


def test_peek_offset_discard_and_clear():
    buffer = mpa.PCMBuffer(8, format=mpa.paUInt8)
    buffer.write(b"abcdefgh")
    assert buffer.peek(3, offset=2) == b"cde"
    buffer.discard(5)
    assert buffer.peek() == b"fgh"
    buffer.clear()
    assert len(buffer) == 0


def test_resize_linearizes_wrapped_data():
    buffer = mpa.PCMBuffer(5, format=mpa.paUInt8)
    buffer.write(b"abcde")
    assert buffer.read(3) == b"abc"
    buffer.write(b"fgh")
    buffer.resize(12)
    assert buffer.capacity_frames == 12
    assert buffer.read(5) == b"defgh"


def test_resize_cannot_discard_buffered_frames():
    buffer = mpa.PCMBuffer(4, format=mpa.paUInt8)
    buffer.write(b"abc")
    with pytest.raises(BufferError):
        buffer.resize(2)
    assert buffer.peek() == b"abc"


def test_transfer_moves_wrapped_frames_directly():
    source = mpa.PCMBuffer(7, format=mpa.paUInt8)
    destination = mpa.PCMBuffer(8, format=mpa.paUInt8)
    source.write(b"abcdefg")
    source.read(5)
    source.write(b"hijkl")
    destination.write(b"12")
    destination.read(1)
    assert source.transfer_to(destination, 6) == 6
    assert source.read(1) == b"l"
    assert destination.read(7) == b"2fghijk"


def test_transfer_rejects_mismatched_frames_and_is_atomic_when_full():
    source = mpa.PCMBuffer(4, format=mpa.paInt16)
    wrong = mpa.PCMBuffer(4, format=mpa.paUInt8)
    source.write(b"abcdefgh")
    with pytest.raises(ValueError):
        source.transfer_to(wrong)
    destination = mpa.PCMBuffer(2, format=mpa.paInt16)
    destination.write(b"1234")
    with pytest.raises(BufferError):
        source.transfer_to(destination, 1)
    assert source.peek() == b"abcdefgh"
    assert destination.peek() == b"1234"


def test_zero_length_operations_do_not_require_non_null_payloads():
    buffer = mpa.PCMBuffer(2, format=mpa.paUInt8)
    buffer.write(b"")
    assert buffer.read(0) == b""
    assert buffer.peek(0) == b""


def test_ffi_rejects_invalid_metadata_and_integer_narrowing():
    from mojopyaudio._lib import copy_bytes, ring_read

    copy_bytes(0, 0, 0)
    with pytest.raises(RuntimeError):
        copy_bytes(0, 1, 1)
    with pytest.raises(RuntimeError):
        ring_read(1, 4, 4, 1, 1)
    with pytest.raises(OverflowError):
        copy_bytes(1 << 64, 1, 1)


@pytest.mark.parametrize("tail", [31, 32, 33])
def test_simd_copy_tail_and_wraparound(tail):
    from mojopyaudio.buffer import _DIRECT_COPY_THRESHOLD

    size = _DIRECT_COPY_THRESHOLD + tail
    buffer = mpa.PCMBuffer(size + 17, format=mpa.paUInt8)
    payload = np.arange(size, dtype=np.uint8)
    suffix = np.arange(19, dtype=np.uint8) + 73
    buffer.write_frames(payload, size)
    assert buffer.read(19) == payload[:19].tobytes()
    buffer.write_frames(suffix, suffix.size)
    destination = np.empty(size, dtype=np.uint8)
    assert buffer.readinto(destination, size) == size
    expected = np.concatenate((payload[19:], suffix))
    assert np.array_equal(destination, expected)


@pytest.mark.parametrize("offset,expected_calls", [(-1, 0), (33, 2)])
def test_parallel_copy_threshold_roundtrip(monkeypatch, offset, expected_calls):
    import mojopyaudio.buffer as buffer_module

    size = buffer_module._PARALLEL_COPY_THRESHOLD + offset
    calls = []
    original = buffer_module._copy_addresses

    def tracked_copy(source_addr, destination_addr, count):
        calls.append(count)
        original(source_addr, destination_addr, count)

    monkeypatch.setattr(buffer_module, "_copy_addresses", tracked_copy)
    buffer = mpa.PCMBuffer(size, format=mpa.paUInt8)
    payload = np.arange(size, dtype=np.uint8)
    destination = np.empty_like(payload)
    buffer.write_frames(payload, size)
    buffer.readinto(destination, size)
    assert np.array_equal(destination, payload)
    assert len(calls) == expected_calls


def test_manager_helpers_and_empty_virtual_device_inventory():
    manager = mpa.PyAudio()
    assert manager.get_sample_size(mpa.paInt24) == 3
    assert manager.get_format_from_width(2) == mpa.paInt16
    assert manager.get_host_api_count() == manager.get_device_count() == 0
    with pytest.raises(IOError) as error:
        manager.get_default_input_device_info()
    assert error.value.args[1] == mpa.paInvalidDevice


def test_virtual_duplex_stream_roundtrip_and_growth():
    manager = mpa.PyAudio()
    stream = manager.open(
        rate=48_000, channels=2, format=mpa.paInt16,
        input=True, output=True, frames_per_buffer=2,
    )
    payload = np.arange(40, dtype=np.int16).tobytes()
    stream.write(payload)
    assert stream.buffer.capacity_frames >= 20
    assert stream.get_read_available() == 20
    assert stream.read(20) == payload
    assert stream.get_write_available() == stream.buffer.capacity_frames
    stream.close()
    assert stream not in manager._streams


def test_virtual_stream_direction_errors_match_upstream_codes():
    manager = mpa.PyAudio()
    input_stream = manager.open(
        rate=8000, channels=1, format=mpa.paInt16, input=True,
    )
    output_stream = manager.open(
        rate=8000, channels=1, format=mpa.paInt16, output=True,
    )
    with pytest.raises(IOError) as write_error:
        input_stream.write(b"\0\0")
    assert write_error.value.args == (
        "Not output stream", mpa.paCanNotWriteToAnInputOnlyStream,
    )
    with pytest.raises(IOError) as read_error:
        output_stream.read(1)
    assert read_error.value.args == (
        "Not input stream", mpa.paCanNotReadFromAnOutputOnlyStream,
    )


def test_stream_short_read_can_raise_or_pad_silence():
    manager = mpa.PyAudio()
    stream = manager.open(
        rate=8000, channels=1, format=mpa.paInt16, input=True, output=True,
    )
    stream.write(b"\x01\x02")
    with pytest.raises(IOError) as error:
        stream.read(2)
    assert error.value.args[1] == mpa.paInputOverflowed
    assert stream.read(2, exception_on_overflow=False) == b"\x01\x02\0\0"


def test_stream_lifecycle_time_close_and_terminate():
    manager = mpa.PyAudio()
    stream = manager.open(
        rate=8000, channels=1, format=mpa.paUInt8, output=True, start=False,
    )
    assert stream.is_stopped()
    stream.start_stream()
    time.sleep(0.001)
    assert stream.is_active() and stream.get_time() > 0
    assert stream.get_input_latency() == 0.0
    assert stream.get_output_latency() == 0.0
    assert stream.get_cpu_load() == 0.0
    stream.stop_stream()
    elapsed = stream.get_time()
    assert stream.is_stopped() and elapsed > 0
    manager.terminate()
    with pytest.raises(IOError):
        stream.is_active()
    with pytest.raises(IOError):
        manager.open(rate=8000, channels=1, format=mpa.paUInt8, output=True)


def test_manager_close_closes_a_registered_stream():
    manager = mpa.PyAudio()
    stream = manager.open(rate=8000, channels=1, format=mpa.paUInt8, output=True)
    manager.close(stream)
    assert stream not in manager._streams
    with pytest.raises(ValueError):
        manager.close(stream)


def test_context_managers_close_resources():
    with mpa.PyAudio() as manager:
        with manager.open(
            rate=8000, channels=1, format=mpa.paUInt8, output=True,
        ) as stream:
            assert stream.is_active()
        assert stream not in manager._streams
    assert manager._terminated


def test_hardware_and_callback_options_fail_explicitly():
    manager = mpa.PyAudio()
    with pytest.raises(NotImplementedError):
        manager.open(
            rate=8000, channels=1, format=mpa.paUInt8,
            output=True, output_device_index=0,
        )
    with pytest.raises(NotImplementedError):
        manager.open(
            rate=8000, channels=1, format=mpa.paUInt8,
            output=True, stream_callback=lambda *args: (b"", mpa.paComplete),
        )
    with pytest.raises(NotImplementedError):
        manager.is_format_supported(8000, output_device=0)
