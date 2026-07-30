from __future__ import annotations

import time
import warnings

from .buffer import PCMBuffer

__version__ = "0.1.0"

paFloat32 = 1
paInt32 = 2
paInt24 = 4
paInt16 = 8
paInt8 = 16
paUInt8 = 32
paCustomFormat = 65536

paInDevelopment = 0
paDirectSound = 1
paMME = 2
paASIO = 3
paSoundManager = 4
paCoreAudio = 5
paOSS = 7
paALSA = 8
paAL = 9
paBeOS = 10
paWDMKS = 11
paJACK = 12
paWASAPI = 13
paNoDevice = -1

paNoError = 0
paNotInitialized = -10000
paUnanticipatedHostError = -9999
paInvalidChannelCount = -9998
paInvalidSampleRate = -9997
paInvalidDevice = -9996
paInvalidFlag = -9995
paSampleFormatNotSupported = -9994
paBadIODeviceCombination = -9993
paInsufficientMemory = -9992
paBufferTooBig = -9991
paBufferTooSmall = -9990
paNullCallback = -9989
paBadStreamPtr = -9988
paTimedOut = -9987
paInternalError = -9986
paDeviceUnavailable = -9985
paIncompatibleHostApiSpecificStreamInfo = -9984
paStreamIsStopped = -9983
paStreamIsNotStopped = -9982
paInputOverflowed = -9981
paOutputUnderflowed = -9980
paHostApiNotFound = -9979
paInvalidHostApi = -9978
paCanNotReadFromACallbackStream = -9977
paCanNotWriteToACallbackStream = -9976
paCanNotReadFromAnOutputOnlyStream = -9975
paCanNotWriteToAnInputOnlyStream = -9974
paIncompatibleStreamHostApi = -9973

paContinue = 0
paComplete = 1
paAbort = 2

paInputUnderflow = 1
paInputOverflow = 2
paOutputUnderflow = 4
paOutputOverflow = 8
paPrimingOutput = 16
paFramesPerBufferUnspecified = 0

_SAMPLE_SIZES = {
    paFloat32: 4,
    paInt32: 4,
    paInt24: 3,
    paInt16: 2,
    paInt8: 1,
    paUInt8: 1,
}


def get_sample_size(format):
    try:
        return _SAMPLE_SIZES[format]
    except (KeyError, TypeError):
        raise ValueError("Sample format not supported", paSampleFormatNotSupported) from None


def get_format_from_width(width, unsigned=True):
    if width == 1:
        return paUInt8 if unsigned else paInt8
    if width == 2:
        return paInt16
    if width == 3:
        return paInt24
    if width == 4:
        return paFloat32
    raise ValueError(f"Invalid width: {width}")


def get_portaudio_version():
    raise NotImplementedError("this port provides virtual PCM streams, not PortAudio")


def get_portaudio_version_text():
    raise NotImplementedError("this port provides virtual PCM streams, not PortAudio")


class PyAudio:
    class Stream:
        def __init__(
            self,
            PA_manager,
            rate,
            channels,
            format,
            input=False,
            output=False,
            input_device_index=None,
            output_device_index=None,
            frames_per_buffer=0,
            start=True,
            input_host_api_specific_stream_info=None,
            output_host_api_specific_stream_info=None,
            stream_callback=None,
        ):
            if not (input or output):
                raise ValueError("Must specify an input or output stream.")
            if rate <= 0:
                raise ValueError("rate must be positive")
            if not isinstance(channels, int) or channels <= 0:
                raise ValueError("channels must be a positive integer")
            get_sample_size(format)
            if (
                input_device_index is not None
                or output_device_index is not None
                or input_host_api_specific_stream_info is not None
                or output_host_api_specific_stream_info is not None
            ):
                raise NotImplementedError("hardware device selection is outside this port")
            if stream_callback is not None:
                raise NotImplementedError("PortAudio callback streams are outside this port")

            self._parent = PA_manager
            self._is_input = bool(input)
            self._is_output = bool(output)
            self._is_running = bool(start)
            self._closed = False
            self._rate = rate
            self._channels = channels
            self._format = format
            self._frames_per_buffer = frames_per_buffer
            capacity = frames_per_buffer if frames_per_buffer and frames_per_buffer > 0 else 4096
            self.buffer = PCMBuffer(capacity, channels, format)
            self._started_at = time.monotonic() if start else None
            self._elapsed = 0.0

        def close(self):
            if self._closed:
                return
            self.stop_stream()
            self._closed = True
            self._parent._remove_stream(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()

        def _require_open(self):
            if self._closed:
                raise IOError("Stream is closed", paBadStreamPtr)

        def get_input_latency(self):
            self._require_open()
            return 0.0

        def get_output_latency(self):
            self._require_open()
            return 0.0

        def get_time(self):
            self._require_open()
            if self._is_running:
                return self._elapsed + time.monotonic() - self._started_at
            return self._elapsed

        def get_cpu_load(self):
            self._require_open()
            return 0.0

        def start_stream(self):
            self._require_open()
            if not self._is_running:
                self._started_at = time.monotonic()
                self._is_running = True

        def stop_stream(self):
            if self._closed or not self._is_running:
                return
            self._elapsed += time.monotonic() - self._started_at
            self._is_running = False
            self._started_at = None

        def is_active(self):
            self._require_open()
            return self._is_running

        def is_stopped(self):
            self._require_open()
            return not self._is_running

        def _grow_for(self, num_frames):
            needed = len(self.buffer) + num_frames
            if needed <= self.buffer.capacity_frames:
                return
            capacity = self.buffer.capacity_frames
            while capacity < needed:
                capacity *= 2
            self.buffer.resize(capacity)

        def write(self, frames, num_frames=None, exception_on_underflow=False):
            self._require_open()
            if not self._is_output:
                raise IOError("Not output stream", paCanNotWriteToAnInputOnlyStream)
            try:
                byte_count = memoryview(frames).nbytes
            except TypeError as exc:
                raise TypeError("frames must be a bytes-like object") from exc
            if num_frames is None:
                num_frames = byte_count // self.buffer.frame_size
            elif not isinstance(num_frames, int):
                raise TypeError("num_frames must be an integer or None")
            elif num_frames < 0:
                raise ValueError("num_frames must be non-negative")
            elif byte_count < num_frames * self.buffer.frame_size:
                raise ValueError("frames does not contain num_frames complete frames")
            self._grow_for(num_frames)
            self.buffer.write(frames, num_frames, exception_on_underflow)

        def read(self, num_frames, exception_on_overflow=True):
            self._require_open()
            if not self._is_input:
                raise IOError("Not input stream", paCanNotReadFromAnOutputOnlyStream)
            try:
                return self.buffer.read(num_frames, exception_on_overflow=True)
            except BufferError:
                if exception_on_overflow:
                    raise IOError(
                        "Insufficient buffered input", paInputOverflowed
                    ) from None
                available = self.buffer.get_read_available()
                data = self.buffer.read(available)
                missing = (num_frames - available) * self.buffer.frame_size
                return data + bytes(missing)

        def get_read_available(self):
            self._require_open()
            return self.buffer.get_read_available()

        def get_write_available(self):
            self._require_open()
            return self.buffer.get_write_available()

    def __init__(self):
        self._streams = set()
        self._terminated = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def terminate(self):
        for stream in self._streams.copy():
            stream.close()
        self._streams.clear()
        self._terminated = True

    def get_sample_size(self, format):
        return get_sample_size(format)

    def get_format_from_width(self, width, unsigned=True):
        return get_format_from_width(width, unsigned)

    def open(self, *args, **kwargs):
        if self._terminated:
            raise IOError("PyAudio instance has been terminated", paNotInitialized)
        stream = PyAudio.Stream(self, *args, **kwargs)
        self._streams.add(stream)
        return stream

    def close(self, stream):
        if stream not in self._streams:
            raise ValueError(f"Stream {stream} not found")
        stream.close()

    def _remove_stream(self, stream):
        self._streams.discard(stream)

    def get_host_api_count(self):
        return 0

    def get_device_count(self):
        return 0

    def _no_devices(self, *args, **kwargs):
        del args, kwargs
        raise IOError("No PortAudio devices in virtual-buffer mode", paInvalidDevice)

    get_default_host_api_info = _no_devices
    get_host_api_info_by_type = _no_devices
    get_host_api_info_by_index = _no_devices
    get_device_info_by_host_api_device_index = _no_devices
    get_default_input_device_info = _no_devices
    get_default_output_device_info = _no_devices
    get_device_info_by_index = _no_devices

    def is_format_supported(
        self,
        rate,
        input_device=None,
        input_channels=None,
        input_format=None,
        output_device=None,
        output_channels=None,
        output_format=None,
    ):
        del (
            rate,
            input_device,
            input_channels,
            input_format,
            output_device,
            output_channels,
            output_format,
        )
        raise NotImplementedError("hardware format probing is outside this port")


class Stream(PyAudio.Stream):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Do not instantiate mojopyaudio.Stream directly. Use "
            "mojopyaudio.PyAudio.open() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
