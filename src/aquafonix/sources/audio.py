"""Audio input: microphones / line-in via sounddevice (PortAudio) and, on
Windows, "what you hear" system audio via WASAPI loopback (PyAudioWPatch)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from aquafonix.sources.base import Source

log = logging.getLogger(__name__)

SOUNDDEVICE = "sounddevice"
LOOPBACK = "loopback"
_HOSTAPI_ORDER = ("Windows WASAPI", "MME", "Windows DirectSound", "Windows WDM-KS",
                  "Core Audio", "PulseAudio", "ALSA", "JACK Audio Connection Kit", "ASIO")


@dataclass(frozen=True)
class AudioDevice:
    backend: str
    index: int
    name: str
    hostapi: str
    channels: int
    default_rate: float
    is_default: bool = False

    @property
    def key(self) -> str:
        """Stable-ish identifier used to remember the choice between runs."""
        return f"{self.backend}|{self.hostapi}|{self.name}"

    @property
    def label(self) -> str:
        if self.backend == LOOPBACK:
            return f"System audio (loopback): {self.name.replace(' [Loopback]', '')}"
        star = "★ " if self.is_default else ""
        return f"{star}{self.name} — {self.hostapi}"


def list_audio_devices() -> tuple[list[AudioDevice], str | None]:
    """All capture devices, loopback first. Returns ``(devices, problem)``."""
    devices: list[AudioDevice] = []
    problems = []
    devices.extend(_loopback_devices())
    try:
        import sounddevice as sd

        hostapis = sd.query_hostapis()
        default_in = sd.default.device[0] if sd.default.device else -1
        inputs = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] <= 0:
                continue
            api = hostapis[dev["hostapi"]]["name"]
            inputs.append(AudioDevice(SOUNDDEVICE, idx, dev["name"], api,
                                      int(dev["max_input_channels"]),
                                      float(dev["default_samplerate"]), idx == default_in))

        def order(d: AudioDevice):
            api_rank = _HOSTAPI_ORDER.index(d.hostapi) if d.hostapi in _HOSTAPI_ORDER else 99
            return (not d.is_default, api_rank, d.name.lower())

        devices.extend(sorted(inputs, key=order))
    except Exception as exc:  # OSError when the PortAudio library is missing
        problems.append(f"audio inputs unavailable: {exc}")
    return devices, "; ".join(problems) or None


def _loopback_devices() -> list[AudioDevice]:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return []
    out = []
    pa = pyaudio.PyAudio()
    try:
        try:
            default_idx = pa.get_default_wasapi_loopback()["index"]
        except Exception:
            default_idx = -1
        for dev in pa.get_loopback_device_info_generator():
            out.append(AudioDevice(LOOPBACK, int(dev["index"]), dev["name"], "Windows WASAPI",
                                   int(dev["maxInputChannels"]), float(dev["defaultSampleRate"]),
                                   int(dev["index"]) == default_idx))
    except Exception:
        log.exception("could not enumerate WASAPI loopback devices")
    finally:
        pa.terminate()
    out.sort(key=lambda d: not d.is_default)
    return out


class AudioSource(Source):
    sample_dtype = np.float32

    def __init__(self, device: AudioDevice, sample_rate: float | None = None,
                 channels: int | None = None, blocksize: int = 0) -> None:
        super().__init__()
        self.device = device
        self.sample_rate = float(sample_rate or device.default_rate or 48000)
        if device.backend == LOOPBACK:  # WASAPI loopback wants the mix format
            self.sample_rate = float(device.default_rate)
            self.channels = device.channels
        else:
            self.channels = max(1, min(int(channels or device.channels), device.channels))
        self.blocksize = int(blocksize)
        self.buffer_capacity = int(self.sample_rate * 30)
        self.names = (["Left", "Right"] if self.channels == 2
                      else [f"ch{i + 1}" for i in range(self.channels)])
        self.overflows = 0
        self._frames = 0
        self._stream = None
        self._pa = None

    def _uses_thread(self) -> bool:
        return False

    def _callback_running(self) -> bool:
        s = self._stream
        if s is None:
            return False
        try:
            return bool(s.active) if self.device.backend == SOUNDDEVICE else s.is_active()
        except Exception:
            return False

    def _begin(self) -> None:
        self._frames = 0
        self.overflows = 0
        if self.device.backend == SOUNDDEVICE:
            import sounddevice as sd

            self._stream = sd.InputStream(device=self.device.index, channels=self.channels,
                                          samplerate=self.sample_rate,
                                          blocksize=self.blocksize, dtype="float32",
                                          callback=self._sd_callback)
            self._stream.start()
        else:
            import pyaudiowpatch as pyaudio

            self._pa = pyaudio.PyAudio()
            self._paContinue = pyaudio.paContinue
            self._stream = self._pa.open(format=pyaudio.paFloat32, channels=self.channels,
                                         rate=int(self.sample_rate), input=True,
                                         input_device_index=self.device.index,
                                         frames_per_buffer=self.blocksize or 1024,
                                         stream_callback=self._pa_callback)
            self._stream.start_stream()
        self.status = (f"Capturing {self.device.label} @ {self.sample_rate:g} Hz, "
                       f"{self.channels} ch")
        if self.device.backend == LOOPBACK:
            self.status += " (no data while nothing is playing)"

    def _emit(self, block: np.ndarray) -> None:
        n = len(block)
        x = (self._frames + np.arange(n)) / self.sample_rate
        self._frames += n
        self.push(x, block, self.names)

    def _sd_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.overflows += 1
        self._emit(indata.copy())

    def _pa_callback(self, in_data, frame_count, time_info, status):
        if status:
            self.overflows += 1
        self._emit(np.frombuffer(in_data, dtype=np.float32).reshape(-1, self.channels))
        return None, self._paContinue

    def _end(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            if self.device.backend == SOUNDDEVICE:
                stream.stop()
                stream.close()
            else:
                stream.stop_stream()
                stream.close()
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
