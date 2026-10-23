"""Listening: replay clips (speed, filter, denoise) or follow the live stream.

Speed works like a bioacoustics "time expansion" control: 0.1x makes
ultrasonic clicks audible, 10x brings infrasonic whale calls up into
hearing range.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from aquafonix.acoustics.filters import BandSpec, StreamingFIR, design_fir, resample
from aquafonix.buffer import RingBuffer

log = logging.getLogger(__name__)

SPEEDS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


class Player:
    def __init__(self) -> None:
        self._stream = None
        self._lock = threading.Lock()
        self._data: np.ndarray | None = None
        self._pos = 0
        self._out_rate = 48000.0
        self.loop = False
        self.speed = 1.0
        self.gain = 1.0
        self.mode: str | None = None  # "clip" | "live"
        self.clip_start = 0.0  # stream time of the clip's first sample
        self.error: str | None = None
        self._live: dict | None = None

    # -- device ------------------------------------------------------------------
    @staticmethod
    def _sd():
        import sounddevice as sd

        return sd

    def output_rate(self) -> float:
        try:
            return float(self._sd().query_devices(kind="output")["default_samplerate"])
        except Exception:
            return 48000.0

    @property
    def playing(self) -> bool:
        s = self._stream
        try:
            return s is not None and bool(s.active)
        except Exception:
            return False

    @property
    def position(self) -> float:
        """Seconds into the clip (in the clip's own time base)."""
        if self.mode != "clip" or self._data is None:
            return 0.0
        return self._pos / self._out_rate * self.speed

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.exception("closing audio output")
        self.mode = None
        self._live = None

    # -- clips ---------------------------------------------------------------------
    def play(self, data: np.ndarray, rate: float, speed: float = 1.0, loop: bool = False,
             normalize: bool = True, start_time: float = 0.0) -> None:
        """Play a clip ``(n,)`` or ``(n, ch)`` recorded at ``rate``."""
        self.stop()
        sd = self._sd()
        data = np.nan_to_num(np.asarray(data, dtype=np.float64))
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[1] > 2:
            data = data.mean(axis=1, keepdims=True)
        out_rate = self.output_rate()
        data = resample(data, rate * speed, out_rate)
        if normalize:
            peak = np.max(np.abs(data)) if data.size else 0.0
            if peak > 0:
                data = data * (0.9 / peak)
        self._data = (data * self.gain).astype(np.float32)
        self._pos = 0
        self._out_rate = out_rate
        self.speed = speed
        self.loop = loop
        self.clip_start = start_time
        self.mode = "clip"
        self._stream = sd.OutputStream(samplerate=out_rate, channels=self._data.shape[1],
                                       dtype="float32", callback=self._clip_cb)
        self._stream.start()

    def _clip_cb(self, outdata, frames, _time, _status) -> None:
        data = self._data
        n = len(data)
        written = 0
        while written < frames:
            chunk = data[self._pos:self._pos + frames - written]
            outdata[written:written + len(chunk)] = chunk
            written += len(chunk)
            self._pos += len(chunk)
            if self._pos >= n:
                if self.loop and n:
                    self._pos = 0
                else:
                    outdata[written:] = 0
                    raise self._sd().CallbackStop
        return None

    # -- live --------------------------------------------------------------------------
    def follow(self, buffer: RingBuffer, rate: float, channels: list[int] | None = None,
               latency: float = 0.3) -> None:
        """Monitor a (raw or band-filtered) history buffer as it fills."""
        self.stop()
        sd = self._sd()
        out_rate = self.output_rate()
        n_ch = min(2, max(1, len(buffer.names) or 1)) if channels is None else len(channels)
        fir = None
        if rate > out_rate * 1.05:  # anti-alias before decimating
            fir = StreamingFIR(design_fir(BandSpec("lowpass", 0.45 * out_rate), rate), n_ch)
        self._live = {"buffer": buffer, "rate": rate, "ratio": rate / out_rate,
                      "read": max(buffer.first_index, buffer.total - int(latency * rate)),
                      "fifo": np.zeros((0, n_ch)), "phase": 0.0, "fir": fir,
                      "channels": channels, "latency": latency}
        self._out_rate = out_rate
        self.mode = "live"
        self._stream = sd.OutputStream(samplerate=out_rate, channels=n_ch, dtype="float32",
                                       callback=self._live_cb)
        self._stream.start()

    def _live_cb(self, outdata, frames, _time, _status) -> None:
        st = self._live
        if st is None:
            outdata[:] = 0
            return
        buf, rate, ratio = st["buffer"], st["rate"], st["ratio"]
        if buf.total - st["read"] > 2 * st["latency"] * rate + rate:  # fell behind: skip
            st["read"] = buf.total - int(st["latency"] * rate)
            st["fifo"] = st["fifo"][:0]
        snap = buf.from_index(st["read"])
        if len(snap):
            st["read"] = snap.first_index + len(snap)
            y = snap.y
            if st["channels"] is not None:
                y = y[:, [c for c in st["channels"] if c < y.shape[1]] or [0]]
            n_ch = outdata.shape[1]
            if y.shape[1] < n_ch:
                y = np.repeat(y[:, :1], n_ch, axis=1)
            y = np.nan_to_num(y[:, :n_ch])
            if st["fir"] is not None:
                y = st["fir"].process(y)
            st["fifo"] = np.concatenate([st["fifo"], y])
        fifo = st["fifo"]
        need = st["phase"] + frames * ratio + 1
        if len(fifo) < need:
            outdata[:] = 0
            return
        pos = st["phase"] + np.arange(frames) * ratio
        idx = np.arange(len(fifo))
        for c in range(outdata.shape[1]):
            outdata[:, c] = np.interp(pos, idx, fifo[:, c]) * self.gain
        consumed = int(st["phase"] + frames * ratio)
        st["phase"] = st["phase"] + frames * ratio - consumed
        st["fifo"] = fifo[consumed:]
