"""Replay a recorded audio file (WAV/FLAC/...) through the whole pipeline."""

from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np

from aquafonix.sources.base import Source


class AudioFileSource(Source):
    """Streams an audio file into the buffer at ``speed`` x real time.

    x is *file time* in seconds, so detections and events refer to positions
    in the file.  ``speed == 0`` means "scan": as fast as the analysis keeps
    up (``throttle`` returns the absolute sample index it has processed).
    """

    sample_dtype = np.float32

    def __init__(self, path: str, speed: float = 1.0, loop: bool = False,
                 start: float = 0.0) -> None:
        super().__init__()
        import soundfile as sf

        if not path:
            raise ValueError("choose an audio file")
        with sf.SoundFile(path) as f:
            self.sample_rate = float(f.samplerate)
            self.channels = int(f.channels)
            self.frames = int(f.frames)
        self.path = path
        self.speed = float(speed)
        self.loop = loop
        self.paused = False
        self.finished = False
        self.throttle: Callable[[], int] | None = None
        self.names = (["Left", "Right"] if self.channels == 2
                      else [f"ch{i + 1}" for i in range(self.channels)])
        self.buffer_capacity = int(self.sample_rate * 60)
        self._seek_to: float | None = start
        self._pos = 0
        self._clock = (time.perf_counter(), 0)
        self._f = None

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    @property
    def position(self) -> float:
        return self._pos / self.sample_rate

    def seek(self, seconds: float) -> None:
        self._seek_to = float(seconds)

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)
        self._clock = (time.perf_counter(), self._pos)

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self._clock = (time.perf_counter(), self._pos)
        if not paused and self.finished:
            self.finished = False
            self.seek(0.0)

    def _begin(self) -> None:
        import soundfile as sf

        self._f = sf.SoundFile(self.path)
        self.finished = False
        self._update_status()

    def _update_status(self) -> None:
        state = "finished" if self.finished else "paused" if self.paused else (
            "scanning" if self.speed <= 0 else f"{self.speed:g}×")
        self.status = (f"{os.path.basename(self.path)}  {self.position:8.2f} / "
                       f"{self.duration:.2f} s  ({state})")

    def _step(self) -> None:
        if self._seek_to is not None:
            frame = int(min(max(self._seek_to, 0.0), self.duration) * self.sample_rate)
            self._f.seek(frame)
            self._pos = frame
            self._seek_to = None
            self.finished = False
            if self.buffer is not None:
                self.buffer.rebase(frame / self.sample_rate)
            self._clock = (time.perf_counter(), frame)
        if self.paused or self.finished:
            self._update_status()
            self._stop.wait(0.05)
            return
        rate = self.sample_rate
        if self.speed > 0:
            t0, pos0 = self._clock
            target = pos0 + int((time.perf_counter() - t0) * self.speed * rate)
            n = min(target - self._pos, int(0.25 * rate * self.speed) + 1)
        else:
            n = int(0.25 * rate)
            if self.throttle is not None and self.buffer is not None:
                limit = self.buffer.capacity // 2
                while self.buffer.total - self.throttle() > limit and not self._stop.is_set():
                    self._stop.wait(0.01)
        if n <= 0:
            self._stop.wait(0.01)
            return
        data = self._f.read(n, dtype="float32", always_2d=True)
        if len(data):
            x = (self._pos + np.arange(len(data))) / rate
            self.push(x, data, self.names)
            self._pos += len(data)
        if len(data) < n:  # end of file
            if self.loop:
                self._seek_to = 0.0
            else:
                self.finished = True
        self._update_status()
        if self.speed > 0:
            self._stop.wait(0.02)

    def _end(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
