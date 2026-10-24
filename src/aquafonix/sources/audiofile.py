"""Replay recorded audio (WAV/FLAC/...) through the whole pipeline.

A recording can be one file or many consecutive files (a recorder's
folder): they are played back as one continuous stream.  Start times are
read from file names (``20231015_134501.wav``, SoundTrap's
``5678.231015134501.wav``, ``2023-10-15T13-45-01.flac``, ...) so detections
and events get real timestamps.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from aquafonix.sources.base import Source

AUDIO_EXTENSIONS = (".wav", ".flac", ".aif", ".aiff", ".ogg", ".w64", ".rf64", ".caf")

_TS_PATTERNS = [
    # 2023-10-15T13-45-01 / 2023-10-15_13.45.01 / 20231015_134501 / 20231015T134501
    re.compile(r"(?<!\d)(\d{4})[-_]?(\d{2})[-_]?(\d{2})[T_\-\s.]?(\d{2})[-_.:]?(\d{2})[-_.:]?"
               r"(\d{2})(?:\.(\d{1,6}))?(?!\d)"),
    # SoundTrap / 12-digit YYMMDDhhmmss
    re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?!\d)"),
]


def parse_file_timestamp(name: str) -> float | None:
    """Start time (epoch seconds, local time) encoded in a file name."""
    stem = Path(name).stem
    for i, rx in enumerate(_TS_PATTERNS):
        for m in rx.finditer(stem):
            g = m.groups()
            try:
                year = int(g[0]) + (2000 if i == 1 else 0)
                frac = float("0." + g[6]) if i == 0 and g[6] else 0.0
                t = dt.datetime(year, int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]))
                if 1980 <= t.year <= 2100:
                    return t.timestamp() + frac
            except (ValueError, TypeError):
                continue
    return None


def expand_paths(spec) -> list[str]:
    """A path, a folder, or ``;``-separated paths -> sorted audio files."""
    items = spec if isinstance(spec, (list, tuple)) else [p for p in str(spec).split(";")]
    out: list[str] = []
    for item in (p.strip() for p in items):
        if not item:
            continue
        if os.path.isdir(item):
            out += sorted(str(p) for p in Path(item).iterdir()
                          if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file())
        else:
            out.append(item)
    return out


@dataclass
class Segment:
    path: str
    start: int  # first frame of this file in the continuous recording
    frames: int
    wall_start: float | None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def scan_recording(paths: list[str]) -> tuple[list[Segment], float, int, list[str]]:
    """``(segments, sample_rate, channels, skipped)``; files whose rate or
    channel count differ from the first one are skipped."""
    import soundfile as sf

    segments, skipped = [], []
    rate = channels = None
    start = 0
    for path in paths:
        try:
            info = sf.info(path)
        except Exception as exc:
            skipped.append(f"{os.path.basename(path)} ({exc})")
            continue
        if rate is None:
            rate, channels = float(info.samplerate), int(info.channels)
        elif info.samplerate != rate or info.channels != channels:
            skipped.append(f"{os.path.basename(path)} ({info.samplerate} Hz, {info.channels} ch)")
            continue
        segments.append(Segment(path, start, int(info.frames), parse_file_timestamp(path)))
        start += int(info.frames)
    if not segments:
        raise ValueError("no readable audio files" + (f": {'; '.join(skipped)}" if skipped else ""))
    return segments, rate, channels, skipped


class AudioFileSource(Source):
    """Streams a recording into the buffer at ``speed`` x real time.

    x is *recording time* in seconds from the start of the first file.
    ``speed == 0`` means "scan": as fast as the analysis keeps up
    (``throttle`` returns the absolute sample index it has processed).
    """

    sample_dtype = np.float32

    def __init__(self, paths, speed: float = 1.0, loop: bool = False,
                 start: float = 0.0) -> None:
        super().__init__()
        files = expand_paths(paths)
        if not files:
            raise ValueError("choose an audio file or a folder of recordings")
        self.segments, self.sample_rate, self.channels, self.skipped = scan_recording(files)
        self.frames = self.segments[-1].start + self.segments[-1].frames
        self.path = self.segments[0].path
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
        self._seg = 0
        self._clock = (time.perf_counter(), 0)
        self._f = None

    # -- recording geometry -------------------------------------------------------
    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    @property
    def position(self) -> float:
        return self._pos / self.sample_rate

    @property
    def current_file(self) -> str:
        return self.segments[self._seg].name

    def segment_at(self, t: float) -> Segment:
        frame = int(t * self.sample_rate)
        for seg in reversed(self.segments):
            if frame >= seg.start:
                return seg
        return self.segments[0]

    def wall_time(self, t: float) -> float | None:
        """Wall-clock time of recording time ``t`` (if file names carry it)."""
        seg = self.segment_at(t)
        if seg.wall_start is None:
            return None
        return seg.wall_start + t - seg.start / self.sample_rate

    def read(self, t0: float, duration: float) -> np.ndarray:
        """Random access across files (used by the recording overview)."""
        import soundfile as sf

        frame, need = int(t0 * self.sample_rate), int(duration * self.sample_rate)
        chunks = []
        for seg in self.segments:
            if need <= 0:
                break
            if frame >= seg.start + seg.frames or frame + need <= seg.start:
                continue
            off = max(0, frame - seg.start)
            n = min(need, seg.frames - off)
            with sf.SoundFile(seg.path) as f:
                f.seek(off)
                chunks.append(f.read(n, dtype="float32", always_2d=True))
            frame += n
            need -= n
        return np.concatenate(chunks) if chunks else np.zeros((0, self.channels), np.float32)

    # -- transport -------------------------------------------------------------------
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

    # -- streaming -------------------------------------------------------------------
    def _open(self, index: int, offset: int = 0) -> None:
        import soundfile as sf

        if self._f is not None:
            self._f.close()
        self._seg = index
        self._f = sf.SoundFile(self.segments[index].path)
        self._f.seek(offset)

    def _begin(self) -> None:
        self._open(0)
        self.finished = False
        self._update_status()

    def _update_status(self) -> None:
        state = "finished" if self.finished else "paused" if self.paused else (
            "scanning" if self.speed <= 0 else f"{self.speed:g}×")
        files = f" [{self._seg + 1}/{len(self.segments)}]" if len(self.segments) > 1 else ""
        self.status = (f"{self.current_file}{files}  {self.position:9.2f} / "
                       f"{self.duration:.2f} s  ({state})")

    def _do_seek(self) -> None:
        frame = int(min(max(self._seek_to, 0.0), self.duration) * self.sample_rate)
        self._seek_to = None
        index = next((i for i in range(len(self.segments) - 1, -1, -1)
                      if frame >= self.segments[i].start), 0)
        self._open(index, min(frame - self.segments[index].start, self.segments[index].frames))
        self._pos = frame
        self.finished = False
        if self.buffer is not None:
            self.buffer.rebase(frame / self.sample_rate)
        self._clock = (time.perf_counter(), frame)

    def _read(self, n: int) -> np.ndarray:
        chunks = []
        while n > 0:
            data = self._f.read(n, dtype="float32", always_2d=True)
            if len(data):
                chunks.append(data)
                n -= len(data)
            if n > 0:  # end of this file
                if self._seg + 1 >= len(self.segments):
                    break
                self._open(self._seg + 1)
        if not chunks:
            return np.zeros((0, self.channels), np.float32)
        return np.concatenate(chunks)

    def _step(self) -> None:
        if self._seek_to is not None:
            self._do_seek()
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
        data = self._read(n)
        if len(data):
            x = (self._pos + np.arange(len(data))) / rate
            self.push(x, data, self.names)
            self._pos += len(data)
        if len(data) < n:  # end of the recording
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
