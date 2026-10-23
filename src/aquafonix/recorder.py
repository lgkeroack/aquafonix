"""Record incoming samples to CSV or WAV.  ``write`` is called from source
threads, ``close`` from the GUI, so both take a lock."""

from __future__ import annotations

import threading
import wave

import numpy as np


class CsvRecorder:
    def __init__(self, path: str, x_label: str = "time_s") -> None:
        self.path = path
        self.x_label = x_label
        self.rows = 0
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._names: list[str] | None = None
        self._lock = threading.Lock()

    def write(self, x: np.ndarray, y: np.ndarray, names: list[str] | None) -> None:
        y = np.asarray(y, dtype=float).reshape(len(x), -1)
        names = list(names) if names else [f"ch{i + 1}" for i in range(y.shape[1])]
        with self._lock:
            if self._f is None:
                return
            if self._names is None:
                self._names = list(names)
                self._f.write(",".join([self.x_label, *self._names]) + "\n")
            new = [n for n in names if n not in self._names]
            if new:  # a channel appeared mid-recording: extend the columns
                self._names.extend(new)
                self._f.write("# columns: " + ",".join([self.x_label, *self._names]) + "\n")
            if names == self._names:
                rows = y
            else:
                rows = np.full((len(x), len(self._names)), np.nan)
                rows[:, [self._names.index(n) for n in names]] = y
            np.savetxt(self._f, np.column_stack([x, rows]), delimiter=",", fmt="%.10g")
            self.rows += len(x)

    def close(self) -> None:
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None


class WavRecorder:
    """16-bit PCM WAV; samples are expected in [-1, 1] (audio sources)."""

    def __init__(self, path: str, sample_rate: float, channels: int) -> None:
        self.path = path
        self.channels = int(channels)
        self.rows = 0
        self._w = wave.open(path, "wb")
        self._w.setnchannels(self.channels)
        self._w.setsampwidth(2)
        self._w.setframerate(int(round(sample_rate)))
        self._lock = threading.Lock()

    def write(self, x: np.ndarray, y: np.ndarray, names: list[str] | None) -> None:
        y = np.nan_to_num(np.asarray(y, dtype=float).reshape(len(x), -1))
        if y.shape[1] < self.channels:
            y = np.hstack([y, np.zeros((len(y), self.channels - y.shape[1]))])
        pcm = (np.clip(y[:, : self.channels], -1, 1) * 32767).astype("<i2")
        with self._lock:
            if self._w is None:
                return
            self._w.writeframes(pcm.tobytes())
            self.rows += len(pcm)

    def close(self) -> None:
        with self._lock:
            if self._w is not None:
                self._w.close()
                self._w = None
