"""Built-in test signal generator (no hardware needed)."""

from __future__ import annotations

import numpy as np

from aquafonix.sources.base import Source

WAVEFORMS = ("sine", "square", "triangle", "sawtooth", "noise", "chirp")


def waveform(kind: str, t: np.ndarray, freq: float) -> np.ndarray:
    phase = (t * freq) % 1.0
    if kind == "sine":
        return np.sin(2 * np.pi * phase)
    if kind == "square":
        return np.where(phase < 0.5, 1.0, -1.0)
    if kind == "triangle":
        return 4 * np.abs(phase - 0.5) - 1
    if kind == "sawtooth":
        return 2 * phase - 1
    if kind == "noise":
        return np.random.default_rng().uniform(-1, 1, len(t))
    if kind == "chirp":  # sweeps freq .. 10*freq every 2 s
        tau = t % 2.0
        return np.sin(2 * np.pi * freq * (tau + 9 * tau**2 / 4))
    raise ValueError(f"unknown waveform {kind!r}")


class GeneratorSource(Source):
    def __init__(self, kind: str = "sine", freq: float = 5.0, amplitude: float = 1.0,
                 sample_rate: float = 1000.0, channels: int = 1, noise: float = 0.0) -> None:
        super().__init__()
        if kind not in WAVEFORMS:
            raise ValueError(f"unknown waveform {kind!r}")
        self.kind = kind
        self.freq = float(freq)
        self.amplitude = float(amplitude)
        self.sample_rate = float(sample_rate)
        self.channels = max(1, int(channels))
        self.noise = float(noise)
        self.buffer_capacity = int(self.sample_rate * 60)
        self._generated = 0
        self._rng = np.random.default_rng()

    def _begin(self) -> None:
        self._generated = 0
        self.status = f"Generating {self.kind} {self.freq:g} Hz @ {self.sample_rate:g} S/s"

    def _step(self) -> None:
        due = int(self.now() * self.sample_rate) - self._generated
        if due > 0:
            t = (self._generated + np.arange(due)) / self.sample_rate
            cols = []
            for ch in range(self.channels):
                shift = ch / (self.channels * self.freq) if self.freq else 0.0
                y = self.amplitude * waveform(self.kind, t + shift, self.freq)
                if self.noise:
                    y = y + self._rng.normal(0, self.noise, due)
                cols.append(y)
            self.push(t, np.column_stack(cols))
            self._generated += due
        self._stop.wait(0.01)
