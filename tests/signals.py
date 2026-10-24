"""Synthetic test signals."""

from __future__ import annotations

import numpy as np


def add_chirp(x: np.ndarray, fs: float, t0: float, f0: float = 1500.0, f1: float = 3000.0,
              dur: float = 0.4, amp: float = 0.1) -> None:
    """Hann-windowed linear sweep added in place."""
    i0 = int(t0 * fs)
    tt = np.arange(int(dur * fs)) / fs
    k = (f1 - f0) / dur
    x[i0:i0 + len(tt)] += amp * np.hanning(len(tt)) * np.sin(
        2 * np.pi * (f0 * tt + 0.5 * k * tt ** 2))


def calls_recording(fs: int = 16000, dur: float = 30.0, calls=(1, 7, 13, 19, 25),
                    click_at: float | None = 10.0, seed: int = 5) -> np.ndarray:
    """Background noise with upsweep calls and one loud click."""
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal(int(dur * fs))
    for c in calls:
        add_chirp(x, fs, c, 1500, 3375)
    if click_at is not None:
        i = int(click_at * fs)
        x[i:i + 30] += 0.9
    return x


def propeller_noise(fs: float, dur: float, shaft_hz: float, blades: int,
                    rng: np.random.Generator, amp: float = 0.05) -> np.ndarray:
    """Broadband noise amplitude-modulated at the shaft and blade rates."""
    t = np.arange(int(fs * dur)) / fs
    mod = (1 + 0.3 * np.cos(2 * np.pi * shaft_hz * t)
           + 0.6 * np.cos(2 * np.pi * shaft_hz * blades * t))
    return amp * rng.standard_normal(len(t)) * mod


def delayed(x: np.ndarray, fs: float, tau: float) -> np.ndarray:
    """``x`` delayed by ``tau`` seconds (fractional, circular)."""
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / fs)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * tau), n)


def pair_from_bearing(src: np.ndarray, fs: float, bearing_deg: float, spacing_m: float = 1.0,
                      c: float = 1500.0, noise: float = 0.01, seed: int = 0):
    """Two hydrophone channels (A, B) hearing ``src`` from ``bearing_deg``
    measured from the A->B axis."""
    rng = np.random.default_rng(seed)
    tau = spacing_m * np.cos(np.radians(bearing_deg)) / c  # A hears it tau after B
    a = delayed(src, fs, tau) + noise * rng.standard_normal(len(src))
    b = src + noise * rng.standard_normal(len(src))
    return a, b
