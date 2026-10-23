"""Small signal-processing helpers used by the display."""

from __future__ import annotations

import numpy as np


def estimate_rate(x: np.ndarray) -> float | None:
    """Sample rate from (roughly evenly spaced) timestamps."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return None
    dt = np.diff(x[np.isfinite(x)])
    dt = dt[dt > 0]
    if not len(dt):
        return None
    return float(1.0 / np.median(dt))


def spectrum(y: np.ndarray, rate: float, nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum in dB of the last ``nfft`` finite samples."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)][-nfft:]
    n = len(y)
    if n < 8 or not rate:
        return np.zeros(0), np.zeros(0)
    window = np.hanning(n)
    mag = np.abs(np.fft.rfft((y - y.mean()) * window)) * 2.0 / window.sum()
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    return freqs, 20 * np.log10(mag + 1e-12)


def find_trigger(x: np.ndarray, y: np.ndarray, level: float, window: float,
                 rising: bool = True, pretrigger: float = 0.1) -> float | None:
    """x position of the latest edge through ``level`` that leaves a full
    ``window`` of data around it (``pretrigger`` fraction before the edge)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None
    a, b = y[:-1], y[1:]
    edges = (a < level) & (b >= level) if rising else (a > level) & (b <= level)
    idx = np.flatnonzero(edges) + 1
    if not len(idx):
        return None
    xa, xb = x[idx - 1], x[idx]
    frac = (level - y[idx - 1]) / (y[idx] - y[idx - 1])
    xt = xa + frac * (xb - xa)
    ok = (xt - pretrigger * window >= x[0]) & (xt + (1 - pretrigger) * window <= x[-1])
    if not ok.any():
        return None
    return float(xt[ok][-1])


def minmax_decimate(x: np.ndarray, y: np.ndarray, max_points: int):
    """Reduce (x, y[n, ch]) to about ``max_points`` while keeping peaks."""
    n = len(x)
    if n <= max_points or max_points < 4:
        return x, y
    bins = max_points // 2
    step = n // bins
    m = bins * step
    xs = x[n - m:].reshape(bins, step)
    ys = y[n - m:].reshape(bins, step, -1)
    with np.errstate(all="ignore"):
        lo = np.nanmin(ys, axis=1)
        hi = np.nanmax(ys, axis=1)
    xo = xs[:, [0, -1]].reshape(-1)
    yo = np.stack([lo, hi], axis=1).reshape(bins * 2, -1)
    return xo, yo
