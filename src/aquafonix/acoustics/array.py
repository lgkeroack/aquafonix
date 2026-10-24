"""Bearing from a pair of hydrophones (time difference of arrival).

GCC-PHAT (generalised cross-correlation with phase transform) gives the
delay between the two channels; with spacing ``d`` and sound speed ``c``
the bearing relative to the array axis is ``theta = arccos(c * tau / d)``:
0 deg = end-fire towards hydrophone B, 90 deg = broadside, 180 deg = end-fire
towards A.  A two-element array cannot tell left from right of its axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEAWATER_SOUND_SPEED = 1500.0  # m/s (typical; 1450-1540 depending on T, S, depth)


@dataclass(frozen=True)
class ArrayGeometry:
    enabled: bool = False
    ch_a: int = 0
    ch_b: int = 1
    spacing_m: float = 1.0
    sound_speed: float = SEAWATER_SOUND_SPEED

    @property
    def max_delay(self) -> float:
        return self.spacing_m / self.sound_speed


def _next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(n, 2))))


def gcc_phat(a: np.ndarray, b: np.ndarray, sample_rate: float, max_delay: float,
             band: tuple[float, float] | None = None, interp: int = 16):
    """Cross-correlation of ``a`` against ``b`` over lags within
    +-``max_delay``.  Returns ``(lags_s, correlation)``; a peak at lag
    ``tau > 0`` means ``a`` hears the sound ``tau`` seconds after ``b``."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    b = np.nan_to_num(np.asarray(b, dtype=np.float64))
    n = min(len(a), len(b))
    if n < 16:
        return np.zeros(0), np.zeros(0)
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    nfft = _next_pow2(2 * n)
    spec = np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft))
    spec /= np.abs(spec) + 1e-12  # phase transform: whiten, sharpen the peak
    if band is not None:
        f = np.fft.rfftfreq(nfft, 1 / sample_rate)
        spec[(f < band[0]) | (f > band[1])] = 0
    cc = np.fft.irfft(spec, nfft * interp)
    fs_i = sample_rate * interp
    max_shift = int(min(np.ceil(max_delay * fs_i) + interp, len(cc) // 2 - 1))
    cc = np.concatenate([cc[-max_shift:], cc[:max_shift + 1]])
    lags = np.arange(-max_shift, max_shift + 1) / fs_i
    return lags, cc


def tdoa(a: np.ndarray, b: np.ndarray, sample_rate: float, max_delay: float,
         band: tuple[float, float] | None = None) -> tuple[float, float]:
    """``(delay seconds, peak quality)``: quality is the peak height over
    the median absolute correlation (roughly, how clear the arrival is)."""
    lags, cc = gcc_phat(a, b, sample_rate, max_delay, band)
    if not len(cc):
        return float("nan"), 0.0
    inside = np.abs(lags) <= max_delay * 1.02
    if not inside.any():
        return float("nan"), 0.0
    i = int(np.argmax(np.where(inside, cc, -np.inf)))
    quality = float(cc[i] / (np.median(np.abs(cc)) + 1e-12))
    return float(lags[i]), quality


def bearing_from_delay(tau: float, geometry: ArrayGeometry) -> float:
    """Bearing in degrees from the array axis (0 = towards B, 180 = towards A)."""
    if not np.isfinite(tau) or geometry.spacing_m <= 0:
        return float("nan")
    cos_t = np.clip(geometry.sound_speed * tau / geometry.spacing_m, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def bearing_scan(a: np.ndarray, b: np.ndarray, sample_rate: float, geometry: ArrayGeometry,
                 bearings_deg: np.ndarray, band: tuple[float, float] | None = None) -> np.ndarray:
    """Correlation strength for each look direction (for a bearing-time
    display): the strongest GCC-PHAT value over the delays each bearing cell
    spans, so the very sharp PHAT peak can't fall between grid points."""
    bearings_deg = np.asarray(bearings_deg, dtype=float)
    lags, cc = gcc_phat(a, b, sample_rate, geometry.max_delay, band)
    if not len(cc):
        return np.zeros(len(bearings_deg))
    point = np.interp(geometry.spacing_m * np.cos(np.radians(bearings_deg)) / geometry.sound_speed,
                      lags, cc)
    if len(bearings_deg) < 2:
        return point
    mid = (bearings_deg[1:] + bearings_deg[:-1]) / 2
    edges = np.clip(np.r_[bearings_deg[0] - (mid[0] - bearings_deg[0]), mid,
                          bearings_deg[-1] + (bearings_deg[-1] - mid[-1])], 0.0, 180.0)
    tau_edges = geometry.spacing_m * np.cos(np.radians(edges)) / geometry.sound_speed
    lo = np.searchsorted(lags, np.minimum(tau_edges[:-1], tau_edges[1:]), side="left")
    hi = np.searchsorted(lags, np.maximum(tau_edges[:-1], tau_edges[1:]), side="right")
    out = point.copy()
    for i in np.flatnonzero(hi > lo):
        out[i] = max(out[i], cc[lo[i]:hi[i]].max())
    return out
