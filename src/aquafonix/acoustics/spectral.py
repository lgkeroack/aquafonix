"""Short-time spectra shared by the live analyzer and template building, so
templates and live data are measured exactly the same way."""

from __future__ import annotations

import numpy as np

SNR_CLIP_DB = 40.0


def auto_nfft(sample_rate: float) -> int:
    """FFT size giving ~20-45 ms frames (good for calls and clicks alike)."""
    target = sample_rate * 0.03
    n = 1 << int(np.round(np.log2(max(target, 32))))
    return int(min(max(n, 32), 16384))


def power_frames(x: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    """Per-bin mean-square power of Hann-windowed frames, shape (frames, bins).

    Scaled so that summing the bins of a frame gives the signal's mean
    square (Parseval), i.e. ``10*log10(sum)`` is the frame's RMS level."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < nfft:
        return np.zeros((0, nfft // 2 + 1))
    win = np.hanning(nfft)
    frames = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop]
    spec = np.fft.rfft(frames * win, axis=1)
    return (2.0 / (nfft * np.sum(win**2))) * (spec.real**2 + spec.imag**2)


def to_db(power) -> np.ndarray:
    return 10 * np.log10(np.maximum(power, 1e-30))


def blur_freq(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """3-bin moving average along frequency: tolerates small pitch shifts."""
    a = np.moveaxis(np.asarray(a, dtype=np.float64), axis, -1)
    if a.shape[-1] < 3:
        return np.moveaxis(a, -1, axis)
    p = np.concatenate([a[..., :1], a, a[..., -1:]], axis=-1)
    out = (p[..., :-2] + p[..., 1:-1] + p[..., 2:]) / 3.0
    return np.moveaxis(out, -1, axis)


def snr_frames(db_frames: np.ndarray, floor_db: np.ndarray) -> np.ndarray:
    """Floor-relative level, clipped to [0, SNR_CLIP_DB]."""
    return np.clip(db_frames - floor_db, 0.0, SNR_CLIP_DB)


class FloorTracker:
    """Per-frequency noise floor in dB.

    The power in each bin is first smoothed over ``smooth_s`` (so the floor
    sits near the background level instead of under its random dips), then
    tracked asymmetrically: decreases are followed quickly (``fall_s``),
    increases slowly (``rise_s``).  Short calls and clicks barely move it; a
    passing ship does."""

    def __init__(self, bins: int, frame_period: float, rise_s: float = 20.0,
                 fall_s: float = 2.0, smooth_s: float = 0.15) -> None:
        self.floor: np.ndarray | None = None
        self._smooth: np.ndarray | None = None
        self.bins = bins
        self.smooth_s = smooth_s
        self.set_times(frame_period, rise_s, fall_s)

    def set_times(self, frame_period: float, rise_s: float, fall_s: float) -> None:
        self.frame_period = frame_period
        self.a_rise = 1 - np.exp(-frame_period / max(rise_s, 1e-3))
        self.a_fall = 1 - np.exp(-frame_period / max(fall_s, 1e-3))
        self.a_smooth = 1 - np.exp(-frame_period / max(self.smooth_s, 1e-3))

    def update(self, db_frames: np.ndarray) -> np.ndarray:
        """Feed frames (n, bins) in dB; returns the floor after each frame."""
        out = np.empty_like(db_frames)
        for i, d in enumerate(db_frames):
            p = 10 ** (d / 10)
            if self._smooth is None:
                self._smooth = p.copy()
                self.floor = to_db(p)
            else:
                self._smooth += self.a_smooth * (p - self._smooth)
                s = to_db(self._smooth)
                diff = s - self.floor
                self.floor += np.where(diff > 0, self.a_rise, self.a_fall) * diff
            out[i] = self.floor
        return out
