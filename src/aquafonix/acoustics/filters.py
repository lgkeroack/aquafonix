"""Band isolation (linear-phase FIR filters), resampling and denoising.

Filters are windowed-sinc FIRs applied by FFT (overlap-save), so they are
fast even with tens of thousands of taps and have exactly (N-1)/2 samples of
delay, which the display compensates for: filtered and raw traces line up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FILTER_TYPES = {
    "bandpass": "Band-pass (keep a band)",
    "lowpass": "Low-pass (keep below)",
    "highpass": "High-pass (keep above)",
    "bandstop": "Band-stop / notch (remove a band)",
}

#: (label, type, low Hz, high Hz)
PRESETS = [
    ("MSFD 63 Hz third-octave", "bandpass", 56.2, 70.8),
    ("MSFD 125 Hz third-octave", "bandpass", 112.0, 141.0),
    ("Shipping band 20–500 Hz", "bandpass", 20.0, 500.0),
    ("Low 10–1000 Hz", "bandpass", 10.0, 1000.0),
    ("Mid 0.5–5 kHz", "bandpass", 500.0, 5000.0),
    ("High 5–20 kHz", "bandpass", 5000.0, 20000.0),
    ("Ultrasonic > 20 kHz", "highpass", 20000.0, 0.0),
    ("Remove rumble < 100 Hz", "highpass", 100.0, 0.0),
    ("Remove 50 Hz hum", "bandstop", 47.0, 53.0),
    ("Remove 60 Hz hum", "bandstop", 57.0, 63.0),
]

MAX_TAPS = 32769


@dataclass(frozen=True)
class BandSpec:
    kind: str = "bandpass"
    low: float = 100.0
    high: float = 1000.0

    def edges(self, sample_rate: float) -> tuple[float, float]:
        """Effective (low, high) pass band in Hz, clamped to Nyquist."""
        nyq = sample_rate / 2
        if self.kind == "lowpass":
            return 0.0, min(self.low, nyq)
        if self.kind == "highpass":
            return min(self.low, nyq), nyq
        lo, hi = sorted((self.low, self.high))
        return max(lo, 0.0), min(hi, nyq)

    def describe(self) -> str:
        if self.kind == "lowpass":
            return f"< {fmt_hz(self.low)}"
        if self.kind == "highpass":
            return f"> {fmt_hz(self.low)}"
        lo, hi = sorted((self.low, self.high))
        word = "notch " if self.kind == "bandstop" else ""
        return f"{word}{fmt_hz(lo)}–{fmt_hz(hi)}"


def fmt_hz(f: float) -> str:
    return f"{f / 1000:g} kHz" if f >= 1000 else f"{f:g} Hz"


def _sinc_lowpass(fc: float, fs: float, n: int, window: np.ndarray) -> np.ndarray:
    m = np.arange(n) - (n - 1) / 2
    return 2 * fc / fs * np.sinc(2 * fc / fs * m) * window


def design_fir(spec: BandSpec, sample_rate: float, transition: float | None = None,
               attenuation_db: float = 60.0) -> np.ndarray:
    """Kaiser-windowed sinc FIR (odd length) for ``spec``."""
    fs = float(sample_rate)
    nyq = fs / 2
    if spec.kind not in FILTER_TYPES:
        raise ValueError(f"unknown filter type {spec.kind!r}")
    if spec.kind in ("lowpass", "highpass"):
        lo = hi = float(spec.low)
        if not 0 < lo < nyq:
            raise ValueError(f"cut-off must be between 0 and {fmt_hz(nyq)}")
    else:
        lo, hi = sorted((float(spec.low), float(spec.high)))
        if not 0 < lo < nyq:
            raise ValueError(f"lower edge must be between 0 and {fmt_hz(nyq)}")
        hi = min(hi, nyq * 0.999)
        if hi <= lo:
            raise ValueError("upper edge must be above the lower edge")
    tw = transition or max(lo * 0.2, fs * 1e-4, 1.0)
    beta = 0.1102 * (attenuation_db - 8.7) if attenuation_db > 50 else 4.0
    n = int(np.ceil((attenuation_db - 8) / (2.285 * 2 * np.pi * tw / fs))) + 1
    n = min(max(n, 31), MAX_TAPS)
    n += (n + 1) % 2  # odd length -> integer delay, any response type
    w = np.kaiser(n, beta)
    centre = (n - 1) // 2
    if spec.kind == "lowpass":
        h = _sinc_lowpass(lo, fs, n, w)
    elif spec.kind == "highpass":
        h = -_sinc_lowpass(lo, fs, n, w)
        h[centre] += 1.0
    elif spec.kind == "bandpass":
        h = _sinc_lowpass(hi, fs, n, w) - _sinc_lowpass(lo, fs, n, w)
    else:  # bandstop
        h = _sinc_lowpass(lo, fs, n, w) - _sinc_lowpass(hi, fs, n, w)
        h[centre] += 1.0
    return h


class StreamingFIR:
    """Overlap-save FFT convolution for blocks of shape ``(n, channels)``.

    Output sample ``k`` is the filtered input at ``k - delay``.
    """

    def __init__(self, taps: np.ndarray, channels: int) -> None:
        self.taps = np.asarray(taps, dtype=np.float64)
        self.n = len(self.taps)
        self.delay = (self.n - 1) // 2
        self.channels = int(channels)
        self.fft_len = 1 << int(np.ceil(np.log2(max(4 * self.n, 4096))))
        self.block = self.fft_len - self.n + 1
        self._H = np.fft.rfft(self.taps, self.fft_len)[:, None]
        self._tail = np.zeros((self.n - 1, self.channels))

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] != self.channels:  # channel count changed: restart
            self.channels = x.shape[1]
            self._tail = np.zeros((self.n - 1, self.channels))
        buf = np.concatenate([self._tail, np.nan_to_num(x)])
        out = np.empty_like(x)
        m = len(x)
        for start in range(0, m, self.block):
            stop = min(start + self.block, m)
            seg = buf[start:stop + self.n - 1]
            y = np.fft.irfft(np.fft.rfft(seg, self.fft_len, axis=0) * self._H, self.fft_len, axis=0)
            out[start:stop] = y[self.n - 1:self.n - 1 + (stop - start)]
        self._tail = buf[len(buf) - (self.n - 1):] if self.n > 1 else buf[:0]
        return out


def filter_clip(x: np.ndarray, spec: BandSpec, sample_rate: float) -> np.ndarray:
    """Zero-delay filtering of a whole clip (edges padded with reflection)."""
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    taps = design_fir(spec, sample_rate)
    pad = min(len(taps), len(x) - 1) if len(x) > 1 else 0
    padded = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]]) if pad else x
    n = len(padded) + len(taps) - 1
    nfft = 1 << int(np.ceil(np.log2(n)))
    y = np.fft.irfft(np.fft.rfft(padded, nfft, axis=0) * np.fft.rfft(taps, nfft)[:, None],
                     nfft, axis=0)
    d = (len(taps) - 1) // 2
    y = y[d + pad:d + pad + len(x)]
    return y[:, 0] if squeeze else y


def resample(x: np.ndarray, src_rate: float, dst_rate: float) -> np.ndarray:
    """Band-limited (FFT) resampling of a clip along axis 0."""
    x = np.asarray(x, dtype=np.float64)
    if abs(src_rate - dst_rate) < 1e-9 or len(x) < 2:
        return x.copy()
    n_out = max(1, int(round(len(x) * dst_rate / src_rate)))
    spec = np.fft.rfft(x, axis=0)
    k = n_out // 2 + 1
    if k <= spec.shape[0]:
        spec = spec[:k]
    else:
        pad = np.zeros((k - spec.shape[0],) + spec.shape[1:], dtype=spec.dtype)
        spec = np.concatenate([spec, pad])
    return np.fft.irfft(spec, n_out, axis=0) * (n_out / len(x))


def denoise(x: np.ndarray, sample_rate: float, strength: float = 1.5,
            noise_profile: np.ndarray | None = None, nfft: int = 1024) -> np.ndarray:
    """Spectral subtraction for listening: estimate the noise spectrum (per-
    frequency 20th percentile, or ``noise_profile`` magnitudes) and attenuate
    time-frequency cells that don't rise above it."""
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    hop = nfft // 4
    win = np.hanning(nfft)
    n = len(x)
    if n < nfft:
        return x[:, 0] if squeeze else x
    pad = np.concatenate([np.zeros((nfft, x.shape[1])), x, np.zeros((nfft, x.shape[1]))])
    frames = np.lib.stride_tricks.sliding_window_view(pad, nfft, axis=0)[::hop]  # (f, ch, nfft)
    spec = np.fft.rfft(frames * win, axis=-1)
    mag = np.abs(spec)
    noise = np.percentile(mag, 20, axis=0) if noise_profile is None else noise_profile
    gain = np.clip(1 - strength * noise / np.maximum(mag, 1e-12), 0.08, 1.0)
    out_frames = np.fft.irfft(spec * gain, nfft, axis=-1) * win
    out = np.zeros_like(pad)
    norm = np.zeros(len(pad))
    for i in range(out_frames.shape[0]):
        s = i * hop
        out[s:s + nfft] += out_frames[i].T
        norm[s:s + nfft] += win**2
    out /= np.maximum(norm, 1e-9)[:, None]
    out = out[nfft:nfft + n]
    return out[:, 0] if squeeze else out
