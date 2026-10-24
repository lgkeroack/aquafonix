"""Standard underwater-acoustic metrics.

Terminology follows ISO 18405:2017 (Underwater acoustics - Terminology);
fractional-octave bands use the base-10 system of IEC 61260-1:2014
(ANSI/ASA S1.11-2014).  All functions take *digital* samples plus a
calibration ``offset_db`` (see ``HydrophoneCalibration.offset_db``): with a
calibrated offset the results are in dB re 1 uPa (SPL, Lpk), dB re 1 uPa^2 s
(SEL) and dB re 1 uPa^2/Hz (PSD); with offset 0 they are relative to
digital full scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METRIC_INFO = {
    "SPL": ("Sound pressure level  L<sub>p,rms</sub>", "dB re 1 µPa"),
    "Lpk": ("Peak sound pressure level  L<sub>p,pk</sub>", "dB re 1 µPa"),
    "Lpkpk": ("Peak-to-peak sound pressure level", "dB re 1 µPa"),
    "SEL": ("Sound exposure level  L<sub>E</sub>", "dB re 1 µPa²·s"),
}


def _db(power) -> np.ndarray | float:
    return 10 * np.log10(np.maximum(power, 1e-30))


# -- broadband metrics ------------------------------------------------------------
def spl_rms(x: np.ndarray, offset_db: float = 0.0) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(_db(np.mean(x * x)) + offset_db) if len(x) else float("nan")


def spl_peak(x: np.ndarray, offset_db: float = 0.0) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(20 * np.log10(max(np.max(np.abs(x)), 1e-15)) + offset_db) if len(x) else float("nan")


def spl_peak_to_peak(x: np.ndarray, offset_db: float = 0.0) -> float:
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return float("nan")
    return float(20 * np.log10(max(np.max(x) - np.min(x), 1e-15)) + offset_db)


def sel(x: np.ndarray, sample_rate: float, offset_db: float = 0.0) -> float:
    """Sound exposure level: 10 log10( integral p^2 dt / (p0^2 * 1 s) )."""
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return float("nan")
    return float(_db(np.sum(x * x) / sample_rate) + offset_db)


def broadband(x: np.ndarray, sample_rate: float, offset_db: float = 0.0) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    out = {"duration": len(x) / sample_rate if sample_rate else float("nan"),
           "SPL": spl_rms(x, offset_db), "Lpk": spl_peak(x, offset_db),
           "Lpkpk": spl_peak_to_peak(x, offset_db), "SEL": sel(x, sample_rate, offset_db)}
    if len(x) > 4:
        xc = x - x.mean()
        rms = np.sqrt(np.mean(xc**2))
        out["crest_db"] = float(20 * np.log10(np.max(np.abs(xc)) / rms)) if rms > 0 else float("nan")
        out["kurtosis"] = float(np.mean(xc**4) / rms**4) if rms > 0 else float("nan")
    return out


# -- spectra -------------------------------------------------------------------------
def psd_frames(x: np.ndarray, sample_rate: float, nfft: int, overlap: float = 0.5):
    """One-sided PSD of each Hann-windowed frame: ``(freqs, P[frames, bins])``
    in units^2/Hz, scaled so that ``sum(P) * df`` equals the mean square."""
    x = np.asarray(x, dtype=np.float64)
    nfft = int(nfft)
    if len(x) < nfft:
        nfft = max(8, 1 << int(np.log2(max(len(x), 8))))
        if len(x) < nfft:
            return np.fft.rfftfreq(nfft, 1 / sample_rate), np.zeros((0, nfft // 2 + 1))
    hop = max(1, int(round(nfft * (1 - overlap))))
    win = np.hanning(nfft)
    frames = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop]
    spec = np.fft.rfft(frames * win, axis=1)
    p = (spec.real**2 + spec.imag**2) * (2.0 / (sample_rate * np.sum(win**2)))
    p[:, 0] /= 2
    if nfft % 2 == 0:
        p[:, -1] /= 2
    return np.fft.rfftfreq(nfft, 1 / sample_rate), p


def welch(x: np.ndarray, sample_rate: float, nfft: int, overlap: float = 0.5):
    f, p = psd_frames(x, sample_rate, nfft, overlap)
    return f, (p.mean(axis=0) if len(p) else np.zeros(len(f)))


def psd_db(psd: np.ndarray, offset_db: float = 0.0) -> np.ndarray:
    return _db(psd) + offset_db


def auto_psd_nfft(sample_rate: float, n_samples: int, resolution_hz: float = 1.0) -> int:
    """Largest power of two that fits ``n_samples`` and gives at least
    ``resolution_hz`` resolution (capped at 65536)."""
    want = 1 << int(np.ceil(np.log2(max(sample_rate / resolution_hz, 8))))
    fit = 1 << int(np.floor(np.log2(max(n_samples, 8))))
    return int(min(want, fit, 65536))


# -- fractional-octave bands --------------------------------------------------------------
@dataclass(frozen=True)
class Band:
    label: str
    centre: float
    low: float
    high: float


def nominal_label(fc: float) -> str:
    """IEC nominal mid-band label (e.g. 63 Hz, 1.25 kHz); octave centres are
    a subset of the third-octave ones, so one table serves both."""
    preferred = [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80]
    decade = 10 ** np.floor(np.log10(fc) - 1)
    cands = [p * decade for p in preferred] + [p * decade * 10 for p in preferred]
    best = min(cands, key=lambda v: abs(np.log(v / fc)))
    if best >= 1000:
        return f"{best / 1000:g} kHz"
    return f"{best:g} Hz"


def fractional_octave_bands(f_max: float, f_min: float = 10.0, fraction: int = 3) -> list[Band]:
    """Base-10 bands (IEC 61260-1) with centres 1000 * 10^(k*3/(10*fraction))."""
    bands = []
    step = 3.0 / (10 * fraction) if fraction != 3 else 0.1
    half = step / 2
    for k in range(-60, 60):
        fc = 1000.0 * 10 ** (k * step)
        lo, hi = fc * 10 ** (-half), fc * 10 ** half
        if fc >= f_min / 10**half and hi <= f_max:
            bands.append(Band(nominal_label(fc), fc, lo, hi))
    return bands


def third_octave_bands(f_max: float, f_min: float = 10.0) -> list[Band]:
    return fractional_octave_bands(f_max, f_min, 3)


def octave_bands(f_max: float, f_min: float = 16.0) -> list[Band]:
    return fractional_octave_bands(f_max, f_min, 1)


def band_mean_square(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mean-square value in [lo, hi] from a PSD (last axis = frequency), with
    each FFT bin weighted by how much of it lies inside the band - important
    for narrow low-frequency bands that are only a few bins wide."""
    if len(freqs) < 2:
        return np.zeros(psd.shape[:-1])
    df = freqs[1] - freqs[0]
    left, right = freqs - df / 2, freqs + df / 2
    overlap = np.clip(np.minimum(right, hi) - np.maximum(left, lo), 0.0, df)
    return np.tensordot(psd, overlap, axes=([-1], [0]))


def band_levels(freqs: np.ndarray, psd: np.ndarray, bands: list[Band],
                offset_db: float = 0.0) -> np.ndarray:
    """Band levels (dB) for each band; works on (..., bins) PSD arrays."""
    out = [band_mean_square(freqs, psd, b.low, b.high) for b in bands]
    return _db(np.stack(out, axis=-1)) + offset_db if out else np.zeros(psd.shape[:-1] + (0,))


def band_resolution_ok(freqs: np.ndarray, band: Band, min_bins: float = 2.0) -> bool:
    """True if the FFT resolution resolves the band (>= ``min_bins`` bins)."""
    return len(freqs) > 1 and (band.high - band.low) / (freqs[1] - freqs[0]) >= min_bins


# -- spectral descriptors ---------------------------------------------------------------
def spectral_descriptors(freqs: np.ndarray, psd: np.ndarray) -> dict[str, float]:
    p = np.asarray(psd, dtype=float)
    if len(p) < 3 or not np.any(p > 0):
        return {}
    i = int(np.argmax(p[1:])) + 1
    total = p.sum()
    centroid = float(np.sum(freqs * p) / total)
    out = {"peak_hz": float(freqs[i]), "centroid_hz": centroid,
           "rms_bandwidth_hz": float(np.sqrt(np.sum((freqs - centroid) ** 2 * p) / total))}
    for db_down, drop in ((3, 0.5), (10, 0.1)):
        thr = p[i] * drop
        lo = i
        while lo > 0 and p[lo - 1] >= thr:
            lo -= 1
        hi = i
        while hi < len(p) - 1 and p[hi + 1] >= thr:
            hi += 1
        out[f"bw{db_down}_hz"] = float(freqs[hi] - freqs[lo] + (freqs[1] - freqs[0]))
        out[f"f_lo{db_down}_hz"] = float(freqs[lo])
        out[f"f_hi{db_down}_hz"] = float(freqs[hi])
    return out


def percentiles_db(values_db: np.ndarray, axis: int = 0, q=(5, 50, 95)) -> dict[int, np.ndarray]:
    """Percentiles of dB values along ``axis`` (computed on the dB values,
    the usual practice for spectral probability / exceedance displays).
    NaNs are ignored.  Much faster than ``np.nanpercentile`` on 2-D data,
    which falls back to a per-column Python loop."""
    v = np.moveaxis(np.asarray(values_db, dtype=float), axis, 0)
    if np.isfinite(v).all():
        res = np.percentile(v, q, axis=0)
        return {p: r for p, r in zip(q, res)}
    s = np.sort(np.where(np.isnan(v), np.inf, v), axis=0)  # NaNs (as +inf) sort last
    n = np.isfinite(v).sum(axis=0)
    out = {}
    for p in q:
        pos = np.maximum(n - 1, 0) * (p / 100.0)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
        w = pos - lo
        a = np.take_along_axis(s, lo[None], axis=0)[0]
        b = np.take_along_axis(s, hi[None], axis=0)[0]
        val = a * (1 - w) + b * w
        out[p] = np.where(n > 0, val, np.nan)
    return out


def interval_rows(x: np.ndarray, sample_rate: float, interval_s: float, offset_db: float,
                  bands: list[Band], nfft: int | None = None):
    """Split ``x`` into consecutive intervals and yield ``(start_s, metrics,
    band_levels)`` for each complete interval (batch processing)."""
    n = max(1, int(round(interval_s * sample_rate)))
    nfft = nfft or auto_psd_nfft(sample_rate, n)
    for start in range(0, len(x) - n + 1, n):
        seg = x[start:start + n]
        f, p = welch(seg, sample_rate, nfft)
        yield start / sample_rate, broadband(seg, sample_rate, offset_db), \
            band_levels(f, p, bands, offset_db)
