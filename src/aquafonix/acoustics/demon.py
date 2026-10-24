"""DEMON analysis (Detection of Envelope Modulation On Noise).

A propeller amplitude-modulates the broadband cavitation noise it makes at
the shaft rate and the blade rate.  Band-pass the noise, square-law
demodulate it, and the low-frequency spectrum of that envelope shows lines
at the shaft rate and its harmonics; the strongest is usually the blade
rate, so blades ~= blade-rate line / shaft rate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from aquafonix.acoustics.filters import BandSpec, filter_clip
from aquafonix.acoustics.metrics import welch


@dataclass
class DemonResult:
    freqs: np.ndarray  # modulation frequency, Hz
    level_db: np.ndarray  # envelope spectrum, dB (relative)
    band: tuple[float, float]
    lines: list[tuple[float, float]] = field(default_factory=list)  # (Hz, prominence dB)
    shaft_hz: float | None = None
    blade_hz: float | None = None
    blades: int | None = None

    def describe(self) -> str:
        if self.shaft_hz is None:
            return "no clear propeller modulation"
        text = f"shaft {self.shaft_hz:.2f} Hz ({self.shaft_hz * 60:.0f} rpm)"
        if self.blades and self.blade_hz:
            text += f", blade rate {self.blade_hz:.2f} Hz → ~{self.blades} blades"
        return text


def default_band(sample_rate: float) -> tuple[float, float]:
    """Cavitation noise is broadband; a few kHz upward is the usual choice."""
    nyq = sample_rate / 2
    return min(1000.0, nyq * 0.2), min(10_000.0, nyq * 0.9)


def demon(x: np.ndarray, sample_rate: float, band: tuple[float, float] | None = None,
          max_mod_hz: float = 100.0, resolution_hz: float = 0.1) -> DemonResult:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    lo, hi = band or default_band(sample_rate)
    hi = min(hi, sample_rate / 2 * 0.95)
    band = (min(lo, hi / 1.5), hi)
    y = filter_clip(x, BandSpec("bandpass", band[0], band[1]), sample_rate)
    env = y * y  # square-law demodulation
    q = max(1, int(sample_rate / (4 * max_mod_hz)))  # decimate to ~4x the top modulation
    n = (len(env) // q) * q
    env = env[:n].reshape(-1, q).mean(axis=1)  # boxcar low-pass + decimation
    fs_d = sample_rate / q
    # Clicks and bangs would add their own comb of lines: clip them.
    med = np.median(env)
    mad = 1.4826 * np.median(np.abs(env - med))
    env = np.minimum(env, med + 6 * mad) if mad > 0 else env
    env = env - env.mean()
    # Fine enough for the requested resolution, but coarse enough that Welch
    # averages ~7 segments: a noise spectrum from 1-2 segments is full of
    # chance "lines".
    want = 1 << int(np.ceil(np.log2(max(fs_d / resolution_hz, 64))))
    fit = 1 << int(np.log2(max(len(env) // 3, 64)))
    nfft = int(min(want, fit))
    f, p = welch(env, fs_d, nfft)
    segments = max(1, (len(env) - nfft) // (nfft // 2) + 1)
    keep = (f > 0) & (f <= max_mod_hz)
    f, level = f[keep], 10 * np.log10(np.maximum(p[keep], 1e-30))
    res = DemonResult(f, level, band)
    res.lines = _lines(f, level, min_prominence=max(6.0, 3.5 * _level_std_db(segments)))
    _harmonics(res)
    return res


def _level_std_db(segments: int) -> float:
    """Standard deviation (dB) of a Welch estimate of noise averaged over
    ``segments`` Hann frames at 50% overlap (chi-square statistics)."""
    x = segments / 1.056  # half the equivalent degrees of freedom
    trigamma = 1 / x + 1 / (2 * x * x) + 1 / (6 * x ** 3)
    return 10 / np.log(10) * float(np.sqrt(trigamma))


def _lines(f: np.ndarray, level: np.ndarray, min_prominence: float = 6.0, max_lines: int = 12):
    """``(Hz, prominence dB)`` of spectral lines above a running-median
    baseline.  Line bins are masked and the baseline recomputed once, so a
    dense family of lines does not lift its own baseline."""
    n = len(level)
    if n < 9:
        return []
    k = max(15, n // 10) | 1
    base = _running_median(level, k)
    hot = level - base > min_prominence
    hot |= np.r_[hot[1:], False] | np.r_[False, hot[:-1]]  # Hann leakage
    if hot.any() and not hot.all():
        base = _running_median(np.where(hot, np.nan, level), k)
        base = np.where(np.isfinite(base), base, np.nanmedian(base))
    prom = level - base
    peak = (prom[1:-1] > prom[:-2]) & (prom[1:-1] >= prom[2:]) & (prom[1:-1] > min_prominence)
    idx = np.flatnonzero(peak) + 1
    idx = idx[np.argsort(prom[idx])[::-1]][:max_lines]
    df = f[1] - f[0]
    out = []
    for i in idx:  # parabolic interpolation of the peak for sub-bin frequency
        a, b, c = level[i - 1], level[i], level[i + 1]
        den = a - 2 * b + c
        shift = 0.5 * (a - c) / den if den < 0 else 0.0
        out.append((float(f[i] + np.clip(shift, -0.5, 0.5) * df), float(prom[i])))
    return out


def _running_median(x: np.ndarray, k: int) -> np.ndarray:
    pad = np.pad(x, k // 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, k)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=1)


def _tolerance(f: float, df: float) -> float:
    """How far a line may sit from an exact harmonic: interpolated line
    frequencies are good to a fraction of a bin; allow ~1% for rate drift."""
    return max(0.6 * df, 0.01 * f)


def _harmonics(res: DemonResult) -> None:
    """Fundamental that explains most lines (harmonic sieve)."""
    lines = res.lines
    if not lines:
        return
    df = res.freqs[1] - res.freqs[0] if len(res.freqs) > 1 else 0.1
    best, best_score = None, 0.0
    for f0 in sorted({f for f, _ in lines} | {f / k for f, _ in lines for k in (2, 3, 4, 5, 6, 7)}):
        if f0 < 2 * df:
            continue
        score, hits = 0.0, 0
        for f, prom in lines:
            k = round(f / f0)
            if k >= 1 and abs(f - k * f0) <= _tolerance(f, df):
                score += prom
                hits += 1
        score *= min(hits, 4) / 4  # prefer families over single lines
        if score > best_score * 1.05 or (best is not None and score >= best_score * 0.95
                                          and f0 > best and hits >= 2):
            best, best_score = f0, score
    if best is None:
        return
    res.shaft_hz = float(best)
    # Blade rate: the strongest (by level) harmonic of the shaft rate above
    # the fundamental, if it is at least about as strong as the fundamental.
    df = max(df, 1e-9)
    family = []
    for f, _ in lines:
        k = round(f / best)
        if k >= 1 and abs(f - k * best) <= _tolerance(f, df):
            family.append((k, f, float(np.interp(f, res.freqs, res.level_db))))
    if family:  # least-squares shaft rate from the whole family
        ks = np.array([k for k, _, _ in family], float)
        fs = np.array([f for _, f, _ in family])
        res.shaft_hz = float(np.sum(ks * fs) / np.sum(ks * ks))
    fundamental = max((lv for k, _, lv in family if k == 1), default=-np.inf)
    upper = [(lv, k, f) for k, f, lv in family if 2 <= k <= 9]
    if upper:
        lv, k, f = max(upper)
        if lv >= fundamental - 3.0:
            res.blade_hz, res.blades = float(f), int(k)
