"""Noise analysis: band levels, exceedance statistics, impulsiveness, tonal
lines and a heuristic guess at what is making the noise.

Level conventions: ``L5`` is the level exceeded 5 % of the time (95th
percentile), ``L50`` the median and ``L95`` the level exceeded 95 % of the
time (5th percentile), which tracks the background/floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def band_power(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Sum of per-bin power (last axis) for bins with lo <= f < hi."""
    sel = (freqs >= lo) & (freqs < hi)
    if not sel.any():
        return np.zeros(power.shape[:-1])
    return power[..., sel].sum(axis=-1)


def db(power) -> np.ndarray:
    return 10 * np.log10(np.maximum(power, 1e-30))


def exceedance_levels(levels_db: np.ndarray) -> dict[str, float]:
    levels_db = np.asarray(levels_db, dtype=float)
    levels_db = levels_db[np.isfinite(levels_db)]
    if not len(levels_db):
        return {}
    p95, p50, p5 = np.percentile(levels_db, [95, 50, 5])
    return {"L5": float(p95), "L50": float(p50), "L95": float(p5),
            "Leq": float(db(np.mean(10 ** (levels_db / 10))))}


def impulsiveness(x: np.ndarray) -> tuple[float, float]:
    """``(crest factor dB, kurtosis)``; kurtosis is 3 for Gaussian noise and
    much higher for clicks, snapping shrimp and pile driving."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 16:
        return float("nan"), float("nan")
    x = x - x.mean()
    rms = np.sqrt(np.mean(x**2))
    if rms <= 0:
        return float("nan"), float("nan")
    crest = 20 * np.log10(np.max(np.abs(x)) / rms)
    kurt = float(np.mean(x**4) / rms**4)
    return float(crest), kurt


def spectral_flatness(power: np.ndarray) -> float:
    """Wiener entropy: ~1 for white noise, ~0 for pure tones."""
    p = np.asarray(power, dtype=float)
    p = p[p > 0]
    if not len(p):
        return float("nan")
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def tonal_lines(ltsa_db: np.ndarray, freqs: np.ndarray, persistence: np.ndarray | None = None,
                min_prominence: float = 6.0, max_lines: int = 12):
    """Narrow persistent peaks in a long-term average spectrum.
    Returns ``[(freq_hz, prominence_db)]`` strongest first."""
    n = len(ltsa_db)
    if n < 9:
        return []
    k = max(5, n // 64) | 1
    pad = np.pad(ltsa_db, k // 2, mode="edge")
    baseline = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    prom = ltsa_db - baseline
    peaks = (prom[1:-1] > prom[:-2]) & (prom[1:-1] >= prom[2:]) & (prom[1:-1] > min_prominence)
    idx = np.flatnonzero(peaks) + 1
    if persistence is not None:
        idx = idx[persistence[idx] >= 0.7]
    idx = idx[freqs[idx] > 0]
    order = idx[np.argsort(prom[idx])[::-1]][:max_lines]
    return [(float(freqs[i]), float(prom[i])) for i in order]


@dataclass
class NoiseReport:
    level_db: float = float("nan")  # broadband SPL of the last interval
    peak_db: float = float("nan")  # Lpk of the last interval
    floor_db: float = float("nan")  # broadband noise floor
    band_db: float = float("nan")  # filter band, if any
    stats: dict[str, float] = field(default_factory=dict)  # L5/L50/L95/Leq of broadband
    crest_db: float = float("nan")
    kurtosis: float = float("nan")
    flatness: float = float("nan")
    tonals: list[tuple[float, float]] = field(default_factory=list)
    third_octaves: dict[str, float] = field(default_factory=dict)
    sources: list[tuple[str, float, str]] = field(default_factory=list)
    units: str = "dBFS"
    demon: object = None  # DemonResult of the recent signal, if computed


def guess_sources(freqs: np.ndarray, ltsa_db: np.ndarray, tonals, kurtosis: float,
                  sample_rate: float) -> list[tuple[str, float, str]]:
    """Heuristic noise-source hints ``[(name, score 0-1, reason)]``.

    These are rules of thumb (spectral shape, tonal lines, impulsiveness),
    not a classifier; use them as pointers for a closer look."""
    out = []
    lin = 10 ** (ltsa_db / 10)

    def level(lo, hi):
        sel = (freqs >= lo) & (freqs < hi)
        return db(lin[sel].mean()) if sel.any() else float("nan")

    for mains in (50.0, 60.0):
        hits = [f for f, _ in tonals if any(abs(f - mains * k) <= max(1.5, freqs[1] * 1.5)
                                            for k in range(1, 6))]
        if hits:
            out.append((f"Electrical hum ({mains:g} Hz)", min(1.0, 0.4 + 0.2 * len(hits)),
                        "tonal lines at " + ", ".join(f"{f:.1f} Hz" for f in hits[:4])))
    machinery = [f for f, _ in tonals if 5 < f < 1000 and
                 not any(abs(f - m * k) <= 1.5 for m in (50, 60) for k in range(1, 6))]
    low, mid = level(20, 500), level(500, 5000)
    if len(machinery) >= 2 or (np.isfinite(low) and np.isfinite(mid) and low - mid > 12):
        reason = []
        if machinery:
            reason.append("tonal lines at " + ", ".join(f"{f:.1f} Hz" for f in machinery[:4]))
        if np.isfinite(low) and np.isfinite(mid) and low - mid > 12:
            reason.append(f"20–500 Hz {low - mid:.0f} dB above 0.5–5 kHz")
        out.append(("Vessel / machinery", min(1.0, 0.3 + 0.15 * len(machinery)
                                                + max(0, low - mid - 12) / 20), "; ".join(reason)))
    vlow = level(1, 20)
    if np.isfinite(vlow) and np.isfinite(low) and vlow - low > 15:
        out.append(("Flow noise / cable strum (< 20 Hz)", min(1.0, (vlow - low - 15) / 15 + 0.4),
                    f"< 20 Hz {vlow - low:.0f} dB above 20–500 Hz"))
    if np.isfinite(kurtosis) and kurtosis > 8:
        hf = level(2000, min(sample_rate / 2, 40000))
        hint = "Snapping shrimp / impulsive transients"
        out.append((hint, min(1.0, 0.3 + (kurtosis - 8) / 30),
                    f"kurtosis {kurtosis:.0f}" + (f", HF level {hf:.0f} dB" if np.isfinite(hf) else "")))
    if sample_rate >= 30000:
        sel = (freqs >= 1000) & (freqs < 15000)
        if sel.sum() > 8:
            flat = spectral_flatness(lin[sel])
            slope = np.polyfit(np.log10(freqs[sel]), ltsa_db[sel], 1)[0]
            if flat > 0.6 and -6 < slope < 3:
                out.append(("Rain / surface agitation (broadband 1–15 kHz)",
                            min(1.0, flat), f"flat spectrum (flatness {flat:.2f})"))
    hump = level(500, 5000)
    if all(np.isfinite(v) for v in (hump, low)) and sample_rate >= 20000:
        high = level(5000, 10000)
        if np.isfinite(high) and hump - high > 6 and hump - low > 0:
            out.append(("Wind / sea-state noise (0.5–5 kHz hump)", 0.4,
                        f"0.5–5 kHz {hump - high:.0f} dB above 5–10 kHz"))
    return sorted(out, key=lambda t: -t[1])
