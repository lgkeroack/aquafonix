"""Batch analysis: run standard levels over many recordings and write CSV
reports (per interval, and a per-file / overall summary with percentiles)."""

from __future__ import annotations

import csv
import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from aquafonix.acoustics import metrics
from aquafonix.acoustics.calibration_io import find_calibration
from aquafonix.sources.audiofile import parse_file_timestamp

log = logging.getLogger(__name__)


@dataclass
class BatchJob:
    paths: list[str]
    out_csv: str
    integration_s: float = 60.0
    band_labels: tuple[str, ...] | None = None  # None = every third-octave band
    band: tuple[float, float] | None = None  # extra band SPL column
    channel: int = 0  # -1 = mix
    offset_db: float = 0.0  # calibration used when no metadata is found
    calibrated: bool = False
    use_metadata: bool = True
    nfft: int = 0
    summary_csv: str = field(default="")

    def __post_init__(self) -> None:
        if not self.summary_csv:
            root, ext = os.path.splitext(self.out_csv)
            self.summary_csv = f"{root}_summary{ext or '.csv'}"


def _leq(levels_db: np.ndarray) -> float:
    v = levels_db[np.isfinite(levels_db)]
    return float(10 * np.log10(np.mean(10 ** (v / 10)))) if len(v) else float("nan")


def _summary(name: str, rows: list[dict], band_names: list[str]) -> dict:
    out = {"file": name, "intervals": len(rows)}
    if not rows:
        return out
    spl = np.array([r["SPL"] for r in rows])
    p95, p50, p5 = np.nanpercentile(spl, [95, 50, 5])
    sel = np.array([r["SEL"] for r in rows])
    out.update({"duration_s": sum(r["duration"] for r in rows),
                "SPL_L5": p95, "SPL_L50": p50, "SPL_L95": p5, "SPL_Leq": _leq(spl),
                "Lpk_max": float(np.nanmax([r["Lpk"] for r in rows])),
                "SEL_total": float(10 * np.log10(np.nansum(10 ** (sel / 10))))})
    for b in band_names:
        v = np.array([r.get(b, np.nan) for r in rows], dtype=float)
        if np.isfinite(v).any():
            out[f"{b}_L50"] = float(np.nanmedian(v))
            out[f"{b}_Leq"] = _leq(v)
    return out


def run_batch(job: BatchJob, progress: Callable[[float, str], None] | None = None,
              cancelled: Callable[[], bool] | None = None) -> dict:
    """Process every file; returns ``{"files": n, "intervals": n, "errors": [...]}``."""
    import soundfile as sf

    progress = progress or (lambda f, m: None)
    cancelled = cancelled or (lambda: False)
    infos = []
    for path in job.paths:
        try:
            i = sf.info(path)
            infos.append((path, float(i.samplerate), int(i.frames)))
        except Exception as exc:
            infos.append((path, 0.0, 0))
            log.warning("skipping %s: %s", path, exc)
    rates = [r for _p, r, _n in infos if r]
    if not rates:
        raise ValueError("none of the files could be read")
    all_bands = metrics.third_octave_bands(min(rates) / 2)
    bands = [b for b in all_bands if job.band_labels is None or b.label in job.band_labels]
    band_names = [f"TOL_{b.label.replace(' ', '')}" for b in bands]
    total_frames = sum(n for _p, _r, n in infos) or 1
    done_frames = 0
    errors: list[str] = []
    summaries = []
    every: list[dict] = []
    units = "dB re 1 uPa" if job.calibrated else "dBFS"
    header = (["file", "interval_start_s", "start_time", "duration_s", "SPL", "Lpk", "Lpkpk",
               "SEL"] + (["band_SPL"] if job.band else []) + band_names + ["calibration"])
    with open(job.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([f"# Aquafonix batch analysis; integration {job.integration_s:g} s; "
                    f"levels in {units} (SEL in {units}^2 s); third-octave bands per "
                    "IEC 61260-1 base-10; metrics per ISO 18405"])
        w.writerow(header)
        for path, rate, frames in infos:
            name = os.path.basename(path)
            if cancelled():
                errors.append("cancelled")
                break
            if not rate:
                errors.append(f"{name}: unreadable")
                continue
            offset, cal_src = job.offset_db, "settings" if job.calibrated else "none"
            if job.use_metadata:
                hint = find_calibration(path)
                if hint is not None:
                    s_db, g_db, vfs = hint.values()
                    offset = 20 * np.log10(vfs) - s_db - g_db
                    cal_src = hint.source
            wall0 = parse_file_timestamp(path)
            n_int = max(1, int(round(job.integration_s * rate)))
            nfft = job.nfft or metrics.auto_psd_nfft(rate, n_int)
            per_block = max(1, int(np.ceil(30 * rate / n_int)))  # read ~30 s at a time
            rows = []
            try:
                with sf.SoundFile(path) as f:
                    pos = 0
                    while pos + n_int <= frames:
                        if cancelled():
                            break
                        block = f.read(n_int * per_block, dtype="float64", always_2d=True)
                        if not len(block):
                            break
                        if job.channel < 0 or job.channel >= block.shape[1]:
                            x = block.mean(axis=1)
                        else:
                            x = block[:, job.channel]
                        x = np.nan_to_num(x)
                        for start, m, levels in metrics.interval_rows(
                                x, rate, n_int / rate, offset, bands, nfft):
                            t = pos / rate + start
                            row = {"file": name, "interval_start_s": round(t, 6),
                                   "start_time": (dt.datetime.fromtimestamp(wall0 + t).isoformat(
                                       timespec="milliseconds") if wall0 else ""),
                                   "duration": m["duration"], "SPL": m["SPL"], "Lpk": m["Lpk"],
                                   "Lpkpk": m["Lpkpk"], "SEL": m["SEL"]}
                            if job.band:
                                seg = x[int(start * rate):int(start * rate) + n_int]
                                fq, psd = metrics.welch(seg, rate, nfft)
                                ms = metrics.band_mean_square(fq, psd, *job.band)
                                row["band_SPL"] = float(10 * np.log10(max(float(ms), 1e-30))
                                                        + offset)
                            for bn, v in zip(band_names, levels):
                                row[bn] = float(v)
                            rows.append(row)
                            w.writerow([row["file"], row["interval_start_s"], row["start_time"],
                                        f"{row['duration']:.3f}"]
                                       + [f"{row[k]:.2f}" for k in ("SPL", "Lpk", "Lpkpk", "SEL")]
                                       + ([f"{row['band_SPL']:.2f}"] if job.band else [])
                                       + [f"{row.get(bn, float('nan')):.2f}" for bn in band_names]
                                       + [cal_src])
                        pos += len(block) - len(block) % n_int
                        done_frames_now = done_frames + min(pos, frames)
                        progress(done_frames_now / total_frames, f"{name}: {pos / rate:.0f} s")
            except Exception as exc:
                log.exception("batch failed on %s", path)
                errors.append(f"{name}: {exc}")
            done_frames += frames
            summaries.append(_summary(name, rows, band_names))
            every += rows
    summaries.append(_summary("ALL FILES", every, band_names))
    keys: list[str] = []
    for srow in summaries:
        for k in srow:
            if k not in keys:
                keys.append(k)
    with open(job.summary_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for srow in summaries:
            w.writerow([f"{srow[k]:.2f}" if isinstance(srow.get(k), float) else srow.get(k, "")
                        for k in keys])
    progress(1.0, "done")
    return {"files": len(summaries) - 1, "intervals": len(every), "errors": errors,
            "csv": job.out_csv, "summary": job.summary_csv}
