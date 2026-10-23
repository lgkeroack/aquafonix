"""Live analysis running beside a source, on its own thread.

It reads new samples from the source's history buffer and produces:

* a band-filtered copy of the stream (band isolation) for display/listening,
* spectrogram frames and a per-frequency noise floor,
* broadband / floor / band levels per frame and per second, plus a noise
  report (exceedance levels, impulsiveness, tonal lines, likely sources),
* detections: energy spikes above the floor ("Unidentified") and matches
  against the user's signatures ("Suspected: <name>").
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from aquafonix.acoustics import noise
from aquafonix.acoustics.filters import BandSpec, StreamingFIR, design_fir
from aquafonix.acoustics.signatures import DEFAULT_THRESHOLD, Matcher
from aquafonix.acoustics.spectral import (FloorTracker, auto_nfft, blur_freq, power_frames,
                                          snr_frames, to_db)
from aquafonix.buffer import RingBuffer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HydrophoneCalibration:
    """Converts digital level to sound pressure: dB re 1 uPa =
    dBFS + 20*log10(full-scale volts) - sensitivity - gain."""

    enabled: bool = False
    sensitivity_db: float = -165.0  # dB re 1 V/uPa
    gain_db: float = 0.0
    full_scale_v: float = 1.0  # volts (peak) at digital full scale

    @property
    def offset_db(self) -> float:
        if not self.enabled:
            return 0.0
        return 20 * np.log10(max(self.full_scale_v, 1e-12)) - self.sensitivity_db - self.gain_db

    @property
    def units(self) -> str:
        return "dB re 1 µPa" if self.enabled else "dBFS"


@dataclass(frozen=True)
class AnalysisSettings:
    channel: int = 0  # analysis channel; -1 = mix of all channels
    nfft: int = 0  # 0 = automatic (~30 ms frames)
    floor_rise_s: float = 20.0
    floor_fall_s: float = 2.0
    calibration: HydrophoneCalibration = HydrophoneCalibration()
    band: BandSpec | None = None
    detect: bool = True
    event_threshold_db: float = 10.0
    min_event_s: float = 0.03
    max_event_s: float = 10.0
    detect_in_band: bool = True
    match: bool = True
    match_threshold: float = DEFAULT_THRESHOLD
    floor_alarm: bool = False
    floor_alarm_db: float = 100.0
    stats_window_s: float = 60.0


@dataclass
class Detection:
    id: int
    t0: float
    t1: float
    f_lo: float
    f_hi: float
    snr_db: float
    kind: str  # "match" | "event"
    sig_id: str | None = None
    label: str | None = None
    score: float = 0.0
    alternatives: list[tuple[str, str, float]] = field(default_factory=list)
    status: str = "unidentified"  # suspected | unidentified | confirmed | corrected | rejected
    wall_time: float | None = None
    saved_event: str | None = None

    @property
    def text(self) -> str:
        if self.status == "suspected":
            return f"Suspected: {self.label} ({self.score:.2f})"
        if self.status == "confirmed":
            return f"✔ {self.label}"
        if self.status == "corrected":
            return f"✎ {self.label}"
        if self.status == "rejected":
            return f"✖ not {self.label}" if self.label else "✖ rejected"
        if self.alternatives:
            sid, lab, sc = self.alternatives[0]
            return f"Unidentified (closest: {lab} {sc:.2f})"
        return "Unidentified"


class _Ring:
    """Fixed-size ring of rows (time + vector) for analysis products."""

    def __init__(self, capacity: int, width: int, dtype=np.float32) -> None:
        self.capacity = max(1, int(capacity))
        self.t = np.zeros(self.capacity)
        self.v = np.zeros((self.capacity, width), dtype=dtype)
        self.count = 0
        self.head = 0  # next write position
        self._lock = threading.Lock()

    def append(self, t: np.ndarray, v: np.ndarray) -> None:
        n = len(t)
        if not n:
            return
        if n > self.capacity:
            t, v, n = t[-self.capacity:], v[-self.capacity:], self.capacity
        with self._lock:
            idx = (self.head + np.arange(n)) % self.capacity
            self.t[idx] = t
            self.v[idx] = v
            self.head = (self.head + n) % self.capacity
            self.count = min(self.capacity, self.count + n)

    def _order(self) -> np.ndarray:
        start = (self.head - self.count) % self.capacity
        return (start + np.arange(self.count)) % self.capacity

    def range(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            idx = self._order()
            t = self.t[idx]
            sel = idx[(t >= t0) & (t <= t1)]
            return self.t[sel].copy(), self.v[sel].astype(np.float64)

    def latest(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            idx = self._order()[-int(n):] if n > 0 else self._order()[:0]
            return self.t[idx].copy(), self.v[idx].astype(np.float64)

    def last_t(self) -> float | None:
        with self._lock:
            return float(self.t[(self.head - 1) % self.capacity]) if self.count else None


class _PeakPicker:
    """Turns per-window match scores into one detection per occurrence."""

    def __init__(self) -> None:
        self.state: dict[str, list] = {}  # sig -> [best, t_end, duration]
        self.last_emit: dict[str, float] = {}

    def feed(self, sig_id: str, t_end: np.ndarray, scores: np.ndarray, duration: float,
             threshold: float):
        out = []
        st = self.state.get(sig_id)
        for t, s in zip(t_end, scores):
            if t < self.last_emit.get(sig_id, -np.inf) + 0.5 * duration:
                continue
            if s >= threshold:
                if st is None or s > st[0]:
                    st = [float(s), float(t), duration]
                elif t - st[1] > 0.5 * duration:
                    out.append(tuple(st))
                    self.last_emit[sig_id] = st[1]
                    st = None
            elif st is not None:
                out.append(tuple(st))
                self.last_emit[sig_id] = st[1]
                st = None
        if st is None:
            self.state.pop(sig_id, None)
        else:
            self.state[sig_id] = st
        return out


class Analyzer:
    CLIP_BUDGET_BYTES = 64 * 1024 * 1024
    MAX_DETECTIONS = 5000

    def __init__(self, buffer: RingBuffer, matcher: Matcher, settings: AnalysisSettings,
                 sample_rate: float | None = None, history_s: float = 60.0,
                 wall_clock: Callable[[float], float | None] | None = None) -> None:
        self.buffer = buffer
        self.matcher = matcher
        self.settings = settings
        self.nominal_rate = sample_rate
        self.history_s = max(10.0, float(history_s))
        self.wall_clock = wall_clock or (lambda _t: None)
        self.rate: float | None = sample_rate
        self.error: str | None = None
        self.processed_index = 0
        self.filtered: RingBuffer | None = None
        self.spectro: _Ring | None = None
        self.levels: _Ring | None = None  # per frame: broadband, floor, band
        self.long_levels = _Ring(24 * 3600, 5)  # per second: bb, floor, band, 63, 125
        self.freqs = np.zeros(0)
        self.nfft = 0
        self.hop = 0
        self.report = noise.NoiseReport()
        self.alarm = False
        self.dropped_samples = 0
        self._det_lock = threading.Lock()
        self.detections: list[Detection] = []
        self.clips: OrderedDict[int, tuple[np.ndarray, float, float]] = OrderedDict()
        self._clip_bytes = 0
        self._next_id = 1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: AnalysisSettings | None = None
        self._epoch = -1
        self._reset_all()

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="Analyzer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def apply(self, settings: AnalysisSettings) -> None:
        """Change settings; picked up by the analysis thread."""
        self._pending = settings

    # -- results for the GUI ---------------------------------------------------
    @property
    def units(self) -> str:
        return self.settings.calibration.units

    @property
    def offset_db(self) -> float:
        return self.settings.calibration.offset_db

    def floor_snapshot(self) -> tuple[np.ndarray, np.ndarray] | None:
        floor = self._floor.floor if self._floor is not None else None
        if floor is None:
            return None
        return self.freqs.copy(), floor.copy()

    def detections_after(self, det_id: int) -> list[Detection]:
        with self._det_lock:
            return [d for d in self.detections if d.id > det_id]

    def all_detections(self) -> list[Detection]:
        with self._det_lock:
            return list(self.detections)

    def detection(self, det_id: int) -> Detection | None:
        with self._det_lock:
            return next((d for d in self.detections if d.id == det_id), None)

    def detection_clip(self, det: Detection, pad: float = 0.25):
        """``(mono clip, rate, t_start)`` from memory or the history buffer."""
        cached = self.clips.get(det.id)
        if cached is not None:
            return cached
        return self._grab_clip(det.t0 - pad, det.t1 + pad)

    def _grab_clip(self, t0: float, t1: float):
        snap = self.buffer.range(t0, t1)
        if not len(snap) or not self.rate:
            return None
        return self.channel_signal(snap.y).astype(np.float32), float(self.rate), float(snap.x[0])

    # -- internals -------------------------------------------------------------
    def _reset_all(self) -> None:
        self._fir: StreamingFIR | None = None
        self._fnext = 0
        self._floor: FloorTracker | None = None
        self._stft_buf = np.zeros(0)
        self._stft_t0: float | None = None
        self._snr_hist = None  # (frames, bins) recent blurred SNR
        self._snr_t = np.zeros(0)
        self._shape_buf = np.zeros(0)
        self._shape_t = np.zeros(0)
        self._event = None
        self._peaks = _PeakPicker()
        self._recent_matches: list[Detection] = []
        self._raw_tail = np.zeros(0)
        self._report_t = 0.0
        self._sec_acc: list = []
        self._sec_start: float | None = None
        self._configured_for = None

    def channel_signal(self, y: np.ndarray) -> np.ndarray:
        """The analysis channel (or the mix) of a block, NaNs as zeros."""
        ch = self.settings.channel
        if y.ndim == 1:
            sig = y
        elif ch < 0 or ch >= y.shape[1]:
            sig = np.nanmean(y, axis=1) if y.shape[1] > 1 else y[:, 0]
        else:
            sig = y[:, ch]
        return np.nan_to_num(sig.astype(np.float64))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                worked = self._process()
                self.error = None
            except Exception as exc:
                log.exception("analysis failed")
                self.error = str(exc)
                self._stop.wait(0.5)
                worked = False
            if not worked:
                self._stop.wait(0.03)

    def _configure(self) -> None:
        """(Re)build everything that depends on rate / FFT size / band."""
        s = self.settings
        rate = self.rate
        nfft = s.nfft or auto_nfft(rate)
        nfft = int(min(max(nfft, 32), 65536))
        self.nfft, self.hop = nfft, nfft // 2
        self.freqs = np.fft.rfftfreq(nfft, 1 / rate)
        frame_period = self.hop / rate
        frames = int(self.history_s / frame_period) + 8
        self.spectro = _Ring(frames, len(self.freqs))
        self.levels = _Ring(frames, 3, dtype=np.float64)
        self._floor = FloorTracker(len(self.freqs), frame_period, s.floor_rise_s, s.floor_fall_s)
        self._stft_buf = np.zeros(0)
        self._stft_t0 = None
        hist = int(max(12.0, 3 * self.matcher.max_frames * frame_period) / frame_period)
        self._snr_hist = np.zeros((0, len(self.freqs)), dtype=np.float32)
        self._snr_cap = max(hist, 64)
        self._snr_t = np.zeros(0)
        self._event = None
        self._peaks = _PeakPicker()
        self._configure_filter()
        self._configured_for = (rate, s.nfft, s.band)

    def _configure_filter(self) -> None:
        s = self.settings
        self._fir = None
        self.filtered = None
        if s.band is None or not self.nominal_rate:
            return
        try:
            taps = design_fir(s.band, self.rate)
        except ValueError as exc:
            self.error = f"filter: {exc}"
            return
        names = self.buffer.names
        self._fir = StreamingFIR(taps, max(1, len(names)))
        start = self.buffer.first_index
        x0 = self.buffer.x0 + (start - self._fir.delay) / self.rate
        self.filtered = RingBuffer(self.buffer.capacity, self.buffer.dtype, self.rate, x0)
        self._fnext = start

    def _apply_pending(self) -> None:
        new, self._pending = self._pending, None
        old = self.settings
        self.settings = new
        if self.rate is None:
            return
        if (new.nfft, new.band, new.channel) != (old.nfft, old.band, old.channel):
            self._configure()
        elif (new.floor_rise_s, new.floor_fall_s) != (old.floor_rise_s, old.floor_fall_s):
            self._floor.set_times(self.hop / self.rate, new.floor_rise_s, new.floor_fall_s)

    def _process(self) -> bool:
        if self._pending is not None:
            self._apply_pending()
        buf = self.buffer
        if buf.epoch != self._epoch:  # history discarded (clear / seek)
            self._epoch = buf.epoch
            self.processed_index = buf.first_index
            if self.rate is not None:
                self._configure()
        worked = self._filter_step()
        snap = buf.from_index(self.processed_index, max_n=self._chunk())
        if not len(snap):
            return worked
        if snap.first_index > self.processed_index:  # fell behind: data lost
            self.dropped_samples += snap.first_index - self.processed_index
            self._stft_buf = np.zeros(0)
            self._stft_t0 = None
        self.processed_index = snap.first_index + len(snap)
        if self.rate is None:
            if buf.total < 64:
                return True
            from aquafonix.dsp import estimate_rate

            self.rate = estimate_rate(buf.latest(512).x)
            if not self.rate:
                return True
            self._configure()
        if self._configured_for is None:
            self._configure()
        sig = self.channel_signal(snap.y)
        self._raw_tail = np.concatenate([self._raw_tail, sig])[-int(self.rate * 1.0) - 1:]
        self.matcher.prepare(self.rate, self.nfft, self.hop)
        self._shape_step(sig, snap.x)
        self._spectral_step(sig, float(snap.x[0]))
        now = time.perf_counter()
        if now - self._report_t > 1.0:
            self._report_t = now
            self._make_report()
        return True

    def _chunk(self) -> int:
        rate = self.rate or 1000.0
        return int(max(rate * 0.5, 4 * (self.nfft or 1024)))

    # -- band filter -------------------------------------------------------------
    def _filter_step(self) -> bool:
        if self._fir is None or self.filtered is None:
            return False
        snap = self.buffer.from_index(self._fnext, max_n=int(self.rate))
        if not len(snap):
            return False
        if snap.first_index != self._fnext:  # lost data: restart the filter
            self._configure_filter()
            return True
        self._fnext = snap.first_index + len(snap)
        self.filtered.append(None, self._fir.process(snap.y), snap.names)
        return True

    # -- spectrogram, floor, levels, events, spectral matching -------------------
    def _spectral_step(self, sig: np.ndarray, t_first: float) -> None:
        if self._stft_t0 is None:
            self._stft_t0 = t_first
            self._stft_buf = sig
        else:
            self._stft_buf = np.concatenate([self._stft_buf, sig])
        nfft, hop, rate = self.nfft, self.hop, self.rate
        if len(self._stft_buf) < nfft:
            return
        p = power_frames(self._stft_buf, nfft, hop)
        n = len(p)
        times = self._stft_t0 + (np.arange(n) * hop + nfft / 2) / rate
        self._stft_buf = self._stft_buf[n * hop:]
        self._stft_t0 += n * hop / rate
        d = to_db(p)
        floors = self._floor.update(d)
        self.spectro.append(times, d.astype(np.float32))
        flin = 10 ** (floors / 10)
        lo, hi = self._band_edges()
        bsel = (self.freqs >= lo) & (self.freqs <= hi)
        bb = to_db(p.sum(axis=1))
        fl = to_db(flin.sum(axis=1))
        band = to_db(p[:, bsel].sum(axis=1)) if bsel.any() else np.full(n, np.nan)
        self.levels.append(times, np.column_stack([bb, fl, band]))
        self._per_second(times, p.sum(axis=1), flin.sum(axis=1),
                         p[:, bsel].sum(axis=1) if bsel.any() else np.full(n, np.nan))
        snr = blur_freq(snr_frames(d, floors), axis=1).astype(np.float32)
        self._snr_hist = np.concatenate([self._snr_hist, snr])[-self._snr_cap:]
        self._snr_t = np.concatenate([self._snr_t, times])[-self._snr_cap:]
        if self.settings.match and self.matcher.spectral:
            self._scan_spectral(n)
        if self.settings.detect:
            self._events(times, p, flin)

    def _band_edges(self) -> tuple[float, float]:
        s = self.settings
        if s.band is not None and s.band.kind in ("bandpass", "lowpass", "highpass"):
            return s.band.edges(self.rate)
        return (self.freqs[1] if len(self.freqs) > 1 else 0.0), self.rate / 2

    def _per_second(self, times, bb, fl, band) -> None:
        for t, a, b, c in zip(times, bb, fl, band):
            if self._sec_start is None:
                self._sec_start = t
            self._sec_acc.append((a, b, c))
            if t - self._sec_start >= 1.0:
                acc = np.array(self._sec_acc)
                tol = self.report.third_octaves
                row = [*to_db(np.nanmean(acc, axis=0)), tol.get("63 Hz", np.nan),
                       tol.get("125 Hz", np.nan)]
                self.long_levels.append(np.array([self._sec_start + 0.5]), np.array([row]))
                self._sec_acc = []
                self._sec_start = None

    def _frame_hop_s(self) -> float:
        return self.hop / self.rate

    def _scan_spectral(self, new_frames: int) -> None:
        m = self.matcher
        tmax = m.max_frames
        block_len = min(len(self._snr_hist), new_frames + tmax - 1)
        if block_len < 2:
            return
        block = self._snr_hist[-block_len:].T.astype(np.float64)  # (bins, frames)
        t_block = self._snr_t[-block_len:]
        n_new = min(new_frames, block_len)
        per_sig: dict[str, tuple[np.ndarray, float]] = {}
        for tpl, scores in m.spectral_scores(block):
            if not len(scores):
                continue
            ends = np.arange(len(scores)) + tpl.frames - 1
            sel = ends >= block_len - n_new
            if not sel.any():
                continue
            ends, sc = ends[sel], scores[sel]
            cur = per_sig.get(tpl.sig_id)
            full = np.full(n_new, -1.0)
            full[ends - (block_len - n_new)] = sc
            if cur is not None:
                full = np.maximum(full, cur[0])
            per_sig[tpl.sig_id] = (full, max(tpl.duration, cur[1] if cur else 0.0))
        t_end = t_block[block_len - n_new:] + self.nfft / (2 * self.rate)
        thr_default = self.settings.match_threshold
        for sig_id, (scores, duration) in per_sig.items():
            thr = m.threshold(sig_id, thr_default)
            for score, t1, dur in self._peaks.feed(sig_id, t_end, scores, duration, thr):
                self._emit_match(sig_id, t1 - dur, t1, score, "spectral")

    def _region_block(self, t0: float, t1: float):
        sel = (self._snr_t >= t0) & (self._snr_t <= t1)
        if not sel.any():
            return None
        return self._snr_hist[sel].T.astype(np.float64)

    def _rank_region(self, t0: float, t1: float, method: str):
        m = self.matcher
        if method == "spectral":
            pad = (m.max_frames + 2) * self._frame_hop_s()
            block = self._region_block(t0 - pad, t1 + pad)
            if block is None:
                return []
            return m.rank(m.spectral_scores(block, "positive"),
                          m.spectral_scores(block, "negative"))
        pad = m.max_shape / self.rate
        snap = self.buffer.range(t0 - pad, t1 + pad)
        if not len(snap):
            return []
        x = self.channel_signal(snap.y)
        return m.rank(m.shape_scores(x, "positive"), m.shape_scores(x, "negative"))

    def _emit_match(self, sig_id: str, t0: float, t1: float, score: float, method: str) -> None:
        ranked = self._rank_region(t0, t1, method)
        vetoed = {sid for sid, _, v in ranked if v}
        if sig_id in vetoed:
            return  # a negative example of this signature fits at least as well
        alts = [(sid, self.matcher.label(sid), sc) for sid, sc, v in ranked
                if sid != sig_id and not v][:3]
        lo, hi = self._extent(t0, t1, sig_id)
        det = Detection(0, t0, t1, lo, hi, self._region_snr(t0, t1), "match", sig_id,
                        self.matcher.label(sig_id), float(score), alts, "suspected",
                        self.wall_clock(t0))
        self._add_detection(det)
        self._recent_matches = [d for d in self._recent_matches if d.t1 > t1 - 30] + [det]

    def _extent(self, t0: float, t1: float, sig_id: str | None) -> tuple[float, float]:
        tpls = [t for t in self.matcher.spectral if t.sig_id == sig_id]
        if tpls and len(self.freqs):
            r0 = min(t.r0 for t in tpls)
            r1 = max(t.r1 for t in tpls) - 1
            return float(self.freqs[r0]), float(self.freqs[min(r1, len(self.freqs) - 1)])
        return 0.0, (self.rate or 0) / 2

    def _region_snr(self, t0: float, t1: float) -> float:
        if self.levels is None:
            return float("nan")
        t, v = self.levels.range(t0, t1)
        if not len(t):
            return float("nan")
        return float(np.nanmax(v[:, 2] if np.isfinite(v[:, 2]).any() else v[:, 0])
                     - np.nanmedian(v[:, 1]))

    def _events(self, times: np.ndarray, p: np.ndarray, flin: np.ndarray) -> None:
        s = self.settings
        lo, hi = self._band_edges() if s.detect_in_band else (0.0, self.rate / 2)
        sel = (self.freqs >= lo) & (self.freqs <= hi)
        if not sel.any():
            return
        e = p[:, sel].sum(axis=1)
        f = flin[:, sel].sum(axis=1)
        band_snr = to_db(e) - to_db(f)
        bin_snr = to_db(p[:, sel]) - to_db(flin[:, sel])
        fsel = self.freqs[sel]
        hop_s = self._frame_hop_s()
        on, off = s.event_threshold_db, s.event_threshold_db - 3.0
        for i, t in enumerate(times):
            ev = self._event
            if ev is None:
                if band_snr[i] >= on:
                    self._event = ev = {"t0": t - hop_s / 2, "t1": t + hop_s / 2,
                                        "snr": band_snr[i], "quiet": 0,
                                        "bins": bin_snr[i] > 6}
                continue
            if band_snr[i] >= off:
                ev["t1"] = t + hop_s / 2
                ev["snr"] = max(ev["snr"], band_snr[i])
                ev["bins"] |= bin_snr[i] > 6
                ev["quiet"] = 0
            else:
                ev["quiet"] += 1
            if ev["quiet"] * hop_s >= 0.08 or ev["t1"] - ev["t0"] >= s.max_event_s:
                self._event = None
                if ev["t1"] - ev["t0"] >= s.min_event_s:
                    active = np.flatnonzero(ev["bins"])
                    f_lo = float(fsel[active[0]]) if len(active) else lo
                    f_hi = float(fsel[active[-1]]) if len(active) else hi
                    self._close_event(ev["t0"], ev["t1"], f_lo, f_hi, float(ev["snr"]))

    def _close_event(self, t0, t1, f_lo, f_hi, snr) -> None:
        for d in self._recent_matches:  # already reported as a match: merge
            if min(d.t1, t1) - max(d.t0, t0) > 0.3 * min(d.t1 - d.t0, t1 - t0):
                with self._det_lock:
                    d.t0, d.t1 = min(d.t0, t0), max(d.t1, t1)
                    d.f_lo, d.f_hi = min(d.f_lo, f_lo), max(d.f_hi, f_hi)
                    d.snr_db = max(d.snr_db, snr) if np.isfinite(d.snr_db) else snr
                return
        ranked = []
        if self.settings.match and not self.matcher.empty:
            ranked = self._rank_region(t0, t1, "spectral" if self.matcher.spectral else "shape")
        alts = [(sid, self.matcher.label(sid), sc) for sid, sc, v in ranked if not v][:3]
        det = Detection(0, t0, t1, f_lo, f_hi, snr, "event", alternatives=alts,
                        wall_time=self.wall_clock(t0))
        if alts and alts[0][2] >= self.matcher.threshold(alts[0][0], self.settings.match_threshold):
            det.kind, det.status = "match", "suspected"
            det.sig_id, det.label, det.score = alts[0]
            det.alternatives = alts[1:]
        self._add_detection(det)

    # -- waveform-shape matching (slow data) --------------------------------------
    def _shape_step(self, sig: np.ndarray, x: np.ndarray) -> None:
        m = self.matcher
        if not (self.settings.match and m.shape):
            return
        keep = m.max_shape - 1
        self._shape_buf = np.concatenate([self._shape_buf, sig])
        self._shape_t = np.concatenate([self._shape_t, x])
        n_new = len(sig)
        per_sig: dict[str, tuple[np.ndarray, float]] = {}
        total = len(self._shape_buf)
        for tpl, scores in m.shape_scores(self._shape_buf):
            if not len(scores):
                continue
            ends = np.arange(len(scores)) + len(tpl.wave) - 1
            sel = ends >= total - n_new
            full = np.full(n_new, -1.0)
            full[ends[sel] - (total - n_new)] = scores[sel]
            cur = per_sig.get(tpl.sig_id)
            if cur is not None:
                full = np.maximum(full, cur[0])
            per_sig[tpl.sig_id] = (full, max(tpl.duration, cur[1] if cur else 0.0))
        t_end = self._shape_t[total - n_new:]
        for sig_id, (scores, duration) in per_sig.items():
            thr = m.threshold(sig_id, self.settings.match_threshold)
            for score, t1, dur in self._peaks.feed("shape:" + sig_id, t_end, scores, duration, thr):
                self._emit_match(sig_id, t1 - dur, t1, score, "shape")
        self._shape_buf = self._shape_buf[-keep:] if keep > 0 else self._shape_buf[:0]
        self._shape_t = self._shape_t[-keep:] if keep > 0 else self._shape_t[:0]

    # -- detections store ----------------------------------------------------------
    def _add_detection(self, det: Detection) -> None:
        clip = self._grab_clip(det.t0 - 0.25, det.t1 + 0.25)
        with self._det_lock:
            det.id = self._next_id
            self._next_id += 1
            self.detections.append(det)
            if len(self.detections) > self.MAX_DETECTIONS:
                old = self.detections.pop(0)
                self._drop_clip(old.id)
        if clip is not None:
            self.clips[det.id] = clip
            self._clip_bytes += clip[0].nbytes
            while self._clip_bytes > self.CLIP_BUDGET_BYTES and self.clips:
                self._drop_clip(next(iter(self.clips)))

    def _drop_clip(self, det_id: int) -> None:
        clip = self.clips.pop(det_id, None)
        if clip is not None:
            self._clip_bytes -= clip[0].nbytes

    def update_detection(self, det: Detection, **changes) -> None:
        with self._det_lock:
            for k, v in changes.items():
                setattr(det, k, v)

    # -- noise report ----------------------------------------------------------------
    def _make_report(self) -> None:
        s = self.settings
        off = s.calibration.offset_db
        rep = noise.NoiseReport(units=s.calibration.units)
        if self.levels is not None and self.levels.count:
            t_last = self.levels.last_t()
            t, v = self.levels.range(t_last - s.stats_window_s, t_last)
            if len(t):
                rep.level_db = float(v[-1, 0] + off)
                rep.floor_db = float(v[-1, 1] + off)
                rep.band_db = float(v[-1, 2] + off) if s.band is not None else float("nan")
                rep.stats = {k: val + off for k, val in noise.exceedance_levels(v[:, 0]).items()}
        raw = self._raw_tail
        if len(raw) >= 64:
            rep.crest_db, rep.kurtosis = noise.impulsiveness(raw)
            rate = self.rate
            nfft = int(min(len(raw), 1 << int(np.log2(max(rate, 64)))))
            pw = power_frames(raw, nfft, max(1, nfft // 2))
            if len(pw):
                psum = pw.mean(axis=0)
                freqs = np.fft.rfftfreq(nfft, 1 / rate)
                for fc, lo, hi in noise.third_octave_bands(rate / 2):
                    rep.third_octaves[noise.nominal(fc)] = float(
                        noise.db(noise.band_power(psum, freqs, lo, hi)) + off)
        if self.spectro is not None and self.spectro.count >= 4:
            t, d = self.spectro.latest(int(10.0 / self._frame_hop_s()))
            ltsa = to_db(np.mean(10 ** (d / 10), axis=0))
            rep.flatness = noise.spectral_flatness(10 ** (ltsa[1:] / 10))
            rep.tonals = noise.tonal_lines(ltsa, self.freqs)
            rep.sources = noise.guess_sources(self.freqs, ltsa, rep.tonals, rep.kurtosis,
                                              self.rate)
        self.report = rep
        self.alarm = bool(s.floor_alarm and np.isfinite(rep.floor_db)
                          and rep.floor_db >= s.floor_alarm_db)

