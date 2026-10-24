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
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from aquafonix.acoustics import metrics, noise
from aquafonix.acoustics.array import ArrayGeometry, bearing_from_delay, bearing_scan, tdoa
from aquafonix.acoustics.demon import demon
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
    min_event_s: float = 0.005
    max_event_s: float = 10.0
    detect_in_band: bool = True
    match: bool = True
    match_threshold: float = DEFAULT_THRESHOLD
    floor_alarm: bool = False
    floor_alarm_db: float = 100.0
    stats_window_s: float = 60.0
    integration_s: float = 1.0  # averaging time for SPL / PSD / third-octave levels
    psd_nfft: int = 0  # 0 = automatic (~1 Hz resolution, fits the integration time)
    tol_bands: tuple[str, ...] = ("63 Hz", "125 Hz", "2 kHz")  # bands plotted over time
    array: ArrayGeometry = ArrayGeometry()  # two hydrophones -> bearings
    btr_step_s: float = 0.25  # bearing-time record resolution
    demon: bool = True  # look for propeller modulation (DEMON) in the noise report
    demon_band: tuple[float, float] | None = None  # None = automatic


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
    bearing: float | None = None  # degrees from the array axis
    bearing_q: float = 0.0  # GCC-PHAT peak quality (higher = clearer)

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


def _nbytes(clip) -> int:
    """Memory held by a detection clip (audio tuple or spectrogram patch)."""
    if isinstance(clip, dict):
        return sum(np.asarray(v).nbytes for v in clip.values())
    return clip[0].nbytes


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
        # per integration interval (ISO 18405 levels, raw dB; add offset_db to display):
        self.intervals: _Ring | None = None  # SPL, Lpk, SEL, floor, band
        self.tol_series: _Ring | None = None  # third-octave levels, one column per band
        self.psd_series: _Ring | None = None  # PSD (dB per Hz) over the statistics window
        self.tol_bands: list[metrics.Band] = []
        self.psd_freqs = np.zeros(0)
        self.psd_nfft = 0
        self.btr: _Ring | None = None  # bearing-time record (time x look direction)
        self.btr_bearings = np.arange(0.0, 181.0, 2.0)
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
        self.external = False  # frames are fed in (spectrogram read off the screen)
        self._frame_q: deque = deque()
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
        return "dB (screen scale)" if self.external else self.settings.calibration.units

    @property
    def offset_db(self) -> float:
        return 0.0 if self.external else self.settings.calibration.offset_db

    def feed_frames(self, times: np.ndarray, freqs: np.ndarray, frames: np.ndarray,
                    reset: bool = False) -> None:
        """Spectrogram frames from elsewhere (source thread); processed on the
        analysis thread exactly like frames computed from audio.  ``reset``
        means the level scale changed: the floor and history start afresh."""
        self.external = True
        self._frame_q.append((np.asarray(times, float), np.asarray(freqs, float),
                              np.asarray(frames, float), reset))

    def patch(self, t0: float, t1: float) -> dict | None:
        """Spectrogram patch (frames x freqs dB) for [t0, t1] plus the floor."""
        if self.spectro is None:
            return None
        t, d = self.spectro.range(t0, t1)
        if len(t) < 1:
            return None
        floor = self._floor.floor if self._floor is not None and self._floor.floor is not None \
            else np.percentile(d, 20, axis=0)
        return {"freqs": self.freqs.copy(), "times": t, "db": d, "floor": np.array(floor),
                "frame_period": self._frame_hop_s()}

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
        if self.external:
            return self.patch(t0, t1)
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
        self._demon_t = 0.0
        self._demon = None
        self._int_buf = np.zeros(0)
        self._int_t0: float | None = None
        self._btr_buf = np.zeros((0, 2))
        self._btr_t0: float | None = None
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
        self._configure_intervals()
        self._configure_btr()
        self._configured_for = (rate, s.nfft, s.band)

    def _configure_btr(self) -> None:
        s = self.settings
        self.btr = (_Ring(int(self.history_s / s.btr_step_s) + 8, len(self.btr_bearings))
                    if s.array.enabled and self.nominal_rate else None)
        self._btr_buf = np.zeros((0, 2))
        self._btr_t0 = None

    def _configure_intervals(self) -> None:
        s, rate = self.settings, self.rate
        n_int = max(8, int(round(s.integration_s * rate)))
        self.psd_nfft = int(min(s.psd_nfft, n_int)) if s.psd_nfft else \
            metrics.auto_psd_nfft(rate, n_int)
        self.psd_freqs = np.fft.rfftfreq(self.psd_nfft, 1 / rate)
        self.tol_bands = metrics.third_octave_bands(rate / 2)
        rows = int(min(200_000, max(1000, 24 * 3600 / s.integration_s)))
        self.intervals = _Ring(rows, 5, dtype=np.float64)
        self.tol_series = _Ring(rows, len(self.tol_bands))
        self.psd_series = _Ring(int(max(8, s.stats_window_s / s.integration_s)) + 1,
                                len(self.psd_freqs))
        self._int_buf = np.zeros(0)
        self._int_t0 = None

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
            return
        if (new.floor_rise_s, new.floor_fall_s) != (old.floor_rise_s, old.floor_fall_s):
            self._floor.set_times(self.hop / self.rate, new.floor_rise_s, new.floor_fall_s)
        if ((new.integration_s, new.psd_nfft, new.stats_window_s)
                != (old.integration_s, old.psd_nfft, old.stats_window_s)):
            self._configure_intervals()
        if (new.array, new.btr_step_s) != (old.array, old.btr_step_s):
            self._configure_btr()
        if (new.demon, new.demon_band) != (old.demon, old.demon_band):
            self._demon_t = 0.0

    def _process(self) -> bool:
        if self._pending is not None:
            self._apply_pending()
        if self.external:
            return self._external_step()
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
        self._integrate_step(sig, float(snap.x[0]))
        self._bearing_step(snap)
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
        snr = blur_freq(snr_frames(d, floors), axis=1).astype(np.float32)
        self._snr_hist = np.concatenate([self._snr_hist, snr])[-self._snr_cap:]
        self._snr_t = np.concatenate([self._snr_t, times])[-self._snr_cap:]
        if self.settings.match and self.matcher.spectral:
            self._scan_spectral(n)
        if self.settings.detect:
            self._events(times, p, flin)

    def _nyquist(self) -> float:
        if self.external:
            return float(self.freqs[-1]) if len(self.freqs) else 0.0
        return self.rate / 2

    def _band_edges(self) -> tuple[float, float]:
        s = self.settings
        if s.band is not None and s.band.kind in ("bandpass", "lowpass", "highpass"):
            lo, hi = s.band.edges(2 * self._nyquist())
            return lo, hi
        return (self.freqs[1] if len(self.freqs) > 1 else 0.0), self._nyquist()

    # -- spectrogram frames fed from outside (screen) ---------------------------------
    def _configure_external(self, freqs: np.ndarray, times: np.ndarray) -> None:
        s = self.settings
        dt_s = float(np.median(np.diff(times))) if len(times) > 1 else 0.05
        dt_s = max(dt_s, 1e-4)
        self.rate, self.hop, self.nfft = 1.0 / dt_s, 1, 1
        self.freqs = freqs.copy()
        frames = int(self.history_s / dt_s) + 8
        self.spectro = _Ring(frames, len(freqs))
        self.levels = _Ring(frames, 3, dtype=np.float64)
        self._floor = FloorTracker(len(freqs), dt_s, s.floor_rise_s, s.floor_fall_s)
        self._snr_hist = np.zeros((0, len(freqs)), dtype=np.float32)
        self._snr_cap = max(int(12.0 / dt_s), 64)
        self._snr_t = np.zeros(0)
        self._event = None
        self._peaks = _PeakPicker()
        self.intervals = self.tol_series = self.psd_series = None
        self._configured_for = ("external", len(freqs), float(freqs[0]), float(freqs[-1]))

    def _external_step(self) -> bool:
        worked = False
        while self._frame_q:
            times, freqs, frames, reset = self._frame_q.popleft()
            worked = True
            if not len(times):
                continue
            key = ("external", len(freqs), float(freqs[0]), float(freqs[-1]))
            if self._configured_for != key or reset:
                self._configure_external(freqs, times)
            self.matcher.prepare_grid(self.freqs, self._frame_hop_s())
            # unreadable pixels (NaN) count as background
            fill = self._floor.floor if self._floor.floor is not None else np.nanmedian(frames, 0)
            fill = np.nan_to_num(fill, nan=float(np.nanmin(frames)) if np.isfinite(frames).any()
                                 else 0.0)
            d = np.where(np.isfinite(frames), frames, fill[None, :])
            floors = self._floor.update(d)
            self.spectro.append(times, d.astype(np.float32))
            p, flin = 10 ** (d / 10), 10 ** (floors / 10)
            lo, hi = self._band_edges()
            bsel = (self.freqs >= lo) & (self.freqs <= hi)
            self.levels.append(times, np.column_stack([
                to_db(p.mean(axis=1)), to_db(flin.mean(axis=1)),
                to_db(p[:, bsel].mean(axis=1)) if bsel.any() else np.full(len(times), np.nan)]))
            snr = blur_freq(snr_frames(d, floors), axis=1).astype(np.float32)
            self._snr_hist = np.concatenate([self._snr_hist, snr])[-self._snr_cap:]
            self._snr_t = np.concatenate([self._snr_t, times])[-self._snr_cap:]
            if self.settings.match and self.matcher.spectral:
                self._scan_spectral(len(times))
            if self.settings.detect:
                self._events(times, p, flin)
        now = time.perf_counter()
        if worked and now - self._report_t > 1.0:
            self._report_t = now
            self._make_report()
        return worked

    # -- levels per integration interval (ISO 18405) -------------------------------
    def _integrate_step(self, sig: np.ndarray, t_first: float) -> None:
        if self.intervals is None:
            return
        if self._int_t0 is None:
            self._int_t0, self._int_buf = t_first, sig
        else:
            self._int_buf = np.concatenate([self._int_buf, sig])
        rate = self.rate
        n = max(8, int(round(self.settings.integration_s * rate)))
        lo, hi = self._band_edges() if self.settings.band is not None else (None, None)
        while len(self._int_buf) >= n:
            seg, self._int_buf = self._int_buf[:n], self._int_buf[n:]
            t0 = self._int_t0
            self._int_t0 += n / rate
            m = metrics.broadband(seg, rate)
            f, psd = metrics.welch(seg, rate, self.psd_nfft)
            band = (float(to_db(metrics.band_mean_square(f, psd, lo, hi)))
                    if lo is not None else float("nan"))
            floor = float("nan")
            if self.levels is not None:
                _t, v = self.levels.range(t0, t0 + n / rate)
                if len(v):
                    floor = float(to_db(np.mean(10 ** (v[:, 1] / 10))))
            self.intervals.append(np.array([t0]),
                                  np.array([[m["SPL"], m["Lpk"], m["SEL"], floor, band]]))
            self.tol_series.append(np.array([t0]),
                                   metrics.band_levels(f, psd, self.tol_bands)[None, :])
            self.psd_series.append(np.array([t0]), to_db(psd)[None, :].astype(np.float32))

    # -- bearings (two-hydrophone array) ------------------------------------------
    def _pair(self, y: np.ndarray) -> np.ndarray | None:
        geo = self.settings.array
        if not geo.enabled or y.ndim != 2 or max(geo.ch_a, geo.ch_b) >= y.shape[1]:
            return None
        return np.nan_to_num(y[:, [geo.ch_a, geo.ch_b]].astype(np.float64))

    def _bearing_band(self) -> tuple[float, float] | None:
        return self._band_edges() if self.settings.band is not None else None

    def _bearing_step(self, snap) -> None:
        if self.btr is None:
            return
        pair = self._pair(snap.y)
        if pair is None:
            return
        if self._btr_t0 is None:
            self._btr_t0, self._btr_buf = float(snap.x[0]), pair
        else:
            self._btr_buf = np.concatenate([self._btr_buf, pair])
        n = max(64, int(self.settings.btr_step_s * self.rate))
        geo, band = self.settings.array, self._bearing_band()
        while len(self._btr_buf) >= n:
            block, self._btr_buf = self._btr_buf[:n], self._btr_buf[n:]
            scan = bearing_scan(block[:, 0], block[:, 1], self.rate, geo, self.btr_bearings, band)
            self.btr.append(np.array([self._btr_t0 + n / (2 * self.rate)]), scan[None, :])
            self._btr_t0 += n / self.rate

    def _detection_bearing(self, det: "Detection") -> None:
        if self.external or not self.settings.array.enabled:
            return
        snap = self.buffer.range(det.t0, det.t1)
        pair = self._pair(snap.y) if len(snap) else None
        if pair is None or len(pair) < 64:
            return
        band = (det.f_lo, det.f_hi) if det.f_hi > det.f_lo else None
        tau, q = tdoa(pair[:, 0], pair[:, 1], self.rate, self.settings.array.max_delay, band)
        det.bearing = bearing_from_delay(tau, self.settings.array)
        det.bearing_q = q

    def latest_psd(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(freqs, PSD dB per Hz, uncalibrated) of the last interval."""
        ring = self.psd_series
        if ring is None or not ring.count:
            return None
        _t, v = ring.latest(1)
        return self.psd_freqs, v[0]

    def psd_percentiles(self, q=(5, 50, 95)) -> dict[int, np.ndarray]:
        ring = self.psd_series
        if ring is None or ring.count < 3:
            return {}
        _t, v = ring.latest(ring.count)
        return metrics.percentiles_db(v, axis=0, q=q)

    def tol_window(self) -> tuple[np.ndarray, np.ndarray]:
        """Third-octave levels over the statistics window: (times, levels)."""
        ring = self.tol_series
        if ring is None or not ring.count:
            return np.zeros(0), np.zeros((0, len(self.tol_bands)))
        n = int(max(1, self.settings.stats_window_s / self.settings.integration_s))
        return ring.latest(n)

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
        return 0.0, self._nyquist() if self.rate else 0.0

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
        lo, hi = self._band_edges() if s.detect_in_band else (0.0, self._nyquist())
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
                                        "peak": bin_snr[i].copy()}
                continue
            if band_snr[i] >= off:
                ev["t1"] = t + hop_s / 2
                ev["snr"] = max(ev["snr"], band_snr[i])
                np.maximum(ev["peak"], bin_snr[i], out=ev["peak"])
                ev["quiet"] = 0
            else:
                ev["quiet"] += 1
            if ev["quiet"] * hop_s >= 0.08 or ev["t1"] - ev["t0"] >= s.max_event_s:
                self._event = None
                if ev["t1"] - ev["t0"] >= s.min_event_s:
                    # frequency extent: bins whose peak rose well above the floor
                    # during the event (follows sweeps; noise almost never reaches
                    # +12 dB in a single frame)
                    pk = ev["peak"]
                    active = np.flatnonzero(pk >= max(on + 2.0, float(pk.max()) - 25.0))
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
        self._detection_bearing(det)
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
            self._clip_bytes += _nbytes(clip)
            while self._clip_bytes > self.CLIP_BUDGET_BYTES and self.clips:
                self._drop_clip(next(iter(self.clips)))

    def _drop_clip(self, det_id: int) -> None:
        clip = self.clips.pop(det_id, None)
        if clip is not None:
            self._clip_bytes -= _nbytes(clip)

    def update_detection(self, det: Detection, **changes) -> None:
        with self._det_lock:
            for k, v in changes.items():
                setattr(det, k, v)

    # -- noise report ----------------------------------------------------------------
    DEMON_WINDOW_S = 10.0

    def _demon_step(self, rep) -> None:
        """Propeller modulation over the last ~10 s, refreshed every few
        seconds (less often when it is slow, e.g. at high sample rates)."""
        if not self.settings.demon or self.external or not self.rate:
            self._demon = None
            return
        now = time.perf_counter()
        if now >= self._demon_t:
            snap = self.buffer.latest(int(self.rate * self.DEMON_WINDOW_S))
            if len(snap) >= self.rate * 2:
                started = time.perf_counter()
                try:
                    self._demon = demon(self.channel_signal(snap.y), self.rate,
                                        self.settings.demon_band)
                except Exception:
                    log.exception("DEMON analysis failed")
                    self._demon = None
                self._demon_t = now + max(5.0, 20 * (time.perf_counter() - started))
        rep.demon = self._demon
        d = self._demon
        if d is not None and d.blades:
            rep.sources.insert(0, ("Vessel propeller (DEMON)", 0.9, d.describe()))

    def _make_report(self) -> None:
        s = self.settings
        off = self.offset_db
        rep = noise.NoiseReport(units=self.units)
        ring = self.intervals
        if ring is not None and ring.count:
            n = int(max(1, s.stats_window_s / s.integration_s))
            _t, v = ring.latest(n)
            rep.level_db = float(v[-1, 0] + off)
            rep.peak_db = float(v[-1, 1] + off)
            rep.floor_db = float(v[-1, 3] + off)
            rep.band_db = float(v[-1, 4] + off) if s.band is not None else float("nan")
            rep.stats = {k: val + off for k, val in noise.exceedance_levels(v[:, 0]).items()}
            _tt, tol = self.tol_series.latest(1)
            if len(tol):
                rep.third_octaves = {b.label: float(tol[0, i] + off)
                                     for i, b in enumerate(self.tol_bands)}
        raw = self._raw_tail
        if len(raw) >= 64:
            rep.crest_db, rep.kurtosis = noise.impulsiveness(raw)
        if self.spectro is not None and self.spectro.count >= 4:
            t, d = self.spectro.latest(int(10.0 / self._frame_hop_s()))
            ltsa = to_db(np.mean(10 ** (d / 10), axis=0))
            rep.flatness = noise.spectral_flatness(10 ** (ltsa[1:] / 10))
            rep.tonals = noise.tonal_lines(ltsa, self.freqs)
            rep.sources = noise.guess_sources(self.freqs, ltsa, rep.tonals, rep.kurtosis,
                                              2 * self._nyquist())
        self._demon_step(rep)
        self.report = rep
        self.alarm = bool(s.floor_alarm and np.isfinite(rep.floor_db)
                          and rep.floor_db >= s.floor_alarm_db)

