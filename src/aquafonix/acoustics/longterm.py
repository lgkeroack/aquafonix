"""Long-term spectral average (LTSA) and SPL overview of a whole recording,
computed in the background so hours of audio can be navigated at a glance.

Each column samples a few FFT frames at that point in the recording instead
of reading everything, so the overview of a multi-gigabyte recording is
ready in seconds.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from aquafonix.acoustics import metrics

log = logging.getLogger(__name__)


class RecordingOverview:
    def __init__(self, source, channel: int = 0, columns: int = 1500, rows: int = 160,
                 frames_per_column: int = 4) -> None:
        self.source = source  # AudioFileSource (provides read(), duration, sample_rate)
        self.channel = channel
        rate = source.sample_rate
        self.nfft = int(min(8192, max(256, 1 << int(np.log2(max(rate / 10, 256))))))
        span = frames_per_column * self.nfft / rate
        self.columns = int(max(2, min(columns, source.duration / max(span, 1e-6))))
        self.frames_per_column = frames_per_column
        self.times = np.linspace(0, source.duration, self.columns, endpoint=False)
        f_lo = max(rate / self.nfft, 5.0)
        self.freqs = np.logspace(np.log10(f_lo), np.log10(rate / 2), rows)
        self.ltsa = np.full((rows, self.columns), np.nan, dtype=np.float32)
        self.spl = np.full(self.columns, np.nan)
        self.done = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="Overview", daemon=True)

    @property
    def complete(self) -> bool:
        return self.done >= self.columns

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        src, rate = self.source, self.source.sample_rate
        n = self.frames_per_column * self.nfft
        f_bins = np.fft.rfftfreq(self.nfft, 1 / rate)
        # map FFT bins onto the log-frequency rows
        edges = np.concatenate([[self.freqs[0] / 1.02], np.sqrt(self.freqs[1:] * self.freqs[:-1]),
                                [self.freqs[-1] * 1.02]])
        rows = len(self.freqs)
        idx = np.clip(np.searchsorted(edges, f_bins) - 1, -1, rows)
        try:
            for i, t in enumerate(self.times):
                if self._stop.is_set():
                    return
                data = src.read(float(t), n / rate)
                if len(data) < self.nfft:
                    self.done = i + 1
                    continue
                ch = min(self.channel, data.shape[1] - 1) if self.channel >= 0 else None
                x = data[:, ch] if ch is not None else data.mean(axis=1)
                x = np.nan_to_num(x.astype(np.float64))
                f, p = metrics.welch(x, rate, self.nfft)
                inside = (idx >= 0) & (idx < rows)
                sums = np.bincount(idx[inside], weights=p[inside], minlength=rows)
                counts = np.bincount(idx[inside], minlength=rows)
                with np.errstate(invalid="ignore", divide="ignore"):
                    col = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
                # rows narrower than one FFT bin: take the bin they fall in
                missing = np.isnan(col)
                if missing.any():
                    col[missing] = np.interp(self.freqs[missing], f, p)
                self.ltsa[:, i] = metrics.psd_db(col)
                self.spl[i] = metrics.spl_rms(x)
                self.done = i + 1
        except Exception as exc:
            log.exception("recording overview failed")
            self.error = str(exc)
