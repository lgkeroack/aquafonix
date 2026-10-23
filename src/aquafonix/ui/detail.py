"""Selection detail window: everything about one stretch of signal."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from aquafonix.acoustics import metrics
from aquafonix.ui.plots import SpectrumPlot, TolPlot, colormap
from aquafonix.ui.widgets import combo

NFFT = {"0": "Auto", "256": "256", "512": "512", "1024": "1024", "2048": "2048", "4096": "4096",
        "8192": "8192", "16384": "16384", "32768": "32768", "65536": "65536"}


class SelectionDetail(QWidget):
    """A separate window (several can be open) analysing a selected span:
    waveform, zoomed spectrogram with its own FFT size, PSD with
    percentiles, third-octave levels and a table of standard metrics."""

    def __init__(self, parent, x: np.ndarray, y: np.ndarray, names: list[str], rate: float,
                 offset_db: float, calibrated: bool, band: tuple[float, float] | None = None,
                 wall_start: float | None = None, play=None, identify=None,
                 save_event=None) -> None:
        super().__init__(parent, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.x, self.y, self.names, self.rate = x, y, names, float(rate)
        self.offset, self.calibrated, self.band = offset_db, calibrated, band
        self.wall_start = wall_start
        self._play, self._identify, self._save_event = play, identify, save_event
        u = "dB re 1 µPa" if calibrated else "dBFS"
        self.units = {"SPL": u, "Lpk": u, "Lpkpk": u,
                      "SEL": "dB re 1 µPa²·s" if calibrated else "dBFS²·s",
                      "PSD": "dB re 1 µPa²/Hz" if calibrated else "dBFS²/Hz"}
        self.setWindowTitle(f"Selection {x[0]:.3f}–{x[-1]:.3f} s")
        self.resize(1100, 760)

        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.channel = QComboBox()
        self.channel.addItems(names or ["ch1"])
        self.nfft = combo(NFFT, "0")
        for w, label in ((self.channel, "Channel"), (self.nfft, "FFT size")):
            bar.addWidget(QLabel(label))
            bar.addWidget(w)
            w.currentIndexChanged.connect(self.recompute)
        bar.addStretch(1)
        for text, slot in (("▶ Play", self._do_play), ("🏷 Identify…", self._do_identify),
                           ("💾 Save as event", self._do_save), ("Export CSV…", self.export_csv),
                           ("Save image…", self.save_image)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            if (text.startswith("▶") and play is None) or (
                    text.startswith("🏷") and identify is None) or (
                    text.startswith("💾") and save_event is None):
                b.setEnabled(False)
            bar.addWidget(b)
        lay.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        plots = QSplitter(Qt.Vertical)
        self.wave = pg.PlotWidget()
        self.wave.showGrid(x=True, y=True, alpha=0.25)
        self.wave.setLabel("bottom", "Time", units="s")
        self.wave_curve = self.wave.plot([], [], pen=pg.mkPen("#4cc9f0", width=1))
        self.spec = pg.PlotWidget()
        self.spec.setLabel("left", "Frequency", units="Hz")
        self.spec.setLabel("bottom", "Time", units="s")
        self.spec.setXLink(self.wave)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.spec.addItem(self.img)
        self.cbar = pg.ColorBarItem(values=(0, 1), colorMap=colormap("inferno"),
                                    interactive=True, width=12, label="dB")
        self.cbar.setImageItem(self.img, insert_in=self.spec.getPlotItem())
        if band is not None:
            self.band_region = pg.LinearRegionItem(band, orientation="horizontal",
                                                   movable=False,
                                                   brush=pg.mkBrush(255, 255, 255, 20))
            self.spec.addItem(self.band_region)
        lower = QSplitter(Qt.Horizontal)
        self.psd = SpectrumPlot()
        self.psd.setLogMode(x=True, y=False)
        self.tol = TolPlot()
        lower.addWidget(self.psd)
        lower.addWidget(self.tol)
        for w, k in ((self.wave, 2), (self.spec, 3), (lower, 3)):
            plots.addWidget(w)
            plots.setStretchFactor(plots.count() - 1, k)
        split.addWidget(plots)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", "Unit"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMinimumWidth(330)
        split.addWidget(self.table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)
        self.rows: list[tuple[str, str, str]] = []
        self.psd_data = None
        self.tol_data = None
        self.recompute()

    def signal(self) -> np.ndarray:
        ch = max(0, self.channel.currentIndex())
        y = self.y[:, ch] if self.y.ndim == 2 and ch < self.y.shape[1] else self.y.reshape(-1)
        return np.nan_to_num(np.asarray(y, dtype=float))

    def recompute(self, *_args) -> None:
        sig, rate, off = self.signal(), self.rate, self.offset
        n = len(sig)
        t = self.x
        self.wave_curve.setData(t, sig)
        nfft = int(self.nfft.currentData()) or metrics.auto_psd_nfft(rate, n, 1.0)
        nfft = int(min(nfft, 1 << int(np.log2(max(n, 8)))))
        # zoomed spectrogram: finer time steps than the main view
        spec_nfft = int(min(nfft, max(64, 1 << int(np.log2(max(n // 16, 64))))))
        f, frames = metrics.psd_frames(sig, rate, spec_nfft, overlap=0.875)
        if len(frames):
            d = metrics.psd_db(frames, off).T
            hi = float(np.percentile(d, 99.7))
            self.img.setImage(d, autoLevels=False)
            self.img.setRect(QRectF(t[0], 0, t[-1] - t[0], f[-1]))
            self.cbar.setLevels((hi - 70, hi))
        # PSD with percentiles across frames
        fp, pf = metrics.psd_frames(sig, rate, nfft, overlap=0.5)
        psd = pf.mean(axis=0) if len(pf) else np.zeros(len(fp))
        pct = metrics.percentiles_db(metrics.psd_db(pf, off), axis=0) if len(pf) >= 3 else {}
        self.psd.set_psd(fp, metrics.psd_db(psd, off), pct, units=self.units["PSD"])
        self.psd_data = (fp, metrics.psd_db(psd, off), pct)
        # third-octave levels (median/5-95 % across frames when several)
        bands = metrics.third_octave_bands(rate / 2)
        tol = metrics.band_levels(fp, psd, bands, off)
        tol_pct = None
        if len(pf) >= 3:
            per = metrics.band_levels(fp, pf, bands, off)
            p5, p50, p95 = np.nanpercentile(per, [5, 50, 95], axis=0)
            tol_pct = {5: p5, 50: p50, 95: p95}
        resolved = [metrics.band_resolution_ok(fp, b) for b in bands]
        self.tol.set_levels(bands, tol, tol_pct, (), resolved, self.units["SPL"])
        self.tol_data = (bands, tol, resolved)
        # metrics table
        bb = metrics.broadband(sig, rate, off)
        desc = metrics.spectral_descriptors(fp, psd)
        rows = []
        if self.wall_start:
            rows.append(("Start (local time)",
                         dt.datetime.fromtimestamp(self.wall_start).isoformat(
                             sep=" ", timespec="milliseconds"), ""))
        rows += [("Start", f"{t[0]:.4f}", "s"), ("End", f"{t[-1]:.4f}", "s"),
                 ("Duration", f"{bb['duration']:.4f}", "s"),
                 ("SPL  L_p,rms", f"{bb['SPL']:.2f}", self.units["SPL"]),
                 ("Peak  L_p,pk", f"{bb['Lpk']:.2f}", self.units["Lpk"]),
                 ("Peak-to-peak", f"{bb['Lpkpk']:.2f}", self.units["Lpkpk"]),
                 ("SEL  L_E", f"{bb['SEL']:.2f}", self.units["SEL"])]
        if self.band is not None:
            lo, hi = self.band
            ms = metrics.band_mean_square(fp, psd, lo, hi)
            rows.append((f"Band SPL {lo:,.0f}–{hi:,.0f} Hz",
                         f"{10 * np.log10(max(float(ms), 1e-30)) + off:.2f}", self.units["SPL"]))
        if "crest_db" in bb:
            rows += [("Crest factor", f"{bb['crest_db']:.2f}", "dB"),
                     ("Kurtosis", f"{bb['kurtosis']:.2f}", "(3 = Gaussian)")]
        if desc:
            rows += [("Peak frequency", f"{desc['peak_hz']:,.1f}", "Hz"),
                     ("Spectral centroid", f"{desc['centroid_hz']:,.1f}", "Hz"),
                     ("−3 dB bandwidth", f"{desc['bw3_hz']:,.1f}  "
                      f"({desc['f_lo3_hz']:,.0f}–{desc['f_hi3_hz']:,.0f})", "Hz"),
                     ("−10 dB bandwidth", f"{desc['bw10_hz']:,.1f}  "
                      f"({desc['f_lo10_hz']:,.0f}–{desc['f_hi10_hz']:,.0f})", "Hz"),
                     ("RMS bandwidth", f"{desc['rms_bandwidth_hz']:,.1f}", "Hz")]
        rows += [("Sample rate", f"{rate:g}", "Hz"),
                 ("PSD resolution", f"{rate / nfft:.3f}", f"Hz (FFT {nfft})"),
                 ("Levels", "calibrated" if self.calibrated else "uncalibrated (full scale)", "")]
        self.rows = rows
        self.table.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, v in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(v))

    # -- actions -------------------------------------------------------------------
    def _clip(self):
        return self.y[:, :2] if self.y.ndim == 2 else self.y, self.rate, float(self.x[0])

    def _do_play(self) -> None:
        if self._play:
            self._play(*self._clip())

    def _do_identify(self) -> None:
        if self._identify:
            self._identify(self.signal(), self.rate, float(self.x[0]), float(self.x[-1]),
                           self.band)

    def _do_save(self) -> None:
        if self._save_event:
            self._save_event(float(self.x[0]), float(self.x[-1]))

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export selection analysis",
                                              "selection.csv", "CSV (*.csv)")
        if not path:
            return
        base = Path(path)
        with open(base, "w", encoding="utf-8") as f:
            f.write("metric,value,unit\n")
            for name, value, unit in self.rows:
                f.write(f'"{name}","{value}","{unit}"\n')
        if self.psd_data is not None:
            fp, psd, pct = self.psd_data
            cols = [fp, psd] + [pct[k] for k in (5, 50, 95) if k in pct]
            hdr = ["freq_hz", "psd_db"] + [f"p{k}_db" for k in (5, 50, 95) if k in pct]
            np.savetxt(base.with_name(base.stem + "_psd.csv"), np.column_stack(cols),
                       delimiter=",", fmt="%.6g", header=",".join(hdr), comments="")
        if self.tol_data is not None:
            bands, tol, resolved = self.tol_data
            with open(base.with_name(base.stem + "_third_octaves.csv"), "w",
                      encoding="utf-8") as f:
                f.write("band,centre_hz,low_hz,high_hz,level_db,resolved\n")
                for b, v, ok in zip(bands, tol, resolved):
                    f.write(f"{b.label},{b.centre:.3f},{b.low:.3f},{b.high:.3f},{v:.3f},{ok}\n")

    def save_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save image", "selection.png",
                                              "PNG image (*.png)")
        if path:
            self.grab().save(path)
