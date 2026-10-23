"""The main window: source selection, live plots, recording and export."""

from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QScrollArea, QSplitter, QStackedWidget, QVBoxLayout, QWidget)

from aquafonix import APP_NAME, __version__
from aquafonix.buffer import RingBuffer
from aquafonix.dsp import estimate_rate, find_trigger, minmax_decimate, spectrum
from aquafonix.recorder import CsvRecorder, WavRecorder
from aquafonix.screen.capture import Region, ScreenSource
from aquafonix.screen.overlay import RegionOutline, RegionSelector
from aquafonix.sources import Source
from aquafonix.ui.panels import PANELS, ScreenPanel, SourcePanel
from aquafonix.ui.plots import PreviewView, SpectrumPlot, TimePlot, channel_color
from aquafonix.ui.widgets import combo, named, restore_state, save_state, set_widget_value, spin

log = logging.getLogger(__name__)
FFT_SIZES = ["256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536"]


def _fmt(v: float) -> str:
    return f"{v:.6g}"


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings) -> None:
        super().__init__()
        self.settings = settings
        self.persist = True  # save settings on close
        self.source: Source | None = None
        self.buffer: RingBuffer | None = None
        self.recorder: CsvRecorder | WavRecorder | None = None
        self.paused = False
        self._last_version = -1
        self._last_frame = None
        self._spec_avg: dict[str, np.ndarray] = {}
        self._view: tuple[np.ndarray, np.ndarray, list[str]] | None = None
        self._rate_mark = (time.perf_counter(), 0)
        self._status_t = 0.0
        self._trigger_waiting = False
        self._outline_region: Region | None = None
        self._last_dir = str(Path.home())
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1280, 800)
        self.selector = RegionSelector(self)
        self.selector.finished.connect(self._region_selected)
        self.outline = RegionOutline()
        self._build()
        self._restore()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(33)

    # ======================================================================
    # layout
    # ======================================================================
    def _build(self) -> None:
        side = QWidget()
        sl = QVBoxLayout(side)

        # -- source --------------------------------------------------------
        box = QGroupBox("Source")
        bl = QVBoxLayout(box)
        self.source_combo = named(QComboBox(), "source")
        self.panels: list[SourcePanel] = [cls() for cls in PANELS]
        self.stack = QStackedWidget()
        for panel in self.panels:
            self.source_combo.addItem(panel.title, panel.key)
            self.stack.addWidget(panel)
        self.screen_panel: ScreenPanel = next(p for p in self.panels if isinstance(p, ScreenPanel))
        bl.addWidget(self.source_combo)
        bl.addWidget(self.stack)
        row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start")
        self.stop_btn = QPushButton("■  Stop")
        self.start_btn.setToolTip("Start (F5)")
        self.stop_btn.setToolTip("Stop (F5)")
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        bl.addLayout(row)
        sl.addWidget(box)

        # -- display -------------------------------------------------------
        box = QGroupBox("Display")
        f = QFormLayout(box)
        self.xaxis = named(combo({"time": "Time (s)", "samples": "Sample number"}, "time"),
                           "display.xaxis")
        f.addRow("X axis", self.xaxis)
        self.window_s = named(spin(0.001, 86_400, 5, decimals=3, suffix=" s"), "display.window")
        self.window_n = named(spin(10, 10_000_000, 1000, suffix=" samples"),
                              "display.windowsamples")
        f.addRow("Window", self.window_s)
        f.addRow("Window", self.window_n)
        self.mode = named(combo({"roll": "Rolling", "trigger": "Triggered (oscilloscope)"},
                                "roll"), "display.mode")
        f.addRow("Mode", self.mode)
        self.trig_channel = named(QComboBox(), "display.trigchannel")
        self.trig_level = named(spin(-1e9, 1e9, 0, decimals=4), "display.triglevel")
        self.trig_edge = named(combo({"rising": "Rising edge", "falling": "Falling edge"},
                                     "rising"), "display.trigedge")
        f.addRow("Trigger on", self.trig_channel)
        f.addRow("Level", self.trig_level)
        f.addRow("Edge", self.trig_edge)
        self.y_auto = named(QCheckBox("Auto-scale Y"), "display.yauto")
        self.y_auto.setChecked(True)
        f.addRow(self.y_auto)
        yrow = QHBoxLayout()
        self.y_min = named(spin(-1e9, 1e9, -1, decimals=4), "display.ymin")
        self.y_max = named(spin(-1e9, 1e9, 1, decimals=4), "display.ymax")
        yrow.addWidget(self.y_min)
        yrow.addWidget(QLabel("to"))
        yrow.addWidget(self.y_max)
        self.y_row = QWidget()
        self.y_row.setLayout(yrow)
        yrow.setContentsMargins(0, 0, 0, 0)
        f.addRow("Y range", self.y_row)
        self.spectrum_on = named(QCheckBox("Show spectrum (FFT)"), "display.spectrum")
        f.addRow(self.spectrum_on)
        self.fft_size = named(combo(FFT_SIZES, "4096"), "display.fftsize")
        self.fft_log = named(QCheckBox("Log frequency axis"), "display.fftlog")
        self.fft_avg = named(spin(0, 0.99, 0.5, decimals=2, step=0.05), "display.fftavg")
        f.addRow("FFT size", self.fft_size)
        f.addRow(self.fft_log)
        f.addRow("Averaging", self.fft_avg)
        row = QHBoxLayout()
        self.pause_btn = QPushButton("⏸  Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setToolTip("Freeze the display; data keeps arriving (F6)")
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        row.addWidget(self.pause_btn)
        row.addWidget(clear)
        f.addRow(row)
        self._display_form = f
        sl.addWidget(box)

        # -- record --------------------------------------------------------
        box = QGroupBox("Record && export")
        rl = QVBoxLayout(box)
        self.record_btn = QPushButton("●  Record to file…")
        self.record_btn.setCheckable(True)
        rl.addWidget(self.record_btn)
        self.record_label = QLabel("")
        self.record_label.setWordWrap(True)
        rl.addWidget(self.record_label)
        row = QHBoxLayout()
        save_csv = QPushButton("Save view as CSV…")
        save_png = QPushButton("Save image…")
        save_csv.clicked.connect(self._save_view_csv)
        save_png.clicked.connect(self._save_image)
        row.addWidget(save_csv)
        row.addWidget(save_png)
        rl.addLayout(row)
        sl.addWidget(box)
        sl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(370)

        # -- plots ---------------------------------------------------------
        self.preview = PreviewView()
        self.time_plot = TimePlot()
        self.time_plot.setAutoVisible(y=True)
        self.spec_plot = SpectrumPlot()
        self.plot_split = QSplitter(Qt.Vertical)
        for w in (self.preview, self.time_plot, self.spec_plot):
            self.plot_split.addWidget(w)
        self.plot_split.setStretchFactor(0, 2)
        self.plot_split.setStretchFactor(1, 3)
        self.plot_split.setStretchFactor(2, 2)
        self.readout = QLabel("")
        self.readout.setTextFormat(Qt.RichText)
        self.readout.setWordWrap(True)
        self.plots = QWidget()
        pl = QVBoxLayout(self.plots)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.plot_split, 1)
        pl.addWidget(self.readout)

        self.main_split = QSplitter(Qt.Horizontal)
        self.main_split.addWidget(scroll)
        self.main_split.addWidget(self.plots)
        self.main_split.setStretchFactor(1, 1)
        self.main_split.setSizes([390, 890])
        self.setCentralWidget(self.main_split)
        self.status_label = QLabel("")
        self.statusBar().addWidget(self.status_label, 1)

        # -- wiring --------------------------------------------------------
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.pause_btn.toggled.connect(self._set_paused)
        self.record_btn.toggled.connect(self._toggle_record)
        for w in (self.xaxis, self.mode):
            w.currentIndexChanged.connect(self._display_changed)
        for w in (self.y_auto, self.spectrum_on, self.fft_log):
            w.toggled.connect(self._display_changed)
        for w in (self.y_min, self.y_max, self.window_s, self.window_n):
            w.valueChanged.connect(self._display_changed)
        self.fft_size.currentIndexChanged.connect(lambda *_: self._spec_avg.clear())
        self.trig_level.valueChanged.connect(self._trigger_level_edited)
        self.time_plot.trigger_line.sigPositionChanged.connect(
            lambda line: self.trig_level.setValue(line.value()))
        QShortcut(QKeySequence(Qt.Key_F5), self, activated=self.toggle_running)
        QShortcut(QKeySequence(Qt.Key_F6), self, activated=self.pause_btn.toggle)

        sp = self.screen_panel
        sp.select_requested.connect(self._select_region)
        sp.save_trace_requested.connect(self._save_trace)
        sp.region_changed.connect(lambda _r: self._update_outline())
        sp.cal_line_changed.connect(self.preview.set_line)
        sp.changed.connect(self._screen_settings_changed)
        self.preview.line_moved.connect(sp.set_cal_px)
        self.preview.pixel_clicked.connect(self._preview_clicked)

        self._source_changed()
        self._display_changed()

    # ======================================================================
    # public API (also used by the command line)
    # ======================================================================
    def current_panel(self) -> SourcePanel:
        return self.panels[self.stack.currentIndex()]

    def select_source(self, key: str) -> None:
        idx = self.source_combo.findData(key)
        if idx < 0:
            keys = ", ".join(p.key for p in self.panels)
            raise ValueError(f"unknown source {key!r} (choose from {keys})")
        self.source_combo.setCurrentIndex(idx)

    def set_control(self, name: str, value: str) -> None:
        w = self.findChild(QWidget, name)
        if w is None or not set_widget_value(w, value):
            raise KeyError(f"no setting called {name!r}")

    @property
    def running(self) -> bool:
        return self.source is not None and self.source.running

    def toggle_running(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self) -> bool:
        if self.running:
            return True
        panel = self.current_panel()
        try:
            src = panel.create_source()
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Can't start: {exc}")
            return False
        buffer = RingBuffer(max(1000, src.buffer_capacity))
        try:
            src.start(buffer)
        except Exception as exc:
            log.exception("failed to start %s", panel.key)
            src.stop()
            QMessageBox.warning(self, APP_NAME, f"Could not start {panel.title}:\n\n{exc}")
            return False
        self.source, self.buffer = src, buffer
        self._reset_view()
        self._update_buttons()
        self._display_changed()
        return True

    def stop(self) -> None:
        self._stop_recording()
        if self.source is not None:
            self.source.stop()
        self._update_buttons()

    def clear(self) -> None:
        if self.buffer is not None:
            self.buffer.clear()
        if isinstance(self.source, ScreenSource):
            self.source.stitcher.reset()
        self._reset_view()

    # ======================================================================
    # display refresh
    # ======================================================================
    def _reset_view(self) -> None:
        self.time_plot.clear_series()
        self.spec_plot.clear_series()
        self._spec_avg.clear()
        self._last_version = -1
        self._last_frame = None
        self._view = None
        self.readout.setText("")

    def _refresh(self) -> None:
        self._update_status()
        src, buf = self.source, self.buffer
        if src is None or buf is None or self.paused:
            return
        if isinstance(src, ScreenSource):
            self._refresh_screen(src)
            if src.output == "trace":
                return
        if buf.version == self._last_version:
            return
        self._last_version = buf.version
        self._refresh_time(buf)
        if self.spectrum_on.isChecked():
            self._refresh_spectrum(src, buf)

    def _refresh_time(self, buf: RingBuffer) -> None:
        samples_axis = self.xaxis.currentData() == "samples"
        triggered = self.mode.currentData() == "trigger"
        if samples_axis:
            span = float(self.window_n.value())
            snap = buf.latest(int(span) * (2 if triggered else 1))
            x = snap.first_index + np.arange(len(snap), dtype=float)
        else:
            last = buf.last_x()
            if last is None:
                return
            span = self.window_s.value()
            snap = buf.since(last - span * (2 if triggered else 1))
            x = snap.x
        if not len(snap):
            return
        y, names = snap.y, snap.names
        self._update_channel_list(names)
        x_range = (x[-1] - span, x[-1])
        self._trigger_waiting = False
        if triggered:
            ch = names.index(self.trig_channel.currentText()) \
                if self.trig_channel.currentText() in names else 0
            xt = find_trigger(x, y[:, ch], self.trig_level.value(), span,
                              rising=self.trig_edge.currentData() == "rising", pretrigger=0.1)
            if xt is None:  # "auto" scope behaviour: show the latest data untriggered
                self._trigger_waiting = True
                keep = x >= x[-1] - span
            else:
                x = x - xt
                x_range = (-0.1 * span, 0.9 * span)
                keep = (x >= x_range[0]) & (x <= x_range[1])
            x, y = x[keep], y[keep]
        self._view = (x, y, names)
        xd, yd = minmax_decimate(x, y, 4 * max(200, self.time_plot.width()))
        self.time_plot.set_series(xd, yd, names)
        self.time_plot.setXRange(*x_range, padding=0)
        self._apply_y_range()
        self._update_readout(names, y)

    def _refresh_spectrum(self, src: Source, buf: RingBuffer) -> None:
        nfft = int(self.fft_size.currentText())
        snap = buf.latest(nfft)
        rate = src.sample_rate or estimate_rate(snap.x)
        if not rate or len(snap) < 16:
            return
        alpha = self.fft_avg.value()
        for i, name in enumerate(snap.names):
            freqs, db = spectrum(snap.y[:, i], rate, nfft)
            if len(freqs) < 2:
                continue
            freqs, db = freqs[1:], db[1:]  # drop DC (log axis can't show 0 Hz)
            prev = self._spec_avg.get(name)
            if prev is not None and prev.shape == db.shape and alpha > 0:
                db = alpha * prev + (1 - alpha) * db
            self._spec_avg[name] = db
            self.spec_plot.curve(name).setData(freqs, db)

    def _refresh_screen(self, src: ScreenSource) -> None:
        frame = src.latest
        if frame is None or frame is self._last_frame:
            return
        self._last_frame = frame
        r = frame.result
        probe_px = None
        if src.output != "stitch" and src.probe == "column" and len(r.x_px):
            probe_px = src.probe_column * (len(r.x_px) - 1) + 0.5
        if self.preview.isVisible():
            mask = r.mask if self.screen_panel.show_mask.isChecked() else None
            self.preview.set_frame(frame.image, r.x_px, r.y_px, mask, probe_px)
        self.screen_panel.show_detected(r.color, r.background)
        if src.output == "trace":
            names = [src.channel_name]
            y = r.y[:, None]
            self._view = (r.x, y, names)
            self.time_plot.set_series(r.x, y, names)
            finite = np.isfinite(r.x)
            if finite.any():
                lo, hi = float(np.min(r.x[finite])), float(np.max(r.x[finite]))
                if hi > lo:
                    self.time_plot.setXRange(lo, hi, padding=0.01)
            self._apply_y_range()
            self._update_readout(names, y)

    def _apply_y_range(self) -> None:
        if self.y_auto.isChecked():
            self.time_plot.enableAutoRange(axis="y")
        else:
            lo, hi = sorted((self.y_min.value(), self.y_max.value()))
            if hi > lo:
                self.time_plot.setYRange(lo, hi, padding=0)

    def _update_readout(self, names: list[str], y: np.ndarray) -> None:
        parts = []
        for i, name in enumerate(names[:16]):
            col = y[:, i]
            fin = col[np.isfinite(col)]
            if not len(fin):
                continue
            parts.append(
                f'<span style="color:{channel_color(i)}">■</span> <b>{name}</b> '
                f"{_fmt(fin[-1])} <span style='color:gray'>"
                f"[{_fmt(fin.min())} … {_fmt(fin.max())}]</span>")
        self.readout.setText("&nbsp;&nbsp;&nbsp;".join(parts))

    def _update_channel_list(self, names: list[str]) -> None:
        current = [self.trig_channel.itemText(i) for i in range(self.trig_channel.count())]
        if current == names:
            return
        keep = self.trig_channel.currentText()
        self.trig_channel.blockSignals(True)
        self.trig_channel.clear()
        self.trig_channel.addItems(names)
        if keep in names:
            self.trig_channel.setCurrentText(keep)
        self.trig_channel.blockSignals(False)

    def _update_status(self) -> None:
        now = time.perf_counter()
        if now - self._status_t < 0.5:
            return
        src = self.source
        if src is None:
            text = "Choose a source and press Start (F5)."
        else:
            t0, n0 = self._rate_mark
            n = src.samples_received
            rate = (n - n0) / max(now - t0, 1e-6)
            self._rate_mark = (now, n)
            parts = [src.status]
            if src.running:
                parts.append(f"{rate:,.0f} samples/s")
            if self.buffer is not None:
                parts.append(f"{len(self.buffer):,} buffered")
            if self._trigger_waiting and self.mode.currentData() == "trigger":
                parts.append("waiting for trigger")
            if self.recorder is not None:
                parts.append(f"● REC {self.recorder.rows:,} samples")
            if self.paused:
                parts.append("display paused")
            text = "  ·  ".join(parts)
        self._status_t = now
        self.status_label.setText(text)
        self._update_buttons()

    def _update_buttons(self) -> None:
        running = self.running
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if not running and self.recorder is not None:
            self._stop_recording()

    # ======================================================================
    # reactions to controls
    # ======================================================================
    def _source_changed(self, *_args) -> None:
        idx = max(0, self.source_combo.currentIndex())
        if self.running and idx != self.stack.currentIndex():
            self.stop()
        self.stack.setCurrentIndex(idx)
        self.preview.setVisible(self.current_panel() is self.screen_panel)
        self._update_outline()
        self._update_axis_label()

    def _display_changed(self, *_args) -> None:
        f = self._display_form
        samples = self.xaxis.currentData() == "samples"
        f.setRowVisible(self.window_s, not samples)
        f.setRowVisible(self.window_n, samples)
        trig = self.mode.currentData() == "trigger"
        for w in (self.trig_channel, self.trig_level, self.trig_edge):
            f.setRowVisible(w, trig)
        self.time_plot.trigger_line.setVisible(trig)
        self.time_plot.trigger_line.setValue(self.trig_level.value())
        self.y_row.setEnabled(not self.y_auto.isChecked())
        spec = self.spectrum_on.isChecked()
        for w in (self.fft_size, self.fft_log, self.fft_avg):
            f.setRowVisible(w, spec)
        self.spec_plot.setVisible(spec)
        self.spec_plot.setLogMode(x=self.fft_log.isChecked(), y=False)
        self._update_axis_label()
        self._last_version = -1  # redraw with the new settings
        self._apply_y_range()

    def _update_axis_label(self) -> None:
        src = self.source
        if isinstance(src, ScreenSource) and src.output == "trace":
            self.time_plot.setLabel("bottom", "X (calibrated)", units="")
        elif self.xaxis.currentData() == "samples":
            self.time_plot.setLabel("bottom", "Sample", units="")
        elif self.mode.currentData() == "trigger":
            self.time_plot.setLabel("bottom", "Time from trigger", units="s")
        else:
            self.time_plot.setLabel("bottom", "Time", units="s")

    def _trigger_level_edited(self, value: float) -> None:
        line = self.time_plot.trigger_line
        line.blockSignals(True)
        line.setValue(value)
        line.blockSignals(False)
        self._last_version = -1

    def _set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.pause_btn.setText("▶  Resume" if paused else "⏸  Pause")

    # ======================================================================
    # screen region
    # ======================================================================
    def _select_region(self) -> None:
        self.outline.hide()
        self.hide()
        QTimer.singleShot(350, self.selector.start)  # let the window disappear first

    def _region_selected(self, region: Region | None) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        if region is not None:
            self.screen_panel.set_region(region)
            self.preview.setVisible(True)
        self._update_outline()

    def _update_outline(self) -> None:
        sp = self.screen_panel
        region = sp.region()
        show = (self.current_panel() is sp and sp.outline.isChecked() and region is not None
                and self.isVisible())
        if not show:
            self.outline.hide()
            return
        if region != self._outline_region or not self.outline.isVisible():
            self._outline_region = region
            self.outline.set_region(region)
            self.outline.show()

    def _screen_settings_changed(self) -> None:
        src = self.source
        if isinstance(src, ScreenSource) and src.running:
            old_output = src.output
            self.screen_panel.apply_to(src)
            if src.output != old_output:
                self.clear()
        self._update_outline()
        self._update_axis_label()

    def _preview_clicked(self, col: int, row: int) -> None:
        sp = self.screen_panel
        frame = self._last_frame
        if not sp.pick.isChecked() or frame is None:
            return
        img = frame.image
        # Anti-aliased lines blend into the background: take the pixel near the
        # click that stands out most from it.
        r0, r1 = max(0, row - 3), min(img.shape[0], row + 4)
        c0, c1 = max(0, col - 3), min(img.shape[1], col + 4)
        patch = img[r0:r1, c0:c1].reshape(-1, 3).astype(int)
        dist = ((patch - np.array(frame.result.background)) ** 2).sum(axis=1)
        sp.set_picked_color(tuple(int(v) for v in patch[int(np.argmax(dist))]))

    def _save_trace(self) -> None:
        frame = self._last_frame
        if frame is None:
            QMessageBox.information(self, APP_NAME, "Start watching a screen region first.")
            return
        path = self._ask_save("Save trace", "trace", "CSV (*.csv)")
        if not path:
            return
        r = frame.result
        data = np.column_stack([r.x, r.y, r.x_px, r.y_px])
        np.savetxt(path, data, delimiter=",", fmt="%.10g", comments="",
                   header="x,y,x_px,y_px")

    # ======================================================================
    # recording / export
    # ======================================================================
    def _ask_save(self, title: str, stem: str, filters: str) -> str:
        default = Path(self._last_dir) / f"{stem}_{dt.datetime.now():%Y%m%d_%H%M%S}"
        ext = filters.split("*")[1].split(")")[0].split()[0] if "*" in filters else ""
        path, chosen = QFileDialog.getSaveFileName(self, title, str(default) + ext, filters)
        if not path:
            return ""
        if not Path(path).suffix and "*." in chosen:
            path += "." + chosen.split("*.")[1].split(")")[0].split()[0]
        self._last_dir = str(Path(path).parent)
        return path

    def _toggle_record(self, checked: bool) -> None:
        if not checked:
            self._stop_recording()
            return
        src = self.source
        if src is None or not src.running:
            QMessageBox.information(self, APP_NAME, "Start a source before recording.")
            self._set_record_checked(False)
            return
        filters = "CSV (*.csv)" + (";;WAV audio (*.wav)" if src.sample_rate else "")
        path = self._ask_save("Record to file", "aquafonix", filters)
        if not path:
            self._set_record_checked(False)
            return
        try:
            if path.lower().endswith(".wav"):
                channels = (getattr(src, "channels", None)
                            or len(self.buffer.names if self.buffer else []) or 1)
                rec = WavRecorder(path, src.sample_rate, channels)
            else:
                cfg = getattr(src, "parser_config", None)
                rec = CsvRecorder(path, "x" if cfg is not None and cfg.x_from_first
                                  else "time_s")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not create {path}:\n{exc}")
            self._set_record_checked(False)
            return
        self.recorder = rec
        src.add_listener(rec.write)
        self.record_btn.setText("■  Stop recording")
        self.record_label.setText(f"Recording to {Path(path).name}")

    def _stop_recording(self) -> None:
        rec, self.recorder = self.recorder, None
        if rec is not None:
            if self.source is not None:
                self.source.remove_listener(rec.write)
            rec.close()
            self.record_label.setText(f"Saved {rec.rows:,} samples to {Path(rec.path).name}")
        self._set_record_checked(False)

    def _set_record_checked(self, on: bool) -> None:
        self.record_btn.blockSignals(True)
        self.record_btn.setChecked(on)
        self.record_btn.blockSignals(False)
        if not on:
            self.record_btn.setText("●  Record to file…")

    def _save_view_csv(self) -> None:
        if self._view is None:
            QMessageBox.information(self, APP_NAME, "Nothing on screen to save yet.")
            return
        x, y, names = self._view
        path = self._ask_save("Save view", "view", "CSV (*.csv)")
        if path:
            np.savetxt(path, np.column_stack([x, y]), delimiter=",", fmt="%.10g", comments="",
                       header=",".join(["x", *names]))

    def _save_image(self) -> None:
        path = self._ask_save("Save image", "aquafonix", "PNG image (*.png)")
        if path:
            self.plots.grab().save(path)

    # ======================================================================
    # settings
    # ======================================================================
    def _restore(self) -> None:
        s = self.settings
        geometry = s.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        for key, split in (("window/main_split", self.main_split),
                           ("window/plot_split", self.plot_split)):
            state = s.value(key)
            if state is not None:
                split.restoreState(state)
        self._last_dir = str(s.value("window/last_dir", self._last_dir))
        s.beginGroup("controls")
        restore_state(self, s)
        s.endGroup()
        for key, box in self.screen_panel.cal_px.items():
            self.preview.set_line(key, box.value())
        self._source_changed()
        self._display_changed()

    def save_settings(self) -> None:
        s = self.settings
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/main_split", self.main_split.saveState())
        s.setValue("window/plot_split", self.plot_split.saveState())
        s.setValue("window/last_dir", self._last_dir)
        s.beginGroup("controls")
        save_state(self, s)
        s.endGroup()
        s.sync()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_outline()

    def closeEvent(self, event) -> None:
        self.stop()
        self.outline.close()
        if self.persist:
            self.save_settings()
        super().closeEvent(event)
