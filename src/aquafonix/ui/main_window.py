"""The main window: sources, live and explorable plots, acoustics, the
identify / confirm / correct loop, event capture, replay and recording."""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
                               QMainWindow, QMessageBox, QProgressDialog, QPushButton,
                               QScrollArea, QSlider, QSplitter, QStackedWidget, QTabWidget,
                               QToolBar, QVBoxLayout, QWidget)

from aquafonix import APP_NAME, __version__
from aquafonix import raven
from aquafonix.acoustics.analyzer import Analyzer, Detection
from aquafonix.acoustics.filters import denoise, filter_clip
from aquafonix.acoustics.signatures import Matcher, SignatureLibrary, estimate_band
from aquafonix.buffer import RingBuffer
from aquafonix.dsp import find_trigger, minmax_decimate
from aquafonix.alerts import Alert, AlertEngine, AlertRule
from aquafonix.audiofile import OPEN_FILTER
from aquafonix.events import EventStore
from aquafonix.outputs import (OutputHub, OutputSettings, alert_message, detection_message,
                               levels_message)
from aquafonix.player import SPEEDS, Player
from aquafonix.recorder import CsvRecorder, WavRecorder
from aquafonix.screen.capture import Region, ScreenSource
from aquafonix.screen.extract import auto_trace_colors
from aquafonix.screen.overlay import RegionOutline, RegionSelector
from aquafonix.sources import Source
from aquafonix.sources.audiofile import AudioFileSource
from aquafonix.ui.acoustics_panel import AcousticsPanel
from aquafonix.ui.alerts_ui import (AlertsDialog, AlertsPanel, OutputsDialog, Toast,
                                    alert_time_text, beep)
from aquafonix.ui.detail import SelectionDetail
from aquafonix.ui.dialogs import IdentifyDialog, SignatureManager, ask_label
from aquafonix.ui.panels import PANELS, AudioFilePanel, ScreenPanel, SourcePanel
from aquafonix.acoustics.metrics import band_resolution_ok, percentiles_db
from aquafonix.acoustics.longterm import RecordingOverview
from aquafonix.ui.plots import (COLORMAPS, BearingView, LevelsPlot, OverviewPlot, PreviewView,
                                RecordingOverviewView, SpectrogramView, SpectrumPlot, TimePlot,
                                TolPlot, channel_color)
from aquafonix.ui.review import DetectionsPanel, EventsPanel
from aquafonix.ui.widgets import combo, named, restore_state, save_state, set_widget_value, spin

log = logging.getLogger(__name__)

SHORTCUTS = """\
F5  start / stop          F6  pause (explore history) / live
F9  capture the last N seconds as an event
+ / -  zoom time in / out     Ctrl+Up / Ctrl+Down  zoom Y     0  fit
S  selection tool (or Shift-drag on the waveform / spectrogram)
Z  zoom to selection     I  identify selection     P  play selection
L  listen live     Esc  stop playback / clear selection
C / X  confirm / reject the selected detection
Mouse wheel zooms, drag pans; right-click a plot for more options."""


def _fmt(v: float) -> str:
    return f"{v:.6g}"


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings, data_dir: Path) -> None:
        super().__init__()
        self.settings = settings
        self.persist = True
        self.data_dir = Path(settings.value("data/dir", str(data_dir)))
        self.library = SignatureLibrary(self.data_dir / "signatures")
        self.matcher = Matcher(self.library)
        self.store = EventStore(self.data_dir / "events")
        self.player = Player()
        self.source: Source | None = None
        self.buffer: RingBuffer | None = None
        self.analyzer: Analyzer | None = None
        self.rec_overview: RecordingOverview | None = None
        self._ov_final = False
        self.recorder: CsvRecorder | WavRecorder | None = None
        self.explore = False  # display frozen; the user navigates the history
        self._wall0: float | None = None
        self._sel: tuple[float, float, float | None, float | None] | None = None
        self._selected_det: int | None = None
        self._last_version = -1
        self._last_frame = None
        self._view: tuple[np.ndarray, np.ndarray, list[str]] | None = None
        self._rate_mark = (time.perf_counter(), 0)
        self._slow_t = 0.0
        self._status_t = 0.0
        self._trigger_waiting = False
        self._outline_region: Region | None = None
        self._last_dir = str(Path.home())
        self._explore_pending = False
        self._seeking = False
        self._last_det_id = 0
        self._det_table_t = 0.0
        self._psd_key = None
        self._tol_key = None
        self._levels_key = None
        self._spectro_t = 0.0
        self._bands_t = 0.0
        self.alert_engine = AlertEngine(self._load_rules())
        self.outputs = OutputHub()
        self._alert_det_id = 0  # detections up to this id were checked / published
        self._report_seen = None
        self._levels_t: float | None = None
        self._pending_saves: list[tuple[float, float, Alert]] = []
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1440, 900)
        self.selector = RegionSelector(self)
        self.selector.finished.connect(self._region_selected)
        self.outline = RegionOutline()
        self._build()
        self._restore()
        self.acoustics.set_profile_store(self.settings)
        self.events_panel.set_store(self.store)
        self.events_panel.set_tag_suggestions(self.library.labels())
        self.outputs.configure(self._load_output_settings())
        self.toast = Toast(self)
        self.toast.clicked.connect(lambda: self.review_tabs.setCurrentWidget(self.alerts_panel))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(33)

    # ======================================================================
    # layout
    # ======================================================================
    def _build(self) -> None:
        self._build_actions()
        self._build_controls()
        self._build_plots()
        self._build_review()
        self.status_label = QLabel("")
        self.statusBar().addWidget(self.status_label, 1)
        self.cursor_label = QLabel("")
        self.cursor_label.setStyleSheet("color:#c8d0d8;")
        self.statusBar().addPermanentWidget(self.cursor_label)
        self.alarm_label = QLabel("")
        self.statusBar().addPermanentWidget(self.alarm_label)
        self._wire()

    def _action(self, text, slot, shortcut=None, checkable=False, tip=None) -> QAction:
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
            act.setShortcutContext(Qt.WindowShortcut)
        act.setCheckable(checkable)
        if tip:
            act.setToolTip(f"{tip} ({shortcut})" if shortcut else tip)
        (act.toggled if checkable else act.triggered).connect(slot)
        self.addAction(act)
        return act

    def _build_actions(self) -> None:
        a = self._action
        self.act_start = a("▶ Start", self.toggle_running, "F5", tip="Start / stop the source")
        self.act_pause = a("⏸ Pause", self._set_explore, "F6", checkable=True,
                           tip="Freeze the display to explore the history; data keeps coming")
        self.act_capture = a("⏺ Capture last 30 s", self.capture_event, "F9",
                             tip="Save the recent history as an event to tag and annotate")
        self.act_zoom_in = a("＋", lambda: self.zoom_x(0.5), "+", tip="Zoom in (time)")
        self.act_zoom_out = a("－", lambda: self.zoom_x(2.0), "-", tip="Zoom out (time)")
        QShortcut(QKeySequence("="), self, activated=lambda: self.zoom_x(0.5))
        self.act_zoom_in_y = a("Y＋", lambda: self.zoom_y(0.5), "Ctrl+Up", tip="Zoom in (Y)")
        self.act_zoom_out_y = a("Y－", lambda: self.zoom_y(2.0), "Ctrl+Down", tip="Zoom out (Y)")
        self.act_fit = a("Fit", self.fit_view, "0", tip="Auto-scale / show the whole history")
        self.act_box = a("⬚ Box zoom", self._set_box_zoom, checkable=True,
                         tip="Drag a rectangle to zoom into it")
        self.act_select = a("⇔ Select", self._set_select_mode, "S", checkable=True,
                            tip="Drag to select a sound (Shift-drag works any time)")
        self.act_zoom_sel = a("Zoom to selection", self.zoom_to_selection, "Z")
        self.act_analyse = a("📊 Analyse", self.analyse_selection, "A",
                             tip="Open a detail window for the selection: SPL, Lpk, SEL, "
                                 "PSD, third-octave levels, bandwidths")
        self.act_identify = a("🏷 Identify…", self.identify_selection, "I",
                              tip="Name the selected sound so later occurrences are flagged")
        self.act_play_sel = a("▶ Play selection", self.play_selection, "P")
        self.act_save_sel = a("💾 Save selection", self.save_selection_event,
                              tip="Save the selection as an event")
        self.act_listen = a("🎧 Listen live", self._set_listen, "L", checkable=True)
        self.act_stop_audio = a("■ Stop audio", self.stop_audio, tip="Stop playback (Esc)")
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self._escape)
        QShortcut(QKeySequence("C"), self, activated=lambda: self._review("confirm"))
        QShortcut(QKeySequence("X"), self, activated=lambda: self._review("reject"))

        tb = QToolBar("Main")
        tb.setObjectName("toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        for act in (self.act_start, self.act_pause, self.act_capture):
            tb.addAction(act)
        tb.addSeparator()
        for act in (self.act_zoom_in, self.act_zoom_out, self.act_zoom_in_y,
                    self.act_zoom_out_y, self.act_fit, self.act_box):
            tb.addAction(act)
        tb.addSeparator()
        for act in (self.act_select, self.act_analyse, self.act_identify, self.act_play_sel,
                    self.act_save_sel):
            tb.addAction(act)
        tb.addSeparator()
        tb.addAction(self.act_listen)
        tb.addAction(self.act_stop_audio)
        tb.addWidget(QLabel(" Speed "))
        self.play_speed = named(combo([f"{s:g}×" for s in SPEEDS], "1×"), "play.speed")
        self.play_speed.setToolTip("Replay speed (time expansion): <1 lowers pitch of "
                                   "ultrasonic sounds, >1 raises infrasonic ones")
        tb.addWidget(self.play_speed)
        self.denoise_on = named(QCheckBox("Denoise"), "play.denoise")
        self.denoise_on.setToolTip("Spectral noise reduction when replaying clips")
        tb.addWidget(self.denoise_on)
        tb.addWidget(QLabel(" Volume "))
        self.volume = named(spin(-40, 60, 0, suffix=" dB"), "play.volume")
        self.volume.setToolTip("Live listening gain (hydrophone signals are often quiet)")
        tb.addWidget(self.volume)

        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction(self._action("Open audio file…", self.open_audio_file, "Ctrl+O"))
        m.addAction(self._action("Open data folder", lambda: self._open_folder(self.data_dir)))
        m.addSeparator()
        m.addAction(self._action("Export detections (Raven table)…", self.export_detections_raven))
        m.addAction(self._action("Export saved events (Raven table)…", self.export_events_raven))
        m.addAction(self._action("Import Raven selections into signatures…", self.import_raven))
        m.addSeparator()
        m.addAction(self._action("Exit", self.close))
        self.view_menu = mb.addMenu("&View")
        m = mb.addMenu("&Tools")
        for act in (self.act_capture, self.act_analyse, self.act_identify, self.act_play_sel,
                    self.act_save_sel, self.act_listen):
            m.addAction(act)
        m.addAction(self._action("Signatures…", self.manage_signatures))
        m.addAction(self._action("Batch analysis of recordings…", self.batch_analysis))
        m.addSeparator()
        m.addAction(self._action("Alerts…", self.edit_alerts))
        m.addAction(self._action("Data outputs (JSON stream)…", self.edit_outputs))
        m = mb.addMenu("&Help")
        m.addAction(self._action("Keyboard shortcuts", lambda: QMessageBox.information(
            self, "Shortcuts", SHORTCUTS)))
        m.addAction(self._action("About", lambda: QMessageBox.about(
            self, APP_NAME, f"{APP_NAME} {__version__}\nReal-time monitor for audio, data "
            "streams and on-screen graphs, with hydrophone analysis tools.")))

    def _scroll(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidget(widget)
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return area

    def _build_controls(self) -> None:
        tabs = QTabWidget()
        # -- Source --------------------------------------------------------
        page = QWidget()
        pl = QVBoxLayout(page)
        self.source_combo = named(QComboBox(), "source")
        self.panels: list[SourcePanel] = [cls() for cls in PANELS]
        self.stack = QStackedWidget()
        for panel in self.panels:
            self.source_combo.addItem(panel.title, panel.key)
            self.stack.addWidget(panel)
        self.screen_panel: ScreenPanel = next(p for p in self.panels if isinstance(p, ScreenPanel))
        self.file_panel: AudioFilePanel = next(p for p in self.panels
                                               if isinstance(p, AudioFilePanel))
        pl.addWidget(self.source_combo)
        pl.addWidget(self.stack)
        row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start")
        self.stop_btn = QPushButton("■  Stop")
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        pl.addLayout(row)
        pl.addStretch(1)
        tabs.addTab(self._scroll(page), "Source")

        # -- Display -------------------------------------------------------
        page = QWidget()
        dl = QVBoxLayout(page)
        box = QGroupBox("Waveform")
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
        yrow.setContentsMargins(0, 0, 0, 0)
        self.y_row = QWidget()
        self.y_row.setLayout(yrow)
        f.addRow("Y range", self.y_row)
        self.overview_on = named(QCheckBox("History overview strip"), "display.overview")
        self.overview_on.setChecked(True)
        f.addRow(self.overview_on)
        self._display_form = f
        dl.addWidget(box)

        box = QGroupBox("Spectrogram")
        f = QFormLayout(box)
        self.spectro_on = named(QCheckBox("Show spectrogram"), "display.spectro")
        self.spectro_on.setChecked(True)
        f.addRow(self.spectro_on)
        self.cmap = named(combo({k: k.capitalize() for k in COLORMAPS}, "inferno"), "display.cmap")
        f.addRow("Colours", self.cmap)
        self.spec_snr = named(QCheckBox("Show relative to noise floor"), "display.snr")
        self.spec_snr.setToolTip("Whitens the background so faint calls stand out")
        f.addRow(self.spec_snr)
        self.db_auto = named(QCheckBox("Auto level range"), "display.dbauto")
        self.db_auto.setChecked(True)
        f.addRow(self.db_auto)
        self.db_range = named(spin(10, 200, 60, suffix=" dB"), "display.dbrange")
        f.addRow("Dynamic range", self.db_range)
        self.f_max = named(spin(0, 1e6, 0, decimals=0, suffix=" Hz"), "display.fmax")
        self.f_max.setSpecialValueText("Nyquist")
        f.addRow("Max frequency", self.f_max)
        self.auto_detail = named(QCheckBox("Open the analysis window when I select on it"),
                                 "display.autodetail")
        f.addRow(self.auto_detail)
        self.spec_log = named(QCheckBox("Logarithmic frequency axis"), "display.speclog")
        f.addRow(self.spec_log)
        self.f_min_log = named(spin(0.1, 1e5, 10, decimals=1, suffix=" Hz"), "display.fminlog")
        f.addRow("Lowest frequency (log)", self.f_min_log)
        dl.addWidget(box)

        box = QGroupBox("PSD, bands && levels")
        f = QFormLayout(box)
        self.spectrum_on = named(QCheckBox("Show power spectral density (PSD)"),
                                 "display.spectrum")
        f.addRow(self.spectrum_on)
        self.fft_log = named(QCheckBox("Log frequency axis"), "display.fftlog")
        self.fft_log.setChecked(True)
        f.addRow(self.fft_log)
        self.psd_pct = named(QCheckBox("Show 5 / 50 / 95 % percentiles"), "display.psdpct")
        self.psd_pct.setChecked(True)
        f.addRow(self.psd_pct)
        self.tol_on = named(QCheckBox("Show third-octave band levels"), "display.tol")
        f.addRow(self.tol_on)
        self.levels_on = named(QCheckBox("Show levels over time"), "display.levels")
        self.levels_on.setChecked(True)
        f.addRow(self.levels_on)
        self.peak_on = named(QCheckBox("Include peak level (Lpk)"), "display.lpk")
        f.addRow(self.peak_on)
        self.btr_on = named(QCheckBox("Show bearing-time display (array)"), "display.btr")
        f.addRow(self.btr_on)
        self._spec_form = f
        dl.addWidget(box)
        dl.addStretch(1)
        tabs.addTab(self._scroll(page), "Display")

        # -- Acoustics -----------------------------------------------------
        self.acoustics = AcousticsPanel()
        tabs.addTab(self._scroll(self.acoustics), "Acoustics")

        # -- Record --------------------------------------------------------
        page = QWidget()
        rl = QVBoxLayout(page)
        box = QGroupBox("History && event capture")
        f = QFormLayout(box)
        self.history_s = named(spin(30, 3600, 60, suffix=" s"), "record.history")
        self.history_s.setToolTip("How much recent data is kept in memory (applies at next Start)")
        f.addRow("Keep history", self.history_s)
        self.pre_roll = named(spin(1, 3600, 30, suffix=" s"), "record.preroll")
        self.post_roll = named(spin(0, 600, 0, suffix=" s"), "record.postroll")
        f.addRow("Capture before press", self.pre_roll)
        f.addRow("…and after press", self.post_roll)
        cap = QPushButton("⏺ Capture now (F9)")
        cap.clicked.connect(self.capture_event)
        f.addRow(cap)
        self.save_reviewed = named(QCheckBox("Save confirmed / corrected detections as events"),
                                   "record.savereviewed")
        self.save_reviewed.setChecked(True)
        f.addRow(self.save_reviewed)
        self.data_label = QLabel(str(self.data_dir))
        self.data_label.setWordWrap(True)
        f.addRow("Data folder", self.data_label)
        row = QHBoxLayout()
        change = QPushButton("Change…")
        change.clicked.connect(self._choose_data_dir)
        opener = QPushButton("Open")
        opener.clicked.connect(lambda: self._open_folder(self.data_dir))
        row.addWidget(change)
        row.addWidget(opener)
        f.addRow(row)
        rl.addWidget(box)
        box = QGroupBox("Continuous recording && export")
        bl = QVBoxLayout(box)
        self.record_btn = QPushButton("●  Record to file…")
        self.record_btn.setCheckable(True)
        bl.addWidget(self.record_btn)
        self.record_label = QLabel("")
        self.record_label.setWordWrap(True)
        bl.addWidget(self.record_label)
        row = QHBoxLayout()
        save_csv = QPushButton("Save view as CSV…")
        save_png = QPushButton("Save image…")
        save_csv.clicked.connect(self._save_view_csv)
        save_png.clicked.connect(self._save_image)
        row.addWidget(save_csv)
        row.addWidget(save_png)
        bl.addLayout(row)
        rl.addWidget(box)
        rl.addStretch(1)
        tabs.addTab(self._scroll(page), "Record")

        self.control_tabs = tabs
        dock = QDockWidget("Controls", self)
        dock.setObjectName("controls")
        dock.setWidget(tabs)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                         | QDockWidget.DockWidgetClosable)
        dock.setMinimumWidth(380)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.view_menu.addAction(dock.toggleViewAction())

    def _build_plots(self) -> None:
        self.preview = PreviewView()
        self.overview = OverviewPlot()
        self.time_plot = TimePlot()
        self.time_plot.setAutoVisible(y=True)
        self.spectro = SpectrogramView()
        self.spectro.setXLink(self.time_plot)
        self.spec_plot = SpectrumPlot()
        self.tol_plot = TolPlot()
        self.bearing_view = BearingView()
        self.levels_plot = LevelsPlot()

        # transport bar for replaying files
        self.transport = QWidget()
        tl = QHBoxLayout(self.transport)
        tl.setContentsMargins(4, 0, 4, 0)
        self.tr_play = QPushButton("⏸")
        self.tr_play.setFixedWidth(40)
        self.tr_play.clicked.connect(self._transport_toggle)
        self.tr_slider = QSlider(Qt.Horizontal)
        self.tr_slider.setRange(0, 10000)
        self.tr_slider.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.tr_slider.sliderReleased.connect(self._transport_seek)
        self.tr_label = QLabel("")
        self.tr_speed = combo(AudioFilePanel.SPEEDS, "1")
        self.tr_speed.activated.connect(self._transport_speed)
        tl.addWidget(self.tr_play)
        tl.addWidget(self.tr_slider, 1)
        tl.addWidget(self.tr_label)
        tl.addWidget(self.tr_speed)
        self.transport.setVisible(False)

        wave = QWidget()
        wl = QVBoxLayout(wave)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)
        wl.addWidget(self.transport)
        self.rec_view = RecordingOverviewView()
        self.rec_view.setVisible(False)
        self.rec_view.seek_requested.connect(self._seek_to)
        wl.addWidget(self.rec_view)
        wl.addWidget(self.overview)
        wl.addWidget(self.time_plot, 1)
        self.readout = QLabel("")
        self.readout.setTextFormat(Qt.RichText)
        self.readout.setWordWrap(True)
        wl.addWidget(self.readout)

        self.plot_split = QSplitter(Qt.Vertical)
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(self.spec_plot)
        bottom.addWidget(self.tol_plot)
        bottom.addWidget(self.bearing_view)
        self.bottom_split = bottom
        for w in (self.preview, wave, self.spectro, bottom, self.levels_plot):
            self.plot_split.addWidget(w)
        for i, stretch in enumerate((2, 3, 3, 2, 2)):
            self.plot_split.setStretchFactor(i, stretch)
        self.plots = self.plot_split
        self.setCentralWidget(self.plot_split)

    def _build_review(self) -> None:
        self.det_panel = DetectionsPanel()
        self.events_panel = EventsPanel()
        self.review_tabs = QTabWidget()
        self.review_tabs.addTab(self.det_panel, "Detections")
        self.review_tabs.addTab(self.events_panel, "Saved events")
        self.alerts_panel = AlertsPanel()
        self.alerts_panel.show_detection.connect(self.show_detection)
        self.alerts_panel.rules_btn.clicked.connect(self.edit_alerts)
        self.review_tabs.addTab(self.alerts_panel, "Alerts")
        dock = QDockWidget("Detections && events", self)
        dock.setObjectName("review")
        dock.setWidget(self.review_tabs)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.view_menu.addAction(dock.toggleViewAction())

    def _wire(self) -> None:
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.record_btn.toggled.connect(self._toggle_record)
        for w in (self.xaxis, self.mode, self.cmap):
            w.currentIndexChanged.connect(self._display_changed)
        for w in (self.y_auto, self.spectrum_on, self.fft_log, self.spectro_on, self.levels_on,
                  self.overview_on, self.spec_snr, self.db_auto, self.spec_log, self.tol_on,
                  self.psd_pct, self.peak_on, self.btr_on):
            w.toggled.connect(self._display_changed)
        for w in (self.y_min, self.y_max, self.window_s, self.window_n, self.db_range,
                  self.f_max, self.f_min_log):
            w.valueChanged.connect(self._display_changed)
        self.db_auto.toggled.connect(self._db_auto_toggled)
        self.spectro.levels_dragged.connect(lambda *_: self.db_auto.setChecked(False))
        for plot in (self.time_plot, self.spectro, self.spec_plot, self.tol_plot,
                     self.levels_plot, self.bearing_view):
            plot.scene().sigMouseMoved.connect(lambda pos, p=plot: self._hover(p, pos))
        self.pre_roll.valueChanged.connect(self._update_capture_text)
        self.trig_level.valueChanged.connect(self._trigger_level_edited)
        self.time_plot.trigger_line.sigPositionChanged.connect(
            lambda line: self.trig_level.setValue(line.value()))
        self.time_plot.vb.sigSelect.connect(self._wave_select)
        self.spectro.vb.sigSelect.connect(self._spectro_select)
        self.time_plot.selection.sigRegionChangeFinished.connect(self._selection_dragged)
        self.time_plot.vb.sigXRangeChanged.connect(self._view_moved)
        self.overview.region_dragged.connect(self._overview_dragged)
        self.acoustics.changed.connect(self._acoustics_changed)
        self.acoustics.band_from_selection.connect(self._band_from_selection)
        self.acoustics.manage_signatures.connect(self.manage_signatures)

        sp = self.screen_panel
        sp.select_requested.connect(self._select_region)
        sp.save_trace_requested.connect(self._save_trace)
        sp.autodetect_requested.connect(self._autodetect_traces)
        sp.region_changed.connect(lambda _r: self._update_outline())
        sp.cal_line_changed.connect(self.preview.set_line)
        sp.changed.connect(self._screen_settings_changed)
        self.preview.line_moved.connect(sp.set_cal_px)
        self.preview.pixel_clicked.connect(self._preview_clicked)
        sp.pick_bar.toggled.connect(self.preview.set_rect_pick)
        self.preview.rect_picked.connect(self._bar_picked)

        d = self.det_panel
        d.confirm.connect(self.confirm_detection)
        d.correct.connect(self.correct_detection)
        d.reject.connect(self.reject_detection)
        d.identify.connect(self.identify_detection)
        d.play.connect(self.play_detection)
        d.zoom.connect(self.show_detection)
        d.save_event.connect(self.save_detection_event)
        d.selected.connect(self._detection_selected)
        self.events_panel.play.connect(self.play_event)
        self.events_panel.open_in_replay.connect(self.replay_event)

        self._source_changed()
        self._display_changed()
        self._update_capture_text()

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
        history = float(self.history_s.value())
        if src.sample_rate:
            buffer = RingBuffer(int(src.sample_rate * history) + 1, src.sample_dtype,
                                src.sample_rate)
        else:
            buffer = RingBuffer(max(1000, src.buffer_capacity))
        try:
            src.start(buffer)
        except Exception as exc:
            log.exception("failed to start %s", panel.key)
            src.stop()
            QMessageBox.warning(self, APP_NAME, f"Could not start {panel.title}:\n\n{exc}")
            return False
        self._stop_analyzer()
        if isinstance(src, AudioFileSource):
            self._apply_file_calibration(src.path)
        self.source, self.buffer = src, buffer
        live = not isinstance(src, AudioFileSource)
        self._wall0 = time.time() if live else None
        if self.rec_overview is not None:
            self.rec_overview.stop()
            self.rec_overview = None
        if isinstance(src, AudioFileSource):
            self.rec_overview = RecordingOverview(src, int(self.acoustics.channel.currentData()))
            self.rec_overview.start()
            self._ov_final = False
        self.rec_view.setVisible(isinstance(src, AudioFileSource))
        self.analyzer = Analyzer(buffer, self.matcher, self.acoustics.settings(),
                                 src.sample_rate, history, self._wall_clock)
        src.add_frame_listener(self.analyzer.feed_frames)
        self.analyzer.start()
        if isinstance(src, AudioFileSource):
            src.throttle = lambda: self.analyzer.processed_index if self.analyzer else 0
            self.tr_speed.setCurrentIndex(max(0, self.tr_speed.findData(
                self.file_panel.speed.currentData())))
        self.det_panel.clear()
        self._last_det_id = 0
        self._alert_det_id = 0
        self._report_seen = None
        self._levels_t = None
        self._pending_saves = []
        self.alert_engine.reset()
        self._sel = None
        self.time_plot.clear_selection()
        self.spectro.clear_selection()
        self.time_plot.overlay.clear()
        self.spectro.overlay.clear()
        self._reset_view()
        self.act_pause.setChecked(False)
        self._update_buttons()
        self._display_changed()
        return True

    def stop(self) -> None:
        self._stop_recording()
        if self.act_listen.isChecked():
            self.act_listen.setChecked(False)
        if self.source is not None:
            self.source.stop()
        if self.analyzer is not None:
            self.analyzer.stop()  # keep results for review
        self._update_buttons()

    def _stop_analyzer(self) -> None:
        if self.analyzer is not None:
            self.analyzer.stop()
            self.analyzer = None

    def clear(self) -> None:
        if self.buffer is not None:
            self.buffer.clear()
        if isinstance(self.source, ScreenSource):
            self.source.stitcher.reset()
        self._reset_view()

    def _wall_clock(self, t: float) -> float | None:
        if isinstance(self.source, AudioFileSource):
            return self.source.wall_time(t)
        return self._wall0 + t if self._wall0 is not None else None

    def _source_desc(self) -> str:
        src = self.source
        if src is None:
            return ""
        if isinstance(src, AudioFileSource):
            return f"file {Path(src.path).name}"
        return f"{self.current_panel().title}: {src.status.split(' · ')[0]}"[:200]

    # ======================================================================
    # display refresh
    # ======================================================================
    def _reset_view(self) -> None:
        self.time_plot.clear_series()
        self.spec_plot.clear_series()
        self._last_version = -1
        self._last_frame = None
        self._view = None
        self.readout.setText("")

    def _display_buffer(self) -> RingBuffer | None:
        """Band-filtered stream when band isolation is on, else the raw one."""
        a = self.analyzer
        if (a is not None and a.filtered is not None and self.acoustics.band() is not None
                and self.acoustics.band_show.isChecked() and len(a.filtered)):
            return a.filtered
        return self.buffer

    def _refresh(self) -> None:
        self._update_status()
        src, buf = self.source, self.buffer
        if src is None or buf is None:
            return
        if isinstance(src, ScreenSource):
            self._refresh_screen(src)
        trace_mode = isinstance(src, ScreenSource) and src.output == "trace"
        if not self.explore and not trace_mode:
            dbuf = self._display_buffer()
            if dbuf is not None and dbuf.version != self._last_version:
                self._last_version = dbuf.version
                self._refresh_live(dbuf)
                now = time.perf_counter()
                if self.spectro.isVisible() and now - self._spectro_t > 0.08:
                    self._spectro_t = now
                    self._render_spectrogram(*self.time_plot.viewRange()[0])
        elif self._explore_pending:
            self._explore_pending = False
            self._render_explore()
        self._refresh_detections()
        if self.player.mode == "clip" and self.player.playing:
            self.time_plot.playhead.setValue(self.player.clip_start + self.player.position)
            self.time_plot.playhead.setVisible(True)
        elif self.time_plot.playhead.isVisible():
            self.time_plot.playhead.setVisible(False)
        now = time.perf_counter()
        if now - self._slow_t > 0.5:
            self._slow_t = now
            self._refresh_slow()

    def _refresh_slow(self) -> None:
        """Twice a second: overview, spectrum, levels, noise report, transport."""
        buf = self._display_buffer()
        if self.overview.isVisible() and buf is not None and len(buf):
            snap = buf.envelope(None, None, 750)
            if len(snap) and snap.y.shape[1]:
                self.overview.set_data(snap.x, snap.y[:, 0])
            self.overview.set_region(*self.time_plot.viewRange()[0])
        a = self.analyzer
        if a is not None:
            now = time.perf_counter()
            if now - self._bands_t >= 1.0:  # these change once per integration interval
                self._bands_t = now
                if self.spec_plot.isVisible():
                    self._refresh_spectrum(a)
                if self.tol_plot.isVisible():
                    self._refresh_tol(a)
            if self.levels_plot.isVisible() and a.intervals is not None and a.intervals.count:
                self._refresh_levels(a)
            if self.bearing_view.isVisible() and a.btr is not None and a.btr.count > 1:
                t, rows = a.btr.latest(a.btr.count)
                self.bearing_view.set_record(t, a.btr_bearings, rows)
                self.bearing_view.set_detections(a.all_detections()[-500:])
            self.acoustics.show_report(a.report, a.alarm)
            self._check_report(a)
            self.alarm_label.setText("⚠ NOISE FLOOR ALARM" if a.alarm else "")
            self.alarm_label.setStyleSheet("color:#ff5050; font-weight:bold;" if a.alarm else "")
        src = self.source
        ov = self.rec_overview
        if isinstance(src, AudioFileSource) and ov is not None:
            if not ov.complete or not self._ov_final:
                self._ov_final = ov.complete
                start = src.segments[0].wall_start
                label = (dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S") + " · "
                         if start else "") + f"{len(src.segments)} file(s), {src.duration / 60:.1f} min"
                self.rec_view.set_overview(ov, a.offset_db if a else 0.0, label)
            self.rec_view.set_position(src.position, self.time_plot.viewRange()[0])
        if isinstance(src, AudioFileSource):
            if not self._seeking and src.duration:
                self.tr_slider.blockSignals(True)
                self.tr_slider.setValue(int(10000 * src.position / src.duration))
                self.tr_slider.blockSignals(False)
            self.tr_label.setText(f"{src.position:8.2f} / {src.duration:.2f} s")
            self.tr_play.setText("▶" if src.paused or src.finished else "⏸")

    def _refresh_live(self, buf: RingBuffer) -> None:
        samples_axis = self.xaxis.currentData() == "samples"
        triggered = self.mode.currentData() == "trigger"
        if not samples_axis and not triggered:  # common case: draw an envelope, no big copy
            last = buf.last_x()
            if last is None:
                return
            span = self.window_s.value()
            snap = buf.envelope(last - span, last, 2 * max(200, self.time_plot.width()))
            if not len(snap):
                return
            self._update_channel_list(snap.names)
            self._show_series(snap.x, snap.y, snap.names)
            self.time_plot.setXRange(last - span, last, padding=0)
            self._apply_y_range()
            self._update_readout(snap.names, snap.y)
            self._trigger_waiting = False
            return
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
        self._show_series(x, y, names)
        self.time_plot.setXRange(*x_range, padding=0)
        self._apply_y_range()
        self._update_readout(names, y)

    def _show_series(self, x, y, names) -> None:
        self._view = (x, y, names)
        xd, yd = minmax_decimate(x, y, 4 * max(200, self.time_plot.width()))
        self.time_plot.set_series(xd, yd, names)

    def _render_explore(self) -> None:
        """Paused: draw whatever part of the history is in view."""
        buf = self._display_buffer()
        if buf is None:
            return
        x0, x1 = self.time_plot.viewRange()[0]
        if self.xaxis.currentData() == "samples":
            i0 = max(int(np.floor(x0)), buf.first_index)
            snap = buf.from_index(i0, max_n=max(0, int(np.ceil(x1)) - i0 + 1))
            x = snap.first_index + np.arange(len(snap), dtype=float)
        else:
            snap = buf.envelope(x0, x1, 2 * max(200, self.time_plot.width()))
            x = snap.x
        if len(snap):
            self._show_series(x, snap.y, snap.names)
            self._update_readout(snap.names, snap.y)
        if self.spectro.isVisible():
            self._render_spectrogram(x0, x1)

    def _render_spectrogram(self, x0: float, x1: float) -> None:
        a = self.analyzer
        if a is None or a.spectro is None or not a.rate:
            return
        t, d = a.spectro.range(x0, x1)
        if len(t) < 2:
            return
        width = max(200, self.spectro.width())
        if len(t) > 2 * width:  # keep peaks when squeezing many frames into few pixels
            step = len(t) // width
            m = (len(t) // step) * step
            d = d[len(t) - m:].reshape(-1, step, d.shape[1]).max(axis=1)
            t = t[len(t) - m:][step - 1::step]
        if self.spec_snr.isChecked():
            floor = a.floor_snapshot()
            if floor is not None:
                d = d - floor[1]
        elif not a.external:
            df = a.rate / a.nfft
            d = d + a.offset_db - 10 * np.log10(df)  # power per bin -> spectral density
        nyq = float(a.freqs[-1]) if a.external else a.rate / 2
        f_max = self.f_max.value() or nyq
        bins = max(int(np.searchsorted(a.freqs, min(f_max, nyq), side="right")), 2)
        img = d[:, :bins].T
        freqs = a.freqs[:bins]
        rows_max = 2 * max(100, self.spectro.height())
        if img.shape[0] > rows_max and not self.spec_log.isChecked():  # keep peaks
            step = int(np.ceil(img.shape[0] / rows_max))
            m = (img.shape[0] // step) * step
            img = img[:m].reshape(-1, step, img.shape[1]).max(axis=1)
            freqs = freqs[:m].reshape(-1, step).mean(axis=1)
        if self.db_auto.isChecked() or self.spec_snr.isChecked():
            hi = float(np.percentile(img, 99.7))
            lo = hi - self.db_range.value() if not self.spec_snr.isChecked() else 0.0
            if self.spec_snr.isChecked():
                hi = max(hi, 10.0)
        else:
            hi = float(np.max(img))
            lo = hi - self.db_range.value()
        half = a.hop / a.rate / 2
        self.spectro.set_log_freq(self.spec_log.isChecked(), self.f_min_log.value())
        self.spectro.set_image(t[0] - half, t[-1] + half, freqs, img, (lo, hi))
        band = self.acoustics.band()
        if band is not None:
            lo_b, hi_b = band.edges(a.rate)
            self.spectro.set_band(lo_b, hi_b)
        else:
            self.spectro.set_band(None, None)

    def _psd_units(self, a: Analyzer) -> str:
        return "dB re 1 µPa²/Hz" if a.settings.calibration.enabled else "dBFS²/Hz"

    def _refresh_spectrum(self, a: Analyzer) -> None:
        ring = a.psd_series
        key = (id(ring), ring.head if ring else None, self.psd_pct.isChecked(), a.offset_db)
        if key == self._psd_key:  # nothing new since the last interval
            return
        self._psd_key = key
        latest = a.latest_psd()
        if latest is None:
            return
        freqs, psd = latest
        off = a.offset_db
        pct = ({k: v + off for k, v in a.psd_percentiles().items()}
               if self.psd_pct.isChecked() else {})
        floor = a.floor_snapshot()
        floor_f = floor_db = None
        if floor is not None and a.rate:  # analysis floor is per bin: convert to per Hz
            floor_f, floor_db = floor[0], floor[1] + off - 10 * np.log10(a.rate / a.nfft)
        self.spec_plot.set_psd(freqs, psd + off, pct, floor_f, floor_db, self._psd_units(a))

    def _refresh_tol(self, a: Analyzer) -> None:
        ring = a.tol_series
        key = (id(ring), ring.head if ring else None, a.offset_db, self.acoustics.tol_bands())
        if key == self._tol_key:
            return
        self._tol_key = key
        t, levels = a.tol_window()
        if not len(t) or not a.tol_bands:
            return
        off = a.offset_db
        pct = {}
        if len(t) >= 3:
            pct = {k: v + off for k, v in percentiles_db(levels, axis=0).items()}
        resolved = [band_resolution_ok(a.psd_freqs, b) for b in a.tol_bands]
        self.tol_plot.set_levels(a.tol_bands, levels[-1] + off, pct,
                                 self.acoustics.tol_bands(), resolved, a.units)

    def _refresh_levels(self, a: Analyzer) -> None:
        key = (id(a.intervals), a.intervals.head, a.offset_db, self.peak_on.isChecked(),
               self.acoustics.tol_bands(), self.acoustics.band() is not None)
        if key == self._levels_key:
            return
        self._levels_key = key
        t, v = a.intervals.latest(a.intervals.count)
        chosen = self.acoustics.tol_bands()
        bands = {}
        if chosen and a.tol_series is not None and a.tol_series.count:
            tt, tol = a.tol_series.latest(a.tol_series.count)
            index = {b.label: i for i, b in enumerate(a.tol_bands)}
            if len(tt) == len(t):
                bands = {name: tol[:, index[name]] for name in chosen if name in index}
        self.levels_plot.set_data(t, v, a.offset_db, a.units, self.acoustics.band() is not None,
                                  self.peak_on.isChecked(), bands)
        st = self.acoustics.settings()
        self.levels_plot.set_alarm(st.floor_alarm_db if st.floor_alarm else None)

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
            self.preview.set_frame(frame.image, r, self.screen_panel.show_mask.isChecked(),
                                   probe_px)
        self.screen_panel.show_detected(r)
        if src.output == "trace" and not self.explore:
            y = r.y
            self._view = (r.x, y, r.names)
            self.time_plot.set_series(r.x, y, r.names)
            finite = np.isfinite(r.x)
            if finite.any():
                lo, hi = float(np.min(r.x[finite])), float(np.max(r.x[finite]))
                if hi > lo:
                    self.time_plot.setXRange(lo, hi, padding=0.01)
            self._apply_y_range()
            self._update_readout(r.names, y)

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
            if self.buffer is not None and self.buffer.sample_rate:
                parts.append(f"history {self.buffer.duration:.0f} s")
            elif self.buffer is not None:
                parts.append(f"{len(self.buffer):,} buffered")
            a = self.analyzer
            if a is not None and a.error:
                parts.append(f"analysis: {a.error}")
            if self._trigger_waiting and self.mode.currentData() == "trigger":
                parts.append("waiting for trigger")
            if self.recorder is not None:
                parts.append(f"● REC {self.recorder.rows:,} samples")
            if self.explore:
                parts.append("PAUSED — exploring history (F6 for live)")
            text = "  ·  ".join(parts)
        self._status_t = now
        self.status_label.setText(text)
        self._update_buttons()

    def _update_buttons(self) -> None:
        running = self.running
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.act_start.setText("■ Stop" if running else "▶ Start")
        if not running and self.recorder is not None:
            self._stop_recording()

    # ======================================================================
    # zoom / explore
    # ======================================================================
    def _set_explore(self, on: bool) -> None:
        self.explore = on
        self.act_pause.setText("▶ Live" if on else "⏸ Pause")
        if on:
            self._explore_pending = True
        else:
            self._last_version = -1

    def _enter_explore(self) -> None:
        if not self.explore:
            self.act_pause.setChecked(True)

    def _view_moved(self, *_args) -> None:
        if self.explore:
            self._explore_pending = True

    def zoom_x(self, factor: float) -> None:
        if not self.explore:
            if self.xaxis.currentData() == "samples":
                self.window_n.setValue(max(10, int(self.window_n.value() * factor)))
            else:
                self.window_s.setValue(min(max(self.window_s.value() * factor, 0.001),
                                           self.window_s.maximum()))
            return
        (x0, x1), _ = self.time_plot.viewRange()
        c, half = (x0 + x1) / 2, (x1 - x0) / 2 * factor
        self.time_plot.setXRange(c - half, c + half, padding=0)

    def zoom_y(self, factor: float) -> None:
        _, (y0, y1) = self.time_plot.viewRange()
        c, half = (y0 + y1) / 2, (y1 - y0) / 2 * factor
        self.y_auto.setChecked(False)
        self.y_min.setValue(c - half)
        self.y_max.setValue(c + half)

    def fit_view(self) -> None:
        self.y_auto.setChecked(True)
        if self.explore and self.buffer is not None and len(self.buffer):
            if self.xaxis.currentData() == "samples":
                self.time_plot.setXRange(self.buffer.first_index, self.buffer.total, padding=0)
            else:
                self.time_plot.setXRange(self.buffer.first_x(), self.buffer.last_x(), padding=0)
        self._apply_y_range()

    def zoom_to_selection(self) -> None:
        if self._sel is None:
            return
        self._enter_explore()
        t0, t1 = self._sel[:2]
        pad = 0.1 * (t1 - t0)
        self.time_plot.setXRange(t0 - pad, t1 + pad, padding=0)

    def _overview_dragged(self, x0: float, x1: float) -> None:
        self._enter_explore()
        self.time_plot.setXRange(x0, x1, padding=0)

    def _set_box_zoom(self, on: bool) -> None:
        mode = self.time_plot.vb.RectMode if on else self.time_plot.vb.PanMode
        self.time_plot.vb.setMouseMode(mode)
        self.spectro.vb.setMouseMode(mode)
        if on:
            self.act_select.setChecked(False)
            self._enter_explore()

    def _set_select_mode(self, on: bool) -> None:
        self.time_plot.vb.select_mode = on
        self.spectro.vb.select_mode = on
        if on:
            self.act_box.setChecked(False)

    # ======================================================================
    # selection, identification and replay of selections
    # ======================================================================
    def _wave_select(self, x0, x1, _y0, _y1, finished) -> None:
        self._enter_explore()
        t0, t1 = sorted((x0, x1))
        self.time_plot.set_selection(t0, t1)
        self.spectro.clear_selection()
        self._sel = (t0, t1, None, None)

    def _spectro_select(self, x0, x1, y0, y1, finished) -> None:
        self._enter_explore()
        t0, t1 = sorted((x0, x1))
        f0, f1 = self.spectro.inv_fy(y0), self.spectro.inv_fy(y1)
        lo, hi = sorted((max(0.0, f0), max(0.0, f1)))
        self.spectro.set_selection(t0, t1, lo, hi)
        self.time_plot.set_selection(t0, t1)
        self._sel = (t0, t1, lo, hi)
        if finished and self.auto_detail.isChecked() and t1 > t0:
            self.analyse_selection()

    def _selection_dragged(self) -> None:
        span = self.time_plot.selected_span()
        if span is not None:
            lo, hi = (self._sel[2], self._sel[3]) if self._sel else (None, None)
            self._sel = (span[0], span[1], lo, hi)
            if lo is not None:
                self.spectro.set_selection(span[0], span[1], lo, hi)

    def _escape(self) -> None:
        if self.player.playing:
            self.stop_audio()
        else:
            self._sel = None
            self.time_plot.clear_selection()
            self.spectro.clear_selection()

    def _need_selection(self) -> tuple | None:
        if self._sel is None or self._sel[1] - self._sel[0] <= 0:
            QMessageBox.information(self, APP_NAME, "Select a sound first: Shift-drag on the "
                                    "waveform (time) or the spectrogram (time and frequency), "
                                    "or turn on the Select tool (S).")
            return None
        return self._sel

    def _clip(self, t0: float, t1: float, all_channels: bool = False):
        """``(clip, rate, t_start)`` from the raw history, a spectrogram patch
        dict for screen spectrograms, or None."""
        if self.analyzer is not None and self.analyzer.external and not all_channels:
            return self.analyzer.patch(t0, t1)
        if self.buffer is None:
            return None
        snap = self.buffer.range(t0, t1)
        if not len(snap):
            return None
        rate = (self.analyzer.rate if self.analyzer and self.analyzer.rate
                else self.buffer.sample_rate)
        if not rate:
            return None
        if all_channels:
            return snap.y, float(rate), float(snap.x[0])
        mono = self.analyzer.channel_signal(snap.y) if self.analyzer else snap.y[:, 0]
        return mono, float(rate), float(snap.x[0])

    def analyse_selection(self) -> None:
        sel = self._need_selection()
        if sel is None or self.buffer is None:
            return
        t0, t1, lo, hi = sel
        snap = self.buffer.range(t0, t1)
        rate = (self.analyzer.rate if self.analyzer and self.analyzer.rate
                else self.buffer.sample_rate)
        if len(snap) < 16 or not rate:
            QMessageBox.information(self, APP_NAME, "Not enough data in that selection.")
            return
        band = (lo, hi) if lo is not None else None
        if band is None and self.acoustics.band() is not None:
            band = self.acoustics.band().edges(rate)
        cal = self.acoustics.calibration()
        floor = self.analyzer.floor_snapshot() if self.analyzer else None

        def identify(clip, r, a, b, bnd):
            lo_, hi_ = bnd if bnd else estimate_band(clip, r, *(floor or (None, None)))
            self._identify(clip, r, lo_, hi_, "", a, b, floor)

        win = SelectionDetail(self, snap.x, snap.y, snap.names, rate, cal.offset_db,
                              cal.enabled, band, self._wall_clock(float(snap.x[0])),
                              play=lambda y, r, t: self._play_clip(y, r, t),
                              identify=identify,
                              save_event=lambda a, b: self._save_event(a, b, "selection"),
                              array=self.acoustics.array())
        win.show()

    def identify_selection(self) -> None:
        sel = self._need_selection()
        if sel is None:
            return
        t0, t1, lo, hi = sel
        got = self._clip(t0, t1)
        if got is None:
            QMessageBox.information(self, APP_NAME, "That part of the history is no longer "
                                    "in memory.")
            return
        clip, rate, _ = self._unpack(got)
        floor = self.analyzer.floor_snapshot() if self.analyzer else None
        if lo is None:
            lo, hi = self._band_of(clip, rate, floor)
        self._identify(clip, rate, lo, hi, "", t0, t1, floor)

    # -- audio clips vs spectrogram patches (screen) ------------------------------
    @staticmethod
    def _unpack(got):
        """Detection/selection data -> (data, rate, t_start); data is a mono
        clip, or a spectrogram patch dict (rate None) for screen spectrograms."""
        if isinstance(got, dict):
            return got, None, float(got["times"][0]) if len(got["times"]) else 0.0
        return got

    @staticmethod
    def _band_of(data, rate, floor) -> tuple[float, float]:
        if isinstance(data, dict):
            snr = np.nan_to_num(data["db"] - data["floor"][None, :]).max(axis=0)
            rows = np.flatnonzero(snr > 6)
            f = data["freqs"]
            return (float(f[rows[0]]), float(f[rows[-1]])) if len(rows) else (float(f[0]),
                                                                              float(f[-1]))
        return estimate_band(data, rate, *(floor or (None, None)))

    def _add_example(self, label, data, rate, f_lo, f_hi, kind="positive", method="auto",
                     category="", notes="", source=None, floor=None):
        source = source or self._source_desc()
        if isinstance(data, dict):
            return self.library.add_patch_example(label, data["freqs"], data["frame_period"],
                                                  data["db"], data["floor"], f_lo, f_hi, kind,
                                                  category, notes, source)
        return self.library.add_example(label, data, rate, f_lo, f_hi, kind, method, category,
                                        notes, source, floor)

    def _identify(self, clip, rate, lo, hi, suggested, t0, t1, floor,
                  det: Detection | None = None) -> str | None:
        dlg = IdentifyDialog(self, self.library.labels(), clip, rate, lo, hi, suggested,
                             play=lambda: self._play_clip(clip, rate, t0))
        if not dlg.exec():
            return None
        v = dlg.values()
        try:
            sig, _ex = self._add_example(
                v["label"], clip, rate, v["f_lo"], v["f_hi"], "positive", v["method"],
                v["category"], v["notes"], floor=floor)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not save the signature:\n{exc}")
            return None
        if v["save_event"]:
            self._save_event(t0, t1, kind="selection" if det is None else "detection",
                             title=v["label"], tags=[v["label"]], notes=v["notes"],
                             dets=[det] if det else None, open_editor=False)
        self.events_panel.set_tag_suggestions(self.library.labels())
        self.statusBar().showMessage(
            f"Saved '{sig.label}' ({len(sig.positives)} example(s)). Later occurrences will be "
            "flagged as suspected.", 8000)
        return sig.label

    def _speed(self) -> float:
        return float(self.play_speed.currentText().rstrip("×"))

    def _play_clip(self, clip, rate, t_start: float, filtered: bool | None = None) -> None:
        clip = np.asarray(clip, dtype=float)
        band = self.acoustics.band()
        use_band = self.acoustics.band_listen.isChecked() if filtered is None else filtered
        try:
            if band is not None and use_band:
                clip = filter_clip(clip, band, rate)
            if self.denoise_on.isChecked():
                clip = denoise(clip, rate)
            self.player.gain = 10 ** (self.volume.value() / 20)
            self.player.play(clip, rate, self._speed(), start_time=t_start)
        except Exception as exc:
            log.exception("playback failed")
            QMessageBox.warning(self, APP_NAME, f"Playback failed:\n{exc}")

    def play_selection(self) -> None:
        sel = self._need_selection()
        if sel is None:
            return
        got = self._clip(sel[0], sel[1], all_channels=True)
        if got is not None:
            self._play_clip(got[0][:, :2], got[1], got[2])

    def stop_audio(self) -> None:
        self.player.stop()
        if self.act_listen.isChecked():
            self.act_listen.blockSignals(True)
            self.act_listen.setChecked(False)
            self.act_listen.blockSignals(False)

    def _set_listen(self, on: bool) -> None:
        if not on:
            self.player.stop()
            return
        if self.buffer is None or not self.buffer.sample_rate or not self.running:
            QMessageBox.information(self, APP_NAME, "Live listening needs a running audio "
                                    "(fixed sample rate) source.")
            self.stop_audio()
            return
        a = self.analyzer
        buf = self.buffer
        if (a is not None and a.filtered is not None and self.acoustics.band() is not None
                and self.acoustics.band_listen.isChecked()):
            buf = a.filtered
        try:
            self.player.gain = 10 ** (self.volume.value() / 20)
            self.player.follow(buf, self.buffer.sample_rate)
        except Exception as exc:
            log.exception("live listening failed")
            QMessageBox.warning(self, APP_NAME, f"Can't open the audio output:\n{exc}")
            self.stop_audio()

    def save_selection_event(self) -> None:
        sel = self._need_selection()
        if sel is not None:
            self._save_event(sel[0], sel[1], kind="selection")

    # ======================================================================
    # event capture
    # ======================================================================
    def _update_capture_text(self) -> None:
        self.act_capture.setText(f"⏺ Capture last {self.pre_roll.value()} s")

    def capture_event(self) -> None:
        buf = self.buffer
        if buf is None or not len(buf):
            QMessageBox.information(self, APP_NAME, "Start a source first; the history buffer "
                                    "is empty.")
            return
        t_press = buf.last_x()
        pre, post = float(self.pre_roll.value()), float(self.post_roll.value())
        if pre > buf.duration + 1 and buf.sample_rate:
            self.statusBar().showMessage(f"Only {buf.duration:.0f} s of history available; "
                                         "saving what there is.", 5000)
        if post > 0:
            self.statusBar().showMessage(f"Capturing… {post:g} s more", int(post * 1000))
            QTimer.singleShot(int(post * 1000),
                              lambda: self._save_event(t_press - pre, t_press + post, "capture"))
        else:
            self._save_event(t_press - pre, t_press, "capture")

    def _save_event(self, t0: float, t1: float, kind: str, title: str = "",
                    tags: list[str] | None = None, notes: str = "",
                    dets: list[Detection] | None = None, open_editor: bool = True) -> None:
        buf = self.buffer
        if buf is None:
            return
        snap = buf.range(t0, t1)
        if not len(snap):
            QMessageBox.information(self, APP_NAME, "That time span is no longer in memory.")
            return
        if dets is None and self.analyzer is not None:
            dets = [d for d in self.analyzer.all_detections() if d.t1 >= t0 and d.t0 <= t1]
        det_info = [{"id": d.id, "t0": d.t0, "t1": d.t1, "text": d.text, "status": d.status,
                     "label": d.label, "score": round(d.score, 3), "f_lo": d.f_lo,
                     "f_hi": d.f_hi} for d in (dets or [])]
        extra = None
        if self.analyzer is not None and self.analyzer.external:
            patch = self.analyzer.patch(t0, t1)
            if patch is not None:
                extra = {"spectrogram.npz": patch}
        try:
            rec = self.store.add(snap.x, snap.y, snap.names, buf.sample_rate, kind, title,
                                 tags, notes, self._source_desc(),
                                 self._wall_clock(float(snap.x[0])), det_info, extra)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not save the event:\n{exc}")
            return
        self.events_panel.refresh(rec.id)
        for d in dets or []:
            d.saved_event = rec.id
        self.statusBar().showMessage(f"Saved {rec.duration:.1f} s as event {rec.id}", 8000)
        if open_editor:
            self.review_tabs.setCurrentWidget(self.events_panel)
            self.events_panel.focus_tags()

    def play_event(self, rec) -> None:
        try:
            x, y, rate = self.store.load_clip(rec)
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Can't read the clip:\n{exc}")
            return
        if not rate:
            QMessageBox.information(self, APP_NAME, "This event is a data-stream clip (CSV), "
                                    "not audio.")
            return
        self._play_clip(y[:, :2], rate, rec.t0, filtered=False)

    def replay_event(self, rec) -> None:
        if not rec.sample_rate:
            QMessageBox.information(self, APP_NAME, "Only audio events can be replayed as a "
                                    "source.")
            return
        self.stop()
        self.select_source(self.file_panel.key)
        self.file_panel.set_path(str(self.store.clip_path(rec)))
        self.start()

    # ======================================================================
    # detections: review loop
    # ======================================================================
    def _refresh_detections(self) -> None:
        a = self.analyzer
        if a is None:
            return
        dets = a.all_detections()
        newest = dets[-1].id if dets else 0
        if newest > self._alert_det_id:
            for det in dets:
                if det.id > self._alert_det_id:
                    self._new_detection(det)
            self._alert_det_id = newest
        now = time.perf_counter()
        if newest != self._last_det_id or now - self._det_table_t > 0.25:
            self._last_det_id = newest
            self._det_table_t = now
            self.det_panel.update_detections(dets)
        self.time_plot.update_detections(dets, self._selected_det)
        if self.spectro.isVisible():
            self.spectro.update_detections(dets, self._selected_det)

    def _detection_selected(self, det_id: int) -> None:
        self._selected_det = det_id

    def _det(self, det_id: int) -> Detection | None:
        return self.analyzer.detection(det_id) if self.analyzer else None

    def _det_clip(self, det: Detection):
        got = self.analyzer.detection_clip(det) if self.analyzer else None
        if got is None:
            QMessageBox.information(self, APP_NAME, "The audio for this detection is no longer "
                                    "in memory.")
        return got

    def _review(self, action: str) -> None:
        det_id = self.det_panel.current_id()
        if det_id is not None:
            {"confirm": self.confirm_detection, "reject": self.reject_detection}[action](det_id)

    def confirm_detection(self, det_id: int) -> None:
        det = self._det(det_id)
        if det is None or det.status != "suspected":
            return
        got = self._det_clip(det)
        sig = self.library.get(det.sig_id) if det.sig_id else None
        if got is None or sig is None:
            return
        clip, rate, t_start = self._unpack(got)
        self._add_example(sig.label, clip, rate, det.f_lo, det.f_hi, "positive",
                                 source=self._source_desc(),
                                 floor=self.analyzer.floor_snapshot())
        sig.confirmed += 1
        self.library.update(sig)
        self.analyzer.update_detection(det, status="confirmed")
        self._publish_update(det)
        if self.save_reviewed.isChecked():
            self._save_event(det.t0 - 0.5, det.t1 + 0.5, "detection", sig.label, [sig.label],
                             dets=[det], open_editor=False)

    def correct_detection(self, det_id: int) -> None:
        det = self._det(det_id)
        if det is None:
            return
        got = self._det_clip(det)
        if got is None:
            return
        clip, rate, t_start = self._unpack(got)
        new_label = ask_label(self, self.library.labels(), "Correct detection",
                              "What is it really? (pick or type a new name)", det.label or "")
        if not new_label:
            return
        floor = self.analyzer.floor_snapshot()
        self._add_example(new_label, clip, rate, det.f_lo, det.f_hi, "positive",
                                 source=self._source_desc(), floor=floor)
        old = self.library.get(det.sig_id) if det.sig_id else None
        if old is not None and old.label.lower() != new_label.lower():
            self._add_example(old.label, clip, rate, det.f_lo, det.f_hi, "negative",
                                     source=self._source_desc(), floor=floor)
            old.rejected += 1
            self.library.update(old)
        new_sig = self.library.find(new_label)
        self.analyzer.update_detection(det, status="corrected", label=new_label,
                                       sig_id=new_sig.id if new_sig else None)
        self._publish_update(det)
        self.events_panel.set_tag_suggestions(self.library.labels())
        if self.save_reviewed.isChecked():
            self._save_event(det.t0 - 0.5, det.t1 + 0.5, "detection", new_label, [new_label],
                             dets=[det], open_editor=False)

    def reject_detection(self, det_id: int) -> None:
        det = self._det(det_id)
        if det is None or det.status not in ("suspected", "unidentified"):
            return
        sig = self.library.get(det.sig_id) if det.sig_id else None
        if sig is not None:
            got = self._det_clip(det)
            if got is not None:
                clip, rate, _ = self._unpack(got)
                self._add_example(sig.label, clip, rate, det.f_lo, det.f_hi, "negative",
                                         source=self._source_desc(),
                                         floor=self.analyzer.floor_snapshot())
                sig.rejected += 1
                self.library.update(sig)
        self.analyzer.update_detection(det, status="rejected")
        self._publish_update(det)

    def identify_detection(self, det_id: int) -> None:
        det = self._det(det_id)
        if det is None:
            return
        got = self._det_clip(det)
        if got is None:
            return
        clip, rate, t_start = self._unpack(got)
        suggested = det.label or (det.alternatives[0][1] if det.alternatives else "")
        floor = self.analyzer.floor_snapshot()
        lo, hi = (det.f_lo, det.f_hi) if det.f_hi > det.f_lo else self._band_of(clip, rate,
                                                                                floor)
        old_label = det.label
        label = self._identify(clip, rate, lo, hi, suggested, det.t0, det.t1, floor, det)
        if label is None:
            return
        old = self.library.get(det.sig_id) if det.sig_id else None
        if old is not None and old_label and old_label.lower() != label.lower():
            self._add_example(old.label, clip, rate, det.f_lo, det.f_hi, "negative",
                                     source=self._source_desc(), floor=floor)
            old.rejected += 1
            self.library.update(old)
            status = "corrected"
        else:
            status = "confirmed"
        sig = self.library.find(label)
        self.analyzer.update_detection(det, status=status, label=label,
                                       sig_id=sig.id if sig else None)

    def play_detection(self, det_id: int) -> None:
        det = self._det(det_id)
        if det is None:
            return
        got = self._det_clip(det)
        if isinstance(got, dict):
            QMessageBox.information(self, APP_NAME, "There is no audio behind a spectrogram "
                                    "read from the screen.")
        elif got is not None:
            self._play_clip(*got)

    def show_detection(self, det_id: int) -> None:
        det = self._det(det_id)
        if det is None:
            return
        self._selected_det = det_id
        self._enter_explore()
        pad = max(0.5, det.t1 - det.t0)
        self.time_plot.setXRange(det.t0 - pad, det.t1 + pad, padding=0)
        self._explore_pending = True

    def save_detection_event(self, det_id: int) -> None:
        det = self._det(det_id)
        if det is not None:
            tags = [det.label] if det.label else []
            self._save_event(det.t0 - 0.5, det.t1 + 0.5, "detection", det.label or "", tags,
                             dets=[det])

    def batch_analysis(self) -> None:
        from aquafonix.ui.batch_dialog import BatchDialog

        BatchDialog(self, self.acoustics).exec()

    def manage_signatures(self) -> None:
        dlg = SignatureManager(self, self.library,
                               play_example=lambda clip, rate: self._play_clip(clip, rate, 0.0,
                                                                               filtered=False),
                               default_threshold=self.acoustics.match_thr.value())
        dlg.exec()
        self.events_panel.set_tag_suggestions(self.library.labels())

    # ======================================================================
    # alerts and data outputs
    # ======================================================================
    def _load_rules(self) -> list[AlertRule]:
        try:
            raw = json.loads(str(self.settings.value("alerts/rules", "[]")))
            return [AlertRule.from_dict(d) for d in raw]
        except (ValueError, TypeError):
            log.warning("ignoring unreadable alert rules")
            return []

    def _load_output_settings(self) -> OutputSettings:
        try:
            return OutputSettings.from_dict(json.loads(str(self.settings.value("outputs/settings",
                                                                               "{}"))))
        except (ValueError, TypeError):
            return OutputSettings()

    def edit_alerts(self) -> None:
        dlg = AlertsDialog(self, self.alert_engine.rules, self.library.labels())
        if dlg.exec():
            self.alert_engine.set_rules(dlg.rules)
            self.settings.setValue("alerts/rules",
                                   json.dumps([r.to_dict() for r in dlg.rules]))

    def edit_outputs(self) -> None:
        def test(settings: OutputSettings) -> None:
            self.outputs.configure(settings)
            self.outputs.publish({"type": "test", "message": f"{APP_NAME} output test"})

        before = self.outputs.settings
        dlg = OutputsDialog(self, before, self.outputs.status, test)
        if dlg.exec():
            settings = dlg.settings()
            self.settings.setValue("outputs/settings", json.dumps(settings.to_dict()))
        else:
            settings = before
        self.outputs.configure(settings)

    def _new_detection(self, det: Detection) -> None:
        """First sight of a detection: publish it and check the alert rules."""
        wall = self._wall_clock(det.t0)
        if self.outputs.wants("detection"):
            self.outputs.publish(detection_message(det, wall, "new", self._source_desc()))
        for alert in self.alert_engine.check_detection(det):
            self._fire_alert(alert, det)

    def _publish_update(self, det: Detection | None) -> None:
        if det is not None and self.outputs.wants("detection"):
            self.outputs.publish(detection_message(det, self._wall_clock(det.t0), "update",
                                                   self._source_desc()))

    def _check_report(self, a: Analyzer) -> None:
        if a.report is not self._report_seen:
            self._report_seen = a.report
            last = self.buffer.last_x() if self.buffer is not None else None
            if last is not None:
                for alert in self.alert_engine.check_report(a.report, float(last)):
                    self._fire_alert(alert)
        if self.outputs.wants("levels") and a.intervals is not None and a.intervals.count:
            if self._levels_t is None:
                t, v = a.intervals.latest(1)
            else:
                t, v = a.intervals.range(self._levels_t + 1e-9, np.inf)
            if len(t):
                self._levels_t = float(t[-1])
                tt, tol = a.tol_series.range(float(t[0]) - 1e-9, float(t[-1]) + 1e-9)
                off = a.offset_db
                for i, ti in enumerate(t):
                    j = np.flatnonzero(np.isclose(tt, ti))
                    bands = ({b.label: float(tol[j[0], k] + off)
                              for k, b in enumerate(a.tol_bands)} if len(j) else {})
                    self.outputs.publish(levels_message(
                        float(ti), self._wall_clock(float(ti)), v[i] + off, a.units,
                        a.settings.band and (a.settings.band.low, a.settings.band.high), bands,
                        a.report.demon, self._source_desc()))
        self._do_pending_saves()

    def _fire_alert(self, alert: Alert, det: Detection | None = None) -> None:
        rule = alert.rule
        wall = self._wall_clock(alert.stream_t)
        self.alerts_panel.add(alert, alert_time_text(alert, wall))
        n = len(self.alerts_panel.alerts)
        self.review_tabs.setTabText(self.review_tabs.indexOf(self.alerts_panel), f"Alerts ({n})")
        self.statusBar().showMessage(f"🔔 {rule.name}: {alert.message}", 10_000)
        if rule.sound and not self.alerts_panel.mute.isChecked():
            beep()
        if rule.popup:
            self.toast.notify(rule.name, alert.message)
        if self.outputs.wants("alert"):
            self.outputs.publish(alert_message(alert, wall, self._source_desc()))
        if rule.save_event:
            if det is not None:
                span = (det.t0 - 5.0, det.t1 + 5.0)
            else:
                span = (alert.stream_t - 15.0, alert.stream_t + 5.0)
            self._pending_saves.append((*span, alert))

    def _do_pending_saves(self) -> None:
        """Alert clips are saved once their post-roll has been recorded."""
        if not self._pending_saves or self.buffer is None:
            return
        last = self.buffer.last_x()
        if last is None:
            return
        keep = []
        for t0, t1, alert in self._pending_saves:
            if last >= t1 or not self.running:
                det = self._det(alert.detection_id) if alert.detection_id is not None else None
                tags = [alert.rule.name] + ([det.label] if det is not None and det.label else [])
                self._save_event(t0, min(t1, last), "alert", alert.message[:120], tags,
                                 f"Alert '{alert.rule.name}': {alert.message}",
                                 open_editor=False)
            else:
                keep.append((t0, t1, alert))
        self._pending_saves = keep

    # -- Raven selection tables ---------------------------------------------------------
    def _raven_path(self, title: str, name: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(self, title, str(Path(self._last_dir) / name),
                                              "Raven selection table (*.txt);;All files (*)")
        if path:
            self._last_dir = str(Path(path).parent)
        return path or None

    def export_detections_raven(self) -> None:
        dets = self.analyzer.all_detections() if self.analyzer is not None else []
        if not dets:
            QMessageBox.information(self, APP_NAME, "There are no detections to export yet.")
            return
        path = self._raven_path("Export detections", "detections.selections.txt")
        if not path:
            return
        src = self.source
        locate = None
        if isinstance(src, AudioFileSource):
            def locate(t: float):
                seg = src.segment_at(t)
                return Path(seg.path).name, t - seg.start / src.sample_rate
        channel = max(0, int(self.acoustics.channel.currentData())) + 1
        rows = raven.detection_rows(dets, channel, self._wall_clock, locate)
        try:
            raven.write_table(path, rows, raven.DETECTION_EXTRA)
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not write the table:\n{exc}")
            return
        self.statusBar().showMessage(f"Exported {len(rows)} detections to {path}", 8000)

    def export_events_raven(self) -> None:
        records = self.store.list()
        if not records:
            QMessageBox.information(self, APP_NAME, "There are no saved events yet.")
            return
        path = self._raven_path("Export saved events", "events.selections.txt")
        if not path:
            return
        try:
            raven.write_table(path, raven.event_rows(records, self.store), raven.EVENT_EXTRA)
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, f"Could not write the table:\n{exc}")
            return
        self.statusBar().showMessage(f"Exported {len(records)} events to {path}", 8000)

    def import_raven(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Raven selection table", self._last_dir,
                                              "Raven selection table (*.txt);;All files (*)")
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        try:
            table = raven.read_table(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, APP_NAME, f"Can't read the table:\n{exc}")
            return
        columns = table.label_columns()
        if not columns:
            QMessageBox.information(self, APP_NAME, "The table has no column with names "
                                    "(such as Annotation or Species) to learn from.")
            return
        if len(columns) > 1:
            col, ok = QInputDialog.getItem(self, "Raven import", "Which column names the "
                                           "sounds?", columns, 0, False)
            if not ok:
                return
            table.set_label_column(col)
        audio = None
        if any(raven.resolve_audio(s, table.path.parent, None)[0] is None
               for s in table.selections if s.label):
            audio, _ = QFileDialog.getOpenFileName(
                self, "Sound file the selections refer to", str(table.path.parent),
                OPEN_FILTER)
            if not audio:
                return
        n = sum(1 for s in table.selections if s.label)
        progress = QProgressDialog("Adding examples…", "Stop", 0, max(n, 1), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(400)

        def step(i: int, total: int) -> bool:
            progress.setValue(i)
            QApplication.processEvents()
            return not progress.wasCanceled()

        res = raven.import_selections(table, self.library, Path(audio) if audio else None,
                                      progress=step)
        progress.setValue(max(n, 1))
        self.events_panel.set_tag_suggestions(self.library.labels())
        lines = [f"Added {res.added} example(s) to {len(res.labels)} signature(s)."]
        lines += [f"  • {k}: {v}" for k, v in sorted(res.labels.items())[:20]]
        if res.skipped:
            lines.append(f"Skipped {res.skipped} selection(s) (no name or no sound).")
        lines += res.problems[:8]
        QMessageBox.information(self, APP_NAME, "\n".join(lines))

    # ======================================================================
    # reactions to controls
    # ======================================================================
    def _source_changed(self, *_args) -> None:
        idx = max(0, self.source_combo.currentIndex())
        if self.running and idx != self.stack.currentIndex():
            self.stop()
        self.stack.setCurrentIndex(idx)
        self.preview.setVisible(self.current_panel() is self.screen_panel)
        self.transport.setVisible(self.current_panel() is self.file_panel)
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
        self.spectro.setVisible(self.spectro_on.isChecked())
        self.spectro.set_colormap(self.cmap.currentData())
        self.db_range.setEnabled(not self.spec_snr.isChecked())
        self.spec_plot.setVisible(self.spectrum_on.isChecked())
        self.spec_plot.setLogMode(x=self.fft_log.isChecked(), y=False)
        self.tol_plot.setVisible(self.tol_on.isChecked())
        self.bearing_view.setVisible(self.btr_on.isChecked())
        self.bottom_split.setVisible(self.spectrum_on.isChecked() or self.tol_on.isChecked()
                                     or self.btr_on.isChecked())
        self.f_min_log.setEnabled(self.spec_log.isChecked())
        self.levels_plot.setVisible(self.levels_on.isChecked())
        self.overview.setVisible(self.overview_on.isChecked())
        self._update_axis_label()
        self._last_version = -1
        self._explore_pending = True
        self._apply_y_range()

    def _db_auto_toggled(self, on: bool) -> None:
        if on:
            self.spectro.manual_levels = None
        self._explore_pending = True

    def _hover(self, plot, scene_pos) -> None:
        vb = plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(scene_pos):
            return
        p = vb.mapSceneToView(scene_pos)
        x, y = p.x(), p.y()
        text = ""
        a = self.analyzer
        if plot is self.spectro:
            f = self.spectro.inv_fy(y)
            v = self.spectro.value_at(x, y)
            unit = "dB re floor" if self.spec_snr.isChecked() else (
                self._psd_units(a) if a else "dB")
            text = f"t {x:.3f} s · f {f:,.1f} Hz" + (f" · {v:.1f} {unit}" if v is not None else "")
            self.spectro.cursor.setValue(x)
        elif plot is self.time_plot:
            text = f"t {x:.4f} s · y {y:.6g}"
            self.spectro.cursor.setValue(x)
        elif plot is self.levels_plot:
            text = f"t {x:.1f} s · {y:.1f} {a.units if a else 'dB'}"
        else:
            got = plot.value_at(x, y)
            text = got or ""
        self.cursor_label.setText(text)

    def _apply_file_calibration(self, path: str) -> None:
        """Calibration for a recording: its metadata, else what was last used
        for files in the same folder."""
        from aquafonix.acoustics.calibration_io import find_calibration

        if self.acoustics.cal_auto.isChecked():
            hint = find_calibration(path)
            if hint is not None:
                self.acoustics.apply_calibration(*hint.values(), hint.describe())
                self.statusBar().showMessage(f"Calibration: {hint.describe()}", 8000)
                return
        import json

        folder = str(Path(path).resolve().parent)
        try:
            remembered = json.loads(self.settings.value("calibration/folders", "{}") or "{}")
        except ValueError:
            remembered = {}
        cal = remembered.get(folder)
        if cal and cal.get("enabled"):
            self.acoustics.apply_calibration(cal["sensitivity_db"], cal["gain_db"],
                                             cal["full_scale_v"], "the last files in this folder")

    def _remember_folder_calibration(self) -> None:
        src = self.source
        if not isinstance(src, AudioFileSource):
            return
        import json

        folder = str(Path(src.path).resolve().parent)
        try:
            remembered = json.loads(self.settings.value("calibration/folders", "{}") or "{}")
        except ValueError:
            remembered = {}
        remembered[folder] = self.acoustics.calibration_values()
        self.settings.setValue("calibration/folders", json.dumps(remembered))

    def _acoustics_changed(self) -> None:
        self._remember_folder_calibration()
        if self.analyzer is not None:
            self.analyzer.apply(self.acoustics.settings())
        self._last_version = -1
        self._explore_pending = True

    def _band_from_selection(self) -> None:
        band = self.spectro.selected_band()
        if band is None:
            QMessageBox.information(self, APP_NAME, "Shift-drag a box on the spectrogram "
                                    "first, then click this button.")
            return
        self.acoustics.set_band(*band, "bandpass")

    def _update_axis_label(self) -> None:
        src = self.source
        if isinstance(src, ScreenSource) and src.output == "trace":
            self.time_plot.setLabel("bottom", "X (calibrated)", units="")
        elif self.xaxis.currentData() == "samples":
            self.time_plot.setLabel("bottom", "Sample", units="")
        elif self.mode.currentData() == "trigger":
            self.time_plot.setLabel("bottom", "Time from trigger", units="s")
        elif isinstance(src, AudioFileSource):
            self.time_plot.setLabel("bottom", "File time", units="s")
        else:
            self.time_plot.setLabel("bottom", "Time", units="s")

    def _trigger_level_edited(self, value: float) -> None:
        line = self.time_plot.trigger_line
        line.blockSignals(True)
        line.setValue(value)
        line.blockSignals(False)
        self._last_version = -1

    # -- file transport ------------------------------------------------------------
    def open_audio_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open recording", self._last_dir, OPEN_FILTER)
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        self.stop()
        self.select_source(self.file_panel.key)
        self.file_panel.set_path(path)
        self.start()

    def _transport_toggle(self) -> None:
        src = self.source
        if isinstance(src, AudioFileSource) and src.running:
            src.set_paused(not (src.paused or src.finished))
        elif self.current_panel() is self.file_panel:
            self.start()

    def _transport_seek(self) -> None:
        self._seeking = False
        src = self.source
        if isinstance(src, AudioFileSource):
            self._seek_to(self.tr_slider.value() / 10000 * src.duration)

    def _seek_to(self, t: float) -> None:
        src = self.source
        if isinstance(src, AudioFileSource):
            src.seek(t)
            self._reset_view()
            if self.explore:
                self.act_pause.setChecked(False)

    def _transport_speed(self) -> None:
        src = self.source
        if isinstance(src, AudioFileSource):
            src.set_speed(float(self.tr_speed.currentData()))

    # -- screen region -------------------------------------------------------------
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
            old = (src.output, tuple(d.name for d in src.settings.traces), src.settings.mode)
            self.screen_panel.apply_to(src)
            new = (src.output, tuple(d.name for d in src.settings.traces), src.settings.mode)
            if new != old:  # channels change: start the history afresh
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
        sp.add_picked_trace(tuple(int(v) for v in patch[int(np.argmax(dist))]))

    def _bar_picked(self, x0: float, y0: float, x1: float, y1: float) -> None:
        from aquafonix.screen.specread import sample_colour_bar

        frame = self._last_frame
        if frame is None or not self.screen_panel.pick_bar.isChecked():
            return
        colors = sample_colour_bar(frame.image, x0, y0, x1, y1)
        if len(colors) < 2:
            QMessageBox.information(self, APP_NAME, "Drag a box along the colour bar.")
            return
        self.screen_panel.set_bar_colors(colors)
        self.preview.set_rect_pick(False)

    def _autodetect_traces(self, count: int) -> None:
        frame = self._last_frame
        if frame is None:
            QMessageBox.information(self, APP_NAME, "Start watching the region first, then "
                                    "auto-detect its lines.")
            return
        st = self.screen_panel.trace_settings()
        colors = auto_trace_colors(frame.image, frame.result.background, st.tolerance, count,
                                   st.ignore_grid)
        if not colors:
            QMessageBox.information(self, APP_NAME, "No lines found; try a lower tolerance.")
            return
        self.screen_panel.set_detected_traces(colors)

    def _save_trace(self) -> None:
        frame = self._last_frame
        if frame is None:
            QMessageBox.information(self, APP_NAME, "Start watching a screen region first.")
            return
        path = self._ask_save("Save trace", "trace", "CSV (*.csv)")
        if not path:
            return
        r = frame.result
        cols = [r.x, *[t.y for t in r.traces], r.x_px, *[t.y_px for t in r.traces]]
        header = ["x", *r.names, "x_px", *[f"{n}_px" for n in r.names]]
        np.savetxt(path, np.column_stack(cols), delimiter=",", fmt="%.10g", comments="",
                   header=",".join(header))

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

    def _choose_data_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Data folder (signatures and events)",
                                                str(self.data_dir))
        if not path:
            return
        self.data_dir = Path(path)
        self.data_label.setText(path)
        self.library = SignatureLibrary(self.data_dir / "signatures")
        self.matcher.library = self.library
        self.matcher._key = None
        self.store = EventStore(self.data_dir / "events")
        self.events_panel.set_store(self.store)
        self.events_panel.set_tag_suggestions(self.library.labels())

    def _open_folder(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ======================================================================
    # settings
    # ======================================================================
    def _restore(self) -> None:
        s = self.settings
        geometry = s.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = s.value("window/state")
        if state is not None:
            self.restoreState(state)
        split = s.value("window/plot_split")
        if split is not None:
            self.plot_split.restoreState(split)
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
        s.setValue("window/state", self.saveState())
        s.setValue("window/plot_split", self.plot_split.saveState())
        s.setValue("window/last_dir", self._last_dir)
        s.setValue("data/dir", str(self.data_dir))
        s.beginGroup("controls")
        save_state(self, s)
        s.endGroup()
        s.sync()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_outline()

    def closeEvent(self, event) -> None:
        self.player.stop()
        self.stop()
        self._stop_analyzer()
        if self.rec_overview is not None:
            self.rec_overview.stop()
        self.outline.close()
        self.outputs.close()
        if self.persist:
            self.save_settings()
        super().closeEvent(event)

