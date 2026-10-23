"""Acoustics controls: calibration, band isolation, noise floor, detection and
signature matching, plus the live noise report."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from aquafonix.acoustics.analyzer import AnalysisSettings, HydrophoneCalibration
from aquafonix.acoustics.filters import FILTER_TYPES, PRESETS, BandSpec
from aquafonix.acoustics.noise import NoiseReport
from aquafonix.acoustics.signatures import DEFAULT_THRESHOLD
from aquafonix.ui.widgets import combo, named, spin

NFFT_CHOICES = {"0": "Automatic (~30 ms)", "256": "256", "512": "512", "1024": "1024",
                "2048": "2048", "4096": "4096", "8192": "8192", "16384": "16384"}


def _group(title: str, parent_layout: QVBoxLayout) -> QFormLayout:
    box = QGroupBox(title)
    f = QFormLayout(box)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    parent_layout.addWidget(box)
    return f


class AcousticsPanel(QWidget):
    changed = Signal()
    band_from_selection = Signal()
    manage_signatures = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        f = _group("Analysis", lay)
        self.channel = named(combo({"0": "Channel 1", "1": "Channel 2", "2": "Channel 3",
                                    "3": "Channel 4", "-1": "Mix of all channels"}, "0"),
                             "ac.channel")
        f.addRow("Analyse", self.channel)
        self.nfft = named(combo(NFFT_CHOICES, "0"), "ac.nfft")
        self.nfft.setToolTip("Spectrogram FFT size: larger = finer frequency, coarser time")
        f.addRow("FFT size", self.nfft)

        f = _group("Hydrophone calibration", lay)
        self.cal_on = named(QCheckBox("Show levels in dB re 1 µPa"), "ac.cal")
        f.addRow(self.cal_on)
        self.sens = named(spin(-300, 0, -165, decimals=1, suffix=" dB re 1 V/µPa"), "ac.sens")
        self.gain = named(spin(-60, 120, 0, decimals=1, suffix=" dB"), "ac.gain")
        self.vfs = named(spin(0.001, 100, 1.0, decimals=3, suffix=" V"), "ac.vfs")
        self.vfs.setToolTip("Input voltage (peak) that reaches digital full scale")
        f.addRow("Sensitivity", self.sens)
        f.addRow("Pre-amp gain", self.gain)
        f.addRow("Full scale", self.vfs)

        f = _group("Band isolation", lay)
        self.band_on = named(QCheckBox("Isolate a frequency band"), "ac.band")
        f.addRow(self.band_on)
        self.preset = combo(["Presets…"] + [p[0] for p in PRESETS])
        self.preset.activated.connect(self._apply_preset)
        f.addRow(self.preset)
        self.band_kind = named(combo(FILTER_TYPES, "bandpass"), "ac.bandkind")
        f.addRow("Type", self.band_kind)
        self.band_lo = named(spin(0.1, 1e6, 100, decimals=1, suffix=" Hz"), "ac.bandlo")
        self.band_hi = named(spin(0.1, 1e6, 1000, decimals=1, suffix=" Hz"), "ac.bandhi")
        row = QHBoxLayout()
        row.addWidget(self.band_lo)
        row.addWidget(QLabel("to"))
        row.addWidget(self.band_hi)
        self.band_row = QWidget()
        self.band_row.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        f.addRow("Band", self.band_row)
        from_sel = QPushButton("Use the spectrogram selection")
        from_sel.setToolTip("Shift-drag a box on the spectrogram, then click here")
        from_sel.clicked.connect(self.band_from_selection)
        f.addRow(from_sel)
        self.band_show = named(QCheckBox("Show filtered waveform"), "ac.bandshow")
        self.band_show.setChecked(True)
        self.band_listen = named(QCheckBox("Listen to the filtered signal"), "ac.bandlisten")
        self.band_listen.setChecked(True)
        self.band_detect = named(QCheckBox("Detect spikes only inside the band"), "ac.banddetect")
        self.band_detect.setChecked(True)
        for w in (self.band_show, self.band_listen, self.band_detect):
            f.addRow(w)
        self._band_form = f

        f = _group("Noise floor", lay)
        self.rise = named(spin(1, 3600, 20, decimals=1, suffix=" s"), "ac.floorrise")
        self.rise.setToolTip("How slowly the floor follows rising noise (calls shouldn't lift it)")
        self.fall = named(spin(0.1, 600, 2, decimals=1, suffix=" s"), "ac.floorfall")
        f.addRow("Rise time", self.rise)
        f.addRow("Fall time", self.fall)
        self.stats_window = named(spin(5, 3600, 60, suffix=" s"), "ac.statswindow")
        f.addRow("Statistics over", self.stats_window)
        self.alarm_on = named(QCheckBox("Alarm when the floor exceeds"), "ac.alarm")
        self.alarm_level = named(spin(-300, 300, 100, decimals=1, suffix=" dB"), "ac.alarmlevel")
        row = QHBoxLayout()
        row.addWidget(self.alarm_on)
        row.addWidget(self.alarm_level)
        f.addRow(row)

        f = _group("Detection & identification", lay)
        self.detect_on = named(QCheckBox("Highlight spikes above the noise floor"), "ac.detect")
        self.detect_on.setChecked(True)
        f.addRow(self.detect_on)
        self.threshold = named(spin(1, 60, 10, decimals=1, suffix=" dB"), "ac.threshold")
        f.addRow("Spike threshold", self.threshold)
        self.min_dur = named(spin(0.001, 60, 0.03, decimals=3, suffix=" s"), "ac.mindur")
        self.max_dur = named(spin(0.01, 600, 10, decimals=2, suffix=" s"), "ac.maxdur")
        f.addRow("Min duration", self.min_dur)
        f.addRow("Max duration", self.max_dur)
        self.match_on = named(QCheckBox("Flag sounds that match my signatures"), "ac.match")
        self.match_on.setChecked(True)
        f.addRow(self.match_on)
        self.match_thr = named(spin(0.05, 0.99, DEFAULT_THRESHOLD, decimals=2, step=0.05),
                               "ac.matchthr")
        self.match_thr.setToolTip("Similarity needed to say \"Suspected: …\" (0–1)")
        f.addRow("Match threshold", self.match_thr)
        manage = QPushButton("Signatures…")
        manage.clicked.connect(self.manage_signatures)
        f.addRow(manage)

        box = QGroupBox("Noise report")
        bl = QVBoxLayout(box)
        self.report = QLabel("Start an audio source to see levels.")
        self.report.setTextFormat(Qt.RichText)
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bl.addWidget(self.report)
        lay.addWidget(box)
        lay.addStretch(1)

        for w in (self.channel, self.nfft, self.band_kind):
            w.currentIndexChanged.connect(self._changed)
        for w in (self.cal_on, self.band_on, self.band_show, self.band_listen, self.band_detect,
                  self.alarm_on, self.detect_on, self.match_on):
            w.toggled.connect(self._changed)
        for w in (self.sens, self.gain, self.vfs, self.band_lo, self.band_hi, self.rise,
                  self.fall, self.stats_window, self.alarm_level, self.threshold, self.min_dur,
                  self.max_dur, self.match_thr):
            w.valueChanged.connect(self._changed)
        self._update_visibility()

    # -- values -----------------------------------------------------------------
    def calibration(self) -> HydrophoneCalibration:
        return HydrophoneCalibration(self.cal_on.isChecked(), self.sens.value(),
                                     self.gain.value(), self.vfs.value())

    def band(self) -> BandSpec | None:
        if not self.band_on.isChecked():
            return None
        return BandSpec(self.band_kind.currentData(), self.band_lo.value(), self.band_hi.value())

    def set_band(self, lo: float, hi: float, kind: str = "bandpass") -> None:
        for w in (self.band_kind, self.band_lo, self.band_hi, self.band_on):
            w.blockSignals(True)
        self.band_kind.setCurrentIndex(self.band_kind.findData(kind))
        self.band_lo.setValue(max(0.1, lo))
        self.band_hi.setValue(max(0.2, hi))
        self.band_on.setChecked(True)
        for w in (self.band_kind, self.band_lo, self.band_hi, self.band_on):
            w.blockSignals(False)
        self._changed()

    def settings(self) -> AnalysisSettings:
        return AnalysisSettings(
            channel=int(self.channel.currentData()), nfft=int(self.nfft.currentData()),
            floor_rise_s=self.rise.value(), floor_fall_s=self.fall.value(),
            calibration=self.calibration(), band=self.band(),
            detect=self.detect_on.isChecked(), event_threshold_db=self.threshold.value(),
            min_event_s=self.min_dur.value(), max_event_s=self.max_dur.value(),
            detect_in_band=self.band_detect.isChecked(), match=self.match_on.isChecked(),
            match_threshold=self.match_thr.value(), floor_alarm=self.alarm_on.isChecked(),
            floor_alarm_db=self.alarm_level.value(), stats_window_s=self.stats_window.value())

    # -- report -----------------------------------------------------------------
    def show_report(self, rep: NoiseReport, alarm: bool) -> None:
        def v(x, unit=" dB"):
            return f"{x:.1f}{unit}" if x is not None and np.isfinite(x) else "–"

        u = rep.units
        lines = [f"<b>Level</b> {v(rep.level_db)} &nbsp; <b>Floor</b> {v(rep.floor_db)}"
                 + (" <span style='color:#ff5050'><b>⚠ ALARM</b></span>" if alarm else "")
                 + f" <span style='color:gray'>({u})</span>"]
        if np.isfinite(rep.band_db):
            lines.append(f"<b>Band</b> {v(rep.band_db)}")
        if rep.stats:
            st = rep.stats
            lines.append(f"L5 {v(st.get('L5'))} · L50 {v(st.get('L50'))} · "
                         f"L95 {v(st.get('L95'))} · Leq {v(st.get('Leq'))}")
        tol = rep.third_octaves
        if tol:
            lines.append(f"⅓-oct 63 Hz {v(tol.get('63 Hz'))} · 125 Hz {v(tol.get('125 Hz'))}")
        lines.append(f"Crest {v(rep.crest_db)} · kurtosis {v(rep.kurtosis, '')} · "
                     f"flatness {v(rep.flatness, '')}")
        if rep.tonals:
            lines.append("<b>Tonal lines</b> " + ", ".join(
                f"{f:.1f} Hz (+{p:.0f})" for f, p in rep.tonals[:6]))
        if rep.sources:
            lines.append("<b>Likely noise sources</b> <span style='color:gray'>(heuristic)</span>")
            for name, score, reason in rep.sources[:4]:
                lines.append(f"• {name} <span style='color:gray'>— {reason}</span>")
        self.report.setText("<br>".join(lines))

    # -- internals ----------------------------------------------------------------
    def _apply_preset(self, index: int) -> None:
        if index <= 0:
            return
        _label, kind, lo, hi = PRESETS[index - 1]
        self.set_band(lo, hi or lo * 2, kind)
        self.preset.setCurrentIndex(0)

    def _changed(self, *_args) -> None:
        self._update_visibility()
        self.changed.emit()

    def _update_visibility(self) -> None:
        on = self.band_on.isChecked()
        single = self.band_kind.currentData() in ("lowpass", "highpass")
        self.band_hi.setVisible(not single)
        for w in (self.band_kind, self.band_row):
            w.setEnabled(on)
        self.alarm_level.setEnabled(self.alarm_on.isChecked())
