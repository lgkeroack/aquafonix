"""Acoustics controls: calibration, band isolation, noise floor, detection and
signature matching, plus the live noise report."""

from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from aquafonix.acoustics.analyzer import AnalysisSettings, HydrophoneCalibration
from aquafonix.acoustics.filters import FILTER_TYPES, PRESETS, BandSpec
from aquafonix.acoustics.metrics import third_octave_bands
from aquafonix.acoustics.noise import NoiseReport
from aquafonix.acoustics.signatures import DEFAULT_THRESHOLD
from aquafonix.ui.widgets import combo, named, spin

NFFT_CHOICES = {"0": "Automatic (~30 ms)", "256": "256", "512": "512", "1024": "1024",
                "2048": "2048", "4096": "4096", "8192": "8192", "16384": "16384"}
PSD_NFFT_CHOICES = {"0": "Automatic (~1 Hz)", "1024": "1024", "2048": "2048", "4096": "4096",
                    "8192": "8192", "16384": "16384", "32768": "32768", "65536": "65536"}
INTEGRATION_CHOICES = {"0.125": "0.125 s", "0.25": "0.25 s", "0.5": "0.5 s", "1": "1 s",
                       "2": "2 s", "5": "5 s", "10": "10 s", "30": "30 s", "60": "1 min",
                       "600": "10 min"}


def parse_bands(text: str) -> tuple[str, ...]:
    """'63, 125 Hz, 2 kHz' -> ('63 Hz', '125 Hz', '2 kHz') matched to nominal labels."""
    labels = {b.label.lower().replace(" ", ""): b.label for b in third_octave_bands(1e6)}
    out = []
    for part in text.replace(";", ",").split(","):
        key = part.strip().lower().replace(" ", "")
        if not key:
            continue
        if key[-1].isdigit():
            key += "hz"
        if key.endswith("khz"):
            pass
        elif key.endswith("hz"):
            try:
                v = float(key[:-2])
                key = f"{v / 1000:g}khz" if v >= 1000 else f"{v:g}hz"
            except ValueError:
                pass
        if key in labels and labels[key] not in out:
            out.append(labels[key])
    return tuple(out)


class BandPicker(QDialog):
    """Tick the third-octave bands to chart over time."""

    def __init__(self, parent, selected: tuple[str, ...], f_max: float = 192_000) -> None:
        super().__init__(parent)
        self.setWindowTitle("Third-octave bands over time")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Bands (IEC 61260-1 base-10) to plot over time:"))
        self.list = QListWidget()
        for b in third_octave_bands(f_max):
            item = QListWidgetItem(f"{b.label}   ({b.low:.1f}–{b.high:.1f} Hz)")
            item.setData(Qt.UserRole, b.label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if b.label in selected else Qt.Unchecked)
            self.list.addItem(item)
        lay.addWidget(self.list)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def selected(self) -> tuple[str, ...]:
        return tuple(self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                     if self.list.item(i).checkState() == Qt.Checked)


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

        f = _group("Levels (ISO 18405)", lay)
        self.integration = named(combo(INTEGRATION_CHOICES, "1"), "ac.integration")
        self.integration.setToolTip("Averaging time for SPL, Lpk, SEL, PSD and band levels")
        f.addRow("Integration time", self.integration)
        self.psd_nfft = named(combo(PSD_NFFT_CHOICES, "0"), "ac.psdnfft")
        self.psd_nfft.setToolTip("FFT size for PSD and third-octave levels (frequency "
                                 "resolution = sample rate / size)")
        f.addRow("PSD FFT size", self.psd_nfft)
        brow = QHBoxLayout()
        self.tol_text = named(QLineEdit("63 Hz, 125 Hz, 2 kHz"), "ac.tolbands")
        self.tol_text.setToolTip("Third-octave bands charted over time (comma-separated)")
        self.tol_text.editingFinished.connect(self._changed)
        pick = QPushButton("…")
        pick.setFixedWidth(30)
        pick.clicked.connect(self._pick_bands)
        brow.addWidget(self.tol_text, 1)
        brow.addWidget(pick)
        f.addRow("Bands over time", brow)
        self.stats_window = named(spin(5, 86_400, 60, suffix=" s"), "ac.statswindow")
        self.stats_window.setToolTip("Window for L5/L50/L95 and PSD/band percentiles")
        f.addRow("Statistics over", self.stats_window)

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
        prow = QHBoxLayout()
        self.profiles = QComboBox()
        self.profiles.setToolTip("Saved hydrophone / recorder calibrations")
        self.profiles.activated.connect(self._load_profile)
        save_p = QPushButton("Save…")
        save_p.clicked.connect(self._save_profile)
        del_p = QPushButton("Delete")
        del_p.clicked.connect(self._delete_profile)
        prow.addWidget(self.profiles, 1)
        prow.addWidget(save_p)
        prow.addWidget(del_p)
        f.addRow("Profile", prow)
        self.cal_auto = named(QCheckBox("Use calibration found in a recording's metadata"),
                              "ac.calauto")
        self.cal_auto.setChecked(True)
        f.addRow(self.cal_auto)
        self.cal_source = QLabel("")
        self.cal_source.setWordWrap(True)
        self.cal_source.setStyleSheet("color: gray;")
        f.addRow(self.cal_source)
        self._store: QSettings | None = None

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
        self.min_dur = named(spin(0.0, 60, 0.005, decimals=3, suffix=" s"), "ac.mindur")
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

        for w in (self.channel, self.nfft, self.band_kind, self.integration, self.psd_nfft):
            w.currentIndexChanged.connect(self._changed)
        for w in (self.cal_on, self.band_on, self.band_show, self.band_listen, self.band_detect,
                  self.alarm_on, self.detect_on, self.match_on):
            w.toggled.connect(self._changed)
        for w in (self.sens, self.gain, self.vfs):
            w.valueChanged.connect(lambda *_: self.cal_source.setText("Entered by hand"))
        for w in (self.sens, self.gain, self.vfs, self.band_lo, self.band_hi, self.rise,
                  self.fall, self.stats_window, self.alarm_level, self.threshold, self.min_dur,
                  self.max_dur, self.match_thr):
            w.valueChanged.connect(self._changed)
        self._update_visibility()

    # -- calibration profiles -------------------------------------------------------
    def set_profile_store(self, settings: QSettings) -> None:
        self._store = settings
        self._refresh_profiles()

    def _profiles(self) -> dict:
        if self._store is None:
            return {}
        try:
            return json.loads(self._store.value("calibration/profiles", "{}") or "{}")
        except ValueError:
            return {}

    def _refresh_profiles(self) -> None:
        self.profiles.clear()
        self.profiles.addItem("— choose a saved profile —")
        for name in sorted(self._profiles(), key=str.lower):
            self.profiles.addItem(name)

    def _save_profile(self) -> None:
        if self._store is None:
            return
        name, ok = QInputDialog.getText(self, "Save calibration profile",
                                        "Name (e.g. hydrophone model + serial, gain setting)")
        if not ok or not name.strip():
            return
        profiles = self._profiles()
        profiles[name.strip()] = {"sensitivity_db": self.sens.value(),
                                  "gain_db": self.gain.value(), "full_scale_v": self.vfs.value()}
        self._store.setValue("calibration/profiles", json.dumps(profiles))
        self._refresh_profiles()
        self.profiles.setCurrentText(name.strip())

    def _delete_profile(self) -> None:
        name = self.profiles.currentText()
        profiles = self._profiles()
        if name in profiles and QMessageBox.question(
                self, "Delete profile", f"Delete '{name}'?") == QMessageBox.Yes:
            del profiles[name]
            self._store.setValue("calibration/profiles", json.dumps(profiles))
            self._refresh_profiles()

    def _load_profile(self, index: int) -> None:
        p = self._profiles().get(self.profiles.itemText(index))
        if p:
            self.apply_calibration(p["sensitivity_db"], p["gain_db"], p["full_scale_v"],
                                   f"profile '{self.profiles.itemText(index)}'")

    def apply_calibration(self, sensitivity_db: float, gain_db: float, full_scale_v: float,
                          source: str = "") -> None:
        for w in (self.sens, self.gain, self.vfs, self.cal_on):
            w.blockSignals(True)
        self.sens.setValue(sensitivity_db)
        self.gain.setValue(gain_db)
        self.vfs.setValue(full_scale_v)
        self.cal_on.setChecked(True)
        for w in (self.sens, self.gain, self.vfs, self.cal_on):
            w.blockSignals(False)
        self.cal_source.setText(f"From {source}" if source else "")
        self._changed()

    def calibration_values(self) -> dict:
        return {"sensitivity_db": self.sens.value(), "gain_db": self.gain.value(),
                "full_scale_v": self.vfs.value(), "enabled": self.cal_on.isChecked()}

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
            floor_alarm_db=self.alarm_level.value(), stats_window_s=self.stats_window.value(),
            integration_s=float(self.integration.currentData()),
            psd_nfft=int(self.psd_nfft.currentData()), tol_bands=self.tol_bands())

    def tol_bands(self) -> tuple[str, ...]:
        return parse_bands(self.tol_text.text())

    def _pick_bands(self) -> None:
        dlg = BandPicker(self, self.tol_bands())
        if dlg.exec():
            self.tol_text.setText(", ".join(dlg.selected()))
            self._changed()

    # -- report -----------------------------------------------------------------
    def show_report(self, rep: NoiseReport, alarm: bool) -> None:
        def v(x, unit=" dB"):
            return f"{x:.1f}{unit}" if x is not None and np.isfinite(x) else "–"

        u = rep.units
        lines = [f"<b>SPL</b> {v(rep.level_db)} &nbsp; <b>L<sub>pk</sub></b> {v(rep.peak_db)}"
                 f" &nbsp; <b>Floor</b> {v(rep.floor_db)}"
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
            shown = [b for b in self.tol_bands() if b in tol] or [b for b in ("63 Hz", "125 Hz")
                                                                    if b in tol]
            lines.append("⅓-oct " + " · ".join(f"{b} {v(tol[b])}" for b in shown))
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
