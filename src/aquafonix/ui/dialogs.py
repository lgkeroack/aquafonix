"""Dialogs for identifying sounds and managing the signature catalog."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from aquafonix.acoustics.signatures import CATEGORIES, METHODS, SignatureLibrary
from aquafonix.acoustics.spectral import auto_nfft, power_frames, to_db
from aquafonix.ui.plots import lookup_table
from aquafonix.ui.widgets import combo, spin


def clip_spectrogram(clip: np.ndarray, rate: float, height: int = 160) -> pg.PlotWidget:
    """A small static spectrogram of a clip."""
    w = pg.PlotWidget()
    w.setMinimumHeight(height)
    w.setLabel("left", "Hz")
    w.setLabel("bottom", "s")
    nfft = min(auto_nfft(rate), max(32, 1 << int(np.log2(max(len(clip) // 4, 32)))))
    p = power_frames(np.asarray(clip, dtype=float), nfft, max(1, nfft // 4))
    if len(p):
        d = to_db(p).T
        img = pg.ImageItem(axisOrder="row-major")
        img.setLookupTable(lookup_table("inferno"))
        hi = np.percentile(d, 99.5)
        img.setImage(d, autoLevels=False, levels=(hi - 60, hi))
        img.setRect(QRectF(0, 0, len(clip) / rate, rate / 2))
        w.addItem(img)
    else:
        t = np.arange(len(clip)) / rate
        w.plot(t, clip, pen=pg.mkPen("#4cc9f0"))
        w.setLabel("left", "value")
    return w


class IdentifyDialog(QDialog):
    """Name a snipped sound; it becomes (or extends) a signature."""

    def __init__(self, parent, labels: list[str], clip: np.ndarray, rate: float,
                 f_lo: float, f_hi: float, suggested: str = "", play=None,
                 title: str = "Identify sound") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        self.preview = clip_spectrogram(clip, rate)
        self.band_lines = []
        if rate >= 2000:
            for f in (f_lo, f_hi):
                line = pg.InfiniteLine(f, angle=0, pen=pg.mkPen("#3ae8ff", dash=[4, 3]))
                self.preview.addItem(line)
                self.band_lines.append(line)
        lay.addWidget(self.preview)
        info = QLabel(f"{len(clip) / rate:.2f} s at {rate:g} Hz")
        info.setStyleSheet("color: gray;")
        lay.addWidget(info)
        f = QFormLayout()
        self.name = QComboBox()
        self.name.setEditable(True)
        self.name.addItems(labels)
        self.name.setCurrentText(suggested)
        self.name.lineEdit().setPlaceholderText("e.g. Harbour seal roar, boat pass, pump")
        f.addRow("Name", self.name)
        self.category = QComboBox()
        self.category.setEditable(True)
        self.category.addItems(CATEGORIES)
        f.addRow("Category", self.category)
        self.notes = QLineEdit()
        f.addRow("Notes", self.notes)
        self.lo = spin(0, rate / 2, max(0.0, f_lo), decimals=1, suffix=" Hz")
        self.hi = spin(0, rate / 2, min(rate / 2, f_hi), decimals=1, suffix=" Hz")
        self.lo.valueChanged.connect(self._band_edited)
        self.hi.valueChanged.connect(self._band_edited)
        row = QHBoxLayout()
        row.addWidget(self.lo)
        row.addWidget(QLabel("to"))
        row.addWidget(self.hi)
        f.addRow("Frequency band", row)
        self.method = combo(METHODS, "auto")
        f.addRow("Match by", self.method)
        self.save_event = QCheckBox("Also save the clip as a tagged event")
        self.save_event.setChecked(True)
        f.addRow(self.save_event)
        lay.addLayout(f)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        if play is not None:
            listen = buttons.addButton("▶ Listen", QDialogButtonBox.ActionRole)
            listen.clicked.connect(play)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _band_edited(self) -> None:
        for line, box in zip(self.band_lines, (self.lo, self.hi)):
            line.setValue(box.value())

    def _accept(self) -> None:
        if not self.name.currentText().strip():
            QMessageBox.information(self, self.windowTitle(), "Give the sound a name.")
            return
        self.accept()

    def values(self) -> dict:
        lo, hi = sorted((self.lo.value(), self.hi.value()))
        return {"label": self.name.currentText().strip(),
                "category": self.category.currentText().strip(),
                "notes": self.notes.text().strip(), "f_lo": lo, "f_hi": hi,
                "method": self.method.currentData(), "save_event": self.save_event.isChecked()}


def ask_label(parent, labels: list[str], title: str, prompt: str, current: str = "") -> str | None:
    """Pick an existing name or type a new one."""
    items = sorted(set(labels), key=str.lower)
    if current and current in items:
        items.remove(current)
    text, ok = QInputDialog.getItem(parent, title, prompt, items, 0, True)
    text = text.strip()
    return text if ok and text else None


class SignatureManager(QDialog):
    """Browse, rename, tune, disable and delete signatures."""

    COLS = ["Name", "Category", "Match by", "Examples", "Not-examples", "Confirmed",
            "Rejected", "Threshold", "On"]

    def __init__(self, parent, library: SignatureLibrary, play_example=None,
                 default_threshold: float = 0.6) -> None:
        super().__init__(parent)
        self.setWindowTitle("Signatures")
        self.resize(820, 420)
        self.library = library
        self.play_example = play_example
        self.default_threshold = default_threshold
        lay = QVBoxLayout(self)
        hint = QLabel("Your catalog grows as you identify, confirm and correct sounds. "
                      "Each confirmation adds an example; each rejection or correction "
                      "adds a 'not-example' that stops false matches.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, slot in (("▶ Play example", self._play), ("Rename…", self._rename),
                           ("Delete", self._delete), ("Open folder", self._open_folder)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)
        self.refresh()

    def _selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        sig_id = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
        return self.library.get(sig_id)

    def refresh(self) -> None:
        sigs = sorted(self.library.signatures, key=lambda s: s.label.lower())
        self.table.setRowCount(len(sigs))
        for r, sig in enumerate(sigs):
            values = [sig.label, sig.category, METHODS.get(sig.method, sig.method),
                      str(len(sig.positives)), str(len(sig.negatives)), str(sig.confirmed),
                      str(sig.rejected)]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                if c == 0:
                    item.setData(Qt.UserRole, sig.id)
                self.table.setItem(r, c, item)
            thr = QDoubleSpinBox()
            thr.setRange(0.0, 0.99)
            thr.setDecimals(2)
            thr.setSingleStep(0.05)
            thr.setSpecialValueText(f"default ({self.default_threshold:.2f})")
            thr.setValue(sig.threshold or 0.0)
            thr.valueChanged.connect(lambda v, s=sig: self._set_threshold(s, v))
            self.table.setCellWidget(r, 7, thr)
            on = QCheckBox()
            on.setChecked(sig.enabled)
            on.toggled.connect(lambda v, s=sig: self._set_enabled(s, v))
            self.table.setCellWidget(r, 8, on)

    def _set_threshold(self, sig, value: float) -> None:
        sig.threshold = value or None
        self.library.update(sig)

    def _set_enabled(self, sig, on: bool) -> None:
        sig.enabled = on
        self.library.update(sig)

    def _play(self) -> None:
        sig = self._selected()
        if sig and sig.positives and self.play_example:
            clip, rate = self.library.load_clip(sig.positives[-1])
            self.play_example(clip, rate)

    def _rename(self) -> None:
        sig = self._selected()
        if sig is None:
            return
        text, ok = QInputDialog.getText(self, "Rename", "New name", text=sig.label)
        if ok and text.strip():
            try:
                self.library.rename(sig, text)
            except ValueError as exc:
                QMessageBox.warning(self, "Rename", str(exc))
            self.refresh()

    def _delete(self) -> None:
        sig = self._selected()
        if sig is None:
            return
        if QMessageBox.question(self, "Delete", f"Delete '{sig.label}' and its "
                                f"{len(sig.examples)} example clip(s)?") == QMessageBox.Yes:
            self.library.remove_signature(sig)
            self.refresh()

    def _open_folder(self) -> None:
        self.library.root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.library.root)))
