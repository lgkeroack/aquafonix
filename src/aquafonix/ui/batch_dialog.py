"""Dialog for batch analysis of many recordings."""

from __future__ import annotations

import threading

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QMessageBox, QProgressBar,
                               QPushButton, QVBoxLayout)

from aquafonix.acoustics.batch import BatchJob, run_batch
from aquafonix.audiofile import OPEN_FILTER
from aquafonix.sources.audiofile import expand_paths
from aquafonix.ui.acoustics_panel import INTEGRATION_CHOICES
from aquafonix.ui.widgets import combo


class BatchDialog(QDialog):
    def __init__(self, parent, acoustics) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch analysis")
        self.resize(640, 520)
        self.acoustics = acoustics
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Computes SPL, peak level, SEL and third-octave levels per "
                             "interval for every recording, plus a summary with L5/L50/L95 "
                             "and Leq per file and overall."))
        self.files = QListWidget()
        lay.addWidget(self.files, 1)
        row = QHBoxLayout()
        for text, slot in (("Add files…", self._add_files), ("Add folder…", self._add_folder),
                           ("Clear", self.files.clear)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        f = QFormLayout()
        self.integration = combo(INTEGRATION_CHOICES, "60")
        f.addRow("Integration time", self.integration)
        self.only_chosen = QCheckBox("Only the third-octave bands chosen in Acoustics")
        f.addRow(self.only_chosen)
        self.with_band = QCheckBox("Add a column for the isolated band (Acoustics tab)")
        f.addRow(self.with_band)
        self.use_meta = QCheckBox("Prefer calibration found in each file's metadata")
        self.use_meta.setChecked(True)
        f.addRow(self.use_meta)
        out_row = QHBoxLayout()
        self.out = QLineEdit("aquafonix_batch.csv")
        browse = QPushButton("…")
        browse.clicked.connect(self._choose_out)
        out_row.addWidget(self.out, 1)
        out_row.addWidget(browse)
        f.addRow("Output CSV", out_row)
        lay.addLayout(f)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        lay.addWidget(self.bar)
        self.status = QLabel("")
        lay.addWidget(self.status)
        row = QHBoxLayout()
        row.addStretch(1)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Close")
        self.cancel_btn.clicked.connect(self._cancel_or_close)
        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
        lay.addLayout(row)
        self._thread: threading.Thread | None = None
        self._cancel = False
        self._progress = (0.0, "")
        self._result = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Recordings", "", OPEN_FILTER)
        self.files.addItems(paths)

    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder of recordings")
        if path:
            self.files.addItems(expand_paths(path))

    def _choose_out(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Output CSV", self.out.text(), "CSV (*.csv)")
        if path:
            self.out.setText(path)

    def _run(self) -> None:
        paths = [self.files.item(i).text() for i in range(self.files.count())]
        if not paths:
            QMessageBox.information(self, "Batch analysis", "Add some recordings first.")
            return
        cal = self.acoustics.calibration()
        band = None
        if self.with_band.isChecked() and self.acoustics.band() is not None:
            b = self.acoustics.band()
            band = (b.low, b.high) if b.kind == "bandpass" else None
        job = BatchJob(paths, self.out.text(), float(self.integration.currentData()),
                       self.acoustics.tol_bands() if self.only_chosen.isChecked() else None,
                       band, int(self.acoustics.channel.currentData()), cal.offset_db,
                       cal.enabled, self.use_meta.isChecked())
        self._cancel = False
        self._result = None
        self.run_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel")

        def work():
            try:
                self._result = run_batch(job, lambda f, m: setattr(self, "_progress", (f, m)),
                                         lambda: self._cancel)
            except Exception as exc:
                self._result = {"error": str(exc)}

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        self._timer.start(200)

    def _poll(self) -> None:
        frac, msg = self._progress
        self.bar.setValue(int(frac * 1000))
        self.status.setText(msg)
        if self._thread is not None and not self._thread.is_alive():
            self._timer.stop()
            self._thread = None
            self.run_btn.setEnabled(True)
            self.cancel_btn.setText("Close")
            r = self._result or {}
            if "error" in r:
                QMessageBox.warning(self, "Batch analysis", r["error"])
            else:
                extra = ("\n\nProblems:\n" + "\n".join(r["errors"])) if r.get("errors") else ""
                QMessageBox.information(self, "Batch analysis",
                                        f"{r['intervals']} intervals from {r['files']} file(s).\n"
                                        f"Rows: {r['csv']}\nSummary: {r['summary']}{extra}")

    def _cancel_or_close(self) -> None:
        if self._thread is not None:
            self._cancel = True
        else:
            self.close()
