"""Bottom panels: detections to review, and saved events to tag/annotate."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from PySide6.QtCore import QStringListModel, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QCompleter, QFormLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from aquafonix.events import EventRecord, EventStore
from aquafonix.ui.plots import STATUS_COLORS


def _time_text(stream_t: float, wall: float | None) -> str:
    if wall:
        return dt.datetime.fromtimestamp(wall).strftime("%H:%M:%S.%f")[:-4]
    return f"{stream_t:.2f} s"


class DetectionsPanel(QWidget):
    confirm = Signal(int)
    correct = Signal(int)
    reject = Signal(int)
    identify = Signal(int)
    play = Signal(int)
    zoom = Signal(int)
    save_event = Signal(int)
    selected = Signal(int)

    COLS = ["Time", "Detection", "Score", "Band", "SNR", "Status"]
    FILTERS = {"all": "All", "suspected": "Suspected", "unidentified": "Unidentified",
               "reviewed": "Reviewed"}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.buttons = {}
        for key, text, tip in (
                ("confirm", "✔ Confirm", "It is what was suspected: add as an example (C)"),
                ("correct", "✎ Correct…", "It is something else: choose the right name"),
                ("reject", "✖ Reject", "Not that: teach the signature not to match this (X)"),
                ("identify", "🏷 Identify…", "Name this sound and add it to the catalog (I)"),
                ("play", "▶ Play", "Listen to the detection"),
                ("zoom", "⤢ Show", "Zoom the plots to the detection (double-click)"),
                ("save_event", "💾 Save as event", "Keep the clip in Events")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self._emit(k))
            bar.addWidget(b)
            self.buttons[key] = b
        bar.addStretch(1)
        self.filter = QComboBox()
        for k, v in self.FILTERS.items():
            self.filter.addItem(v, k)
        self.filter.currentIndexChanged.connect(self._apply_filter)
        bar.addWidget(QLabel("Show"))
        bar.addWidget(self.filter)
        lay.addLayout(bar)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._emit("zoom"))
        lay.addWidget(self.table, 1)
        self._rows: dict[int, int] = {}
        self._state: dict[int, tuple] = {}
        self._status: dict[int, str] = {}
        self._update_buttons()

    def current_id(self) -> int | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).data(Qt.UserRole)

    def select(self, det_id: int) -> None:
        row = self._rows.get(det_id)
        if row is not None:
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0))

    def clear(self) -> None:
        self.table.setRowCount(0)
        self._rows.clear()
        self._state.clear()
        self._status.clear()

    def update_detections(self, detections) -> None:
        follow = self.table.verticalScrollBar().value() >= self.table.verticalScrollBar().maximum() - 2
        for det in detections:
            state = (det.status, det.label, round(det.score, 3), det.t0, det.t1)
            if self._state.get(det.id) == state:
                continue
            self._state[det.id] = state
            self._status[det.id] = det.status
            row = self._rows.get(det.id)
            if row is None:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self._rows[det.id] = row
            band = (f"{det.f_lo:.0f}–{det.f_hi:.0f} Hz" if det.f_hi > det.f_lo else "")
            values = [_time_text(det.t0, det.wall_time), det.text,
                      f"{det.score:.2f}" if det.score else "",
                      band, f"{det.snr_db:.1f} dB" if det.snr_db == det.snr_db else "",
                      det.status]
            color = QColor(*STATUS_COLORS.get(det.status, (200, 200, 200)))
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                if c == 0:
                    item.setData(Qt.UserRole, det.id)
                if c in (1, 5):
                    item.setForeground(color)
                self.table.setItem(row, c, item)
            self.table.setRowHidden(row, not self._passes(det.status))
        if follow:
            self.table.scrollToBottom()
        self._update_buttons()

    def _passes(self, status: str) -> bool:
        f = self.filter.currentData()
        if f == "all":
            return True
        if f == "reviewed":
            return status in ("confirmed", "corrected", "rejected")
        return status == f

    def _apply_filter(self) -> None:
        for det_id, row in self._rows.items():
            self.table.setRowHidden(row, not self._passes(self._status.get(det_id, "")))

    def _selection_changed(self) -> None:
        self._update_buttons()
        det_id = self.current_id()
        if det_id is not None:
            self.selected.emit(det_id)

    def _update_buttons(self) -> None:
        det_id = self.current_id()
        status = self._status.get(det_id) if det_id is not None else None
        for key, b in self.buttons.items():
            b.setEnabled(det_id is not None)
        if det_id is not None:
            self.buttons["confirm"].setEnabled(status == "suspected")
            self.buttons["reject"].setEnabled(status in ("suspected", "unidentified"))

    def _emit(self, key: str) -> None:
        det_id = self.current_id()
        if det_id is not None:
            getattr(self, key).emit(det_id)


class TagCompleter(QCompleter):
    """Completes the last of several comma-separated tags."""

    def splitPath(self, path: str) -> list[str]:
        return [path.split(",")[-1].strip()]

    def pathFromIndex(self, index) -> str:
        text = self.widget().text()
        head = text.rsplit(",", 1)[0] + ", " if "," in text else ""
        return head + index.data()


class EventsPanel(QWidget):
    play = Signal(object)  # EventRecord
    open_in_replay = Signal(object)

    COLS = ["Created", "Title / tags", "Length", "Kind", "✔", "Notes"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.store: EventStore | None = None
        self._records: list[EventRecord] = []
        self._extra_tags: list[str] = []
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by tag, title or notes…")
        self.search.textChanged.connect(self._apply_filter)
        bar.addWidget(self.search, 1)
        for text, slot in (("Refresh", self.refresh), ("Export CSV…", self._export),
                           ("Open folder", self._open_root)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        ll.addLayout(bar)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._load_selected)
        ll.addWidget(self.table, 1)
        split.addWidget(left)

        right = QWidget()
        f = QFormLayout(right)
        self.title = QLineEdit()
        f.addRow("Title", self.title)
        self.tags = QLineEdit()
        self.tags.setPlaceholderText("comma-separated, e.g. seal, near buoy")
        self._tag_model = QStringListModel()
        completer = TagCompleter(self._tag_model, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.tags.setCompleter(completer)
        f.addRow("Tags", self.tags)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Annotation: what happened, conditions, observer…")
        f.addRow("Notes", self.notes)
        self.verified = QCheckBox("Verified")
        f.addRow(self.verified)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray;")
        f.addRow(self.info)
        row = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save)
        play = QPushButton("▶ Play")
        play.clicked.connect(lambda: self._with_current(self.play.emit))
        replay = QPushButton("Open in replay")
        replay.setToolTip("Load the clip as an audio-file source to inspect it")
        replay.clicked.connect(lambda: self._with_current(self.open_in_replay.emit))
        folder = QPushButton("Folder")
        folder.clicked.connect(lambda: self._with_current(
            lambda r: QDesktopServices.openUrl(QUrl.fromLocalFile(r.folder))))
        delete = QPushButton("Delete")
        delete.clicked.connect(self._delete)
        for b in (self.save_btn, play, replay, folder, delete):
            row.addWidget(b)
        f.addRow(row)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        self._set_editor_enabled(False)

    def set_store(self, store: EventStore) -> None:
        self.store = store
        self.refresh()

    def set_tag_suggestions(self, tags: list[str]) -> None:
        self._extra_tags = list(tags)
        self._update_tag_model()

    def _update_tag_model(self) -> None:
        known = {t for r in self._records for t in r.tags} | set(self._extra_tags)
        self._tag_model.setStringList(sorted(known, key=str.lower))

    def refresh(self, select_id: str | None = None) -> None:
        current = select_id or (self._current().id if self._current() else None)
        self._records = self.store.list() if self.store else []
        self.table.setRowCount(len(self._records))
        for r, rec in enumerate(self._records):
            title = rec.title or ""
            tags = ", ".join(rec.tags)
            values = [rec.created.replace("T", " "), " · ".join(v for v in (title, tags) if v),
                      f"{rec.duration:.1f} s", rec.kind, "✔" if rec.verified else "",
                      rec.notes.replace("\n", " ")[:120]]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                if c == 0:
                    item.setData(Qt.UserRole, rec.id)
                self.table.setItem(r, c, item)
        self._update_tag_model()
        self._apply_filter()
        if current:
            self.select(current)

    def select(self, rec_id: str) -> None:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.UserRole) == rec_id:
                self.table.selectRow(r)
                return

    def focus_tags(self) -> None:
        self.tags.setFocus()
        self.tags.selectAll()

    def _current(self) -> EventRecord | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        rec_id = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
        return next((r for r in self._records if r.id == rec_id), None)

    def _with_current(self, fn) -> None:
        rec = self._current()
        if rec is not None:
            fn(rec)

    def _set_editor_enabled(self, on: bool) -> None:
        for w in (self.title, self.tags, self.notes, self.verified, self.save_btn):
            w.setEnabled(on)

    def _load_selected(self) -> None:
        rec = self._current()
        self._set_editor_enabled(rec is not None)
        if rec is None:
            return
        self.title.setText(rec.title)
        self.tags.setText(", ".join(rec.tags))
        self.notes.setPlainText(rec.notes)
        self.verified.setChecked(rec.verified)
        when = rec.wall_start.replace("T", " ") if rec.wall_start else f"t = {rec.t0:.2f} s"
        dets = ", ".join(d.get("text", "") for d in rec.detections[:5])
        self.info.setText(f"{rec.kind} · {when} · {rec.duration:.2f} s · {rec.source}"
                          + (f"<br>Detections: {dets}" if dets else ""))

    def _save(self) -> None:
        rec = self._current()
        if rec is None or self.store is None:
            return
        rec.title = self.title.text().strip()
        rec.tags = [t.strip() for t in self.tags.text().split(",") if t.strip()]
        rec.notes = self.notes.toPlainText().strip()
        rec.verified = self.verified.isChecked()
        self.store.save(rec)
        self.refresh(rec.id)

    def _delete(self) -> None:
        rec = self._current()
        if rec is None or self.store is None:
            return
        if QMessageBox.question(self, "Delete event",
                                f"Delete {rec.id} and its clip?") == QMessageBox.Yes:
            self.store.delete(rec)
            self.refresh()

    def _apply_filter(self) -> None:
        needle = self.search.text().strip().lower()
        for r in range(self.table.rowCount()):
            rec = self._records[r] if r < len(self._records) else None
            text = (" ".join([rec.title, rec.notes, *rec.tags]).lower() if rec else "")
            self.table.setRowHidden(r, bool(needle) and needle not in text)

    def _export(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        if self.store is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export events", "events.csv", "CSV (*.csv)")
        if path:
            n = self.store.export_csv(path)
            QMessageBox.information(self, "Export", f"Wrote {n} events to {Path(path).name}.")

    def _open_root(self) -> None:
        if self.store is not None:
            self.store.root.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.root)))
