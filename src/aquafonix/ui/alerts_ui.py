"""Alert rules editor, the alert log, pop-up notices and the data-outputs
dialog."""

from __future__ import annotations

import datetime as dt
import sys

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from aquafonix.alerts import DETECTION_KINDS, KINDS, Alert, AlertRule
from aquafonix.outputs import OutputSettings
from aquafonix.ui.widgets import spin


def beep() -> None:
    if sys.platform == "win32":
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            return
        except Exception:
            pass
    QApplication.beep()


class RuleEditor(QDialog):
    def __init__(self, parent, rule: AlertRule, labels: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Alert rule")
        f = QFormLayout(self)
        self.name = QLineEdit(rule.name)
        f.addRow("Name", self.name)
        self.kind = QComboBox()
        for k, v in KINDS.items():
            self.kind.addItem(v, k)
        self.kind.setCurrentIndex(max(0, self.kind.findData(rule.kind)))
        f.addRow("Alert on", self.kind)
        self.labels = QLineEdit(", ".join(rule.labels))
        self.labels.setPlaceholderText("any signature")
        self.labels.setToolTip("Signature names, comma-separated. Known: " + ", ".join(labels))
        f.addRow("Signatures", self.labels)
        self.min_score = spin(0.0, 1.0, rule.min_score, decimals=2, step=0.05)
        f.addRow("Min score", self.min_score)
        self.min_snr = spin(0.0, 100.0, rule.min_snr_db, decimals=1, suffix=" dB")
        f.addRow("Min SNR", self.min_snr)
        self.level = spin(-300.0, 300.0, rule.level_db, decimals=1, suffix=" dB")
        f.addRow("Threshold", self.level)
        srow = QHBoxLayout()
        self.sector_on = QCheckBox("Only from bearings")
        self.sector_lo = spin(0.0, 180.0, rule.sector[0] if rule.sector else 0.0, decimals=0,
                              suffix="°")
        self.sector_hi = spin(0.0, 180.0, rule.sector[1] if rule.sector else 180.0, decimals=0,
                              suffix="°")
        self.sector_on.setChecked(rule.sector is not None)
        for w in (self.sector_on, self.sector_lo, QLabel("–"), self.sector_hi):
            srow.addWidget(w)
        f.addRow("Sector", srow)
        self.cooldown = spin(0.0, 86_400.0, rule.cooldown_s, decimals=1, suffix=" s")
        self.cooldown.setToolTip("Quiet time after an alert before this rule can fire again")
        f.addRow("Cooldown", self.cooldown)
        arow = QHBoxLayout()
        self.sound = QCheckBox("Sound")
        self.popup = QCheckBox("Pop-up")
        self.save = QCheckBox("Save as event")
        self.save.setToolTip("Keep a clip from 5 s before to 5 s after, tagged with the rule")
        for w, on in ((self.sound, rule.sound), (self.popup, rule.popup),
                      (self.save, rule.save_event)):
            w.setChecked(on)
            arow.addWidget(w)
        f.addRow("Then", arow)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self._rows = {"labels": self.labels, "score": self.min_score, "snr": self.min_snr,
                      "level": self.level, "sector": srow}
        self._form = f
        self.kind.currentIndexChanged.connect(self._update)
        self.sector_on.toggled.connect(self._update)
        self._update()

    def _update(self) -> None:
        k = self.kind.currentData()
        det = k in DETECTION_KINDS
        visible = {"labels": k == "signature", "score": k == "signature", "snr": det,
                   "level": k in ("level", "floor"), "sector": det}
        for key, w in self._rows.items():
            self._form.setRowVisible(w, visible[key])
        for w in (self.sector_lo, self.sector_hi):
            w.setEnabled(self.sector_on.isChecked())

    def rule(self, enabled: bool = True) -> AlertRule:
        labels = tuple(s.strip() for s in self.labels.text().split(",") if s.strip())
        sector = (tuple(sorted((self.sector_lo.value(), self.sector_hi.value())))
                  if self.sector_on.isChecked() else None)
        return AlertRule(self.name.text().strip() or "Alert", enabled, self.kind.currentData(),
                         labels, self.min_score.value(), self.min_snr.value(),
                         self.level.value(), sector, self.cooldown.value(),
                         self.sound.isChecked(), self.popup.isChecked(), self.save.isChecked())


class AlertsDialog(QDialog):
    COLS = ["On", "Name", "When", "Sound", "Pop-up", "Save"]

    def __init__(self, parent, rules: list[AlertRule], labels: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Alerts")
        self.resize(720, 360)
        self.rules = [AlertRule.from_dict(r.to_dict()) for r in rules]
        self.labels = labels
        lay = QVBoxLayout(self)
        hint = QLabel("Alerts call your attention to chosen sounds or conditions. They are "
                      "listed in the Alerts tab and sent to the data outputs, if any.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._edit(r))
        self.table.itemChanged.connect(self._toggled)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, slot in (("Add…", self._add), ("Edit…", lambda: self._edit()),
                           ("Remove", self._remove)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        row.addWidget(bb)
        lay.addLayout(row)
        self._fill()

    def _fill(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.rules))
        for r, rule in enumerate(self.rules):
            on = QTableWidgetItem()
            on.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            on.setCheckState(Qt.Checked if rule.enabled else Qt.Unchecked)
            self.table.setItem(r, 0, on)
            for c, text in enumerate((rule.name, rule.describe(), "✔" if rule.sound else "",
                                      "✔" if rule.popup else "", "✔" if rule.save_event else ""),
                                     1):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.resizeColumnToContents(1)
        self.table.blockSignals(False)

    def _toggled(self, item: QTableWidgetItem) -> None:
        if item.column() == 0 and item.row() < len(self.rules):
            self.rules[item.row()].enabled = item.checkState() == Qt.Checked

    def _add(self) -> None:
        dlg = RuleEditor(self, AlertRule(), self.labels)
        if dlg.exec():
            self.rules.append(dlg.rule())
            self._fill()

    def _edit(self, row: int | None = None) -> None:
        row = self.table.currentRow() if row is None else row
        if not 0 <= row < len(self.rules):
            return
        dlg = RuleEditor(self, self.rules[row], self.labels)
        if dlg.exec():
            self.rules[row] = dlg.rule(self.rules[row].enabled)
            self._fill()

    def _remove(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.rules):
            del self.rules[row]
            self._fill()


class AlertsPanel(QWidget):
    """Log of alerts raised (newest first)."""

    show_detection = Signal(int)
    COLS = ["Time", "Rule", "Alert"]
    MAX_ROWS = 2000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.show_btn = QPushButton("⤢ Show")
        self.show_btn.clicked.connect(self._show)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        self.rules_btn = QPushButton("Rules…")
        self.mute = QCheckBox("Mute sounds")
        for w in (self.show_btn, self.clear_btn, self.rules_btn, self.mute):
            bar.addWidget(w)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._show())
        lay.addWidget(self.table, 1)
        self.alerts: list[Alert] = []

    def add(self, alert: Alert, when: str) -> None:
        self.alerts.insert(0, alert)
        self.table.insertRow(0)
        for c, text in enumerate((when, alert.rule.name, alert.message)):
            self.table.setItem(0, c, QTableWidgetItem(text))
        if len(self.alerts) > self.MAX_ROWS:
            self.alerts.pop()
            self.table.removeRow(self.table.rowCount() - 1)

    def clear(self) -> None:
        self.alerts = []
        self.table.setRowCount(0)

    def _show(self) -> None:
        r = self.table.currentRow()
        if 0 <= r < len(self.alerts) and self.alerts[r].detection_id is not None:
            self.show_detection.emit(self.alerts[r].detection_id)


class Toast(QWidget):
    """A small notice in the corner of the main window; click to dismiss."""

    clicked = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet("QWidget{background:#3a1d1d;border:1px solid #ff6b5a;}"
                           "QLabel{border:none;color:#fff;padding:2px;}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight:bold;")
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setMaximumWidth(420)
        lay.addWidget(self.title)
        lay.addWidget(self.body)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def notify(self, title: str, text: str, seconds: float = 6.0) -> None:
        self.title.setText("🔔 " + title)
        self.body.setText(text)
        self.adjustSize()
        host = self.parentWidget()
        corner = host.mapToGlobal(QPoint(host.width(), host.height()))
        self.move(corner - QPoint(self.width() + 24, self.height() + 48))
        self.show()
        self.raise_()
        self.timer.start(int(seconds * 1000))

    def mousePressEvent(self, _event) -> None:
        self.hide()
        self.clicked.emit()


def alert_time_text(alert: Alert, wall: float | None) -> str:
    if wall:
        return dt.datetime.fromtimestamp(wall).strftime("%Y-%m-%d %H:%M:%S")
    return f"{alert.stream_t:.2f} s"


class OutputsDialog(QDialog):
    """Where to send detections, alerts and levels as JSON."""

    def __init__(self, parent, settings: OutputSettings, status=None, test=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Data outputs")
        self._status, self._test = status, test
        lay = QVBoxLayout(self)
        hint = QLabel("Stream detections, alerts and levels as JSON to other software "
                      "(plotters, loggers, vessel-tracking or automation systems). Each "
                      "message is one JSON object with a \"type\" field.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        f = QFormLayout()
        s = settings
        self.udp = QCheckBox("UDP datagrams to")
        self.udp.setChecked(s.udp)
        self.udp_host = QLineEdit(s.udp_host)
        self.udp_port = spin(1, 65535, s.udp_port)
        row = QHBoxLayout()
        for w in (self.udp_host, QLabel("port"), self.udp_port):
            row.addWidget(w)
        f.addRow(self.udp, row)
        self.tcp = QCheckBox("TCP server on port")
        self.tcp.setChecked(s.tcp)
        self.tcp_port = spin(1, 65535, s.tcp_port)
        f.addRow(self.tcp, self.tcp_port)
        self.webhook = QCheckBox("HTTP POST (webhook) to")
        self.webhook.setChecked(s.webhook)
        self.webhook_url = QLineEdit(s.webhook_url)
        self.webhook_url.setPlaceholderText("https://example.org/hook")
        f.addRow(self.webhook, self.webhook_url)
        self.file = QCheckBox("Append to JSON-lines file")
        self.file.setChecked(s.file)
        frow = QHBoxLayout()
        self.file_path = QLineEdit(s.file_path)
        browse = QPushButton("…")
        browse.clicked.connect(self._browse)
        frow.addWidget(self.file_path, 1)
        frow.addWidget(browse)
        f.addRow(self.file, frow)
        what = QHBoxLayout()
        self.send_det = QCheckBox("Detections")
        self.send_alerts = QCheckBox("Alerts")
        self.send_levels = QCheckBox("Levels (each integration interval)")
        for w, on in ((self.send_det, s.detections), (self.send_alerts, s.alerts),
                      (self.send_levels, s.levels)):
            w.setChecked(on)
            what.addWidget(w)
        f.addRow("Send", what)
        lay.addLayout(f)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#9aa4ae;")
        lay.addWidget(self.status)
        row = QHBoxLayout()
        test_btn = QPushButton("Apply && send test message")
        test_btn.clicked.connect(self._send_test)
        test_btn.setEnabled(test is not None)
        row.addWidget(test_btn)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        row.addWidget(bb)
        lay.addLayout(row)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(500)
        self._refresh()

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "JSON-lines file",
                                              self.file_path.text() or "aquafonix.jsonl",
                                              "JSON lines (*.jsonl *.ndjson);;All files (*)")
        if path:
            self.file_path.setText(path)
            self.file.setChecked(True)

    def settings(self) -> OutputSettings:
        return OutputSettings(self.udp.isChecked(), self.udp_host.text().strip() or "127.0.0.1",
                              self.udp_port.value(), self.tcp.isChecked(), self.tcp_port.value(),
                              self.webhook.isChecked(), self.webhook_url.text().strip(),
                              self.file.isChecked(), self.file_path.text().strip(),
                              self.send_det.isChecked(), self.send_alerts.isChecked(),
                              self.send_levels.isChecked())

    def _send_test(self) -> None:
        if self._test is not None:
            self._test(self.settings())

    def _refresh(self) -> None:
        if self._status is not None:
            self.status.setText(self._status())
