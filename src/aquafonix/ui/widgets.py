"""Small widget helpers: named controls, settings persistence, colour button."""

from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractButton, QColorDialog, QComboBox, QDoubleSpinBox,
                               QLineEdit, QPushButton, QSlider, QSpinBox, QWidget)

TRUE_STRINGS = ("1", "true", "yes", "on")


def named(widget: QWidget, name: str) -> QWidget:
    """Give a control an objectName so its value is saved, restored and
    settable from the command line (``--set name=value``)."""
    widget.setObjectName(name)
    return widget


def spin(lo: float, hi: float, value: float, decimals: int = 0, step: float | None = None,
         suffix: str = "") -> QSpinBox | QDoubleSpinBox:
    box = QDoubleSpinBox() if decimals else QSpinBox()
    if decimals:
        box.setDecimals(decimals)
    box.setRange(lo, hi)
    box.setValue(value)
    if step:
        box.setSingleStep(step)
    if suffix:
        box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    box.setMinimumWidth(64)  # wide ranges would otherwise force a very wide panel
    return box


def combo(items, current=None) -> QComboBox:
    """``items`` is a dict {data: label} or a list of labels."""
    box = QComboBox()
    if isinstance(items, dict):
        for data, label in items.items():
            box.addItem(label, data)
        if current is not None:
            box.setCurrentIndex(max(0, box.findData(current)))
    else:
        box.addItems([str(i) for i in items])
        if current is not None:
            box.setCurrentText(str(current))
    return box


class ColorButton(QPushButton):
    colorChanged = Signal(tuple)

    def __init__(self, color=(255, 0, 0), parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(64)
        self._color = tuple(color)
        self.clicked.connect(self._choose)
        self._paint()

    def color(self) -> tuple[int, int, int]:
        return self._color

    def setColor(self, color) -> None:
        color = tuple(int(c) for c in color[:3])
        if color != self._color:
            self._color = color
            self._paint()
            self.colorChanged.emit(color)

    def _paint(self) -> None:
        r, g, b = self._color
        text = "#000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#fff"
        self.setStyleSheet(f"background:rgb({r},{g},{b}); color:{text}; border:1px solid #888;")
        self.setText(f"#{r:02x}{g:02x}{b:02x}")

    def _choose(self) -> None:
        c = QColorDialog.getColor(QColor(*self._color), self, "Trace colour")
        if c.isValid():
            self.setColor((c.red(), c.green(), c.blue()))


def widget_value(w: QWidget):
    if isinstance(w, ColorButton):
        return "#%02x%02x%02x" % w.color()
    if isinstance(w, QAbstractButton):
        return w.isChecked() if w.isCheckable() else None
    if isinstance(w, (QSpinBox, QDoubleSpinBox, QSlider)):
        return w.value()
    if isinstance(w, QComboBox):
        if w.isEditable() or w.currentData() is None:
            return w.currentText()
        return w.currentData()
    if isinstance(w, QLineEdit):
        return w.text()
    return None


def set_widget_value(w: QWidget, value) -> bool:
    """Set a control from a (possibly string) value. Returns False if the
    widget type isn't supported."""
    if isinstance(w, ColorButton):
        c = QColor(str(value))
        if c.isValid():
            w.setColor((c.red(), c.green(), c.blue()))
        return True
    if isinstance(w, QAbstractButton):
        if w.isCheckable():
            w.setChecked(str(value).strip().lower() in TRUE_STRINGS)
            return True
        return False
    if isinstance(w, (QSpinBox, QSlider)):
        w.setValue(int(float(value)))
        return True
    if isinstance(w, QDoubleSpinBox):
        w.setValue(float(value))
        return True
    if isinstance(w, QComboBox):
        value = str(value)
        idx = w.findData(value)
        if idx < 0:
            idx = w.findText(value)
        if idx < 0:  # allow prefix match, e.g. "COM3" -> "COM3 — USB Serial"
            idx = next((i for i in range(w.count()) if w.itemText(i).startswith(value)), -1)
        if idx >= 0:
            w.setCurrentIndex(idx)
        elif w.isEditable():
            w.setEditText(value)
        return True
    if isinstance(w, QLineEdit):
        w.setText(str(value))
        return True
    return False


def _named_controls(root: QWidget):
    for w in root.findChildren(QWidget):
        name = w.objectName()
        if name and not name.startswith("qt_") and widget_value(w) is not None:
            yield name, w


def save_state(root: QWidget, settings: QSettings) -> None:
    for name, w in _named_controls(root):
        settings.setValue(name, widget_value(w))


def restore_state(root: QWidget, settings: QSettings) -> None:
    for name, w in _named_controls(root):
        if settings.contains(name):
            try:
                set_widget_value(w, settings.value(name))
            except (TypeError, ValueError):
                pass
