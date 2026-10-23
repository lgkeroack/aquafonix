"""Qt pieces of screen monitoring: mapping between Qt's logical coordinates
and native pixels, the snipping-style region selector, and the outline that
marks the watched region on screen."""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QObject, QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QGuiApplication, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from aquafonix.screen.capture import Region, monitor_rects

log = logging.getLogger(__name__)

NativeRect = tuple[int, int, int, int]


def native_screen_rect(screen, monitors: list[NativeRect] | None = None,
                       size: tuple[int, int] | None = None) -> NativeRect:
    """Native-pixel rectangle of a QScreen.

    Qt 6 keeps each screen's top-left at its native position and scales only
    the size, so the native origin equals ``geometry().topLeft()``.  When the
    OS monitor list is available we snap to the matching entry to be safe.
    """
    g = screen.geometry()
    if sys.platform == "darwin":  # mss works in points on macOS, like Qt
        return g.x(), g.y(), g.width(), g.height()
    dpr = screen.devicePixelRatio()
    w, h = size or (round(g.width() * dpr), round(g.height() * dpr))
    guess = (g.x(), g.y(), w, h)
    if monitors:
        if guess in monitors:
            return guess
        same = [m for m in monitors if (m[2], m[3]) == (w, h)]
        if same:
            def dist(m):
                return min(abs(m[0] - g.x()) + abs(m[1] - g.y()),
                           abs(m[0] - g.x() * dpr) + abs(m[1] - g.y() * dpr))
            return min(same, key=dist)
    return guess


def local_to_region(local: QRect, widget_size: tuple[int, int], native: NativeRect) -> Region:
    """Widget-local logical rect on a full-screen overlay -> native Region."""
    sx = native[2] / max(widget_size[0], 1)
    sy = native[3] / max(widget_size[1], 1)
    left = native[0] + round(local.x() * sx)
    top = native[1] + round(local.y() * sy)
    right = native[0] + round((local.x() + local.width()) * sx)
    bottom = native[1] + round((local.y() + local.height()) * sy)
    return Region(left, top, max(1, right - left), max(1, bottom - top))


def region_to_logical(region: Region, monitors: list[NativeRect] | None = None) -> QRect:
    """Native Region -> global logical QRect (for placing windows over it)."""
    cx = region.left + region.width / 2
    cy = region.top + region.height / 2
    best = None
    for screen in QGuiApplication.screens():
        nat = native_screen_rect(screen, monitors)
        inside = nat[0] <= cx < nat[0] + nat[2] and nat[1] <= cy < nat[1] + nat[3]
        if inside or best is None:
            best = (screen, nat)
            if inside:
                break
    screen, nat = best
    g = screen.geometry()
    sx = nat[2] / max(g.width(), 1)
    sy = nat[3] / max(g.height(), 1)
    x = g.x() + (region.left - nat[0]) / sx
    y = g.y() + (region.top - nat[1]) / sy
    return QRect(round(x), round(y), max(1, round(region.width / sx)),
                 max(1, round(region.height / sy)))


class _Overlay(QWidget):
    """Full-screen frozen screenshot of one monitor; drag to select."""

    def __init__(self, selector: "RegionSelector", screen, pixmap: QPixmap,
                 native: NativeRect) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._selector = selector
        self._pix = pixmap
        self._native = native
        self._origin: QPoint | None = None
        self._sel = QRect()
        self._hover = QPoint(-100, -100)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())

    def region(self) -> Region:
        return local_to_region(self._sel, (self.width(), self.height()), self._native)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.drawPixmap(self.rect(), self._pix)
        p.fillRect(self.rect(), QColor(0, 0, 0, 110))
        sx = self._pix.width() / max(self.width(), 1)
        sy = self._pix.height() / max(self.height(), 1)
        if not self._sel.isEmpty():
            s = self._sel
            p.drawPixmap(QRectF(s), self._pix,
                         QRectF(s.x() * sx, s.y() * sy, s.width() * sx, s.height() * sy))
            p.setPen(QPen(QColor(0, 200, 255), 1))
            p.drawRect(s.adjusted(0, 0, -1, -1))
            r = self.region()
            self._label(p, s.bottomLeft() + QPoint(0, 6), f"{r.width} × {r.height} px")
        else:
            p.setPen(QPen(QColor(0, 200, 255, 140), 1, Qt.DashLine))
            p.drawLine(0, self._hover.y(), self.width(), self._hover.y())
            p.drawLine(self._hover.x(), 0, self._hover.x(), self.height())
        if self.rect().contains(self._hover) or not self._sel.isEmpty():
            self._label(p, QPoint(self.width() // 2 - 210, 16),
                        "Drag a box around the graph   ·   Esc / right-click to cancel")

    def _label(self, p: QPainter, at: QPoint, text: str) -> None:
        font = QFont(self.font())
        font.setPointSizeF(max(9.0, font.pointSizeF()))
        p.setFont(font)
        rect = p.fontMetrics().boundingRect(text).adjusted(-6, -3, 6, 3)
        rect.moveTopLeft(at)
        p.fillRect(rect, QColor(20, 20, 20, 220))
        p.setPen(QColor(240, 240, 240))
        p.drawText(rect, Qt.AlignCenter, text)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self._selector.finish(None)
        elif event.button() == Qt.LeftButton:
            self._origin = event.position().toPoint()
            self._sel = QRect(self._origin, self._origin)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        self._hover = event.position().toPoint()
        if self._origin is not None:
            self._sel = QRect(self._origin, self._hover).normalized()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._origin is None:
            return
        self._sel = QRect(self._origin, event.position().toPoint()).normalized()
        self._origin = None
        if self._sel.width() >= 3 and self._sel.height() >= 3:
            self._selector.finish(self.region())
        else:
            self._sel = QRect()
            self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self._selector.finish(None)


class RegionSelector(QObject):
    """Shows a frozen screenshot on every monitor and lets the user drag a
    rectangle.  Emits ``finished(Region)`` or ``finished(None)``."""

    finished = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._overlays: list[_Overlay] = []

    def start(self) -> None:
        try:
            monitors = monitor_rects()
        except Exception:
            log.warning("could not list monitors via mss", exc_info=True)
            monitors = []
        cursor = QCursor.pos()
        for screen in QGuiApplication.screens():
            pix = screen.grabWindow(0)
            pix.setDevicePixelRatio(1.0)
            native = native_screen_rect(screen, monitors, (pix.width(), pix.height()))
            overlay = _Overlay(self, screen, pix, native)
            self._overlays.append(overlay)
            overlay.show()
            if screen.geometry().contains(cursor):
                overlay.raise_()
                overlay.activateWindow()
                overlay.setFocus()

    @property
    def overlays(self) -> list[_Overlay]:
        return list(self._overlays)

    def finish(self, region: Region | None) -> None:
        overlays, self._overlays = self._overlays, []
        for overlay in overlays:
            overlay.close()
        self.finished.emit(region)


class RegionOutline(QWidget):
    """Click-through dashed frame drawn just outside the watched region."""

    BORDER = 3

    def __init__(self) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.color = QColor(255, 150, 0)

    def set_region(self, region: Region) -> None:
        try:
            monitors = monitor_rects()
        except Exception:
            monitors = []
        b = self.BORDER
        self.setGeometry(region_to_logical(region, monitors).adjusted(-b, -b, b, b))
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(QPen(QColor(0, 0, 0, 200), 2))
        p.drawRect(r)
        p.setPen(QPen(self.color, 2, Qt.DashLine))
        p.drawRect(r)
