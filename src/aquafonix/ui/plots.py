"""pyqtgraph-based views: waveform, overview, spectrogram, spectrum, levels
and the screen-capture preview."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsRectItem

PALETTE = ["#4cc9f0", "#f72585", "#b5e48c", "#ffd166", "#f8961e", "#c77dff", "#06d6a0",
           "#ef476f", "#90e0ef", "#ffafcc", "#e9ff70", "#a0c4ff"]

STATUS_COLORS = {
    "suspected": (255, 170, 0),
    "unidentified": (200, 200, 200),
    "confirmed": (60, 220, 120),
    "corrected": (120, 180, 255),
    "rejected": (255, 80, 80),
}

COLORMAPS = {
    "inferno": [(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66),
                (245, 125, 21), (250, 193, 39), (252, 255, 164)],
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142),
                (31, 158, 137), (53, 183, 121), (110, 206, 88), (181, 222, 43), (253, 231, 37)],
    "ocean": [(0, 0, 0), (0, 20, 60), (0, 60, 120), (0, 120, 160), (40, 180, 180),
              (160, 230, 200), (255, 255, 255)],
    "grey": [(0, 0, 0), (255, 255, 255)],
}


def channel_color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def lookup_table(name: str, n: int = 256) -> np.ndarray:
    anchors = np.array(COLORMAPS.get(name, COLORMAPS["inferno"]), dtype=float)
    pos = np.linspace(0, 1, len(anchors))
    t = np.linspace(0, 1, n)
    return np.column_stack([np.interp(t, pos, anchors[:, c]) for c in range(3)]).astype(np.uint8)


class SelectViewBox(pg.ViewBox):
    """ViewBox where left-drag selects (in select mode, or with Shift held)
    instead of panning."""

    sigSelect = Signal(float, float, float, float, bool)  # x0, x1, y0, y1, finished

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.select_mode = False

    def mouseDragEvent(self, ev, axis=None) -> None:
        selecting = self.select_mode or bool(ev.modifiers() & Qt.ShiftModifier)
        if selecting and ev.button() == Qt.LeftButton and axis is None:
            ev.accept()
            p0 = self.mapSceneToView(ev.buttonDownScenePos())
            p1 = self.mapSceneToView(ev.scenePos())
            self.sigSelect.emit(p0.x(), p1.x(), p0.y(), p1.y(), ev.isFinish())
        else:
            super().mouseDragEvent(ev, axis)


class _MultiCurvePlot(pg.PlotWidget):
    def __init__(self, parent=None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.legend = self.addLegend(offset=(8, 8))
        self.curves: dict[str, pg.PlotDataItem] = {}

    def curve(self, name: str) -> pg.PlotDataItem:
        item = self.curves.get(name)
        if item is None:
            item = self.plot([], [], name=name, pen=pg.mkPen(channel_color(len(self.curves)),
                                                                 width=1.5))
            self.curves[name] = item
        return item

    def set_series(self, x: np.ndarray, y: np.ndarray, names: list[str]) -> None:
        for i, name in enumerate(names):
            self.curve(name).setData(x, y[:, i], connect="finite")
        for name in list(self.curves):
            if name not in names:
                self.curves[name].setData([], [])

    def clear_series(self) -> None:
        for item in self.curves.values():
            self.removeItem(item)
        self.legend.clear()
        self.curves.clear()


class _DetectionOverlay:
    """Shaded spans / boxes with labels for detections on a plot."""

    def __init__(self, plot: pg.PlotWidget, boxes: bool) -> None:
        self.plot = plot
        self.boxes = boxes
        self.items: dict[int, tuple] = {}

    def update(self, detections, x_range, y_top: float, selected: int | None = None) -> None:
        x0, x1 = x_range
        visible = {d.id: d for d in detections if d.t1 >= x0 and d.t0 <= x1}
        if len(visible) > 300:
            visible = dict(list(visible.items())[-300:])
        for det_id in list(self.items):
            if det_id not in visible:
                for item in self.items.pop(det_id):
                    self.plot.removeItem(item)
        for det in visible.values():
            r, g, b = STATUS_COLORS.get(det.status, (200, 200, 200))
            alpha = 90 if det.id == selected else 45
            pen_w = 2 if det.id == selected else 1
            items = self.items.get(det.id)
            if items is None:
                if self.boxes:
                    shape = QGraphicsRectItem()
                    shape.setZValue(5)
                    self.plot.addItem(shape, ignoreBounds=True)
                else:
                    shape = pg.LinearRegionItem(movable=False)
                    shape.setZValue(-5)
                    self.plot.addItem(shape, ignoreBounds=True)
                label = pg.TextItem(anchor=(0, 1) if self.boxes else (0, 0))
                label.setZValue(6)
                self.plot.addItem(label, ignoreBounds=True)
                items = self.items[det.id] = (shape, label)
            shape, label = items
            if self.boxes:
                shape.setRect(QRectF(det.t0, det.f_lo, det.t1 - det.t0, max(det.f_hi - det.f_lo, 1)))
                pen = QPen(QColor(r, g, b, 230), pen_w)
                pen.setCosmetic(True)
                shape.setPen(pen)
                shape.setBrush(QColor(r, g, b, alpha // 3))
                label.setPos(det.t0, det.f_hi)
            else:
                shape.setRegion((det.t0, det.t1))
                shape.setBrush(pg.mkBrush(r, g, b, alpha))
                for line in shape.lines:
                    line.setPen(pg.mkPen((r, g, b, 160), width=pen_w))
                label.setPos(det.t0, y_top)
            label.setText(det.text, color=(r, g, b))

    def clear(self) -> None:
        for items in self.items.values():
            for item in items:
                self.plot.removeItem(item)
        self.items.clear()


class TimePlot(_MultiCurvePlot):
    """The waveform: curves, a time selection, detection highlights,
    a trigger level line and a playback cursor."""

    def __init__(self, parent=None) -> None:
        self.vb = SelectViewBox()
        super().__init__(parent, viewBox=self.vb)
        self.setLabel("bottom", "Time", units="s")
        self.trigger_line = pg.InfiniteLine(angle=0, movable=True,
                                            pen=pg.mkPen("#ffffff", width=1, dash=[4, 4]))
        self.trigger_line.setVisible(False)
        self.addItem(self.trigger_line, ignoreBounds=True)
        self.selection = pg.LinearRegionItem(movable=True, brush=pg.mkBrush(80, 160, 255, 50))
        self.selection.setZValue(10)
        self.selection.setVisible(False)
        self.addItem(self.selection, ignoreBounds=True)
        self.playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffea00", width=2))
        self.playhead.setVisible(False)
        self.addItem(self.playhead, ignoreBounds=True)
        self.overlay = _DetectionOverlay(self, boxes=False)

    def selected_span(self) -> tuple[float, float] | None:
        if not self.selection.isVisible():
            return None
        a, b = self.selection.getRegion()
        return (a, b) if b > a else None

    def set_selection(self, x0: float, x1: float) -> None:
        self.selection.setRegion(sorted((x0, x1)))
        self.selection.setVisible(True)

    def clear_selection(self) -> None:
        self.selection.setVisible(False)

    def update_detections(self, detections, selected: int | None = None) -> None:
        (x0, x1), (_y0, y1) = self.viewRange()
        self.overlay.update(detections, (x0, x1), y1, selected)


class OverviewPlot(pg.PlotWidget):
    """The whole history, with the visible window as a draggable region."""

    region_dragged = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMaximumHeight(80)
        self.setMinimumHeight(50)
        self.hideAxis("left")
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.curve = self.plot([], [], pen=pg.mkPen("#7f8c9a", width=1))
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(76, 201, 240, 60))
        self.region.setZValue(10)
        self.addItem(self.region, ignoreBounds=True)
        self._updating = False
        self.region.sigRegionChanged.connect(self._dragged)

    def set_data(self, x: np.ndarray, y: np.ndarray) -> None:
        self.curve.setData(x, y, connect="finite")
        if len(x) > 1:
            self.setXRange(float(x[0]), float(x[-1]), padding=0)

    def set_region(self, x0: float, x1: float) -> None:
        self._updating = True
        self.region.setRegion((x0, x1))
        self._updating = False

    def _dragged(self) -> None:
        if not self._updating:
            self.region_dragged.emit(*self.region.getRegion())


class SpectrogramView(pg.PlotWidget):
    """Scrolling spectrogram (time x frequency) with detection boxes, the
    filter band and a time-frequency selection box."""

    def __init__(self, parent=None) -> None:
        self.vb = SelectViewBox()
        super().__init__(parent, viewBox=self.vb)
        self.setLabel("left", "Frequency", units="Hz")
        self.setLabel("bottom", "Time", units="s")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.addItem(self.image)
        self.set_colormap("inferno")
        self.band = pg.LinearRegionItem(orientation="horizontal", movable=False,
                                        brush=pg.mkBrush(255, 255, 255, 18))
        for line in self.band.lines:
            line.setPen(pg.mkPen((255, 255, 255, 120), dash=[3, 3]))
        self.band.setVisible(False)
        self.addItem(self.band, ignoreBounds=True)
        self.sel_box = QGraphicsRectItem()
        pen = QPen(QColor(80, 200, 255), 1, Qt.DashLine)
        pen.setCosmetic(True)
        self.sel_box.setPen(pen)
        self.sel_box.setBrush(QColor(80, 160, 255, 40))
        self.sel_box.setZValue(10)
        self.sel_box.setVisible(False)
        self.addItem(self.sel_box, ignoreBounds=True)
        self.overlay = _DetectionOverlay(self, boxes=True)
        self.f_max = 1.0

    def set_colormap(self, name: str) -> None:
        self.image.setLookupTable(lookup_table(name))

    def set_image(self, t0: float, t1: float, f_max: float, data: np.ndarray,
                  levels: tuple[float, float]) -> None:
        """``data`` is (bins, frames) in dB."""
        if data.size == 0 or t1 <= t0:
            return
        self.image.setImage(data, autoLevels=False, levels=levels)
        self.image.setRect(QRectF(t0, 0.0, t1 - t0, f_max))
        if f_max != self.f_max:
            self.f_max = f_max
            self.setYRange(0, f_max, padding=0)

    def set_band(self, lo: float | None, hi: float | None) -> None:
        if lo is None:
            self.band.setVisible(False)
            return
        self.band.setRegion((lo, hi))
        self.band.setVisible(True)

    def set_selection(self, t0, t1, f0, f1) -> None:
        self.sel_box.setRect(QRectF(min(t0, t1), min(f0, f1), abs(t1 - t0), abs(f1 - f0)))
        self.sel_box.setVisible(True)

    def clear_selection(self) -> None:
        self.sel_box.setVisible(False)

    def selected_band(self) -> tuple[float, float] | None:
        if not self.sel_box.isVisible():
            return None
        r = self.sel_box.rect()
        return max(0.0, r.top()), r.bottom()

    def update_detections(self, detections, selected: int | None = None) -> None:
        (x0, x1), (_y0, y1) = self.viewRange()
        self.overlay.update(detections, (x0, x1), y1, selected)


class SpectrumPlot(_MultiCurvePlot):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Level", units="dB")
        self.floor_curve = self.plot([], [], name="noise floor",
                                     pen=pg.mkPen("#bbbbbb", width=1, dash=[5, 4]))


class LevelsPlot(pg.PlotWidget):
    """Levels over time: broadband, noise floor, filter band, 63/125 Hz."""

    SERIES = [("Broadband", "#4cc9f0"), ("Noise floor", "#bbbbbb"), ("Filter band", "#ffd166"),
              ("63 Hz ⅓-oct", "#b5e48c"), ("125 Hz ⅓-oct", "#c77dff")]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.addLegend(offset=(8, 8))
        self.setLabel("bottom", "Time", units="s")
        self.curves = [self.plot([], [], name=n, pen=pg.mkPen(c, width=1.5))
                       for n, c in self.SERIES]
        self.curves[1].setPen(pg.mkPen("#bbbbbb", width=1.5, dash=[5, 4]))
        self.alarm = pg.InfiniteLine(angle=0, movable=False,
                                     pen=pg.mkPen("#ff5050", width=1, dash=[2, 2]),
                                     label="alarm", labelOpts={"color": "#ff5050",
                                                               "position": 0.02})
        self.alarm.setVisible(False)
        self.addItem(self.alarm, ignoreBounds=True)

    def set_data(self, t: np.ndarray, v: np.ndarray, offset: float, units: str,
                 show_band: bool) -> None:
        self.setLabel("left", "Level", units=units)
        for i, curve in enumerate(self.curves):
            if i < v.shape[1] and (i != 2 or show_band):
                curve.setData(t, v[:, i] + offset, connect="finite")
            else:
                curve.setData([], [])

    def set_alarm(self, level: float | None) -> None:
        self.alarm.setVisible(level is not None)
        if level is not None:
            self.alarm.setValue(level)


class PreviewView(pg.GraphicsLayoutWidget):
    """The captured region with the detected lines and calibration lines."""

    pixel_clicked = Signal(int, int)  # column, row
    line_moved = Signal(str, float)  # "ya" | "yb" | "xa" | "xb", pixel position

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.vb = self.addViewBox(lockAspect=True, invertY=True)
        self.vb.setMenuEnabled(False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.mask = pg.ImageItem(axisOrder="row-major")
        self.mask.setZValue(1)
        for item in (self.image, self.mask):
            self.vb.addItem(item)
        self.traces: dict[str, pg.PlotDataItem] = {}
        self.lines: dict[str, pg.InfiniteLine] = {}
        for key, angle, color, text in (("ya", 0, "#ff4fd8", "A"), ("yb", 0, "#ff4fd8", "B"),
                                        ("xa", 90, "#3ae8ff", "A"), ("xb", 90, "#3ae8ff", "B")):
            line = pg.InfiniteLine(angle=angle, movable=True,
                                   pen=pg.mkPen(color, width=1, dash=[6, 4]),
                                   hoverPen=pg.mkPen(color, width=3),
                                   label=text, labelOpts={"color": color, "position": 0.04})
            line.setZValue(3)
            line.sigPositionChanged.connect(lambda ln, k=key: self.line_moved.emit(k, ln.value()))
            self.vb.addItem(line, ignoreBounds=True)
            self.lines[key] = line
        self.probe = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen("#ffffff", width=1, dash=[2, 3]))
        self.probe.setZValue(3)
        self.probe.setVisible(False)
        self.vb.addItem(self.probe, ignoreBounds=True)
        self._shape: tuple[int, int] | None = None
        self.scene().sigMouseClicked.connect(self._clicked)

    def set_frame(self, image: np.ndarray, result, show_mask: bool = False,
                  probe_px: float | None = None) -> None:
        self.image.setImage(image, autoLevels=False, levels=(0, 255))
        mask = result.mask if show_mask and result.traces else None
        if mask is not None:
            rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
            rgba[mask] = (255, 0, 255, 170)
            self.mask.setImage(rgba, autoLevels=False, levels=(0, 255))
        self.mask.setVisible(mask is not None)
        names = result.names
        for i, tr in enumerate(result.traces):
            item = self.traces.get(tr.name)
            if item is None:
                item = pg.PlotDataItem(pen=pg.mkPen(channel_color(i), width=2))
                item.setZValue(2)
                self.vb.addItem(item)
                self.traces[tr.name] = item
            item.setData(result.x_px, tr.y_px, connect="finite")
        for name in list(self.traces):
            if name not in names:
                self.vb.removeItem(self.traces.pop(name))
        self.probe.setVisible(probe_px is not None)
        if probe_px is not None:
            self.probe.setValue(probe_px)
        shape = image.shape[:2]
        if shape != self._shape:
            self._shape = shape
            self.vb.setRange(xRange=(0, shape[1]), yRange=(0, shape[0]), padding=0.02)

    def set_line(self, key: str, px: float) -> None:
        line = self.lines[key]
        line.blockSignals(True)
        line.setValue(px)
        line.blockSignals(False)

    def _clicked(self, event) -> None:
        if self._shape is None:
            return
        pos = self.vb.mapSceneToView(event.scenePos())
        col, row = int(np.floor(pos.x())), int(np.floor(pos.y()))
        if 0 <= row < self._shape[0] and 0 <= col < self._shape[1]:
            self.pixel_clicked.emit(col, row)
