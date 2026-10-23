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

    def __init__(self, plot: pg.PlotWidget, boxes: bool, fy=None) -> None:
        self.plot = plot
        self.boxes = boxes
        self.fy = fy or (lambda f: f)
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
                y0, y1 = self.fy(det.f_lo), self.fy(det.f_hi)
                shape.setRect(QRectF(det.t0, y0, det.t1 - det.t0, max(y1 - y0, 1e-6)))
                pen = QPen(QColor(r, g, b, 230), pen_w)
                pen.setCosmetic(True)
                shape.setPen(pen)
                shape.setBrush(QColor(r, g, b, alpha // 3))
                label.setPos(det.t0, y1)
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


def colormap(name: str) -> pg.ColorMap:
    anchors = np.array(COLORMAPS.get(name, COLORMAPS["inferno"]), dtype=np.ubyte)
    return pg.ColorMap(np.linspace(0, 1, len(anchors)), anchors)


class SpectrogramView(pg.PlotWidget):
    """Scrolling spectrogram (time x frequency) with an interactive colour
    bar, linear or logarithmic frequency axis, detection boxes, the filter
    band and a time-frequency selection box."""

    levels_dragged = Signal(float, float)

    def __init__(self, parent=None) -> None:
        self.vb = SelectViewBox()
        super().__init__(parent, viewBox=self.vb)
        self.setLabel("left", "Frequency", units="Hz")
        self.setLabel("bottom", "Time", units="s")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.addItem(self.image)
        self.cbar = pg.ColorBarItem(values=(0, 1), colorMap=colormap("inferno"),
                                    interactive=True, width=14, label="dB")
        self.cbar.setImageItem(self.image, insert_in=self.getPlotItem())
        self.cbar.sigLevelsChangeFinished.connect(self._levels_dragged)
        self.manual_levels: tuple[float, float] | None = None
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
        self.cursor = pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen((255, 255, 255, 90), dash=[2, 4]))
        self.addItem(self.cursor, ignoreBounds=True)
        self.log_freq = False
        self.f_min_log = 10.0
        self.f_max = 1.0
        self._sel_hz: tuple[float, float] | None = None
        self._band_hz: tuple[float, float] | None = None
        self._img: tuple | None = None  # (data rows x frames, t0, t1, y0, y1)
        self.overlay = _DetectionOverlay(self, boxes=True, fy=self.fy)

    # -- frequency mapping (Hz <-> view y) -----------------------------------------
    def fy(self, f: float) -> float:
        if self.log_freq:
            return float(np.log10(max(f, self.f_min_log)))
        return float(f)

    def inv_fy(self, y: float) -> float:
        return float(10 ** y) if self.log_freq else float(y)

    def set_log_freq(self, on: bool, f_min: float = 10.0) -> None:
        if on == self.log_freq and f_min == self.f_min_log:
            return
        self.log_freq, self.f_min_log = on, max(f_min, 0.1)
        self.getPlotItem().setLogMode(False, on)
        self.f_max = -1.0  # force a y-range reset on the next image
        self.overlay.clear()
        if self._sel_hz is not None:
            r = self.sel_box.rect()
            self.set_selection(r.left(), r.right(), *self._sel_hz)
        if self._band_hz is not None:
            self.set_band(*self._band_hz)

    def set_colormap(self, name: str) -> None:
        self.cbar.setColorMap(colormap(name))

    def _levels_dragged(self, bar) -> None:
        lo, hi = bar.levels()
        self.manual_levels = (lo, hi)
        self.levels_dragged.emit(lo, hi)

    def set_image(self, t0: float, t1: float, freqs: np.ndarray, data: np.ndarray,
                  levels: tuple[float, float]) -> None:
        """``data`` is (bins, frames) in dB for bin frequencies ``freqs``."""
        if data.size == 0 or t1 <= t0 or len(freqs) < 2:
            return
        f_top = float(freqs[-1])
        if self.log_freq:
            lo_f = max(self.f_min_log, float(freqs[1]))
            rows = min(len(freqs), 600)
            grid = np.logspace(np.log10(lo_f), np.log10(f_top), rows)
            pos = np.interp(grid, freqs, np.arange(len(freqs)))
            i0 = np.floor(pos).astype(int)
            i1 = np.minimum(i0 + 1, len(freqs) - 1)
            w = (pos - i0)[:, None]
            data = data[i0] * (1 - w) + data[i1] * w
            y0, y1 = np.log10(lo_f), np.log10(f_top)
        else:
            half = (freqs[1] - freqs[0]) / 2
            y0, y1 = max(0.0, float(freqs[0]) - half), f_top + half
        self.image.setImage(data, autoLevels=False)
        self.image.setRect(QRectF(t0, y0, t1 - t0, y1 - y0))
        self.cbar.setLevels(self.manual_levels or levels)
        self._img = (data, t0, t1, y0, y1)
        if f_top != self.f_max:
            self.f_max = f_top
            self.setYRange(y0, y1, padding=0)

    def value_at(self, x: float, y: float) -> float | None:
        if self._img is None:
            return None
        data, t0, t1, y0, y1 = self._img
        if not (t0 <= x <= t1 and y0 <= y <= y1):
            return None
        col = min(int((x - t0) / (t1 - t0) * data.shape[1]), data.shape[1] - 1)
        row = min(int((y - y0) / (y1 - y0) * data.shape[0]), data.shape[0] - 1)
        return float(data[row, col])

    def set_band(self, lo: float | None, hi: float | None) -> None:
        if lo is None:
            self._band_hz = None
            self.band.setVisible(False)
            return
        self._band_hz = (lo, hi)
        self.band.setRegion((self.fy(lo), self.fy(hi)))
        self.band.setVisible(True)

    def set_selection(self, t0, t1, f0, f1) -> None:
        """Times in s, frequencies in Hz."""
        f0, f1 = sorted((f0, f1))
        self._sel_hz = (f0, f1)
        y0, y1 = self.fy(f0), self.fy(f1)
        self.sel_box.setRect(QRectF(min(t0, t1), y0, abs(t1 - t0), y1 - y0))
        self.sel_box.setVisible(True)

    def clear_selection(self) -> None:
        self._sel_hz = None
        self.sel_box.setVisible(False)

    def selected_band(self) -> tuple[float, float] | None:
        return self._sel_hz if self.sel_box.isVisible() else None

    def update_detections(self, detections, selected: int | None = None) -> None:
        (x0, x1), (_y0, y1) = self.viewRange()
        self.overlay.update(detections, (x0, x1), y1, selected)


class SpectrumPlot(pg.PlotWidget):
    """Power spectral density of the last integration interval, with the
    5th/50th/95th percentiles over the statistics window and the noise floor."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.addLegend(offset=(8, 8))
        self.setLabel("bottom", "Frequency", units="Hz")
        self.p95 = self.plot([], [], name="95th pct", pen=pg.mkPen((247, 37, 133, 170), width=1))
        self.p5 = self.plot([], [], name="5th pct", pen=pg.mkPen((76, 201, 240, 170), width=1))
        self.p50 = self.plot([], [], name="median", pen=pg.mkPen((230, 230, 230, 200), width=1,
                                                                   dash=[4, 3]))
        self.current = self.plot([], [], name="PSD (latest)", pen=pg.mkPen("#ffd166", width=1.5))
        self.floor_curve = self.plot([], [], name="noise floor",
                                     pen=pg.mkPen("#9a9a9a", width=1, dash=[5, 4]))

    MAX_POINTS = 1200

    def _reduce(self, freqs: np.ndarray, values: np.ndarray, keep_min: bool = False):
        """Thin a dense spectrum to ~MAX_POINTS (log-spaced buckets on a log
        axis), keeping each bucket's peak so tonal lines stay visible."""
        f, v = freqs[1:], values[1:]  # no DC (log axes can't show 0 Hz)
        n = len(f)
        if n <= self.MAX_POINTS:
            return f, v
        if self.getPlotItem().ctrl.logXCheck.isChecked():
            edges = np.unique(np.round(np.logspace(0, np.log10(n), self.MAX_POINTS)).astype(int)) - 1
        else:
            edges = np.linspace(0, n, self.MAX_POINTS + 1).astype(int)[:-1]
        edges = np.unique(np.clip(edges, 0, n - 1))
        red = np.minimum.reduceat if keep_min else np.maximum.reduceat
        return f[edges], red(np.nan_to_num(v, nan=-300.0), edges)

    def set_psd(self, freqs, current, pct: dict, floor_f=None, floor_db=None,
                units: str = "dB") -> None:
        self.setLabel("left", "PSD", units=units)
        if current is not None:
            self.current.setData(*self._reduce(freqs, current))
        for key, item in ((5, self.p5), (50, self.p50), (95, self.p95)):
            if key in pct:
                item.setData(*self._reduce(freqs, pct[key], keep_min=key == 5))
            else:
                item.setData([], [])
        if floor_f is not None:
            self.floor_curve.setData(*self._reduce(floor_f, floor_db))

    def clear_series(self) -> None:
        for item in (self.current, self.p5, self.p50, self.p95, self.floor_curve):
            item.setData([], [])

    def value_at(self, x: float, _y: float) -> str | None:
        data = self.current.getData()
        if data[0] is None or not len(data[0]):
            return None
        f = 10 ** x if self.getPlotItem().ctrl.logXCheck.isChecked() else x
        i = int(np.argmin(np.abs(data[0] - f)))
        return f"{data[0][i]:.1f} Hz · {data[1][i]:.1f} dB"


class TolPlot(pg.PlotWidget):
    """Third-octave band levels: bars for the last interval, with the median
    and the 5-95 % range over the statistics window."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.showGrid(x=False, y=True, alpha=0.25)
        self.setLabel("bottom", "Third-octave band (IEC 61260-1)")
        self.bars = pg.BarGraphItem(x=[], height=[], width=0.75, brush=pg.mkBrush("#4cc9f0"))
        self.addItem(self.bars)
        self.err = pg.ErrorBarItem(x=np.zeros(0), y=np.zeros(0), pen=pg.mkPen("#f72585"))
        self.addItem(self.err)
        self.median = pg.ScatterPlotItem(size=6, brush=pg.mkBrush("#ffffff"), pen=None)
        self.addItem(self.median)
        self.labels: list[str] = []
        self.values = np.zeros(0)

    def set_levels(self, bands, current: np.ndarray, pct: dict | None, chosen=(),
                   resolved=None, units: str = "dB") -> None:
        n = len(bands)
        if not n or current is None or not len(current):
            return
        x = np.arange(n, dtype=float)
        finite = current[np.isfinite(current)]
        base = (np.min(finite) - 10) if len(finite) else 0.0
        brushes = []
        for i, b in enumerate(bands):
            if resolved is not None and not resolved[i]:
                brushes.append(pg.mkBrush(110, 110, 110))
            elif b.label in chosen:
                brushes.append(pg.mkBrush("#ffd166"))
            else:
                brushes.append(pg.mkBrush("#4cc9f0"))
        self.bars.setOpts(x=x, height=np.nan_to_num(current - base), y0=base, width=0.75,
                          brushes=brushes)
        if pct and all(k in pct for k in (5, 50, 95)):
            self.err.setData(x=x, y=pct[50], top=pct[95] - pct[50], bottom=pct[50] - pct[5],
                             beam=0.3)
            self.median.setData(x=x, y=pct[50])
        self.labels = [b.label for b in bands]
        self.values = current
        step = max(1, n // 12)
        self.getAxis("bottom").setTicks([[(i, bands[i].label) for i in range(0, n, step)],
                                         [(i, "") for i in range(n)]])
        self.setLabel("left", "Band level", units=units)
        self.setXRange(-0.6, n - 0.4, padding=0)

    def value_at(self, x: float, _y: float) -> str | None:
        i = int(round(x))
        if 0 <= i < len(self.labels):
            return f"{self.labels[i]} band · {self.values[i]:.1f} dB"
        return None


class LevelsPlot(pg.PlotWidget):
    """Levels per integration interval: SPL, Lpk, noise floor, filter band
    and the chosen third-octave bands."""

    FIXED = [("SPL (L_p,rms)", "#4cc9f0", None), ("L_pk", "#f72585", [2, 3]),
             ("Noise floor", "#9a9a9a", [5, 4]), ("Filter band", "#ffd166", None)]
    BAND_COLORS = ["#b5e48c", "#c77dff", "#f8961e", "#06d6a0", "#ef476f", "#90e0ef"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.legend = self.addLegend(offset=(8, 8))
        self.setLabel("bottom", "Time", units="s")
        self.fixed = []
        for name, color, dash in self.FIXED:
            pen = pg.mkPen(color, width=1.5, dash=dash) if dash else pg.mkPen(color, width=1.5)
            self.fixed.append(self.plot([], [], name=name, pen=pen))
        self.band_curves: dict[str, pg.PlotDataItem] = {}
        self.alarm = pg.InfiniteLine(angle=0, movable=False,
                                     pen=pg.mkPen("#ff5050", width=1, dash=[2, 2]),
                                     label="alarm", labelOpts={"color": "#ff5050",
                                                               "position": 0.02})
        self.alarm.setVisible(False)
        self.addItem(self.alarm, ignoreBounds=True)

    def set_data(self, t, intervals, offset: float, units: str, show_band: bool,
                 show_peak: bool, bands: dict[str, np.ndarray]) -> None:
        """``intervals`` columns: SPL, Lpk, SEL, floor, band (raw dB)."""
        self.setLabel("left", "Level", units=units)
        cols = {0: 0, 1: 1, 2: 3, 3: 4}
        for i, curve in enumerate(self.fixed):
            visible = not ((i == 1 and not show_peak) or (i == 3 and not show_band))
            if visible and len(t):
                curve.setData(t, intervals[:, cols[i]] + offset, connect="finite")
            else:
                curve.setData([], [])
        for name in list(self.band_curves):
            if name not in bands:
                self.removeItem(self.band_curves.pop(name))
                self.legend.removeItem(name)
        for i, (name, values) in enumerate(bands.items()):
            curve = self.band_curves.get(name)
            if curve is None:
                curve = self.plot([], [], name=f"⅓-oct {name}",
                                  pen=pg.mkPen(self.BAND_COLORS[i % len(self.BAND_COLORS)],
                                               width=1.2))
                self.band_curves[name] = curve
            curve.setData(t, values + offset, connect="finite")

    def set_alarm(self, level: float | None) -> None:
        self.alarm.setVisible(level is not None)
        if level is not None:
            self.alarm.setValue(level)


class RecordingOverviewView(pg.GraphicsLayoutWidget):
    """Whole-recording navigator: long-term spectrogram on top, SPL below.
    Click or drag to jump; shows the playback position and the visible window."""

    seek_requested = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setMaximumHeight(170)
        self.top = self.addPlot(row=0, col=0)
        self.bottom = self.addPlot(row=1, col=0)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 1)
        for plot in (self.top, self.bottom):
            plot.setMouseEnabled(x=False, y=False)
            plot.setMenuEnabled(False)
            plot.hideButtons()
        self.top.hideAxis("bottom")
        self.top.setLogMode(False, True)
        self.top.setLabel("left", "Hz")
        self.bottom.setLabel("left", "SPL")
        self.bottom.setXLink(self.top)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setLookupTable(lookup_table("inferno"))
        self.top.addItem(self.image)
        self.spl = self.bottom.plot([], [], pen=pg.mkPen("#4cc9f0", width=1))
        self.window = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(76, 201, 240, 50))
        self.top.addItem(self.window, ignoreBounds=True)
        self.pos_lines = [pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffea00"))
                          for _ in range(2)]
        self.top.addItem(self.pos_lines[0], ignoreBounds=True)
        self.bottom.addItem(self.pos_lines[1], ignoreBounds=True)
        self.title = pg.LabelItem(justify="left")
        self.addItem(self.title, row=2, col=0)
        self.duration = 0.0
        self.scene().sigMouseClicked.connect(self._clicked)

    def set_overview(self, ov, offset: float = 0.0, label: str = "") -> None:
        valid = ov.ltsa[:, :max(ov.done, 1)]
        finite = valid[np.isfinite(valid)]
        if len(finite):
            hi = float(np.percentile(finite, 99.5))
            self.image.setImage(np.nan_to_num(ov.ltsa, nan=float(finite.min())) + offset,
                                autoLevels=False, levels=(hi - 60 + offset, hi + offset))
        lf0, lf1 = np.log10(ov.freqs[0]), np.log10(ov.freqs[-1])
        self.duration = float(ov.source.duration)
        self.image.setRect(QRectF(0, lf0, self.duration, lf1 - lf0))
        self.top.setXRange(0, self.duration, padding=0)
        self.top.setYRange(lf0, lf1, padding=0)
        self.spl.setData(ov.times, ov.spl + offset, connect="finite")
        progress = "" if ov.complete else f"   (building overview {ov.done}/{ov.columns}…)"
        self.title.setText(f"<span style='color:#9aa'>{label}{progress}</span>")

    def set_position(self, t: float, window: tuple[float, float] | None) -> None:
        for line in self.pos_lines:
            line.setValue(t)
        if window is not None:
            self.window.setRegion(window)

    def _clicked(self, event) -> None:
        for plot in (self.top, self.bottom):
            if plot.sceneBoundingRect().contains(event.scenePos()):
                t = plot.vb.mapSceneToView(event.scenePos()).x()
                if 0 <= t <= self.duration:
                    self.seek_requested.emit(float(t))
                return


class PreviewView(pg.GraphicsLayoutWidget):
    """The captured region with the detected lines and calibration lines."""

    pixel_clicked = Signal(int, int)  # column, row
    line_moved = Signal(str, float)  # "ya" | "yb" | "xa" | "xb", pixel position
    rect_picked = Signal(float, float, float, float)  # x0, y0, x1, y1 in image pixels

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.vb = SelectViewBox(lockAspect=True, invertY=True)
        self.addItem(self.vb)
        self.vb.setMenuEnabled(False)
        self.pick_rect = QGraphicsRectItem()
        pen = QPen(QColor(80, 200, 255), 1, Qt.DashLine)
        pen.setCosmetic(True)
        self.pick_rect.setPen(pen)
        self.pick_rect.setZValue(4)
        self.pick_rect.setVisible(False)
        self.vb.addItem(self.pick_rect, ignoreBounds=True)
        self.vb.sigSelect.connect(self._dragging)
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

    def set_rect_pick(self, on: bool) -> None:
        """Drag a rectangle (e.g. over a colour bar) instead of panning."""
        self.vb.select_mode = on
        if not on:
            self.pick_rect.setVisible(False)

    def _dragging(self, x0, x1, y0, y1, finished) -> None:
        self.pick_rect.setRect(QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))
        self.pick_rect.setVisible(True)
        if finished:
            self.rect_picked.emit(x0, y0, x1, y1)

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
