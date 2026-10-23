"""pyqtgraph-based views: time plot, spectrum and screen-capture preview."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal

PALETTE = ["#4cc9f0", "#f72585", "#b5e48c", "#ffd166", "#f8961e", "#c77dff", "#06d6a0",
           "#ef476f", "#90e0ef", "#ffafcc", "#e9ff70", "#a0c4ff"]


def channel_color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


class _MultiCurvePlot(pg.PlotWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
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


class TimePlot(_MultiCurvePlot):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setLabel("bottom", "Time", units="s")
        self.trigger_line = pg.InfiniteLine(angle=0, movable=True,
                                            pen=pg.mkPen("#ffffff", width=1, dash=[4, 4]))
        self.trigger_line.setVisible(False)
        self.addItem(self.trigger_line, ignoreBounds=True)
        self.getPlotItem().setMenuEnabled(True)


class SpectrumPlot(_MultiCurvePlot):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Level", units="dB")


class PreviewView(pg.GraphicsLayoutWidget):
    """The captured region with the detected line and calibration lines."""

    pixel_clicked = Signal(int, int)  # column, row
    line_moved = Signal(str, float)  # "ya" | "yb" | "xa" | "xb", pixel position

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.vb = self.addViewBox(lockAspect=True, invertY=True)
        self.vb.setMenuEnabled(False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.mask = pg.ImageItem(axisOrder="row-major")
        self.mask.setZValue(1)
        self.trace = pg.PlotDataItem(pen=pg.mkPen("#ffea00", width=2))
        self.trace.setZValue(2)
        for item in (self.image, self.mask, self.trace):
            self.vb.addItem(item)
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

    def set_frame(self, image: np.ndarray, x_px: np.ndarray, y_px: np.ndarray,
                  mask: np.ndarray | None = None, probe_px: float | None = None) -> None:
        self.image.setImage(image, autoLevels=False, levels=(0, 255))
        if mask is not None:
            rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
            rgba[mask] = (255, 0, 255, 170)
            self.mask.setImage(rgba, autoLevels=False, levels=(0, 255))
        self.mask.setVisible(mask is not None)
        self.trace.setData(x_px, y_px, connect="finite")
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
