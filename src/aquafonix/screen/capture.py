"""Screen grabbing (mss) and the screen-region source."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from aquafonix.screen.extract import (FrameTraces, ScrollStitcher, TraceSettings,
                                      extract_traces, probe_value)
from aquafonix.sources.base import Source

OUTPUTS = {
    "trace": "Live trace (whole curve)",
    "history": "Value history (one value per frame)",
    "stitch": "Scrolling chart (stitch history)",
}


@dataclass(frozen=True)
class Region:
    """Rectangle in physical (native) desktop pixels."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("region must be at least 1x1 pixel")

    def as_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __str__(self) -> str:
        return f"{self.left},{self.top},{self.width},{self.height}"

    @classmethod
    def parse(cls, text: str) -> "Region":
        """Accepts ``x,y,w,h`` (also space or ``x``-separated)."""
        parts = [p for p in re.split(r"[,\s×x]+", text.strip()) if p]
        if len(parts) != 4:
            raise ValueError(f"region must be 'x,y,width,height', got {text!r}")
        return cls(*(int(float(p)) for p in parts))


class Grabber(Protocol):
    def grab(self, region: Region) -> np.ndarray: ...
    def close(self) -> None: ...


def _mss_factory():
    import mss

    return getattr(mss, "MSS", None) or mss.mss  # mss >= 10.2 renamed the class


class MssGrabber:
    """Grabs RGB arrays with mss.  Create and use it on one thread."""

    def __init__(self) -> None:
        self._sct = _mss_factory()()

    def grab(self, region: Region) -> np.ndarray:
        shot = self._sct.grab(region.as_dict())
        bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return bgra[:, :, 2::-1].copy()

    def close(self) -> None:
        self._sct.close()


def monitor_rects() -> list[tuple[int, int, int, int]]:
    """Native ``(left, top, width, height)`` of every monitor."""
    sct = _mss_factory()()
    try:
        return [(m["left"], m["top"], m["width"], m["height"]) for m in sct.monitors[1:]]
    finally:
        sct.close()


@dataclass
class Frame:
    image: np.ndarray
    result: FrameTraces
    t: float
    region: Region


class ScreenSource(Source):
    """Captures a screen region at a fixed rate and extracts the plotted line.

    The GUI may replace ``region``, ``settings``, ``output``, ``probe`` and
    ``probe_column`` while running; they are read once per frame.
    """

    def __init__(self, region: Region, fps: float = 20.0,
                 settings: TraceSettings | None = None, output: str = "trace",
                 probe: str = "right", probe_column: float = 0.5,
                 grabber_factory: Callable[[], Grabber] | None = None) -> None:
        super().__init__()
        self.region = region
        self.fps = float(fps)
        self.settings = settings or TraceSettings()
        self.output = output
        self.probe = probe
        self.probe_column = probe_column
        self.buffer_capacity = 200_000
        self.latest: Frame | None = None
        self.frames = 0
        self.measured_fps = 0.0
        self.stitcher = ScrollStitcher()
        self._factory = grabber_factory or MssGrabber
        self._grabber: Grabber | None = None
        self._stitch_key = None

    def _begin(self) -> None:
        self.frames = 0
        self.stitcher.reset()
        self.status = f"Watching {self.region.width}×{self.region.height} region"

    def _run(self) -> None:
        try:
            self._grabber = self._factory()
        except Exception as exc:
            self.error = f"screen capture unavailable: {exc}"
            self.status = f"Error: {self.error}"
            return
        try:
            super()._run()
        finally:
            try:
                self._grabber.close()
            except Exception:
                pass
            self._grabber = None

    def _step(self) -> None:
        started = time.perf_counter()
        region, settings, output = self.region, self.settings, self.output
        image = self._grabber.grab(region)
        t = self.now()
        result = extract_traces(image, settings)
        self.latest = Frame(image, result, t, region)
        names = result.names
        if output == "stitch":
            key = (region, settings.calibration, tuple(names))
            if key != self._stitch_key:
                self.stitcher.reset()
                self._stitch_key = key
            ref = result.reference()
            if ref is not None:
                times, values = self.stitcher.update(ref.y_px, result.y, t)
                if len(times):
                    self.push(times, values, names)
        elif names:
            values = [probe_value(tr.y, self.probe, self.probe_column) for tr in result.traces]
            self.push([t], [values], names)
        self.frames += 1
        elapsed = time.perf_counter() - started
        period = 1.0 / max(self.fps, 0.1)
        rate = 1.0 / max(elapsed, period)
        self.measured_fps = rate if self.frames == 1 else 0.9 * self.measured_fps + 0.1 * rate
        self.status = (f"Watching {region.width}×{region.height} at {self.measured_fps:.1f} fps"
                       f" · line found in {result.coverage:.0%} of columns")
        self._stop.wait(max(0.0, period - elapsed))
