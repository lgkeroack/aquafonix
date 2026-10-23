"""Find a plotted line in a screenshot and turn it into numbers.

Everything here is plain numpy so it can be unit-tested without a display.
Image arrays are ``(height, width, 3)`` uint8 RGB.  Pixel coordinates are
continuous: row ``r`` spans ``[r, r + 1)`` so the top edge is 0 and the
bottom edge is ``height``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MODES = {
    "auto": "Auto (most prominent colour)",
    "color": "Picked colour",
    "contrast": "Anything unlike the background",
}
ESTIMATORS = {
    "center": "Centre of line",
    "top": "Top edge (upper envelope)",
    "bottom": "Bottom edge (lower envelope)",
    "track": "Follow the line (ignore stray marks)",
}
PROBES = {
    "right": "Newest point (rightmost)",
    "left": "Oldest point (leftmost)",
    "column": "At a fixed column",
    "mean": "Mean of the trace",
    "max": "Maximum of the trace",
    "min": "Minimum of the trace",
}


@dataclass(frozen=True)
class Calibration:
    """Pixel -> value mapping from two reference lines per axis."""

    y_px_a: float = 1.0
    y_val_a: float = 0.0
    y_px_b: float = 0.0
    y_val_b: float = 1.0
    x_px_a: float = 0.0
    x_val_a: float = 0.0
    x_px_b: float = 1.0
    x_val_b: float = 1.0
    log_y: bool = False

    @classmethod
    def for_size(cls, width: int, height: int, bottom: float = 0.0, top: float = 1.0,
                 left: float = 0.0, right: float = 1.0, log_y: bool = False) -> "Calibration":
        """Reference lines on the region edges."""
        return cls(float(height), bottom, 0.0, top, 0.0, left, float(width), right, log_y)

    def y_value(self, y_px):
        y_px = np.asarray(y_px, dtype=float)
        span = self.y_px_b - self.y_px_a
        if span == 0:
            return np.full_like(y_px, np.nan)
        f = (y_px - self.y_px_a) / span
        if self.log_y and self.y_val_a > 0 and self.y_val_b > 0:
            la, lb = np.log10(self.y_val_a), np.log10(self.y_val_b)
            return 10 ** (la + f * (lb - la))
        return self.y_val_a + f * (self.y_val_b - self.y_val_a)

    def x_value(self, x_px):
        x_px = np.asarray(x_px, dtype=float)
        span = self.x_px_b - self.x_px_a
        if span == 0:
            return np.full_like(x_px, np.nan)
        return self.x_val_a + (x_px - self.x_px_a) / span * (self.x_val_b - self.x_val_a)


@dataclass(frozen=True)
class TraceSettings:
    mode: str = "auto"
    color: tuple[int, int, int] = (255, 0, 0)
    tolerance: float = 60.0
    ignore_grid: bool = True
    estimator: str = "center"
    bridge_gaps: bool = True
    calibration: Calibration | None = None


@dataclass
class TraceResult:
    x_px: np.ndarray  # (W,) column centres
    y_px: np.ndarray  # (W,) line position per column, NaN where none
    x: np.ndarray  # calibrated
    y: np.ndarray  # calibrated
    mask: np.ndarray  # (H, W) bool, pixels classified as trace
    color: tuple[int, int, int] | None  # colour that was matched
    background: tuple[int, int, int]
    coverage: float  # fraction of columns where the line was found


# -- pixel classification -------------------------------------------------------
def background_color(img: np.ndarray) -> np.ndarray:
    """Most common colour (5 bits per channel), averaged within its bin."""
    step = max(1, int(np.sqrt(img.shape[0] * img.shape[1] / 40_000)))
    px = img[::step, ::step].reshape(-1, 3)
    q = (px >> 3).astype(np.int32)
    key = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    best = np.bincount(key, minlength=1 << 15).argmax()
    return px[key == best].mean(axis=0)


# Per-channel elementwise ops: much faster than reducing over the size-3 axis.
def color_distance_sq(img: np.ndarray, color) -> np.ndarray:
    out = np.zeros(img.shape[:2], dtype=np.int32)
    for i in range(3):
        d = img[..., i].astype(np.int32) - int(round(float(color[i])))
        out += d * d
    return out


def chroma(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return np.maximum(np.maximum(r, g), b).astype(np.int16) - np.minimum(np.minimum(r, g), b)


def neutral_pixels(img: np.ndarray, neutral_chroma: int = 40) -> np.ndarray:
    """Grey/black/white pixels (what grids, axes and text are usually made of)."""
    return chroma(img) < neutral_chroma


def suppress_grid(mask: np.ndarray, img: np.ndarray, frac: float = 0.4,
                  neutral: np.ndarray | None = None) -> np.ndarray:
    """Drop grey/black/white pixels on rows/columns that are mostly covered:
    grid lines and axes.  Coloured lines survive even when flat."""
    if neutral is None:
        neutral = neutral_pixels(img)
    m = mask & neutral
    h, w = mask.shape
    rows = m.sum(axis=1) > frac * w
    cols = m.sum(axis=0) > frac * h
    if not (rows.any() or cols.any()):
        return mask
    kill = np.zeros_like(mask)
    kill[rows] = True
    kill[:, cols] = True
    return mask & ~(kill & neutral)


def auto_trace_color(img: np.ndarray, background, tolerance: float,
                     ignore_grid: bool = True,
                     neutral: np.ndarray | None = None) -> tuple[int, int, int] | None:
    """Pick the trace colour: the most common non-background colour, with
    saturated colours strongly preferred over greys (grid, text)."""
    fg = color_distance_sq(img, background) > tolerance**2
    if ignore_grid:
        fg = suppress_grid(fg, img, neutral=neutral)
    px = img[fg]
    if not len(px):
        return None
    if len(px) > 200_000:
        px = px[:: len(px) // 200_000 + 1]
    c = chroma(px[None])[0]
    q = (px >> 4).astype(np.int32)
    key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    score = np.bincount(key, weights=0.15 + c / 255.0, minlength=4096)
    best = score.argmax()
    return tuple(int(v) for v in px[key == best].mean(axis=0).round())


def trace_mask(img: np.ndarray, settings: TraceSettings):
    """Returns ``(mask, matched_colour, background)``."""
    bg = background_color(img)
    tol2 = float(settings.tolerance) ** 2
    neutral = neutral_pixels(img) if settings.ignore_grid else None
    color = None
    if settings.mode == "contrast":
        mask = color_distance_sq(img, bg) > tol2
    else:
        if settings.mode == "color":
            color = tuple(int(v) for v in settings.color)
        else:
            color = auto_trace_color(img, bg, settings.tolerance, settings.ignore_grid, neutral)
        if color is None:
            mask = np.zeros(img.shape[:2], dtype=bool)
        else:
            mask = color_distance_sq(img, color) <= tol2
    if settings.ignore_grid:
        mask = suppress_grid(mask, img, neutral=neutral)
    return mask, color, tuple(int(v) for v in bg.round())


# -- line position per column ---------------------------------------------------
def column_positions(mask: np.ndarray, estimator: str = "center") -> np.ndarray:
    h, w = mask.shape
    counts = mask.sum(axis=0)
    has = counts > 0
    if estimator == "top":
        y = np.argmax(mask, axis=0) + 0.5
    elif estimator == "bottom":
        y = h - np.argmax(mask[::-1], axis=0) - 0.5
    elif estimator == "track":
        return _track(mask)
    else:
        rows = np.arange(h, dtype=float) + 0.5
        y = rows @ mask / np.maximum(counts, 1)
    y = y.astype(float)
    y[~has] = np.nan
    return y


def _track(mask: np.ndarray) -> np.ndarray:
    """Per column pick the run of pixels closest to the previous column's
    pick, so labels/markers away from the line are ignored."""
    h, w = mask.shape
    pad = np.zeros((h + 2, w), dtype=np.int8)
    pad[1:-1] = mask
    d = np.diff(pad, axis=0).T  # (w, h + 1)
    s_col, s_row = np.nonzero(d == 1)
    _, e_row = np.nonzero(d == -1)
    y = np.full(w, np.nan)
    if not len(s_col):
        return y
    centers = (s_row + e_row) / 2.0
    bounds = np.searchsorted(s_col, np.arange(w + 1))
    prev = float(np.median(centers))
    for j in range(w):
        a, b = bounds[j], bounds[j + 1]
        if a == b:
            continue
        c = centers[a] if b - a == 1 else centers[a:b][np.argmin(np.abs(centers[a:b] - prev))]
        y[j] = prev = c
    return y


def bridge_gaps(y: np.ndarray) -> np.ndarray:
    """Linearly interpolate interior NaNs (leading/trailing NaNs stay)."""
    valid = np.isfinite(y)
    if valid.sum() < 2 or valid.all():
        return y
    idx = np.arange(len(y))
    first, last = idx[valid][0], idx[valid][-1]
    inner = ~valid & (idx > first) & (idx < last)
    out = y.copy()
    out[inner] = np.interp(idx[inner], idx[valid], y[valid])
    return out


def extract_trace(img: np.ndarray, settings: TraceSettings | None = None) -> TraceResult:
    settings = settings or TraceSettings()
    img = np.ascontiguousarray(img[..., :3])
    h, w = img.shape[:2]
    mask, color, bg = trace_mask(img, settings)
    y_px = column_positions(mask, settings.estimator)
    coverage = float(np.isfinite(y_px).mean()) if w else 0.0
    if settings.bridge_gaps:
        y_px = bridge_gaps(y_px)
    x_px = np.arange(w, dtype=float) + 0.5
    cal = settings.calibration or Calibration.for_size(w, h)
    return TraceResult(x_px, y_px, cal.x_value(x_px), cal.y_value(y_px), mask, color, bg,
                       coverage)


def probe_value(values: np.ndarray, probe: str = "right", column_frac: float = 0.5) -> float:
    """Reduce one extracted trace to a single number."""
    values = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return float("nan")
    if probe == "right":
        return float(values[finite[-1]])
    if probe == "left":
        return float(values[finite[0]])
    if probe == "column":
        col = int(round(min(max(column_frac, 0.0), 1.0) * (len(values) - 1)))
        return float(values[col])
    if probe == "mean":
        return float(values[finite].mean())
    if probe == "max":
        return float(values[finite].max())
    if probe == "min":
        return float(values[finite].min())
    raise ValueError(f"unknown probe {probe!r}")


# -- scrolling charts -------------------------------------------------------------
def shift_errors(prev: np.ndarray, cur: np.ndarray, max_shift: int,
                 min_overlap: int) -> np.ndarray:
    """Mean |cur[j] - prev[j + s]| for each scroll shift ``s`` (inf if the
    overlap is too small)."""
    w = len(cur)
    max_shift = max(0, min(max_shift, w - 1))
    padded = np.concatenate([prev, np.full(max_shift, np.nan)])
    windows = sliding_window_view(padded, w)[: max_shift + 1]
    diff = np.abs(windows - cur[None, :])
    ok = np.isfinite(diff)
    counts = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        errs = np.where(ok, diff, 0).sum(axis=1) / counts
    errs[counts < min_overlap] = np.inf
    return errs


class ScrollStitcher:
    """Rebuilds the full history of a chart that scrolls right-to-left by
    measuring how far the trace moved between frames and keeping only the
    newly revealed columns."""

    def __init__(self, max_shift: int = 256, min_overlap_frac: float = 0.25,
                 ambiguity_px: float = 0.75, max_error_px: float = 4.0) -> None:
        self.max_shift = max_shift
        self.min_overlap_frac = min_overlap_frac
        self.ambiguity_px = ambiguity_px
        self.max_error_px = max_error_px
        self.reset()

    def reset(self) -> None:
        self._prev_px: np.ndarray | None = None
        self._prev_val: np.ndarray | None = None
        self._prev_t = 0.0
        self._col_t: float | None = None  # time given to the newest emitted column
        self._last_shift_t = 0.0
        self.rate: float | None = None  # scroll speed, px/s
        self.last_shift = 0
        self.resyncs = 0

    def update(self, y_px: np.ndarray, values: np.ndarray, t: float):
        """Feed one frame; returns ``(times, values)`` of new columns."""
        empty = (np.zeros(0), np.zeros(0))
        y_px = np.asarray(y_px, dtype=float)
        values = np.asarray(values, dtype=float)
        if self._prev_px is None or len(self._prev_px) != len(y_px):
            self.reset()
            self._prev_px, self._prev_val, self._prev_t = y_px, values, t
            self._last_shift_t = t
            return empty
        w = len(y_px)
        errs = shift_errors(self._prev_px, y_px, min(self.max_shift, w // 2),
                            max(2, int(w * self.min_overlap_frac)))
        prev_val, prev_t = self._prev_val, self._prev_t
        self._prev_px, self._prev_val, self._prev_t = y_px, values, t
        if not np.isfinite(errs).any():
            return empty
        best = int(np.argmin(errs))
        if errs[best] > self.max_error_px:  # the picture changed, not scrolled
            self.resyncs += 1
            self._col_t = None
            self._last_shift_t = t
            return empty
        candidates = np.flatnonzero(errs <= errs[best] + self.ambiguity_px)
        shift = int(candidates[0])
        predicted = None
        if len(candidates) > 1 and self.rate:
            # Featureless (flat) or periodic line: trust the measured speed.
            predicted = self.rate * (t - self._last_shift_t)
            shift = int(candidates[np.argmin(np.abs(candidates - predicted))])
        self.last_shift = shift
        if shift <= 0:
            return empty
        if predicted is not None:  # keep the rounding remainder for next time
            self._last_shift_t = t - (predicted - shift) / self.rate
        else:
            if len(candidates) == 1 and t > self._last_shift_t:
                inst = shift / (t - self._last_shift_t)
                self.rate = inst if self.rate is None else 0.7 * self.rate + 0.3 * inst
            self._last_shift_t = t
        times, vals = [], []
        if self._col_t is None:  # first movement: also emit what was on screen
            col_dt = (t - prev_t) / shift
            times.append(prev_t - (w - 1 - np.arange(w)) * col_dt)
            vals.append(prev_val)
            self._col_t = prev_t
        times.append(np.linspace(self._col_t, t, shift + 1)[1:])
        vals.append(values[w - shift:])
        self._col_t = t
        return np.concatenate(times), np.concatenate(vals)
