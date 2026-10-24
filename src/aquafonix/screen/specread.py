"""Read a spectrogram drawn by another program: colours back to levels.

The colour scale is either a known colormap, greyscale, or sampled from the
program's own colour bar (drag a box over it in the preview).  Every
quantised colour (5 bits per channel) is mapped once to the nearest colour of
the scale, so decoding a frame is a single table lookup.  Pixels whose
colour is far from the scale (text, grid lines, cursors) become NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# (position, (r, g, b)) anchors
_MAPS = {
    "inferno": [(0, (0, 0, 4)), (0.14, (40, 11, 84)), (0.29, (101, 21, 110)),
                (0.43, (159, 42, 99)), (0.57, (212, 72, 66)), (0.71, (245, 125, 21)),
                (0.86, (250, 193, 39)), (1, (252, 255, 164))],
    "viridis": [(0, (68, 1, 84)), (0.11, (72, 40, 120)), (0.22, (62, 74, 137)),
                (0.33, (49, 104, 142)), (0.44, (38, 130, 142)), (0.56, (31, 158, 137)),
                (0.67, (53, 183, 121)), (0.78, (110, 206, 88)), (0.89, (181, 222, 43)),
                (1, (253, 231, 37))],
    "jet": [(0, (0, 0, 128)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)),
            (0.625, (255, 255, 0)), (0.875, (255, 0, 0)), (1, (128, 0, 0))],
    "hot": [(0, (0, 0, 0)), (0.375, (255, 0, 0)), (0.75, (255, 255, 0)), (1, (255, 255, 255))],
    "ocean": [(0, (0, 0, 0)), (0.17, (0, 20, 60)), (0.33, (0, 60, 120)), (0.5, (0, 120, 160)),
              (0.67, (40, 180, 180)), (0.83, (160, 230, 200)), (1, (255, 255, 255))],
}
COLOUR_SCALES = {
    "inferno": "Inferno / magma", "viridis": "Viridis", "jet": "Jet / rainbow", "hot": "Hot",
    "ocean": "Ocean", "grey": "Greyscale (bright = loud)", "grey_inv": "Greyscale (dark = loud)",
    "bar": "Sampled from the colour bar",
}
ORIENTATIONS = {
    "right": "Time → (newest on the right)",
    "down": "Time ↓ (newest at the top, waterfall)",
    "up": "Time ↑ (newest at the bottom)",
}


@dataclass(frozen=True)
class ColourScale:
    kind: str = "inferno"
    colors: tuple = ()  # sampled colour bar, low -> high (kind == "bar")
    low_db: float = 0.0
    high_db: float = 100.0
    tolerance: float = 40.0

    def lut(self, n: int = 128) -> tuple[np.ndarray, np.ndarray]:
        values = np.linspace(self.low_db, self.high_db, n)
        if self.kind == "bar" and len(self.colors) >= 2:
            c = np.array(self.colors, dtype=float)
            pos = np.linspace(0, 1, len(c))
        else:
            anchors = _MAPS.get(self.kind, _MAPS["inferno"])
            pos = np.array([a[0] for a in anchors], dtype=float)
            c = np.array([a[1] for a in anchors], dtype=float)
        t = np.linspace(0, 1, n)
        colors = np.column_stack([np.interp(t, pos, c[:, k]) for k in range(3)])
        return colors, values


@lru_cache(maxsize=8)
def _table(scale: ColourScale) -> np.ndarray:
    """Level for every 5-bit-per-channel colour (NaN = not on the scale)."""
    q = np.arange(1 << 15)
    rgb = np.column_stack([(q >> 10) & 31, (q >> 5) & 31, q & 31]).astype(float) * 8 + 4
    colors, values = scale.lut()
    best = np.full(len(rgb), np.inf)
    out = np.full(len(rgb), np.nan, dtype=np.float32)
    for c, v in zip(colors, values):  # 128 passes over 32k colours: fast enough, once
        d = ((rgb - c) ** 2).sum(axis=1)
        closer = d < best
        best[closer] = d[closer]
        out[closer] = v
    out[best > scale.tolerance**2] = np.nan
    return out


def decode_levels(img: np.ndarray, scale: ColourScale) -> np.ndarray:
    """(H, W, 3) uint8 image -> (H, W) levels in dB (NaN where unreadable)."""
    img = np.asarray(img[..., :3])
    if scale.kind in ("grey", "grey_inv"):
        r, g, b = (img[..., i].astype(np.float32) for i in range(3))
        bright = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        if scale.kind == "grey_inv":
            bright = 1.0 - bright
        levels = scale.low_db + bright * (scale.high_db - scale.low_db)
        chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        levels[chroma > max(scale.tolerance, 20)] = np.nan  # coloured overlays
        return levels
    q = img >> 3
    key = (q[..., 0].astype(np.int32) << 10) | (q[..., 1].astype(np.int32) << 5) | q[..., 2]
    return _table(scale)[key]


def sample_colour_bar(img: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> tuple:
    """Colours along a colour bar region, low -> high.  A tall box is read
    bottom (low) to top (high); a wide box left (low) to right (high)."""
    h, w = img.shape[:2]
    c0, c1 = sorted((int(max(0, min(x0, x1))), int(min(w, max(x0, x1)))))
    r0, r1 = sorted((int(max(0, min(y0, y1))), int(min(h, max(y0, y1)))))
    crop = img[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1), :3].astype(float)
    if crop.shape[0] >= crop.shape[1]:
        line = crop.mean(axis=1)[::-1]  # rows, bottom first
    else:
        line = crop.mean(axis=0)
    if len(line) > 128:
        line = line[np.linspace(0, len(line) - 1, 128).astype(int)]
    return tuple(tuple(int(v) for v in c) for c in line.round())


def canonical(levels: np.ndarray, orientation: str) -> np.ndarray:
    """Rotate so rows are frequency (top = high) and columns are time
    (oldest on the left, newest on the right)."""
    if orientation == "down":  # rows = time (row 0 newest), columns = frequency (low left)
        return levels[::-1, :].T[::-1, :]
    if orientation == "up":  # rows = time (last row newest)
        return levels.T[::-1, :]
    return levels


def to_frequency_grid(levels: np.ndarray, row_freqs: np.ndarray, n_bins: int = 256):
    """Resample a canonical (rows, W) image whose row ``i`` shows frequency
    ``row_freqs[i]`` onto an ascending uniform grid.  Returns
    ``(grid_hz, frames[W, n_bins])``."""
    order = np.argsort(row_freqs)
    f_sorted = row_freqs[order]
    rows = levels[order]
    grid = np.linspace(max(f_sorted[0], 0.0), f_sorted[-1], n_bins)
    pos = np.interp(grid, f_sorted, np.arange(len(f_sorted)))
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(f_sorted) - 1)
    w = (pos - i0)[:, None]
    grid_levels = rows[i0] * (1 - w) + rows[i1] * w  # (n_bins, W)
    return grid, grid_levels.T
