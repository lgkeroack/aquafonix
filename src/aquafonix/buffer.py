"""Thread-safe ring buffer for multi-channel sample streams."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class Snapshot:
    """An ordered copy of part of a :class:`RingBuffer`."""

    x: np.ndarray  # (n,)
    y: np.ndarray  # (n, channels)
    names: list[str]
    first_index: int  # absolute sample number of x[0]

    def __len__(self) -> int:
        return len(self.x)


class RingBuffer:
    """Fixed-capacity history of ``(x, y[channels])`` samples.

    Source threads call :meth:`append`; the GUI and the analysis thread read
    ordered copies.  Every sample has an *absolute index* (0 for the first
    sample ever appended) so readers can pick up exactly where they left off.

    For uniformly sampled streams pass ``sample_rate``: x is then implicit
    (``x = x0 + index / sample_rate``) and not stored, which halves memory
    for long audio histories.  Channels can be added at runtime (e.g. a new
    label in a text stream); earlier samples of a new channel read as NaN.
    """

    def __init__(self, capacity: int = 500_000, dtype=np.float64,
                 sample_rate: float | None = None, x0: float = 0.0):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.dtype = np.dtype(dtype)
        self.sample_rate = float(sample_rate) if sample_rate else None
        self.x0 = float(x0)
        self._lock = threading.Lock()
        self._x = None if self.sample_rate else np.zeros(self.capacity)
        self._y = np.full((self.capacity, 0), np.nan, dtype=self.dtype)
        self._names: list[str] = []
        self._start = 0  # physical index of the oldest sample
        self._count = 0  # number of valid samples
        self._total = 0  # samples ever appended
        self.version = 0  # bumped on every change
        self.epoch = 0  # bumped when history is discarded (clear / rebase)

    # -- properties ---------------------------------------------------------
    @property
    def names(self) -> list[str]:
        with self._lock:
            return list(self._names)

    @property
    def total(self) -> int:
        """Absolute index one past the newest sample."""
        return self._total

    @property
    def first_index(self) -> int:
        """Absolute index of the oldest sample still held."""
        return self._total - self._count

    @property
    def duration(self) -> float:
        """Seconds of history held (uniform streams) or x span."""
        with self._lock:
            if self._count < 2:
                return 0.0
            return self._x_at(self._count - 1) - self._x_at(0)

    def __len__(self) -> int:
        return self._count

    def _x_at(self, i: int) -> float:
        """x of logical index i (0 = oldest). Caller holds the lock."""
        if self.sample_rate:
            return self.x0 + (self._total - self._count + i) / self.sample_rate
        return float(self._x[(self._start + i) % self.capacity])

    def last_x(self) -> float | None:
        with self._lock:
            return self._x_at(self._count - 1) if self._count else None

    def first_x(self) -> float | None:
        with self._lock:
            return self._x_at(0) if self._count else None

    # -- writing ------------------------------------------------------------
    def clear(self) -> None:
        """Drop all samples. Absolute indices keep counting up."""
        with self._lock:
            self._start = self._count = 0
            self._names = []
            self._y = np.full((self.capacity, 0), np.nan, dtype=self.dtype)
            self.version += 1
            self.epoch += 1

    def rebase(self, x_next: float) -> None:
        """Clear, and make the next appended sample have ``x == x_next``
        (implicit-x buffers; used when a file player seeks)."""
        with self._lock:
            self._start = self._count = 0
            self._y[:] = np.nan
            if self.sample_rate:
                self.x0 = x_next - self._total / self.sample_rate
            self.version += 1
            self.epoch += 1

    def _channel_index(self, name: str) -> int:
        try:
            return self._names.index(name)
        except ValueError:
            self._names.append(name)
            self._y = np.hstack([self._y, np.full((self.capacity, 1), np.nan, dtype=self.dtype)])
            return len(self._names) - 1

    def append(self, x, y, names: list[str] | None = None) -> None:
        """Append ``n`` samples. ``y`` is ``(n,)`` or ``(n, columns)``.

        ``names`` labels the columns of ``y``; without it the columns map onto
        the existing channels in order (new ones are called ``ch<N>``).  With
        an implicit-x buffer ``x`` only supplies the sample count.
        """
        y = np.asarray(y, dtype=self.dtype)
        if y.ndim == 1:
            y = y[:, None]
        n = y.shape[0]
        if n == 0:
            return
        if not self.sample_rate:
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            if len(x) != n:
                raise ValueError(f"x has {len(x)} samples but y has {n}")
        with self._lock:
            if names is None:
                names = self._names[: y.shape[1]] + [
                    f"ch{i + 1}" for i in range(len(self._names), y.shape[1])
                ]
            cols = [self._channel_index(str(nm)) for nm in names]
            added = n
            if n > self.capacity:
                y, n = y[-self.capacity:], self.capacity
                if not self.sample_rate:
                    x = x[-self.capacity:]
            if cols == list(range(self._y.shape[1])):
                rows = y
            else:
                rows = np.full((n, self._y.shape[1]), np.nan, dtype=self.dtype)
                rows[:, cols] = y
            end = (self._start + self._count) % self.capacity
            first = min(n, self.capacity - end)
            self._y[end:end + first] = rows[:first]
            if not self.sample_rate:
                self._x[end:end + first] = x[:first]
            rest = n - first
            if rest:
                self._y[:rest] = rows[first:]
                if not self.sample_rate:
                    self._x[:rest] = x[first:]
            self._count += n
            if self._count > self.capacity:
                self._start = (self._start + self._count - self.capacity) % self.capacity
                self._count = self.capacity
            self._total += added
            self.version += 1

    # -- reading ------------------------------------------------------------
    def _copy(self, i0: int, i1: int) -> Snapshot:
        """Ordered copy of logical indices ``[i0, i1)``. Caller holds the lock."""
        i0, i1 = max(0, i0), min(self._count, i1)
        n = max(0, i1 - i0)
        p0 = (self._start + i0) % self.capacity
        first = min(n, self.capacity - p0)
        if first == n:
            y = self._y[p0:p0 + n].astype(np.float64)
        else:
            y = np.concatenate([self._y[p0:], self._y[:n - first]]).astype(np.float64)
        first_index = self._total - self._count + i0
        if self.sample_rate:
            x = self.x0 + (first_index + np.arange(n)) / self.sample_rate
        elif first == n:
            x = self._x[p0:p0 + n].copy()
        else:
            x = np.concatenate([self._x[p0:], self._x[:n - first]])
        return Snapshot(x, y, list(self._names), first_index)

    def _logical_at_or_after(self, x_min: float) -> int:
        """First logical index with x >= x_min (x non-decreasing)."""
        if self.sample_rate:
            idx = math.ceil((x_min - self.x0) * self.sample_rate - 1e-9)
            return int(min(max(idx - (self._total - self._count), 0), self._count))
        lo, hi = 0, self._count
        while lo < hi:
            mid = (lo + hi) // 2
            if self._x[(self._start + mid) % self.capacity] < x_min:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def latest(self, n: int | None = None) -> Snapshot:
        """The newest ``n`` samples (all of them if ``n`` is None)."""
        with self._lock:
            n = self._count if n is None else max(0, min(int(n), self._count))
            return self._copy(self._count - n, self._count)

    def since(self, x_min: float, max_points: int | None = None) -> Snapshot:
        """Samples with ``x >= x_min``, assuming x is non-decreasing."""
        with self._lock:
            lo = self._logical_at_or_after(x_min)
            if max_points is not None:
                lo = max(lo, self._count - int(max_points))
            return self._copy(lo, self._count)

    def range(self, x_min: float, x_max: float) -> Snapshot:
        """Samples with ``x_min <= x <= x_max``."""
        with self._lock:
            lo = self._logical_at_or_after(x_min)
            hi = self._logical_at_or_after(np.nextafter(x_max, np.inf))
            return self._copy(lo, max(lo, hi))

    def envelope(self, x_min: float | None, x_max: float | None, buckets: int):
        """Min/max envelope of ``[x_min, x_max]`` (whole history if None) in
        about ``buckets`` buckets, computed on the stored samples without
        copying them: what a plot needs to draw millions of samples.
        Returns a Snapshot with 2 points per bucket (min, max), or the raw
        samples when there are fewer than ``2 * buckets``."""
        with self._lock:
            lo = 0 if x_min is None else self._logical_at_or_after(x_min)
            hi = self._count if x_max is None else self._logical_at_or_after(
                np.nextafter(x_max, np.inf))
            n = hi - lo
            if n <= 2 * buckets or buckets < 2:
                return self._copy(lo, hi)
            step = int(np.ceil(n / buckets))
            xs, ys = [], []
            p0 = (self._start + lo) % self.capacity
            first = min(n, self.capacity - p0)
            for seg_lo, seg_n, logical in ((p0, first, lo), (0, n - first, lo + first)):
                if seg_n <= 0:
                    continue
                seg = self._y[seg_lo:seg_lo + seg_n]
                starts = np.arange(0, seg_n, step)
                with np.errstate(all="ignore"):
                    mn = np.fmin.reduceat(seg, starts, axis=0).astype(np.float64)
                    mx = np.fmax.reduceat(seg, starts, axis=0).astype(np.float64)
                idx = logical + starts
                if self.sample_rate:
                    bx = self.x0 + (self._total - self._count + idx) / self.sample_rate
                else:
                    bx = self._x[(self._start + idx) % self.capacity]
                xs.append(np.repeat(bx, 2))
                ys.append(np.stack([mn, mx], axis=1).reshape(-1, seg.shape[1]))
            first_index = self._total - self._count + lo
            return Snapshot(np.concatenate(xs), np.concatenate(ys), list(self._names), first_index)

    def from_index(self, index: int, max_n: int | None = None) -> Snapshot:
        """Samples from absolute ``index`` onwards (clamped to what is held).
        ``snapshot.first_index`` tells the caller where the data really starts."""
        with self._lock:
            lo = max(0, index - (self._total - self._count))
            hi = self._count if max_n is None else min(self._count, lo + int(max_n))
            return self._copy(lo, hi)
