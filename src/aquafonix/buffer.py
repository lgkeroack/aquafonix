"""Thread-safe ring buffer for multi-channel sample streams."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class Snapshot:
    """An ordered copy of part of a :class:`RingBuffer`."""

    x: np.ndarray  # (n,)
    y: np.ndarray  # (n, channels)
    names: list[str]
    first_index: int  # running sample number of x[0]

    def __len__(self) -> int:
        return len(self.x)


class RingBuffer:
    """Fixed-capacity buffer of ``(x, y[channels])`` samples.

    Source threads call :meth:`append`; the GUI reads ordered copies with
    :meth:`latest` / :meth:`since`.  Channels can be added at runtime (e.g.
    when a new label shows up in a text stream); earlier samples of a new
    channel read back as NaN.
    """

    def __init__(self, capacity: int = 500_000):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._lock = threading.Lock()
        self._x = np.zeros(self.capacity)
        self._y = np.full((self.capacity, 0), np.nan)
        self._names: list[str] = []
        self._start = 0  # physical index of the oldest sample
        self._count = 0  # number of valid samples
        self._total = 0  # samples ever appended
        self.version = 0  # bumped on every change

    # -- properties ---------------------------------------------------------
    @property
    def names(self) -> list[str]:
        with self._lock:
            return list(self._names)

    @property
    def total(self) -> int:
        return self._total

    def __len__(self) -> int:
        return self._count

    def last_x(self) -> float | None:
        with self._lock:
            if not self._count:
                return None
            return float(self._x[(self._start + self._count - 1) % self.capacity])

    # -- writing ------------------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            self._start = self._count = self._total = 0
            self._names = []
            self._y = np.full((self.capacity, 0), np.nan)
            self.version += 1

    def _channel_index(self, name: str) -> int:
        try:
            return self._names.index(name)
        except ValueError:
            self._names.append(name)
            self._y = np.hstack([self._y, np.full((self.capacity, 1), np.nan)])
            return len(self._names) - 1

    def append(self, x, y, names: list[str] | None = None) -> None:
        """Append ``n`` samples. ``y`` is ``(n,)`` or ``(n, columns)``.

        ``names`` labels the columns of ``y``; without it the columns map onto
        the existing channels in order (new ones are called ``ch<N>``).
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        n = len(x)
        if n == 0:
            return
        if y.shape[0] != n:
            raise ValueError(f"x has {n} samples but y has {y.shape[0]}")
        with self._lock:
            if names is None:
                names = self._names[: y.shape[1]] + [
                    f"ch{i + 1}" for i in range(len(self._names), y.shape[1])
                ]
            cols = [self._channel_index(str(nm)) for nm in names]
            added = n
            if n > self.capacity:
                x, y, n = x[-self.capacity :], y[-self.capacity :], self.capacity
            if cols == list(range(self._y.shape[1])):
                rows = y
            else:
                rows = np.full((n, self._y.shape[1]), np.nan)
                rows[:, cols] = y
            end = (self._start + self._count) % self.capacity
            first = min(n, self.capacity - end)
            self._x[end : end + first] = x[:first]
            self._y[end : end + first] = rows[:first]
            rest = n - first
            if rest:
                self._x[:rest] = x[first:]
                self._y[:rest] = rows[first:]
            self._count += n
            if self._count > self.capacity:
                self._start = (self._start + self._count - self.capacity) % self.capacity
                self._count = self.capacity
            self._total += added
            self.version += 1

    # -- reading ------------------------------------------------------------
    def _copy(self, i0: int, i1: int) -> Snapshot:
        """Ordered copy of logical indices ``[i0, i1)``. Caller holds the lock."""
        n = max(0, i1 - i0)
        p0 = (self._start + i0) % self.capacity
        first = min(n, self.capacity - p0)
        if first == n:
            x = self._x[p0 : p0 + n].copy()
            y = self._y[p0 : p0 + n].copy()
        else:
            x = np.concatenate([self._x[p0:], self._x[: n - first]])
            y = np.concatenate([self._y[p0:], self._y[: n - first]])
        first_index = self._total - self._count + i0
        return Snapshot(x, y, list(self._names), first_index)

    def latest(self, n: int | None = None) -> Snapshot:
        """The newest ``n`` samples (all of them if ``n`` is None)."""
        with self._lock:
            n = self._count if n is None else max(0, min(int(n), self._count))
            return self._copy(self._count - n, self._count)

    def since(self, x_min: float, max_points: int | None = None) -> Snapshot:
        """Samples with ``x >= x_min``, assuming x is non-decreasing."""
        with self._lock:
            lo, hi = 0, self._count
            while lo < hi:  # first logical index with x >= x_min
                mid = (lo + hi) // 2
                if self._x[(self._start + mid) % self.capacity] < x_min:
                    lo = mid + 1
                else:
                    hi = mid
            if max_points is not None:
                lo = max(lo, self._count - int(max_points))
            return self._copy(lo, self._count)
