"""Common machinery for data sources.

A source produces blocks of samples on a background thread and pushes them
into a :class:`~aquafonix.buffer.RingBuffer`.  Listeners (e.g. a recorder)
get every block too, on the source's thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from aquafonix.buffer import RingBuffer
from aquafonix.parsing import BinaryParser, ParserConfig

log = logging.getLogger(__name__)

Listener = Callable[[np.ndarray, np.ndarray, "list[str] | None"], None]


class Source:
    """Base class.  Loop-driven subclasses implement :meth:`_step`; callback
    driven ones (audio) override :meth:`_begin` / :meth:`_end` instead."""

    #: nominal sample rate in Hz when the source has one (audio, generator).
    #: Such sources produce x = sample_index / sample_rate.
    sample_rate: float | None = None
    #: suggested ring-buffer capacity in samples
    buffer_capacity: int = 500_000
    #: storage type for the history buffer (audio uses float32)
    sample_dtype = np.float64

    def __init__(self) -> None:
        self.buffer: RingBuffer | None = None
        self.status = "Idle"
        self.error: str | None = None
        self.samples_received = 0
        self._listeners: list[Listener] = []
        self._listeners_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = time.perf_counter()
        self._last_push_t = 0.0

    # -- lifecycle ------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() or self._callback_running()

    def _callback_running(self) -> bool:
        return False

    def start(self, buffer: RingBuffer) -> None:
        """Open the device/socket (errors raise here) and start streaming."""
        self.buffer = buffer
        self.error = None
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._last_push_t = 0.0
        self._begin()
        if self._uses_thread():
            self._thread = threading.Thread(target=self._run, name=type(self).__name__,
                                            daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        try:
            self._end()
        except Exception:  # closing must never raise into the GUI
            log.exception("error while stopping %s", type(self).__name__)
        if not self.error:
            self.status = "Stopped"

    def _uses_thread(self) -> bool:
        return True

    def _begin(self) -> None:
        """Open resources. Runs on the caller's thread so errors surface."""

    def _end(self) -> None:
        """Release resources after the thread has stopped."""

    def _step(self) -> None:
        raise NotImplementedError

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._step()
        except Exception as exc:  # report, don't crash the app
            log.exception("%s failed", type(self).__name__)
            self.error = str(exc) or type(exc).__name__
            self.status = f"Error: {self.error}"

    # -- data -----------------------------------------------------------------
    def now(self) -> float:
        """Seconds since :meth:`start`."""
        return time.perf_counter() - self._t0

    def add_listener(self, fn: Listener) -> None:
        with self._listeners_lock:
            self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        with self._listeners_lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def push(self, x, y, names: list[str] | None = None) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.buffer is not None:
            self.buffer.append(x, y, names)
        self.samples_received += len(x)
        with self._listeners_lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(x, y, names)
            except Exception:
                log.exception("listener failed")

    def arrival_times(self, n: int, max_spread: float = 0.25) -> np.ndarray:
        """Timestamps for ``n`` samples that arrived together, spread evenly
        since the previous chunk (at most ``max_spread`` s back)."""
        now = self.now()
        start = max(self._last_push_t, now - max_spread)
        self._last_push_t = now
        return np.linspace(start, now, n + 1)[1:]


class ByteStreamSource(Source):
    """A source that reads raw bytes (serial, socket, file) and parses them."""

    def __init__(self, parser_config: ParserConfig | None = None) -> None:
        super().__init__()
        self.parser_config = parser_config or ParserConfig()
        self.parser = self.parser_config.make_parser()
        self.bytes_received = 0
        self._binary_index = 0
        if self.parser_config.mode == "binary" and self.parser_config.sample_rate > 0:
            self.sample_rate = self.parser_config.sample_rate

    def reset_parser(self) -> None:
        self.parser.reset()

    def handle_bytes(self, data: bytes) -> None:
        self.bytes_received += len(data)
        block = self.parser.feed(data)
        if block is None or not len(block):
            return
        n = len(block)
        if block.x is not None:
            x = block.x
            missing = np.isnan(x)
            if missing.any():
                x = x.copy()
                x[missing] = self.now()
        elif isinstance(self.parser, BinaryParser) and self.parser_config.sample_rate > 0:
            x = (self._binary_index + np.arange(n)) / self.parser_config.sample_rate
            self._binary_index += n
        else:
            x = self.arrival_times(n)
        self.push(x, block.y, block.names)
