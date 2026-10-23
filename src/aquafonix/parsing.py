"""Parsers that turn raw bytes from serial/network/file streams into samples.

Text streams are line based.  Each line may be

* plain numbers with any separators:      ``1.5, 2.25, -3``  /  ``1.5 2.25``
* labelled values (Arduino plotter style): ``temp:21.5 hum:40``  /  ``a=1;b=2``
* a header naming the channels:            ``time,temp,hum`` (before any data)

Binary streams are fixed-size frames of interleaved channels, optionally
preceded by a sync word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])"
NUMBER_RE = re.compile(_NUM, re.IGNORECASE)
LABELLED_RE = re.compile(
    r"(?P<label>[A-Za-z_][\w .\-/%()\[\]]*?)\s*[:=]\s*(?P<num>" + _NUM + ")", re.IGNORECASE
)
HEADER_SPLIT_RE = re.compile(r"\s*[,;\t|]\s*")


@dataclass
class Block:
    """Parsed samples: ``y`` is (n, len(names)); ``x`` is set if the stream
    carried its own x/time column."""

    y: np.ndarray
    names: list[str]
    x: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.y)


def parse_line(line: str) -> list[float] | dict[str, float] | None:
    """Parse one text line into a list (unlabelled) or dict (labelled)."""
    pairs = LABELLED_RE.findall(line)
    if pairs:
        return {label.strip(): float(num) for label, num in pairs}
    nums = NUMBER_RE.findall(line)
    if nums:
        return [float(v) for v in nums]
    return None


class TextLineParser:
    """Incremental line parser; :meth:`feed` accepts arbitrary byte chunks."""

    def __init__(self, x_from_first: bool = False, names: list[str] | None = None,
                 max_line: int = 65536):
        self.x_from_first = x_from_first
        self.fixed_names = [n for n in (names or []) if n]
        self.header: list[str] = []
        self.max_line = max_line
        self.lines_parsed = 0
        self.lines_skipped = 0
        self._pending = b""

    def reset(self) -> None:
        self._pending = b""

    def _column_name(self, i: int) -> str:
        names = self.fixed_names or self.header
        if self.x_from_first:
            names = names[1:] if names and not self.fixed_names else names
        return names[i] if i < len(names) else f"ch{i + 1}"

    def feed(self, data: bytes) -> Block | None:
        buf = self._pending + data
        lines = buf.split(b"\n")
        self._pending = lines.pop()
        if len(self._pending) > self.max_line:  # no newline in sight: drop it
            self._pending = b""
            self.lines_skipped += 1
        rows: list[dict[str, float]] = []
        xs: list[float] = []
        for raw in lines:
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            parsed = parse_line(text)
            if parsed is None:
                if not self.lines_parsed and not self.header:
                    self.header = [h for h in HEADER_SPLIT_RE.split(text) if h] \
                        if HEADER_SPLIT_RE.search(text) else text.split()
                else:
                    self.lines_skipped += 1
                continue
            if isinstance(parsed, list):
                if self.x_from_first:
                    if len(parsed) < 2:
                        self.lines_skipped += 1
                        continue
                    xs.append(parsed[0])
                    parsed = parsed[1:]
                parsed = {self._column_name(i): v for i, v in enumerate(parsed)}
            elif self.x_from_first:
                xs.append(float("nan"))
            rows.append(parsed)
            self.lines_parsed += 1
        if not rows:
            return None
        names: list[str] = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
        index = {name: i for i, name in enumerate(names)}
        y = np.full((len(rows), len(names)), np.nan)
        for r, row in enumerate(rows):
            for key, value in row.items():
                y[r, index[key]] = value
        x = np.array(xs) if self.x_from_first else None
        return Block(y, names, x)


BINARY_TYPES = {
    "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
}


class BinaryParser:
    """Fixed-size frames of ``channels`` interleaved values of ``dtype``."""

    def __init__(self, dtype: str = "int16", channels: int = 1, big_endian: bool = False,
                 sync: bytes = b""):
        if dtype not in BINARY_TYPES:
            raise ValueError(f"unknown binary type {dtype!r}")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        self.dtype = np.dtype(BINARY_TYPES[dtype]).newbyteorder(">" if big_endian else "<")
        self.channels = int(channels)
        self.frame_bytes = self.dtype.itemsize * self.channels
        self.sync = bytes(sync)
        self.names = [f"ch{i + 1}" for i in range(self.channels)]
        self._pending = b""

    def reset(self) -> None:
        self._pending = b""

    def feed(self, data: bytes) -> Block | None:
        buf = self._pending + data
        if not self.sync:
            usable = len(buf) - len(buf) % self.frame_bytes
            self._pending = buf[usable:]
            if not usable:
                return None
            raw = np.frombuffer(buf[:usable], dtype=self.dtype)
        else:
            frames = []
            pos, step = 0, len(self.sync) + self.frame_bytes
            while True:
                idx = buf.find(self.sync, pos)
                if idx < 0:
                    pos = max(pos, len(buf) - len(self.sync) + 1)
                    break
                if len(buf) - idx < step:
                    pos = idx
                    break
                frames.append(buf[idx + len(self.sync): idx + step])
                pos = idx + step
            self._pending = buf[pos:]
            if not frames:
                return None
            raw = np.frombuffer(b"".join(frames), dtype=self.dtype)
        return Block(raw.reshape(-1, self.channels).astype(np.float64), list(self.names))


@dataclass
class ParserConfig:
    """User-facing parser settings shared by all byte-stream sources."""

    mode: str = "text"  # "text" or "binary"
    x_from_first: bool = False
    names: list[str] = field(default_factory=list)
    dtype: str = "int16"
    channels: int = 1
    big_endian: bool = False
    sync_hex: str = ""
    sample_rate: float = 0.0  # binary: if > 0, x = sample index / rate

    def make_parser(self) -> TextLineParser | BinaryParser:
        if self.mode == "binary":
            try:
                sync = bytes.fromhex(self.sync_hex.replace("0x", "").replace(" ", ""))
            except ValueError as exc:
                raise ValueError(f"sync word must be hex bytes, e.g. 'AA 55' ({exc})") from None
            parser = BinaryParser(self.dtype, self.channels, self.big_endian, sync)
            if self.names:
                parser.names = (self.names + parser.names[len(self.names):])[: self.channels]
            return parser
        return TextLineParser(self.x_from_first, self.names)
