"""Ring buffer and stream parsers."""

import struct

import numpy as np
import pytest

from aquafonix.buffer import RingBuffer
from aquafonix.parsing import BinaryParser, ParserConfig, TextLineParser, parse_line


def test_ring_buffer_wraps_and_keeps_absolute_indices():
    buf = RingBuffer(10)
    buf.append(np.arange(7.0), np.arange(7.0)[:, None] * 2, ["a"])
    buf.append(np.arange(7.0, 15.0), np.arange(7.0, 15.0)[:, None] * 2)
    assert len(buf) == 10 and buf.total == 15 and buf.first_index == 5
    snap = buf.latest(3)
    np.testing.assert_array_equal(snap.x, [12, 13, 14])
    np.testing.assert_array_equal(snap.y[:, 0], [24, 26, 28])
    later = buf.from_index(13)
    assert later.first_index == 13 and list(later.x) == [13, 14]


def test_uniform_buffer_has_implicit_x_and_compact_storage():
    buf = RingBuffer(1000, np.float32, sample_rate=100.0)
    buf.append(None, np.ones((250, 2)), ["L", "R"])
    snap = buf.range(1.0, 1.5)
    assert buf.dtype == np.float32  # stored compactly; reads come back as float64
    assert snap.x[0] == pytest.approx(1.0) and np.allclose(np.diff(snap.x), 0.01)
    assert buf.duration == pytest.approx(2.49)


def test_new_channel_appears_with_nan_history():
    buf = RingBuffer(100)
    buf.append([0, 1], [[1.0], [2.0]], ["a"])
    buf.append([2], [[3.0, 9.0]], ["a", "b"])
    snap = buf.latest()
    assert snap.names == ["a", "b"]
    assert np.isnan(snap.y[0, 1]) and snap.y[2, 1] == 9.0


def test_envelope_keeps_extremes():
    buf = RingBuffer(100_000, sample_rate=1000.0)
    y = np.zeros(50_000)
    y[12_345] = 5.0
    y[40_000] = -3.0
    buf.append(None, y)
    env = buf.envelope(None, None, 100)
    assert env.y.max() == 5.0 and env.y.min() == -3.0 and len(env) <= 200


@pytest.mark.parametrize("line,want", [
    ("1.5, -2, 3e2", [1.5, -2.0, 300.0]),
    ("temp=21.5 hum:40", {"temp": 21.5, "hum": 40.0}),
    ("no numbers here", None),
])
def test_parse_line(line, want):
    assert parse_line(line) == want


def test_text_parser_handles_split_chunks_and_header():
    p = TextLineParser()
    assert p.feed(b"time pressure\n1 10\n2 2") is not None
    block = p.feed(b"0\n3 30\n")
    assert block.names == ["time", "pressure"]
    np.testing.assert_array_equal(block.y, [[2, 20], [3, 30]])


def test_text_parser_x_from_first_column():
    p = TextLineParser(x_from_first=True)
    block = p.feed(b"0.0 1 2\n0.5 3 4\n")
    np.testing.assert_array_equal(block.x, [0.0, 0.5])
    assert block.y.shape == (2, 2)


def test_binary_parser_with_sync_word_resyncs():
    p = BinaryParser("int16", 2, sync=b"\xaa\x55")
    frame = lambda a, b: b"\xaa\x55" + struct.pack("<hh", a, b)  # noqa: E731
    data = b"junk" + frame(1, 2) + frame(3, 4) + frame(5, 6)[:3]
    block = p.feed(data)
    np.testing.assert_array_equal(block.y, [[1, 2], [3, 4]])
    block = p.feed(frame(5, 6)[3:])
    np.testing.assert_array_equal(block.y, [[5, 6]])


def test_parser_config_rejects_bad_sync():
    with pytest.raises(ValueError):
        ParserConfig(mode="binary", sync_hex="zz").make_parser()
