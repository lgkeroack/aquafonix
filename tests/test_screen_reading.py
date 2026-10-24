"""Reading graphs off the screen: several coloured traces in one region,
and a scrolling spectrogram image decoded back into levels, stitched and
matched against a snipped patch (no real screen needed: a fake grabber
serves the images)."""

import time

import numpy as np
import pytest

from aquafonix.acoustics.analyzer import AnalysisSettings, Analyzer
from aquafonix.acoustics.signatures import Matcher, SignatureLibrary
from aquafonix.buffer import RingBuffer
from aquafonix.screen.capture import Region, ScreenSource
from aquafonix.screen.extract import (Calibration, TraceDef, TraceSettings, auto_trace_colors,
                                      background_color, extract_traces)
from aquafonix.screen.specread import ColourScale, decode_levels, sample_colour_bar

SCALE = ColourScale("inferno", low_db=40, high_db=120, tolerance=40)
LUT, _ = SCALE.lut()


def render(levels, lut=LUT, lo=40, hi=120):
    idx = np.clip(((levels - lo) / (hi - lo) * (len(lut) - 1)).round().astype(int), 0,
                  len(lut) - 1)
    return np.ascontiguousarray(lut[idx].round().astype(np.uint8))


@pytest.fixture(scope="module")
def two_lines():
    h, w = 300, 600
    img = np.full((h, w, 3), 255, np.uint8)
    img[::50] = 210
    img[:, ::60] = 210  # grey grid
    cols = np.arange(w)
    ya = (150 - 80 * np.sin(cols / 40)).astype(int)
    yb = (150 + 60 * np.cos(cols / 25)).astype(int)
    for y, c in ((ya, (220, 40, 40)), (yb, (30, 90, 220))):
        for dy in (0, 1):
            img[np.clip(y + dy, 0, h - 1), cols] = c
    return img, ya, yb


def test_auto_detects_line_colours(two_lines):
    img = two_lines[0]
    found = auto_trace_colors(img, background_color(img), 60, 2)
    assert len(found) == 2
    assert any(abs(c[0] - 220) < 30 for c in found) and any(abs(c[2] - 220) < 30 for c in found)


def test_extracts_both_traces(two_lines):
    img, ya, yb = two_lines
    st = TraceSettings(mode="colors", traces=(TraceDef("red", (220, 40, 40)),
                                              TraceDef("blue", (30, 90, 220))))
    r = extract_traces(img, st)
    assert r.names == ["red", "blue"]
    assert np.nanmean(np.abs(r.traces[0].y_px - (ya + 1))) < 1.5
    assert np.nanmean(np.abs(r.traces[1].y_px - (yb + 1))) < 1.5
    assert min(t.coverage for t in r.traces) > 0.95


ROWS, TOTAL, VIEW = 200, 1800, 400
CALLS = [100, 500, 900, 1300, 1700]


@pytest.fixture(scope="module")
def truth():
    t = np.full((ROWS, TOTAL), 55.0) + np.random.default_rng(0).normal(0, 2, (ROWS, TOTAL))
    for c0 in CALLS:  # upsweep 2 -> 4 kHz over 40 columns (0..10 kHz, top = high)
        for k in range(40):
            r = int(round(ROWS - 0.5 - (2000 + 2000 * k / 40) / 10000 * ROWS))
            t[max(r - 2, 0):r + 3, c0 + k] = 100.0
    return np.clip(t, 40, 120)


def test_colours_decode_to_levels(truth):
    dec = decode_levels(render(truth), SCALE)
    assert np.nanmean(np.abs(dec - truth)) < 1.5 and np.isfinite(dec).mean() > 0.99


def test_levels_from_picked_colour_bar(truth):
    bar = np.repeat(render(np.linspace(120, 40, 120))[:, None, :], 12, axis=1)
    picked = sample_colour_bar(bar, 0, 0, 12, 120)
    dec = decode_levels(render(truth), ColourScale("bar", picked, 40, 120, 40))
    assert np.nanmean(np.abs(dec - truth)) < 2.0


def test_scrolling_spectrogram_is_stitched_and_matched(truth, tmp_path):
    class FakeScreen:
        pos = VIEW

        def grab(self, region):
            im = render(truth[:, self.pos - VIEW:self.pos])
            self.pos = min(TOTAL, self.pos + 4)  # scrolls 4 px per frame
            return im

        def close(self):
            pass

    fake = FakeScreen()
    cal = Calibration(float(ROWS), 0.0, 0.0, 10000.0, 0.0, 0.0, float(VIEW), 1.0)
    src = ScreenSource(Region(0, 0, VIEW, ROWS), fps=100, settings=TraceSettings(calibration=cal),
                       output="spectrogram", grabber_factory=lambda: fake)
    src.colour_scale = SCALE
    lib = SignatureLibrary(tmp_path / "sig")
    buf = RingBuffer(200_000)
    an = Analyzer(buf, Matcher(lib), AnalysisSettings(), None, 120)
    src.add_frame_listener(an.feed_frames)
    src.start(buf)
    an.start()
    added = False
    t0 = time.perf_counter()
    try:
        while fake.pos < TOTAL and time.perf_counter() - t0 < 60:
            time.sleep(0.05)
            if not added and an.spectro is not None and an.spectro.count > VIEW + 300:
                t_all, _ = an.spectro.latest(an.spectro.count)
                p = an.patch(t_all[498], t_all[542])  # snip the second call
                lib.add_patch_example("screen call", p["freqs"], p["frame_period"], p["db"],
                                      p["floor"], 1500, 4500)
                added = True
        time.sleep(1.0)
    finally:
        src.stop()
        an.stop()
    assert abs(an.spectro.count - TOTAL) <= 8
    t_all, d_all = an.spectro.latest(an.spectro.count)
    r3k = np.argmin(np.abs(an.freqs - 3000))
    assert d_all[520, r3k] > 90 and d_all[300, r3k] < 65
    sus = [d for d in an.all_detections() if d.status == "suspected"]
    later = [t_all[c + 20] for c in (900, 1300, 1700) if c + 20 < len(t_all)]
    assert added and later
    assert all(any(d.t0 - 0.2 <= tc <= d.t1 + 0.2 for d in sus) for tc in later)
