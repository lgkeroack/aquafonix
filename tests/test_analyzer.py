"""The analysis pipeline on synthetic hydrophone data: band filter, levels,
noise floor, spike detection and "Suspected: …" matching."""

import time

import numpy as np
import pytest

from aquafonix.acoustics.analyzer import AnalysisSettings, Analyzer
from aquafonix.acoustics.array import ArrayGeometry
from aquafonix.acoustics.filters import BandSpec, StreamingFIR, design_fir
from aquafonix.acoustics.signatures import Matcher, SignatureLibrary
from aquafonix.buffer import RingBuffer
from signals import add_chirp, pair_from_bearing

FS = 48000
CALLS = [2, 8, 14, 20, 26, 32]


@pytest.mark.parametrize("f0,passes", [(2000, True), (300, False), (8000, False)])
def test_band_pass_response(f0, passes):
    taps = design_fir(BandSpec("bandpass", 1000, 3000), FS)
    fir = StreamingFIR(taps, 1)
    tone = np.sin(2 * np.pi * f0 * np.arange(FS) / FS)[:, None]
    out = np.concatenate([fir.process(tone[i:i + 4096]) for i in range(0, len(tone), 4096)])
    gain = 20 * np.log10(np.std(out[len(taps):]) / np.std(tone))
    assert gain > -0.5 if passes else gain < -50


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """40 s with six upsweeps, two different downsweeps and a click; the first
    call is identified after 4 s, the rest is then analysed."""
    rng = np.random.default_rng(7)
    dur = 40.0
    x = 0.01 * rng.standard_normal(int(dur * FS))
    for c in CALLS:
        add_chirp(x, FS, c, 1500, 3000, 0.4, 0.1)
    for c in (5, 17):
        add_chirp(x, FS, c, 6000, 4000, 0.4, 0.1)
    x[int(11 * FS):int(11 * FS) + 40] += 0.8 * rng.standard_normal(40)
    root = tmp_path_factory.mktemp("analyzer")
    lib = SignatureLibrary(root / "signatures")
    buf = RingBuffer(int(60 * FS), np.float32, FS)
    settings = AnalysisSettings(band=BandSpec("bandpass", 1000, 3000), detect_in_band=False)
    an = Analyzer(buf, Matcher(lib), settings, FS, 60.0, lambda tt: 1_700_000_000 + tt)

    def feed(t_end):
        i1 = int(t_end * FS)
        for i in range(buf.total, i1, 4800):
            buf.append(None, x[i:min(i + 4800, i1)])
            while an._process():
                pass

    started = time.perf_counter()
    feed(4.0)
    clip = buf.range(1.95, 2.45)
    lib.add_example("test call", clip.y[:, 0], FS, 1400, 3200, floor=an.floor_snapshot())
    feed(dur)
    elapsed = time.perf_counter() - started
    an._make_report()  # normally refreshed once per wall-clock second
    return an, buf, elapsed, dur


def test_faster_than_real_time(run):
    _an, _buf, elapsed, dur = run
    assert elapsed < dur / 2


def test_later_calls_are_suspected(run):
    an = run[0]
    sus = [d for d in an.all_detections() if d.status == "suspected"]
    later = [c for c in CALLS if c > 4]
    hits = [c for c in later if any(abs(d.t0 - c) < 0.5 or d.t0 <= c + 0.2 <= d.t1 for d in sus)]
    assert hits == later
    assert all(d.label == "test call" and d.wall_time > 1_700_000_000 for d in sus)


def test_no_false_matches(run):
    an = run[0]
    later = [c for c in CALLS if c > 4]
    false = [d for d in an.all_detections() if d.status == "suspected"
             and not any(abs(d.t0 - c) < 1.0 for c in later)]
    assert not false


def test_other_sounds_are_unidentified_spikes(run):
    unid = [d for d in run[0].all_detections() if d.status == "unidentified"]
    assert any(abs(d.t0 - 11) < 0.3 for d in unid)  # the click
    assert any(abs(d.t0 - 17) < 0.5 for d in unid)  # a downsweep


def test_levels_and_report(run):
    an = run[0]
    assert an.intervals.count >= 39
    _t, v = an.intervals.latest(5)
    assert np.all(np.abs(v[:, 0] + 37) < 6)
    assert an.tol_series.count == an.intervals.count and len(an.tol_bands) == 34
    assert set(an.psd_percentiles()) == {5, 50, 95}
    rep = an.report
    assert np.isfinite(rep.level_db) and np.isfinite(rep.floor_db) and rep.stats


def test_filter_delay_compensated(run):
    an, buf = run[0], run[1]
    fx, rx = an.filtered.range(8.0, 8.5), buf.range(8.0, 8.5)
    lag = np.argmax(np.correlate(fx.y[:, 0], rx.y[:, 0], "full")) - (len(rx) - 1)
    assert abs(lag) <= 2


def test_bearings_per_detection_and_record(tmp_path):
    fs = 48000
    rng = np.random.default_rng(2)
    src = 0.02 * rng.standard_normal(fs * 12)
    for c in (3.0, 7.0, 10.0):
        i = int(c * fs)
        src[i:i + 200] += 0.8 * np.hanning(200) * np.sin(2 * np.pi * 5000 * np.arange(200) / fs)
    a, b = pair_from_bearing(src, fs, 125.0)
    buf = RingBuffer(fs * 20, np.float32, fs)
    geo = ArrayGeometry(True, 0, 1, 1.0)
    an = Analyzer(buf, Matcher(SignatureLibrary(tmp_path / "s")), AnalysisSettings(array=geo),
                  fs, 20.0)
    y = np.column_stack([a, b])
    for i in range(0, len(y), 4800):
        buf.append(None, y[i:i + 4800], ["A", "B"])
        while an._process():
            pass
    clicks = [d for d in an.all_detections() if any(abs(d.t0 - c) < 0.3 for c in (3, 7, 10))]
    assert len(clicks) == 3
    assert all(abs(d.bearing - 125.0) < 2 for d in clicks)
    _t, rows = an.btr.latest(an.btr.count)
    assert an.btr.count > 30
    assert abs(an.btr_bearings[np.argmax(np.median(rows, axis=0))] - 125.0) <= 2
