"""Recordings: multi-file replay with clock times, calibration found in file
metadata, the whole-recording overview, batch reports and saved events."""

import csv
import datetime as dt
import time

import numpy as np
import pytest
import soundfile as sf

from aquafonix.acoustics.analyzer import AnalysisSettings, Analyzer
from aquafonix.acoustics.batch import BatchJob, run_batch
from aquafonix.acoustics.calibration_io import find_calibration
from aquafonix.acoustics.longterm import RecordingOverview
from aquafonix.acoustics.signatures import Matcher, SignatureLibrary
from aquafonix.buffer import RingBuffer
from aquafonix.events import EventStore
from aquafonix.sources.audiofile import AudioFileSource, parse_file_timestamp

FS = 48000
TONE_DB = 20 * np.log10(0.05 / np.sqrt(2))


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    """Two 10 s files: 500 Hz then 1 kHz, with calibration in a sidecar
    text file and in a WAV INFO comment."""
    d = tmp_path_factory.mktemp("recorder")
    rng = np.random.default_rng(3)
    for i, name in enumerate(["20231015_120000.wav", "20231015_120010.wav"]):
        tt = np.arange(10 * FS) / FS
        y = 0.05 * np.sin(2 * np.pi * (500 + 500 * i) * tt) + 0.005 * rng.standard_normal(len(tt))
        with sf.SoundFile(d / name, "w", FS, 1, subtype="PCM_24") as f:
            if i == 1:
                f.comment = "SN1234 hydrophone sensitivity -168.0 dB re 1V/uPa, gain 6 dB"
            f.write(y)
    (d / "20231015_120000.txt").write_text(
        "Hydrophone sensitivity: -170.5 dB re 1 V/uPa\nPreamp gain: 20 dB\n")
    return d


@pytest.mark.parametrize("name,want", [
    ("20231015_120000.wav", dt.datetime(2023, 10, 15, 12, 0, 0)),
    ("SoundTrap.231015120000.wav", dt.datetime(2023, 10, 15, 12, 0, 0)),
    ("AMAR_20231015T120000.250Z.flac", dt.datetime(2023, 10, 15, 12, 0, 0, 250000)),
])
def test_file_timestamps(name, want):
    ts = parse_file_timestamp(name)
    assert ts is not None
    got = dt.datetime.fromtimestamp(ts) if "Z" not in name else \
        dt.datetime.fromtimestamp(ts, dt.timezone.utc).replace(tzinfo=None)
    assert got == want


def test_calibration_from_metadata(rec):
    h0 = find_calibration(str(rec / "20231015_120000.wav"))
    assert h0 is not None and h0.sensitivity_db == -170.5 and h0.gain_db == 20
    h1 = find_calibration(str(rec / "20231015_120010.wav"))
    assert h1 is not None and h1.sensitivity_db == -168 and h1.gain_db == 6


def test_folder_replays_as_one_recording(rec, tmp_path):
    src = AudioFileSource(str(rec), speed=0)
    assert len(src.segments) == 2 and src.duration == pytest.approx(20)
    w = src.wall_time(12.5)
    assert dt.datetime.fromtimestamp(w) == dt.datetime(2023, 10, 15, 12, 0, 12, 500000)
    buf = RingBuffer(int(30 * FS), np.float32, FS)
    an = Analyzer(buf, Matcher(SignatureLibrary(tmp_path / "sig")), AnalysisSettings(), FS, 30,
                  src.wall_time)
    src.throttle = lambda: an.processed_index
    src.start(buf)
    an.start()
    t0 = time.perf_counter()
    while not src.finished and time.perf_counter() - t0 < 60:
        time.sleep(0.05)
    time.sleep(0.5)
    src.stop()
    an.stop()
    assert buf.total == 20 * FS
    assert np.allclose(np.diff(buf.range(9.99, 10.01).x), 1 / FS)  # continuous across files
    assert an.intervals.count >= 19
    _t, v = an.intervals.latest(an.intervals.count)
    assert np.median(v[:9, 0]) == pytest.approx(TONE_DB, abs=0.2)


def test_recording_overview(rec):
    ov = RecordingOverview(AudioFileSource(str(rec), speed=1), 0, columns=200)
    ov.start()
    t0 = time.perf_counter()
    while not ov.complete and time.perf_counter() - t0 < 30:
        time.sleep(0.05)
    assert ov.complete and np.isfinite(ov.ltsa).mean() > 0.95
    half = ov.columns // 2
    r500 = np.argmin(np.abs(ov.freqs - 500))
    r1k = np.argmin(np.abs(ov.freqs - 1000))
    assert ov.ltsa[r500, :half - 2].mean() > ov.ltsa[r500, half + 2:].mean() + 20
    assert ov.ltsa[r1k, half + 2:].mean() > ov.ltsa[r1k, :half - 2].mean() + 20


def test_batch_report_uses_file_calibration(rec, tmp_path):
    out = tmp_path / "batch.csv"
    res = run_batch(BatchJob([str(p) for p in sorted(rec.glob("*.wav"))], str(out), 1.0))
    rows = [r for r in csv.reader(open(out)) if r and not r[0].startswith("#")]
    assert res["intervals"] == 20 and len(rows) == 21
    first = dict(zip(rows[0], rows[1]))
    assert float(first["SPL"]) == pytest.approx(TONE_DB + 170.5 - 20, abs=0.3)
    summary = list(csv.reader(open(res["summary"])))
    assert summary[0][:4] == ["file", "intervals", "duration_s", "SPL_L5"]
    assert len(summary) == 4  # two files + overall


def test_event_store(tmp_path):
    store = EventStore(tmp_path / "events")
    x = np.arange(2 * FS) / FS + 5
    y = 0.1 * np.random.default_rng(0).standard_normal((len(x), 1))
    store.add(x, y, ["ch1"], FS, "capture", "", ["seal", "near buoy"], "a note", "test",
              1_697_364_005.0)
    recs = store.list()
    assert len(recs) == 1 and recs[0].tags == ["seal", "near buoy"]
    _x, y2, rate = store.load_clip(recs[0])
    assert rate == FS and abs(len(y2) - 2 * FS) <= 1
    recs[0].notes, recs[0].verified = "edited", True
    store.save(recs[0])
    assert store.list()[0].notes == "edited" and store.list()[0].verified
    assert store.export_csv(str(tmp_path / "ev.csv")) == 1
