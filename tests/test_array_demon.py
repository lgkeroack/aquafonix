"""Bearings from a hydrophone pair and DEMON propeller analysis."""

import numpy as np
import pytest

from aquafonix.acoustics.array import (ArrayGeometry, bearing_from_delay, bearing_scan, gcc_phat,
                                       tdoa)
from aquafonix.acoustics.demon import demon
from signals import delayed, pair_from_bearing, propeller_noise

FS = 48000
GEO = ArrayGeometry(True, 0, 1, 1.0)


def test_gcc_phat_finds_a_fractional_delay():
    x = np.random.default_rng(0).standard_normal(FS)
    tau = 123.4e-6
    lags, cc = gcc_phat(delayed(x, FS, tau), x, FS, GEO.max_delay)
    assert lags[np.argmax(cc)] == pytest.approx(tau, abs=2e-6)


@pytest.mark.parametrize("bearing", [0.0, 33.0, 60.0, 90.0, 137.0, 180.0])
def test_bearing_from_delay(bearing):
    src = np.random.default_rng(1).standard_normal(FS // 2)
    a, b = pair_from_bearing(src, FS, bearing, noise=0.3)
    tau, quality = tdoa(a, b, FS, GEO.max_delay)
    assert bearing_from_delay(tau, GEO) == pytest.approx(bearing, abs=3.0 if bearing in (0, 180)
                                                         else 1.0)
    assert quality > 5
    scan = bearing_scan(a, b, FS, GEO, np.arange(0, 181, 2.0))
    assert abs(np.arange(0, 181, 2.0)[np.argmax(scan)] - bearing) <= 2.0


@pytest.mark.parametrize("fs,shaft,blades,dur", [(48000, 3.2, 5, 20), (96000, 2.0, 4, 15),
                                                 (48000, 4.0, 3, 5), (48000, 1.6, 7, 10)])
def test_demon_finds_shaft_and_blades(fs, shaft, blades, dur):
    x = propeller_noise(fs, dur, shaft, blades, np.random.default_rng(1))
    r = demon(x, fs)
    assert r.blades == blades
    assert r.shaft_hz == pytest.approx(shaft, abs=0.1)
    assert "blades" in r.describe()


def test_demon_ignores_clicks_in_the_window():
    fs = 48000
    rng = np.random.default_rng(4)
    x = propeller_noise(fs, 10, 3.0, 4, rng)
    for c in (2.0, 7.0):
        i = int(c * fs)
        x[i:i + 200] += 0.8 * np.hanning(200) * np.sin(2 * np.pi * 5000 * np.arange(200) / fs)
    r = demon(x, fs)
    assert r.blades == 4 and r.shaft_hz == pytest.approx(3.0, abs=0.1)


def test_demon_no_false_propeller_on_noise():
    calls = 0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        x = rng.standard_normal(FS * 10)
        if seed % 2:
            x[FS * 3:FS * 3 + 200] += 30 * np.hanning(200)  # a bang
        calls += demon(x, FS).blades is not None
    assert calls == 0
