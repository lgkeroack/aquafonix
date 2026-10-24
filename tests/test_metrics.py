"""Standard metrics against analytic values (ISO 18405 levels, IEC 61260-1
fractional-octave bands, Welch PSD scaling, calibration)."""

import numpy as np
import pytest

from aquafonix.acoustics import metrics as M
from aquafonix.acoustics.analyzer import HydrophoneCalibration

FS = 48000.0
T = np.arange(int(2 * FS)) / FS
SINE = 0.5 * np.sin(2 * np.pi * 1000 * T)
BANDS = M.third_octave_bands(FS / 2)


def levels(x, nfft):
    f, p = M.welch(x, FS, nfft)
    return f, p, dict(zip([b.label for b in BANDS], M.band_levels(f, p, BANDS)))


def test_sine_levels():
    rms_db = 20 * np.log10(0.5 / np.sqrt(2))
    assert M.spl_rms(SINE) == pytest.approx(rms_db, abs=0.01)
    assert M.spl_peak(SINE) == pytest.approx(20 * np.log10(0.5), abs=0.01)
    assert M.spl_peak_to_peak(SINE) == pytest.approx(0.0, abs=0.01)
    assert M.sel(SINE, FS) == pytest.approx(rms_db + 10 * np.log10(2), abs=0.01)


def test_welch_integral_is_mean_square():
    f, p = M.welch(SINE, FS, 32768)
    assert 10 * np.log10(p.sum() * (f[1] - f[0])) == pytest.approx(M.spl_rms(SINE), abs=0.05)


@pytest.mark.parametrize("f0,label", [(1000.0, "1 kHz"), (60.0, "63 Hz"), (75.0, "80 Hz"),
                                      (130.0, "125 Hz")])
def test_tone_lands_in_its_third_octave(f0, label):
    y = 0.1 * np.sin(2 * np.pi * f0 * T)
    _f, _p, lv = levels(y, 32768)
    assert lv[label] == pytest.approx(M.spl_rms(y), abs=0.3)


def test_white_noise_psd_and_band_level():
    n = 0.1 * np.random.default_rng(1).standard_normal(int(10 * FS))
    assert M.spl_rms(n) == pytest.approx(-20.0, abs=0.05)
    f, p, lv = levels(n, 4096)
    psd_db = -20 - 10 * np.log10(FS / 2)
    assert 10 * np.log10(np.median(p[10:-10])) == pytest.approx(psd_db, abs=0.3)
    b1k = next(b for b in BANDS if b.label == "1 kHz")
    assert lv["1 kHz"] == pytest.approx(psd_db + 10 * np.log10(b1k.high - b1k.low), abs=0.3)


def test_band_centres_follow_iec_61260():
    labels = [b.label for b in BANDS]
    assert labels[0] == "10 Hz" and labels[-1] == "20 kHz" and len(BANDS) == 34
    assert BANDS[-1].high <= FS / 2
    b = next(b for b in BANDS if b.label == "1 kHz")
    assert b.centre == pytest.approx(1000.0, rel=1e-6)
    assert b.high / b.low == pytest.approx(10 ** 0.1, rel=1e-6)


@pytest.mark.parametrize("sens,gain,vfs,offset", [(-165.0, 0.0, 1.0, 165.0),
                                                  (-170.0, 20.0, 2.0, 20 * np.log10(2) + 150)])
def test_calibration_offset(sens, gain, vfs, offset):
    cal = HydrophoneCalibration(True, sens, gain, vfs)
    assert cal.offset_db == pytest.approx(offset, abs=1e-9)
    assert cal.units == "dB re 1 µPa"


def test_spectral_descriptors_peak():
    f, p = M.welch(SINE, FS, 32768)
    assert M.spectral_descriptors(f, p)["peak_hz"] == pytest.approx(1000.0, abs=f[1] - f[0])


def test_fast_percentiles_match_numpy():
    rng = np.random.default_rng(3)
    d = rng.normal(-60, 5, (400, 300))
    d[rng.random(d.shape) < 0.05] = np.nan
    got = M.percentiles_db(d, axis=0)
    want = np.nanpercentile(d, [5, 50, 95], axis=0)
    for k, w in zip((5, 50, 95), want):
        np.testing.assert_allclose(got[k], w, atol=0.6)
