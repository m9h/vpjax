"""Tests for in-scanner EEG cleaning: synthetic artifacts with known answers."""

import numpy as np
import pytest

from vpjax.validation.eeg_artifacts import (
    band_envelope,
    detect_r_peaks,
    downsample,
    subtract_bcg,
    subtract_gradient,
    subtract_locked_artifact,
)


def _synthetic(fs=1000.0, duration=60.0, tr=2.0, hr_bpm=72.0, seed=0):
    """EEG-like noise plus a trigger-locked gradient waveform and an R-locked BCG."""
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    eeg = rng.normal(scale=10e-6, size=(2, n))
    alpha = 20e-6 * np.sin(2 * np.pi * 10 * t)
    eeg[0] += alpha

    period = int(tr * fs)
    triggers = np.arange(period // 2, n - period, period)
    grad_wave = 2e-3 * np.sin(2 * np.pi * 37 * np.arange(period) / fs) ** 3
    gradient = np.zeros(n)
    for s in triggers:
        gradient[s:s + period] += grad_wave

    rr = int(60.0 / hr_bpm * fs)
    r_peaks = np.arange(rr, n - rr, rr)
    bcg_wave = 100e-6 * np.exp(-((np.arange(rr) / fs - 0.25) / 0.08) ** 2)
    bcg = np.zeros(n)
    ecg = np.zeros(n)
    for s in r_peaks:
        bcg[s:s + rr] += bcg_wave
        ecg[s:s + 80] += 1e-3 * np.hanning(80)   # ~80 ms QRS
    ecg += rng.normal(scale=20e-6, size=n)

    return {
        "clean": eeg, "data": eeg + gradient + bcg, "ecg": ecg, "fs": fs,
        "triggers": triggers, "r_peaks": r_peaks, "t": t,
    }


class TestLockedSubtraction:
    def test_removes_periodic_artifact(self):
        s = _synthetic()
        period = int(2.0 * s["fs"])
        out = subtract_locked_artifact(s["data"], s["triggers"], period, half_width=5)
        lo, hi = s["triggers"][0], s["triggers"][-1] + period
        resid = out[:, lo:hi] - s["clean"][:, lo:hi]
        # BCG remains (not locked to the trigger); gradient should be gone.
        assert np.sqrt(np.mean(resid**2)) < 1e-4
        before = np.sqrt(np.mean((s["data"][:, lo:hi] - s["clean"][:, lo:hi]) ** 2))
        assert before > 1e-3

    def test_rejects_too_few_repetitions(self):
        s = _synthetic(duration=10.0)
        with pytest.raises(ValueError, match="too few"):
            subtract_locked_artifact(s["data"], s["triggers"], 2000, half_width=10)

    def test_exclude_self_matches_within_tolerance(self):
        s = _synthetic()
        period = int(2.0 * s["fs"])
        a = subtract_locked_artifact(s["data"], s["triggers"], period, 5)
        b = subtract_locked_artifact(s["data"], s["triggers"], period, 5, exclude_self=True)
        assert np.allclose(a, b, atol=5e-5)


class TestGradient:
    def test_attenuation_and_metadata(self):
        s = _synthetic()
        out, info = subtract_gradient(s["data"], s["triggers"], half_width=5)
        assert info["period_samples"] == 2000
        assert info["trigger_jitter_samples"] == 0
        assert info["attenuation_db"] > 20.0

    def test_jittered_triggers_rejected(self):
        s = _synthetic()
        trig = s["triggers"].copy()
        trig[3] += 5
        with pytest.raises(ValueError, match="not locked"):
            subtract_gradient(s["data"], trig)


class TestBCG:
    def test_r_peaks_found(self):
        s = _synthetic()
        r = detect_r_peaks(s["ecg"], s["fs"])
        assert r.size == s["r_peaks"].size
        # The synthetic QRS is a Hanning bump, peak at sample 40 of 80.
        assert np.all(np.abs(r - (s["r_peaks"] + 40)) <= 5)

    def test_inverted_ecg_still_found(self):
        s = _synthetic()
        r = detect_r_peaks(-s["ecg"], s["fs"])
        assert r.size == s["r_peaks"].size

    def test_bcg_removed_after_gradient(self):
        s = _synthetic()
        period = int(2.0 * s["fs"])
        x = subtract_locked_artifact(s["data"], s["triggers"], period, 5)
        r = detect_r_peaks(s["ecg"], s["fs"])
        out, info = subtract_bcg(x, r, s["fs"], half_width=5)
        assert abs(info["heart_rate_bpm"] - 72.0) < 1.0
        lo = s["r_peaks"][6]
        hi = s["r_peaks"][-6]
        resid = out[:, lo:hi] - s["clean"][:, lo:hi]
        assert np.sqrt(np.mean(resid**2)) < 1.5e-5   # EEG noise is 1e-5


class TestResampling:
    def test_downsample_keeps_alpha(self):
        s = _synthetic()
        y, fs2 = downsample(s["clean"], s["fs"], 250.0)
        assert fs2 == 250.0
        assert y.shape[1] == s["clean"].shape[1] // 4
        t_c, env = band_envelope(y[0], fs2, (8.0, 13.0), out_dt=0.1)
        # Envelope of a 20 µV sinusoid is 20 µV (edges excluded); the
        # 10 µV broadband noise leaves ~1 µV in the alpha band.
        inner = env[5:-5]
        assert np.isclose(np.median(inner), 20e-6, rtol=0.1)
        assert np.allclose(inner, 20e-6, rtol=0.25)
        assert np.isclose(t_c[1] - t_c[0], 0.1)

    def test_downsample_requires_integer_ratio(self):
        s = _synthetic()
        with pytest.raises(ValueError, match="divide"):
            downsample(s["clean"], s["fs"], 300.0)
