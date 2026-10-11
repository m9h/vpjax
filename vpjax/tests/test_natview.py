"""Tests for the NATVIEW loader's alignment and binning, on synthetic streams."""

import gzip
import json

import numpy as np
import pytest

from vpjax.validation.natview import _align, load_eye, load_respiration


class TestAlign:
    def test_recovers_offset_and_tolerates_quantisation(self):
        eeg = 10.0 + 2.1 * np.arange(200)
        stream = np.round((eeg - 3.25) * 62.5) / 62.5        # 62.5 Hz timestamps
        a, b, spread = _align(stream, eeg, fs=62.5)
        assert abs(a - 3.25) < 0.02 and abs(b - 1.0) < 1e-4
        assert spread <= 2.0 / 62.5 + 0.002

    def test_missing_leading_triggers_are_fine(self):
        eeg = 10.0 + 2.1 * np.arange(200)
        stream = eeg[5:] - 1.0
        a, b, _ = _align(stream, eeg, fs=250.0)
        assert abs(a - 1.0) < 1e-6

    def test_unlocked_clocks_rejected(self):
        eeg = 10.0 + 2.1 * np.arange(200)
        rng = np.random.default_rng(0)
        stream = eeg + rng.normal(scale=0.05, size=eeg.size)
        with pytest.raises(ValueError, match="not on locked clocks"):
            _align(stream, eeg, fs=250.0)

    def test_too_few_triggers(self):
        with pytest.raises(ValueError, match="too few"):
            _align(np.arange(5.0), np.arange(5.0), fs=250.0)


@pytest.fixture
def eye_run(tmp_path):
    """A 60 s eye-tracking file at 250 Hz with blinks in the middle third."""
    fs, dur = 250.0, 60.0
    t = np.arange(0, dur, 1 / fs) - 5.0        # eye clock starts 5 s before the EEG's
    pupil = 2000.0 + 200.0 * np.sin(2 * np.pi * t / 20.0)
    blink = np.zeros_like(t)
    blink[(t > 15) & (t < 35)] = 1.0
    pupil[blink > 0] = 0.0
    eeg_trig = 2.1 * np.arange(1, 25)                     # EEG clock
    trig = np.zeros_like(t)
    for tt in eeg_trig - 5.0:
        trig[np.argmin(np.abs(t - tt))] = 1.0
    cols = ["Time", "Gaze_X", "Gaze_Y", "Pupil_Area", "Resolution_X", "Resolution_Y",
            "Fixations", "Saccades", "Blinks", "Task_Start_End_Trigger",
            "Timer_Trigger_1_second", "fMRI_Volume_Trigger"]
    arr = np.zeros((t.size, len(cols)))
    arr[:, 0], arr[:, 3], arr[:, 8], arr[:, 11] = t, pupil, blink, trig
    tsv = tmp_path / "eye.tsv.gz"
    with gzip.open(tsv, "wt") as f:
        np.savetxt(f, arr, delimiter="\t")
    js = tmp_path / "eye.json"
    js.write_text(json.dumps({"SamplingFrequency": fs, "Columns": cols}))
    return {"eye": tsv, "eye_json": js}, eeg_trig


class TestEye:
    def test_bins_land_on_eeg_clock_and_mask_blinks(self, eye_run):
        paths, eeg_trig = eye_run
        out = load_eye(paths, eeg_trig, out_dt=0.25)
        assert abs(out["offset_s"] - 5.0) < 0.01
        t, v, valid = out["time_s"], out["values"], out["valid"]
        # Blink interval was 15–35 s on the eye clock = 20–40 s on the EEG clock.
        assert not valid[(t > 21) & (t < 39)].any()
        assert valid[(t > 1) & (t < 19)].all()
        assert np.isnan(v[~valid]).all()
        assert 0.6 < out["fraction_valid"] < 0.7


class TestRespiration:
    def test_amplitude_envelope_tracks_breathing_depth(self, tmp_path):
        fs = 62.5
        t = np.arange(0, 120, 1 / fs)
        depth = np.where(t < 60, 1.0, 2.0)
        resp = depth * np.sin(2 * np.pi * 0.25 * t)
        trig_times = 2.1 * np.arange(1, 50) - 0.7          # BIOPAC clock
        eeg_trig = trig_times + 0.7
        with gzip.open(tmp_path / "resp.tsv.gz", "wt") as f:
            np.savetxt(f, np.c_[t, resp], delimiter="\t")
        rows = np.repeat(np.round(trig_times * fs) / fs, 2)   # each pulse sampled twice
        with gzip.open(tmp_path / "trig.tsv.gz", "wt") as f:
            np.savetxt(f, np.c_[rows, 5.0 * np.ones_like(rows)], delimiter="\t")
        (tmp_path / "resp.json").write_text(json.dumps({"SamplingFrequency": fs}))
        paths = {"resp": tmp_path / "resp.tsv.gz", "resp_json": tmp_path / "resp.json",
                 "trigger": tmp_path / "trig.tsv.gz"}
        out = load_respiration(paths, eeg_trig, out_dt=1.0)
        assert abs(out["offset_s"] - 0.7) < 0.02
        tt, env = out["time_s"], out["values"]
        early, late = env[(tt > 10) & (tt < 50)], env[(tt > 70) & (tt < 110)]
        assert np.isclose(late.mean() / early.mean(), 2.0, rtol=0.1)
