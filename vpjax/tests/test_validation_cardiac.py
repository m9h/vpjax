"""Tests for cardiac extraction from EEG-embedded ECG and vasomotion prediction."""

import os
import pytest
import jax.numpy as jnp
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ds003768")
SUB23_DIR = os.path.join(DATA_DIR, "sub-23")
STAGING_FILE = os.path.join(DATA_DIR, "sourcedata", "sub-23-sleep-stage.tsv")

has_data = os.path.isdir(SUB23_DIR) and os.path.isfile(STAGING_FILE)
skip_no_data = pytest.mark.skipif(not has_data, reason="ds003768 sub-23 not downloaded")


class TestVasomotionPrediction:
    """Vasomotion-enhanced BOLD spectrum prediction."""

    def test_n3_more_low_freq_than_wake(self):
        """N3 predicted spectrum should have more low-frequency power."""
        from vpjax.validation.sleep_eeg_fmri import predict_bold_spectrum_with_vasomotion
        freqs = np.linspace(0.001, 0.1, 100)
        spec_w = np.array(predict_bold_spectrum_with_vasomotion("W", freqs, 2.0))
        spec_n3 = np.array(predict_bold_spectrum_with_vasomotion("3", freqs, 2.0))
        # Low-freq band (< 0.03 Hz) should be stronger in N3
        low_mask = freqs < 0.03
        assert np.sum(spec_n3[low_mask]) > np.sum(spec_w[low_mask])

    def test_vasomotion_peak_in_n3(self):
        """N3 should show a spectral peak near 0.02 Hz."""
        from vpjax.validation.sleep_eeg_fmri import predict_bold_spectrum_with_vasomotion
        freqs = np.linspace(0.005, 0.1, 200)
        spec_n3 = np.array(predict_bold_spectrum_with_vasomotion("3", freqs, 2.0))
        # Peak should be in the 0.01-0.04 Hz range
        peak_idx = np.argmax(spec_n3)
        peak_freq = freqs[peak_idx]
        assert 0.005 < peak_freq < 0.05

    def test_scales_with_depth(self):
        """Low-freq fraction should increase with sleep depth."""
        from vpjax.validation.sleep_eeg_fmri import predict_bold_spectrum_with_vasomotion
        freqs = np.linspace(0.005, 0.2, 200)
        low_mask = freqs < 0.03
        low_frac = {}
        for stage in ["W", "1", "2", "3"]:
            spec = np.array(predict_bold_spectrum_with_vasomotion(stage, freqs, 2.0))
            low_frac[stage] = float(np.sum(spec[low_mask]) / np.sum(spec))
        assert low_frac["3"] > low_frac["2"] > low_frac["1"]

    def test_output_finite(self):
        from vpjax.validation.sleep_eeg_fmri import predict_bold_spectrum_with_vasomotion
        freqs = np.linspace(0.01, 0.2, 50)
        for stage in ["W", "1", "2", "3", "R"]:
            spec = predict_bold_spectrum_with_vasomotion(stage, freqs, 2.0)
            assert jnp.all(jnp.isfinite(spec))
            assert jnp.all(spec >= 0)


class TestCardiacExtraction:
    """Heart rate extraction from ECG channel."""

    @skip_no_data
    def test_extract_heart_rate(self):
        """Should extract physiological HR from ECG channel."""
        from vpjax.validation.cardiac_from_eeg import extract_heart_rate
        vhdr = os.path.join(SUB23_DIR, "eeg", "sub-23_task-sleep_run-1_eeg.vhdr")
        hr = extract_heart_rate(vhdr)
        assert hr.ndim == 1
        assert len(hr) > 0
        # Valid epochs should have HR in physiological range
        valid = hr[~np.isnan(hr)]
        assert len(valid) > 0
        assert np.all(valid > 30)
        assert np.all(valid < 150)

    def test_cardiac_spectrum_peak(self):
        """Cardiac spectrum should peak near the cardiac frequency."""
        from vpjax.validation.cardiac_from_eeg import cardiac_bold_spectrum
        freqs = np.linspace(0.01, 3.0, 300)
        spec = np.array(cardiac_bold_spectrum(70.0, freqs))
        peak_freq = freqs[np.argmax(spec)]
        # Should peak near 70/60 ≈ 1.17 Hz
        assert 0.8 < peak_freq < 1.5

    def test_cardiac_spectrum_scales_with_hr(self):
        """Higher HR → peak shifts to higher frequency."""
        from vpjax.validation.cardiac_from_eeg import cardiac_bold_spectrum
        freqs = np.linspace(0.5, 2.5, 200)
        peak_60 = freqs[np.argmax(np.array(cardiac_bold_spectrum(60.0, freqs)))]
        peak_90 = freqs[np.argmax(np.array(cardiac_bold_spectrum(90.0, freqs)))]
        assert peak_90 > peak_60


class TestAllRunsValidation:

    @skip_no_data
    def test_run_validation(self):
        """Should produce results for multiple runs and stages."""
        from vpjax.validation.run_all_sleep_runs import run_validation
        results = run_validation(subject=23, use_vasomotion=False)
        assert len(results) > 0
        for r in results:
            assert "run" in r
            assert "stage" in r
            assert "r_balloon" in r
            assert np.isfinite(r["r_balloon"])
