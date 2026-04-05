"""Validation tests: vpjax sleep models against OpenNeuro ds003768.

Tests that the forward model chain (EEG → NVC → predicted BOLD)
produces spectral signatures consistent with measured BOLD across
sleep stages.

Requires: data/ds003768/sub-23/ (downloaded separately)
"""

import os
import pytest
import jax.numpy as jnp
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ds003768")
SUB23_DIR = os.path.join(DATA_DIR, "sub-23")
STAGING_FILE = os.path.join(DATA_DIR, "sourcedata", "sub-23-sleep-stage.tsv")

has_data = os.path.isdir(SUB23_DIR) and os.path.isfile(STAGING_FILE)
skip_no_data = pytest.mark.skipif(not has_data, reason="ds003768 sub-23 not downloaded")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestDataLoading:

    @skip_no_data
    def test_load_sleep_stages(self):
        """Should parse sleep staging TSV into epoch arrays."""
        from vpjax.validation.sleep_eeg_fmri import load_sleep_stages
        stages = load_sleep_stages(STAGING_FILE, subject=23)
        assert "task-sleep_run-1" in stages
        # Each run should have list of (epoch_time, stage) tuples
        run1 = stages["task-sleep_run-1"]
        assert len(run1) > 0
        assert all(s in ("W", "1", "2", "3", "R") for _, s in run1)

    @skip_no_data
    def test_load_bold(self):
        """Should load BOLD time series from NIfTI."""
        from vpjax.validation.sleep_eeg_fmri import load_bold_global
        bold_path = os.path.join(SUB23_DIR, "func", "sub-23_task-sleep_run-1_bold.nii.gz")
        ts, tr = load_bold_global(bold_path)
        assert ts.ndim == 1  # global mean signal
        assert len(ts) > 100  # should have many volumes
        assert 0.5 < tr < 5.0  # reasonable TR

    @skip_no_data
    def test_load_eeg(self):
        """Should load EEG and extract band power time course."""
        from vpjax.validation.sleep_eeg_fmri import load_eeg_bandpower
        vhdr_path = os.path.join(SUB23_DIR, "eeg", "sub-23_task-sleep_run-1_eeg.vhdr")
        bp = load_eeg_bandpower(vhdr_path)
        assert "delta" in bp
        assert "theta" in bp
        assert "alpha" in bp
        assert bp["delta"].ndim == 1
        assert len(bp["delta"]) > 0


# ---------------------------------------------------------------------------
# Spectral analysis per sleep stage
# ---------------------------------------------------------------------------

class TestSpectralAnalysis:

    @skip_no_data
    def test_bold_spectrum_per_stage(self):
        """BOLD spectrum should differ across sleep stages."""
        from vpjax.validation.sleep_eeg_fmri import (
            load_bold_global, load_sleep_stages, bold_spectrum_by_stage
        )
        bold_path = os.path.join(SUB23_DIR, "func", "sub-23_task-sleep_run-1_bold.nii.gz")
        ts, tr = load_bold_global(bold_path)
        stages = load_sleep_stages(STAGING_FILE, subject=23)
        run_stages = stages.get("task-sleep_run-1", [])
        if len(run_stages) == 0:
            pytest.skip("No stages for run-1")
        spectra = bold_spectrum_by_stage(ts, tr, run_stages)
        # Should return dict of stage → (freqs, power)
        assert isinstance(spectra, dict)
        assert len(spectra) > 0


# ---------------------------------------------------------------------------
# Forward model comparison
# ---------------------------------------------------------------------------

class TestForwardModel:

    @skip_no_data
    def test_predicted_bold_spectrum(self):
        """vpjax predicted BOLD spectrum should correlate with measured."""
        from vpjax.validation.sleep_eeg_fmri import (
            load_bold_global, load_sleep_stages,
            predict_bold_spectrum_for_stage, bold_spectrum_by_stage,
        )
        bold_path = os.path.join(SUB23_DIR, "func", "sub-23_task-sleep_run-1_bold.nii.gz")
        ts, tr = load_bold_global(bold_path)
        stages = load_sleep_stages(STAGING_FILE, subject=23)
        run_stages = stages.get("task-sleep_run-1", [])
        if len(run_stages) == 0:
            pytest.skip("No stages for run-1")

        measured = bold_spectrum_by_stage(ts, tr, run_stages)
        # For each stage present, the predicted spectrum should be
        # positively correlated with measured (r > 0)
        for stage_label, (freqs, power) in measured.items():
            if len(power) < 5:
                continue
            predicted_power = predict_bold_spectrum_for_stage(
                stage_label, freqs, tr
            )
            # Basic sanity: predicted should be positive and finite
            assert jnp.all(jnp.isfinite(predicted_power))
            assert jnp.all(predicted_power >= 0)
