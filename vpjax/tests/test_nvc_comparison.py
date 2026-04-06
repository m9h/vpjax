"""Tests for NVC model comparison on WAND task data.

Compares four neurovascular coupling models:
1. Linear convolution (canonical HRF)
2. Balloon-Windkessel (Friston/Buxton)
3. Riera multi-compartment NVC
4. Riera + calibrated (with pCASL/TRUST constraints)

Evaluated on WAND sub-08033 category localiser (block design, 6s blocks).
"""

import os
import pytest
import jax.numpy as jnp
import numpy as np

WAND_DIR = os.path.expanduser("~/dev/wand")
SUB_DIR = os.path.join(WAND_DIR, "sub-08033", "ses-03")
BOLD_PATH = os.path.join(SUB_DIR, "func", "sub-08033_ses-03_task-categorylocaliser_run-1_bold.nii.gz")
EVENTS_PATH = os.path.join(SUB_DIR, "func", "sub-08033_ses-03_task-categorylocaliser_run-1_events.tsv")

has_wand = os.path.isfile(BOLD_PATH) and os.path.isfile(EVENTS_PATH)
skip_no_wand = pytest.mark.skipif(not has_wand, reason="WAND sub-08033 not available")


# ---------------------------------------------------------------------------
# Model definitions (synthetic — no data needed)
# ---------------------------------------------------------------------------

class TestNVCModels:
    """All four NVC models should produce valid BOLD predictions from a stimulus."""

    def test_linear_hrf_convolution(self):
        """Linear convolution with canonical HRF."""
        from vpjax.validation.nvc_comparison import predict_bold_linear
        stimulus = np.zeros(150)
        stimulus[10:13] = 1.0  # 6s block at TR=2
        bold = predict_bold_linear(stimulus, tr=2.0)
        assert bold.shape == stimulus.shape
        assert float(np.max(bold)) > 0

    def test_balloon_prediction(self):
        """Balloon-Windkessel model prediction."""
        from vpjax.validation.nvc_comparison import predict_bold_balloon
        stimulus = np.zeros(150)
        stimulus[10:13] = 1.0
        bold = predict_bold_balloon(stimulus, tr=2.0)
        assert bold.shape == stimulus.shape
        assert float(np.max(bold)) > 0

    def test_riera_prediction(self):
        """Riera multi-compartment model prediction."""
        from vpjax.validation.nvc_comparison import predict_bold_riera
        stimulus = np.zeros(150)
        stimulus[10:13] = 1.0
        bold = predict_bold_riera(stimulus, tr=2.0)
        assert bold.shape == stimulus.shape
        assert float(np.max(bold)) > 0

    def test_all_models_different(self):
        """Models should produce different predictions (not identical)."""
        from vpjax.validation.nvc_comparison import (
            predict_bold_linear, predict_bold_balloon, predict_bold_riera,
        )
        stimulus = np.zeros(150)
        stimulus[10:13] = 1.0
        b_lin = predict_bold_linear(stimulus, tr=2.0)
        b_bal = predict_bold_balloon(stimulus, tr=2.0)
        b_rie = predict_bold_riera(stimulus, tr=2.0)
        # Not identical
        assert not np.allclose(b_lin, b_bal, atol=1e-4)
        assert not np.allclose(b_bal, b_rie, atol=1e-4)

    def test_all_return_to_baseline(self):
        """All models should return near baseline after stimulus ends."""
        from vpjax.validation.nvc_comparison import (
            predict_bold_linear, predict_bold_balloon, predict_bold_riera,
        )
        stimulus = np.zeros(150)
        stimulus[5:8] = 1.0  # early block
        for predict_fn in [predict_bold_linear, predict_bold_balloon, predict_bold_riera]:
            bold = predict_fn(stimulus, tr=2.0)
            assert abs(float(bold[-1])) < 0.01


# ---------------------------------------------------------------------------
# WAND data loading
# ---------------------------------------------------------------------------

class TestWANDDataLoading:

    @skip_no_wand
    def test_load_bold_and_events(self):
        """Should load BOLD and parse events TSV."""
        from vpjax.validation.nvc_comparison import load_wand_task
        bold_ts, events, tr = load_wand_task(BOLD_PATH, EVENTS_PATH)
        assert bold_ts.ndim == 1
        assert len(bold_ts) == 125  # 125 volumes
        assert tr == 2.0
        assert len(events) > 10
        assert "trial_type" in events[0]

    @skip_no_wand
    def test_build_stimulus_from_events(self):
        """Should build a stimulus regressor from events."""
        from vpjax.validation.nvc_comparison import load_wand_task, build_stimulus
        bold_ts, events, tr = load_wand_task(BOLD_PATH, EVENTS_PATH)
        stim = build_stimulus(events, n_vols=len(bold_ts), tr=tr, condition="adult")
        assert stim.shape == bold_ts.shape
        assert float(np.max(stim)) > 0
        assert float(np.min(stim)) == 0.0


# ---------------------------------------------------------------------------
# Full comparison
# ---------------------------------------------------------------------------

class TestFullComparison:

    @skip_no_wand
    def test_run_comparison(self):
        """Should produce correlation results for each model × condition."""
        from vpjax.validation.nvc_comparison import run_nvc_comparison
        results = run_nvc_comparison(BOLD_PATH, EVENTS_PATH)
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "model" in r
            assert "condition" in r
            assert "r_value" in r
            assert np.isfinite(r["r_value"])

    @skip_no_wand
    def test_riera_beats_or_matches_balloon(self):
        """Riera should perform at least as well as Balloon on average."""
        from vpjax.validation.nvc_comparison import run_nvc_comparison
        results = run_nvc_comparison(BOLD_PATH, EVENTS_PATH)
        balloon_rs = [r["r_value"] for r in results if r["model"] == "balloon"]
        riera_rs = [r["r_value"] for r in results if r["model"] == "riera"]
        if balloon_rs and riera_rs:
            # Riera mean r should be >= balloon mean r (or close)
            assert np.mean(riera_rs) >= np.mean(balloon_rs) - 0.1
