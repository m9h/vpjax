"""Tests for vpjax.hemodynamics.cvr.

Synthetic round-trip:
* Build a tiny 4D BOLD volume from known per-region Balloon parameters
  driven by a block-design stimulus.
* Use the data-driven stimulus extractor to recover a CO2 proxy.
* Fit per-region Balloon params and check loss + scalar CVR shape.

Tests stay small (3 regions, 60 timepoints) so the JIT compile and
optimization complete in seconds.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vpjax._types import BalloonParams
from vpjax.hemodynamics.balloon import solve_balloon
from vpjax.hemodynamics.bold import observe_bold
from vpjax.hemodynamics.cvr import (
    CVRFitResult,
    extract_global_stimulus,
    fit_cvr_per_region,
    resample_etco2_stimulus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_cvr_volume():
    """Generate a 4D BOLD volume with 3 regions of differing kappa.

    Spatial layout: 4×4×3 voxels, three labels (1, 2, 3) of 4 voxels
    each, plus background.  Time series length 60 at TR=2s, with a
    20s block hypercapnia stimulus starting at t=20s.

    Region 1: kappa=0.6 (slow vasodilation)
    Region 2: kappa=1.0 (DCM default)
    Region 3: kappa=1.5 (fast vasodilation)
    """
    rng = np.random.default_rng(0)

    tr = 2.0
    dt = 0.1
    n_t = 60
    sub = int(round(tr / dt))
    t_total = n_t * sub
    stim = jnp.zeros(t_total).at[int(20.0 / dt):int(40.0 / dt)].set(1.0)

    region_kappa = {1: 0.6, 2: 1.0, 3: 1.5}
    region_bold = {}
    for rid, k in region_kappa.items():
        params = BalloonParams(kappa=jnp.array(k))
        _, traj = solve_balloon(params, stim, dt=dt)
        bold = observe_bold(traj)[::sub][:n_t]
        region_bold[rid] = np.asarray(bold)

    # Synth 4D with 3 distinct labels.  Baseline 1000.
    X, Y, Z = 4, 4, 3
    bold_4d = np.zeros((X, Y, Z, n_t), dtype=np.float32)
    region_volume = np.zeros((X, Y, Z), dtype=np.int32)
    brain_mask = np.zeros((X, Y, Z), dtype=np.float32)

    layout = {
        1: [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1)],
        2: [(2, 0, 1), (2, 1, 1), (3, 0, 1), (3, 1, 1), (2, 2, 1), (3, 2, 1)],
        3: [(0, 2, 2), (0, 3, 2), (1, 2, 2), (1, 3, 2), (2, 2, 2), (2, 3, 2)],
    }

    for rid, voxels in layout.items():
        ts = region_bold[rid]
        for x, y, z in voxels:
            region_volume[x, y, z] = rid
            brain_mask[x, y, z] = 1.0
            # bring into "raw" intensity space + per-voxel noise
            noise = rng.normal(0, 0.001, size=n_t).astype(np.float32)
            bold_4d[x, y, z, :] = 1000.0 * (1.0 + ts + noise)

    return {
        "bold_4d": bold_4d,
        "brain_mask": brain_mask,
        "region_volume": region_volume,
        "tr": tr,
        "dt": dt,
        "stimulus": stim,
        "true_kappa": region_kappa,
    }


# ---------------------------------------------------------------------------
# Stimulus extraction
# ---------------------------------------------------------------------------

class TestExtractGlobalStimulus:

    def test_shape_matches_ode_grid(self, synth_cvr_volume):
        d = synth_cvr_volume
        stim, g = extract_global_stimulus(
            d["bold_4d"], d["brain_mask"], tr=d["tr"], dt=d["dt"],
        )
        n_t = d["bold_4d"].shape[-1]
        sub = int(round(d["tr"] / d["dt"]))
        assert stim.shape == (n_t * sub,)
        assert g.shape == (n_t,)

    def test_zero_mean_and_amplitude(self, synth_cvr_volume):
        d = synth_cvr_volume
        stim, _ = extract_global_stimulus(
            d["bold_4d"], d["brain_mask"], tr=d["tr"], dt=d["dt"],
        )
        assert abs(float(jnp.mean(stim))) < 1e-3
        # Default amplitude 0.3 keeps the Balloon ODE stable.
        assert float(jnp.max(jnp.abs(stim))) <= 0.31

    def test_tracks_block_stimulus(self, synth_cvr_volume):
        """Peak of recovered stimulus should fall in the block window."""
        d = synth_cvr_volume
        stim, _ = extract_global_stimulus(
            d["bold_4d"], d["brain_mask"], tr=d["tr"], dt=d["dt"],
        )
        peak_t = int(jnp.argmax(stim)) * d["dt"]
        # block was 20-40 s; with HRF lag the peak should be 20-50 s
        assert 20.0 <= peak_t <= 55.0


# ---------------------------------------------------------------------------
# etCO2 resampler
# ---------------------------------------------------------------------------

class TestResampleEtco2:

    def test_shape(self):
        et = np.linspace(35.0, 45.0, 50)  # 50 samples
        stim = resample_etco2_stimulus(
            et, etco2_tr=2.0, n_volumes=60, bold_tr=2.0, dt=0.1,
        )
        assert stim.shape == (60 * 20,)

    def test_zero_mean(self):
        et = np.array([35.0] * 25 + [45.0] * 25)
        stim = resample_etco2_stimulus(
            et, etco2_tr=2.0, n_volumes=50, bold_tr=2.0, dt=0.1,
        )
        assert abs(float(jnp.mean(stim))) < 1e-3


# ---------------------------------------------------------------------------
# Per-region fitting
# ---------------------------------------------------------------------------

class TestFitCvrPerRegion:

    def test_shapes(self, synth_cvr_volume):
        d = synth_cvr_volume
        result = fit_cvr_per_region(
            d["bold_4d"], d["brain_mask"], d["region_volume"],
            tr=d["tr"], dt=d["dt"],
            stimulus=d["stimulus"],
            n_steps=20,
        )
        assert isinstance(result, CVRFitResult)
        assert result.region_ids.shape == (3,)
        assert result.kappa.shape == (3,)
        assert result.cvr_scalar.shape == (3,)
        assert result.bold_predicted.shape == (3, d["bold_4d"].shape[-1])

    def test_cvr_scalar_positive(self, synth_cvr_volume):
        """Hypercapnia → positive BOLD swing in every region."""
        d = synth_cvr_volume
        result = fit_cvr_per_region(
            d["bold_4d"], d["brain_mask"], d["region_volume"],
            tr=d["tr"], dt=d["dt"],
            stimulus=d["stimulus"],
            n_steps=20,
        )
        assert (result.cvr_scalar > 0).all()

    def test_loss_finite_after_fit(self, synth_cvr_volume):
        d = synth_cvr_volume
        result = fit_cvr_per_region(
            d["bold_4d"], d["brain_mask"], d["region_volume"],
            tr=d["tr"], dt=d["dt"],
            stimulus=d["stimulus"],
            n_steps=20,
        )
        assert np.all(np.isfinite(result.loss))

    def test_data_driven_stimulus_path(self, synth_cvr_volume):
        """Fitting without a supplied stimulus must not raise."""
        d = synth_cvr_volume
        result = fit_cvr_per_region(
            d["bold_4d"], d["brain_mask"], d["region_volume"],
            tr=d["tr"], dt=d["dt"],
            n_steps=10,
        )
        assert result.region_ids.shape == (3,)

    def test_min_voxels_filter(self, synth_cvr_volume):
        d = synth_cvr_volume
        with pytest.raises(ValueError):
            fit_cvr_per_region(
                d["bold_4d"], d["brain_mask"], d["region_volume"],
                tr=d["tr"], dt=d["dt"],
                stimulus=d["stimulus"],
                n_steps=10,
                min_voxels=999,
            )
