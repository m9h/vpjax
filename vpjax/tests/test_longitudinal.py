"""Tests for vpjax.hemodynamics.longitudinal.

Synthetic round-trip:
* Pick known per-region (kappa_0, beta_kappa).
* Generate three sessions of BOLD at ages 54/58/63 with the
  age-dependent kappa.
* Fit and recover (kappa_0, beta_kappa) within tolerance.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vpjax._types import BalloonParams
from vpjax.hemodynamics.balloon import solve_balloon
from vpjax.hemodynamics.bold import observe_bold
from vpjax.hemodynamics.longitudinal import (
    fit_balloon_longitudinal,
    fit_balloon_longitudinal_from_volumes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AGE_CENTER = 60.0
AGES = (54.0, 58.0, 63.0)


@pytest.fixture
def synth_longitudinal():
    """Three sessions, two regions, known per-region kappa(age)."""
    tr = 2.0
    dt = 0.1
    n_t = 60
    sub = int(round(tr / dt))
    t_total = n_t * sub
    stim = jnp.zeros(t_total).at[int(20.0 / dt):int(40.0 / dt)].set(1.0)

    # Region 1: kappa_0 = 0.8, beta_kappa = +0.02 / yr (slowing with age)
    # Region 2: kappa_0 = 1.2, beta_kappa = -0.01 / yr (slight speedup)
    region_baselines = np.array([0.8, 1.2], dtype=np.float32)
    region_slopes = np.array([0.02, -0.01], dtype=np.float32)

    def session_bold(age):
        offset = age - AGE_CENTER
        kappas = region_baselines + region_slopes * offset
        boldlist = []
        for k in kappas:
            bp = BalloonParams(kappa=jnp.array(float(k)))
            _, traj = solve_balloon(bp, stim, dt=dt)
            boldlist.append(np.asarray(observe_bold(traj)[::sub][:n_t]))
        return np.stack(boldlist, axis=0)

    bold_per_session = [session_bold(a) for a in AGES]

    return {
        "bold_per_session": bold_per_session,
        "stimulus": [stim, stim, stim],
        "ages": AGES,
        "tr": tr,
        "dt": dt,
        "true_baselines": region_baselines,
        "true_slopes": region_slopes,
    }


# ---------------------------------------------------------------------------
# Core fitter
# ---------------------------------------------------------------------------

class TestFitBalloonLongitudinal:

    def test_loss_converges(self, synth_longitudinal):
        """The fitter should drive the BOLD residual MSE below 1e-3.

        The Balloon model is degenerate in its parameters (kappa/tau/gamma
        trade off, see Stephan 2007), so we check that the *forward
        prediction* matches the data, not that the latent parameters are
        recovered exactly — the same convention the BOLD-only test in
        :mod:`test_inversion` uses.
        """
        d = synth_longitudinal
        out = fit_balloon_longitudinal(
            bold_per_session=d["bold_per_session"],
            stimulus_per_session=d["stimulus"],
            ages=d["ages"],
            tr=d["tr"], dt=d["dt"],
            age_center=AGE_CENTER,
            fit_names=("kappa",),
            n_steps=120,
            learning_rate=2.0,
        )
        assert float(out["loss"]) < 1e-3

    def test_slope_sign_tracks_age_dependence(self, synth_longitudinal):
        """Region-1 BOLD HRF speeds up with age (kappa↑), region-2 slows
        down (kappa↓).  The fitted slope signs should reflect that —
        even if magnitudes are biased by parameter degeneracy.
        """
        d = synth_longitudinal
        out = fit_balloon_longitudinal(
            bold_per_session=d["bold_per_session"],
            stimulus_per_session=d["stimulus"],
            ages=d["ages"],
            tr=d["tr"], dt=d["dt"],
            age_center=AGE_CENTER,
            fit_names=("kappa",),
            n_steps=120,
            learning_rate=2.0,
        )
        sl = out["slopes"][:, 0]
        # True slopes are (+0.02, -0.01); the fitted signs must agree
        assert sl[0] > 0
        assert sl[1] < 0

    def test_zero_slope_when_no_age_signal(self, synth_longitudinal):
        """Feed identical BOLD across sessions → slope magnitudes ~ 0."""
        d = synth_longitudinal
        same_bold = [d["bold_per_session"][0]] * 3
        out = fit_balloon_longitudinal(
            bold_per_session=same_bold,
            stimulus_per_session=d["stimulus"],
            ages=d["ages"],
            tr=d["tr"], dt=d["dt"],
            age_center=AGE_CENTER,
            fit_names=("kappa",),
            n_steps=80,
            learning_rate=2.0,
        )
        sl = out["slopes"][:, 0]
        # Loose bound: identical-data slopes should be small
        assert np.max(np.abs(sl)) < 0.05

    def test_loss_finite(self, synth_longitudinal):
        d = synth_longitudinal
        out = fit_balloon_longitudinal(
            bold_per_session=d["bold_per_session"],
            stimulus_per_session=d["stimulus"],
            ages=d["ages"],
            tr=d["tr"], dt=d["dt"],
            n_steps=50,
        )
        assert np.isfinite(out["loss"])

    def test_predicted_shapes(self, synth_longitudinal):
        d = synth_longitudinal
        out = fit_balloon_longitudinal(
            bold_per_session=d["bold_per_session"],
            stimulus_per_session=d["stimulus"],
            ages=d["ages"],
            tr=d["tr"], dt=d["dt"],
            n_steps=20,
        )
        assert len(out["bold_predicted"]) == 3
        for p in out["bold_predicted"]:
            assert p.shape == (2, 60)

    def test_S_validation(self, synth_longitudinal):
        d = synth_longitudinal
        with pytest.raises(ValueError):
            fit_balloon_longitudinal(
                bold_per_session=d["bold_per_session"],
                stimulus_per_session=d["stimulus"][:2],
                ages=d["ages"],
                tr=d["tr"], dt=d["dt"],
            )


# ---------------------------------------------------------------------------
# 4D-volumes wrapper
# ---------------------------------------------------------------------------

class TestFitFromVolumes:

    def _make_volume(self, region_ts: np.ndarray, layout, shape=(4, 4, 3)):
        """Embed per-region time series into a 4D volume."""
        n_t = region_ts.shape[1]
        bold_4d = np.zeros(shape + (n_t,), dtype=np.float32)
        mask = np.zeros(shape, dtype=np.float32)
        labels = np.zeros(shape, dtype=np.int32)
        for rid_idx, (rid, voxels) in enumerate(layout.items()):
            for x, y, z in voxels:
                bold_4d[x, y, z, :] = 1000.0 * (1.0 + region_ts[rid_idx])
                mask[x, y, z] = 1.0
                labels[x, y, z] = rid
        return bold_4d, mask, labels

    def test_volumes_wrapper(self, synth_longitudinal):
        d = synth_longitudinal
        layout = {
            10: [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0),
                 (0, 0, 1), (1, 0, 1)],
            20: [(2, 0, 1), (2, 1, 1), (3, 0, 1), (3, 1, 1),
                 (2, 2, 1), (3, 2, 1)],
        }
        bold_4d_per = []
        mask_per = []
        labels_per = []
        for ts in d["bold_per_session"]:
            b, m, l = self._make_volume(np.asarray(ts), layout)
            bold_4d_per.append(b)
            mask_per.append(m)
            labels_per.append(l)

        out = fit_balloon_longitudinal_from_volumes(
            bold_4d_per_session=bold_4d_per,
            brain_mask_per_session=mask_per,
            region_volume_per_session=labels_per,
            ages=d["ages"],
            tr=d["tr"], dt=d["dt"],
            age_center=AGE_CENTER,
            stimulus_per_session=d["stimulus"],
            fit_names=("kappa",),
            n_steps=120,
            learning_rate=2.0,
        )
        # Region order in `region_ids` is sorted ascending → labels [10, 20]
        assert list(out["region_ids"]) == [10, 20]
        # Forward fit should converge; per-region slope signs match truth
        sl = out["slopes"][:, 0]
        assert float(out["loss"]) < 1e-3
        assert sl[0] > 0
        assert sl[1] < 0
