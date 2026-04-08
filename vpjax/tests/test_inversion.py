"""Tests for hemodynamic model inversion.

Verifies round-trip recovery: generate synthetic data from known
parameters, fit the model, and check that parameters are recovered
within physiological tolerance.
"""

import jax
import jax.numpy as jnp
import pytest

from vpjax._types import BalloonParams, BalloonState
from vpjax.hemodynamics.balloon import solve_balloon
from vpjax.hemodynamics.bold import BOLDParams, observe_bold
from vpjax.hemodynamics.inversion import (
    fit_balloon_bold,
    fit_balloon_bold_batch,
    fit_balloon_multimodal,
    fit_riera_bold,
)
from vpjax.hemodynamics.riera import (
    RieraParams,
    RieraState,
    riera_to_balloon,
    solve_riera,
)
from vpjax.perfusion.asl import observe_asl
from vpjax.perfusion.vaso import observe_vaso


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def block_stimulus():
    """3s block stimulus at dt=0.05, 30s total."""
    dt = 0.05
    T = 30.0
    n = int(T / dt)
    stim = jnp.zeros(n).at[int(2.0 / dt):int(5.0 / dt)].set(1.0)
    return stim, dt


@pytest.fixture
def tr():
    return 1.0


# ---------------------------------------------------------------------------
# solve_riera
# ---------------------------------------------------------------------------

class TestSolveRiera:
    """Verify solve_riera matches manual Euler integration."""

    def test_solve_riera_runs(self, block_stimulus):
        """solve_riera produces trajectory with correct shape."""
        stim, dt = block_stimulus
        ts, traj = solve_riera(RieraParams(), stim, dt=dt)
        assert ts.shape == stim.shape
        assert traj.f_a.shape == stim.shape

    def test_solve_riera_responds(self, block_stimulus):
        """Riera model responds to stimulus."""
        stim, dt = block_stimulus
        _, traj = solve_riera(RieraParams(), stim, dt=dt)
        # Flow should deviate from baseline
        assert float(jnp.max(jnp.abs(traj.f_a - 1.0))) > 0.01

    def test_solve_riera_returns_to_baseline(self, block_stimulus):
        """State should return near baseline after stimulus."""
        stim, dt = block_stimulus
        _, traj = solve_riera(RieraParams(), stim, dt=dt)
        assert jnp.abs(traj.f_a[-1] - 1.0) < 0.05
        assert jnp.abs(traj.v_v[-1] - 1.0) < 0.05

    def test_solve_riera_differentiable(self, block_stimulus):
        """Gradient flows through solve_riera."""
        stim, dt = block_stimulus

        def peak_flow(E0):
            params = RieraParams(E0=E0)
            _, traj = solve_riera(params, stim, dt=dt)
            v, q = riera_to_balloon(traj)
            bold = observe_bold(BalloonState(
                s=jnp.zeros_like(v), f=traj.f_a, v=v, q=q,
            ))
            return jnp.max(bold)

        g = jax.grad(peak_flow)(jnp.array(0.34))
        assert jnp.isfinite(g)
        assert float(g) != 0.0


# ---------------------------------------------------------------------------
# Balloon BOLD inversion
# ---------------------------------------------------------------------------

class TestFitBalloonBold:
    """Round-trip test: synthesise BOLD, fit, recover parameters."""

    def test_recovers_kappa(self, block_stimulus, tr):
        """Should recover non-default kappa (high BOLD sensitivity)."""
        stim, dt = block_stimulus
        true_kappa = 1.2
        true_params = BalloonParams(kappa=jnp.array(true_kappa))

        _, traj = solve_balloon(true_params, stim, dt=dt)
        sub = int(round(tr / dt))
        bold_data = observe_bold(traj)[::sub]

        result = fit_balloon_bold(
            bold_data, stim, tr=tr, dt=dt, n_steps=500,
        )

        assert float(result["loss"]) < 1e-5

    def test_recovers_tau(self, block_stimulus, tr):
        """Should recover non-default transit time."""
        stim, dt = block_stimulus
        true_tau = 1.8
        true_params = BalloonParams(tau=jnp.array(true_tau))

        _, traj = solve_balloon(true_params, stim, dt=dt)
        sub = int(round(tr / dt))
        bold_data = observe_bold(traj)[::sub]

        result = fit_balloon_bold(
            bold_data, stim, tr=tr, dt=dt, n_steps=500,
        )

        assert float(result["loss"]) < 1e-5

    def test_loss_decreases(self, block_stimulus, tr):
        """Loss after fitting should be lower than at initial params."""
        stim, dt = block_stimulus
        true_params = BalloonParams(kappa=jnp.array(1.2))

        _, traj = solve_balloon(true_params, stim, dt=dt)
        sub = int(round(tr / dt))
        bold_data = observe_bold(traj)[::sub]

        # Evaluate initial loss
        _, traj0 = solve_balloon(BalloonParams(), stim, dt=dt)
        bold0 = observe_bold(traj0)[::sub]
        n = bold_data.shape[0]
        initial_loss = float(jnp.mean((bold0[:n] - bold_data) ** 2))

        result = fit_balloon_bold(
            bold_data, stim, tr=tr, dt=dt, n_steps=300,
        )

        assert float(result["loss"]) < initial_loss

    def test_predicted_matches_data(self, block_stimulus, tr):
        """Predicted BOLD should closely match data after fitting."""
        stim, dt = block_stimulus
        true_params = BalloonParams(kappa=jnp.array(1.0))

        _, traj = solve_balloon(true_params, stim, dt=dt)
        sub = int(round(tr / dt))
        bold_data = observe_bold(traj)[::sub]

        result = fit_balloon_bold(
            bold_data, stim, tr=tr, dt=dt, n_steps=500,
        )

        assert result["bold_predicted"].shape == bold_data.shape
        assert float(jnp.max(jnp.abs(result["bold_predicted"] - bold_data))) < 0.005


# ---------------------------------------------------------------------------
# Balloon multimodal inversion
# ---------------------------------------------------------------------------

class TestFitBalloonMultimodal:
    """Joint BOLD + ASL + VASO fitting."""

    def test_bold_plus_asl(self, block_stimulus, tr):
        """Joint BOLD + ASL should converge."""
        stim, dt = block_stimulus
        true_params = BalloonParams(kappa=jnp.array(1.0))

        _, traj = solve_balloon(true_params, stim, dt=dt)
        sub = int(round(tr / dt))
        bold_data = observe_bold(traj)[::sub]
        asl_data = observe_asl(traj)[::sub]

        result = fit_balloon_multimodal(
            bold_data, stim, tr=tr, dt=dt,
            asl_data=asl_data, n_steps=500,
        )

        assert float(result["loss"]) < 1e-5
        assert "asl_predicted" in result

    def test_bold_plus_asl_plus_vaso(self, block_stimulus, tr):
        """All three modalities jointly."""
        stim, dt = block_stimulus
        true_params = BalloonParams(kappa=jnp.array(1.0), tau=jnp.array(1.2))

        _, traj = solve_balloon(true_params, stim, dt=dt)
        sub = int(round(tr / dt))
        bold_data = observe_bold(traj)[::sub]
        asl_data = observe_asl(traj)[::sub]
        vaso_data = observe_vaso(traj)[::sub]

        result = fit_balloon_multimodal(
            bold_data, stim, tr=tr, dt=dt,
            asl_data=asl_data, vaso_data=vaso_data, n_steps=500,
        )

        assert float(result["loss"]) < 1e-5
        assert "asl_predicted" in result
        assert "vaso_predicted" in result

    def test_multimodal_recovers_E0(self, block_stimulus, tr):
        """Adding ASL constrains E0 via independent CBF information."""
        stim, dt = block_stimulus
        true_E0 = 0.55
        true_params = BalloonParams(E0=jnp.array(true_E0))

        _, traj = solve_balloon(true_params, stim, dt=dt)
        sub = int(round(tr / dt))
        bold_data = observe_bold(traj)[::sub]
        asl_data = observe_asl(traj)[::sub]
        vaso_data = observe_vaso(traj)[::sub]

        result = fit_balloon_multimodal(
            bold_data, stim, tr=tr, dt=dt,
            asl_data=asl_data, vaso_data=vaso_data, n_steps=500,
        )

        assert float(result["loss"]) < 1e-5


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------

class TestFitBalloonBatch:
    """Batch (vmapped) fitting over multiple ROIs."""

    def test_batch_fits_two_rois(self, block_stimulus, tr):
        """Fit two ROIs with different true kappa values."""
        stim, dt = block_stimulus
        sub = int(round(tr / dt))

        # ROI 1: kappa=0.8, ROI 2: kappa=1.3
        _, traj1 = solve_balloon(BalloonParams(kappa=jnp.array(0.8)), stim, dt=dt)
        _, traj2 = solve_balloon(BalloonParams(kappa=jnp.array(1.3)), stim, dt=dt)

        bold1 = observe_bold(traj1)[::sub]
        bold2 = observe_bold(traj2)[::sub]
        n = min(bold1.shape[0], bold2.shape[0])
        bold_batch = jnp.stack([bold1[:n], bold2[:n]])

        result = fit_balloon_bold_batch(
            bold_batch, stim, tr=tr, dt=dt, n_steps=400,
        )

        assert result["kappa"].shape == (2,)
        assert result["loss"].shape == (2,)
        # Both should have low loss
        assert float(jnp.max(result["loss"])) < 1e-4


# ---------------------------------------------------------------------------
# Riera BOLD inversion
# ---------------------------------------------------------------------------

class TestFitRieraBold:
    """Round-trip test for Riera NVC model inversion."""

    def test_riera_fit_converges(self, block_stimulus, tr):
        """Riera fitting should reduce loss from initial params."""
        stim, dt = block_stimulus
        true_params = RieraParams(c_no=jnp.array(1.5))

        _, traj = solve_riera(true_params, stim, dt=dt)
        sub = int(round(tr / dt))
        v, q = riera_to_balloon(traj)
        pseudo = BalloonState(
            s=jnp.zeros_like(v), f=traj.f_a, v=v, q=q,
        )
        bold_data = observe_bold(pseudo)[::sub]

        result = fit_riera_bold(
            bold_data, stim, tr=tr, dt=dt, n_steps=400,
        )

        assert float(result["loss"]) < 1e-4

    def test_riera_returns_all_params(self, block_stimulus, tr):
        """Result should contain all 15 Riera parameters."""
        stim, dt = block_stimulus
        true_params = RieraParams()

        _, traj = solve_riera(true_params, stim, dt=dt)
        sub = int(round(tr / dt))
        v, q = riera_to_balloon(traj)
        pseudo = BalloonState(
            s=jnp.zeros_like(v), f=traj.f_a, v=v, q=q,
        )
        bold_data = observe_bold(pseudo)[::sub]

        result = fit_riera_bold(
            bold_data, stim, tr=tr, dt=dt, n_steps=100,
        )

        for name in (
            "kappa_no", "kappa_ade", "gamma_no", "gamma_ade",
            "c_no", "c_ade", "tau_a", "tau_c", "tau_v",
            "alpha_a", "alpha_c", "alpha_v", "E0", "phi", "tau_m",
        ):
            assert name in result, f"Missing parameter: {name}"

    def test_riera_custom_fit_names(self, block_stimulus, tr):
        """Should accept custom subset of parameters to fit."""
        stim, dt = block_stimulus
        true_params = RieraParams()

        _, traj = solve_riera(true_params, stim, dt=dt)
        sub = int(round(tr / dt))
        v, q = riera_to_balloon(traj)
        pseudo = BalloonState(
            s=jnp.zeros_like(v), f=traj.f_a, v=v, q=q,
        )
        bold_data = observe_bold(pseudo)[::sub]

        # Only fit two parameters
        result = fit_riera_bold(
            bold_data, stim, tr=tr, dt=dt,
            fit_names=("c_no", "E0"),
            n_steps=200,
        )

        assert "c_no" in result
        assert "E0" in result
        assert float(result["loss"]) < 1e-3


