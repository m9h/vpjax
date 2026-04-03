"""Tests for the Balloon-Windkessel model and observation functions."""

import jax
import jax.numpy as jnp
import pytest

from vpjax._types import BalloonParams, BalloonState
from vpjax.hemodynamics.balloon import BalloonWindkessel, solve_balloon
from vpjax.hemodynamics.bold import BOLDParams, observe_bold
from vpjax.perfusion.asl import observe_asl
from vpjax.perfusion.vaso import observe_vaso


# ---------------------------------------------------------------------------
# Steady-state tests
# ---------------------------------------------------------------------------

class TestSteadyState:
    """At rest (zero stimulus), all variables should remain at baseline."""

    def test_balloon_state_defaults(self):
        y0 = BalloonState.steady_state()
        assert float(y0.s) == 0.0
        assert float(y0.f) == 1.0
        assert float(y0.v) == 1.0
        assert float(y0.q) == 1.0

    def test_rhs_at_rest(self):
        """RHS should be ~zero at baseline with zero stimulus."""
        model = BalloonWindkessel(params=BalloonParams())
        y0 = BalloonState.steady_state()
        dy = model(jnp.array(0.0), y0, jnp.array(0.0))
        assert jnp.allclose(dy.s, 0.0, atol=1e-6)
        assert jnp.allclose(dy.f, 0.0, atol=1e-6)
        assert jnp.allclose(dy.v, 0.0, atol=1e-6)
        assert jnp.allclose(dy.q, 0.0, atol=1e-6)

    def test_bold_at_rest(self):
        """BOLD signal should be zero at baseline."""
        y0 = BalloonState.steady_state()
        signal = observe_bold(y0)
        assert jnp.allclose(signal, 0.0, atol=1e-6)

    def test_asl_at_rest(self):
        y0 = BalloonState.steady_state()
        assert jnp.allclose(observe_asl(y0), 0.0, atol=1e-6)

    def test_vaso_at_rest(self):
        y0 = BalloonState.steady_state()
        assert jnp.allclose(observe_vaso(y0), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestBalloonIntegration:
    """Test solving the Balloon ODE with a stimulus."""

    @pytest.fixture
    def impulse_response(self):
        """3s block stimulus, integrated for 30s."""
        dt = 0.01
        T = 30.0
        n = int(T / dt)
        stim = jnp.zeros(n)
        # 3s block starting at t=1s
        stim = stim.at[int(1.0 / dt) : int(4.0 / dt)].set(1.0)
        ts, traj = solve_balloon(BalloonParams(), stim, dt=dt)
        return ts, traj, stim

    def test_returns_correct_length(self, impulse_response):
        ts, traj, stim = impulse_response
        assert ts.shape[0] == stim.shape[0]
        assert traj.v.shape[0] == stim.shape[0]

    def test_flow_increases_during_stimulus(self, impulse_response):
        ts, traj, _ = impulse_response
        # Flow should peak somewhere after stimulus onset
        peak_idx = jnp.argmax(traj.f)
        peak_time = ts[peak_idx]
        assert 1.0 < float(peak_time) < 10.0  # peak between 1-10s

    def test_bold_positive_response(self, impulse_response):
        """BOLD should show a positive response to a stimulus."""
        ts, traj, _ = impulse_response
        bold = observe_bold(traj)
        # Peak BOLD should be positive
        assert float(jnp.max(bold)) > 0.0

    def test_bold_initial_dip(self, impulse_response):
        """With standard params, there may be an initial dip or not,
        but BOLD should eventually return to baseline."""
        ts, traj, _ = impulse_response
        bold = observe_bold(traj)
        # At t=30s, BOLD should have largely returned to baseline
        assert jnp.abs(bold[-1]) < 0.01

    def test_returns_to_baseline(self, impulse_response):
        """All state variables should return near baseline after stimulus."""
        _, traj, _ = impulse_response
        assert jnp.abs(traj.f[-1] - 1.0) < 0.01
        assert jnp.abs(traj.v[-1] - 1.0) < 0.01
        assert jnp.abs(traj.q[-1] - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Differentiability tests
# ---------------------------------------------------------------------------

class TestDifferentiability:
    """Ensure gradients flow through the model."""

    def test_grad_bold_wrt_params(self):
        """Gradient of peak BOLD w.r.t. kappa should exist."""
        def peak_bold(kappa):
            params = BalloonParams(kappa=kappa)
            dt = 0.05
            n = int(20.0 / dt)
            stim = jnp.zeros(n).at[int(1.0 / dt) : int(4.0 / dt)].set(1.0)
            _, traj = solve_balloon(params, stim, dt=dt)
            bold = observe_bold(traj)
            return jnp.max(bold)

        grad_fn = jax.grad(peak_bold)
        g = grad_fn(jnp.array(0.65))
        assert jnp.isfinite(g)
        assert float(g) != 0.0

    def test_grad_asl(self):
        """Gradient of ASL signal w.r.t. flow."""
        y = BalloonState(
            s=jnp.array(0.0),
            f=jnp.array(1.5),
            v=jnp.array(1.0),
            q=jnp.array(1.0),
        )
        grad_fn = jax.grad(lambda f: observe_asl(
            BalloonState(s=y.s, f=f, v=y.v, q=y.q)
        ))
        g = grad_fn(jnp.array(1.5))
        assert jnp.allclose(g, 1.0)  # d(f-1)/df = 1


# ---------------------------------------------------------------------------
# Optical properties tests
# ---------------------------------------------------------------------------

class TestOpticalProperties:
    def test_to_optical_properties_shape(self):
        from vpjax.hemodynamics.optics import to_optical_properties
        hbo = jnp.array(0.06)  # mM
        hbr = jnp.array(0.04)  # mM
        wl = jnp.array([760.0, 850.0])
        mua, musp = to_optical_properties(hbo, hbr, wl)
        assert mua.shape == (2,)
        assert musp.shape == (2,)
        # mua should be positive
        assert jnp.all(mua > 0)
        assert jnp.all(musp > 0)

    def test_isosbestic_point(self):
        """At 800nm, HbO and HbR extinction are similar."""
        from vpjax.hemodynamics.optics import _get_extinction
        wl = jnp.array([800.0])
        eps_hbo, eps_hbr = _get_extinction(wl)
        # They should be close (within ~20% at isosbestic)
        ratio = float((eps_hbo / eps_hbr)[0])
        assert 0.8 < ratio < 1.5
