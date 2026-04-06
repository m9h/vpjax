"""Tests for stochastic hemodynamic models.

Three components:
1. SDE Balloon/Riera (state noise in the ODE)
2. Fokker-Planck mean+covariance evolution (Marreiros/Friston)
3. Sleep state transition dynamics (noise-driven bifurcations)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. SDE formulation: Balloon with state noise
# ---------------------------------------------------------------------------

class TestStochasticBalloon:
    """Balloon-Windkessel with additive state noise (SDE)."""

    def test_sde_params(self):
        from vpjax.stochastic.sde_balloon import SDEBalloonParams
        p = SDEBalloonParams()
        assert float(p.sigma_neural) > 0
        assert float(p.sigma_hemo) > 0

    def test_sde_step_changes_state(self):
        """SDE step should produce different states from deterministic."""
        from vpjax.stochastic.sde_balloon import sde_balloon_step, SDEBalloonParams
        from vpjax._types import BalloonState
        y0 = BalloonState.steady_state()
        params = SDEBalloonParams()
        key = jax.random.PRNGKey(42)
        y1 = sde_balloon_step(y0, jnp.array(1.0), params, key, dt=0.01)
        # Should have moved from steady state
        assert not jnp.allclose(y1.s, 0.0, atol=1e-8)

    def test_sde_produces_variability(self):
        """Multiple SDE runs should produce different trajectories."""
        from vpjax.stochastic.sde_balloon import sde_balloon_solve, SDEBalloonParams
        params = SDEBalloonParams()
        stimulus = jnp.zeros(200).at[20:30].set(1.0)
        traj1 = sde_balloon_solve(params, stimulus, dt=0.1, key=jax.random.PRNGKey(0))
        traj2 = sde_balloon_solve(params, stimulus, dt=0.1, key=jax.random.PRNGKey(1))
        # Should not be identical (stochastic)
        assert not jnp.allclose(traj1.v, traj2.v)

    def test_sde_mean_matches_deterministic(self):
        """Average of many SDE runs should approximate deterministic solution."""
        from vpjax.stochastic.sde_balloon import sde_balloon_solve, SDEBalloonParams
        from vpjax.hemodynamics.balloon import solve_balloon
        from vpjax._types import BalloonParams
        stimulus = jnp.zeros(100).at[10:15].set(1.0)
        # Deterministic
        _, det_traj = solve_balloon(BalloonParams(), stimulus, dt=0.1)
        # Average of stochastic runs
        params = SDEBalloonParams(sigma_neural=jnp.array(0.01), sigma_hemo=jnp.array(0.001))
        bold_runs = []
        for i in range(20):
            traj = sde_balloon_solve(params, stimulus, dt=0.1, key=jax.random.PRNGKey(i))
            bold_runs.append(traj.v)
        mean_v = jnp.mean(jnp.stack(bold_runs), axis=0)
        # Mean should be close to deterministic (within noise)
        r = float(jnp.corrcoef(mean_v, det_traj.v)[0, 1])
        assert r > 0.9


# ---------------------------------------------------------------------------
# 2. Fokker-Planck: mean + covariance evolution
# ---------------------------------------------------------------------------

class TestFokkerPlanck:
    """Population density dynamics under Laplace assumption."""

    def test_fp_state(self):
        from vpjax.stochastic.fokker_planck import FPState
        state = FPState.from_balloon_steady_state()
        assert state.mean.shape == (4,)   # s, f, v, q
        assert state.cov.shape == (4, 4)
        # Covariance should be positive definite
        eigvals = jnp.linalg.eigvalsh(state.cov)
        assert jnp.all(eigvals > 0)

    def test_fp_step_propagates_uncertainty(self):
        """One FP step should increase covariance (uncertainty grows)."""
        from vpjax.stochastic.fokker_planck import FPState, fp_step, FPParams
        state = FPState.from_balloon_steady_state()
        initial_trace = float(jnp.trace(state.cov))
        new_state = fp_step(state, jnp.array(0.0), FPParams(), dt=0.1)
        new_trace = float(jnp.trace(new_state.cov))
        # Diffusion should increase total variance
        assert new_trace >= initial_trace - 0.01

    def test_fp_mean_evolves_like_ode(self):
        """FP mean should follow the deterministic ODE trajectory."""
        from vpjax.stochastic.fokker_planck import FPState, fp_step, FPParams
        state = FPState.from_balloon_steady_state()
        params = FPParams()
        # Apply stimulus for several steps
        for _ in range(50):
            state = fp_step(state, jnp.array(1.0), params, dt=0.1)
        # Mean flow (f) should have increased from 1.0
        assert float(state.mean[1]) > 1.0  # f index

    def test_fp_covariance_symmetric(self):
        """Covariance should remain symmetric."""
        from vpjax.stochastic.fokker_planck import FPState, fp_step, FPParams
        state = FPState.from_balloon_steady_state()
        for _ in range(10):
            state = fp_step(state, jnp.array(0.5), FPParams(), dt=0.1)
        assert jnp.allclose(state.cov, state.cov.T, atol=1e-6)

    def test_differentiable(self):
        from vpjax.stochastic.fokker_planck import FPState, fp_step, FPParams
        def loss(sigma):
            params = FPParams(D_neural=sigma)
            state = FPState.from_balloon_steady_state()
            for _ in range(5):
                state = fp_step(state, jnp.array(1.0), params, dt=0.1)
            return jnp.trace(state.cov)
        g = jax.grad(loss)(jnp.array(0.01))
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# 3. Sleep state transitions as noise-driven dynamics
# ---------------------------------------------------------------------------

class TestSleepTransitions:
    """Noise-driven transitions between sleep attractors."""

    def test_sleep_potential(self):
        """Should define an energy landscape with wake/NREM/REM attractors."""
        from vpjax.stochastic.sleep_transitions import sleep_potential
        # Wake attractor near x=0, NREM near x=1, REM near x=2
        V_wake = sleep_potential(jnp.array(0.0))
        V_nrem = sleep_potential(jnp.array(1.0))
        V_rem = sleep_potential(jnp.array(2.0))
        # Attractors should be local minima (lower potential)
        V_between = sleep_potential(jnp.array(0.5))
        assert float(V_wake) < float(V_between)
        assert float(V_nrem) < float(V_between)

    def test_transition_probability(self):
        """Transition rate should depend on noise amplitude (Kramers rate)."""
        from vpjax.stochastic.sleep_transitions import kramers_transition_rate
        rate_low_noise = kramers_transition_rate(
            barrier_height=jnp.array(1.0), noise=jnp.array(0.1)
        )
        rate_high_noise = kramers_transition_rate(
            barrier_height=jnp.array(1.0), noise=jnp.array(0.5)
        )
        # Higher noise → faster transitions
        assert float(rate_high_noise) > float(rate_low_noise)

    def test_simulate_sleep_hypnogram(self):
        """Should produce a hypnogram-like state sequence."""
        from vpjax.stochastic.sleep_transitions import simulate_sleep_states
        key = jax.random.PRNGKey(42)
        t = jnp.linspace(0, 3600, 360)  # 1 hour, 10s steps
        states = simulate_sleep_states(t, key=key)
        assert states.shape == t.shape
        # Should visit at least 2 different states
        unique = jnp.unique(jnp.round(states))
        assert len(unique) >= 2

    def test_differentiable(self):
        from vpjax.stochastic.sleep_transitions import sleep_potential
        g = jax.grad(sleep_potential)(jnp.array(0.5))
        assert jnp.isfinite(g)
