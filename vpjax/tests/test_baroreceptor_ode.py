"""Tests for Pulse-style baroreceptor ODE model.

Replaces the cosine approximation with sigmoid sympathetic/parasympathetic
fractions and 4 ODEs for HR, elastance, resistance, compliance.
Based on Kitware Pulse nervous methodology.
"""

import jax
import jax.numpy as jnp
import pytest


class TestBaroreflexODE:
    """Baroreflex ODE: MAP → sympathetic/parasympathetic → CV adjustments."""

    def test_params(self):
        from vpjax.cardiac.baroreceptor import BaroreflexParams
        p = BaroreflexParams()
        assert float(p.map_setpoint) > 0
        assert float(p.nu) > 0

    def test_sympathetic_sigmoid(self):
        """Sympathetic fraction should decrease with increasing MAP."""
        from vpjax.cardiac.baroreceptor import sympathetic_fraction
        eta_low = sympathetic_fraction(jnp.array(70.0))
        eta_high = sympathetic_fraction(jnp.array(120.0))
        assert float(eta_low) > float(eta_high)

    def test_parasympathetic_sigmoid(self):
        """Parasympathetic fraction should increase with increasing MAP."""
        from vpjax.cardiac.baroreceptor import parasympathetic_fraction
        eta_low = parasympathetic_fraction(jnp.array(70.0))
        eta_high = parasympathetic_fraction(jnp.array(120.0))
        assert float(eta_high) > float(eta_low)

    def test_setpoint_balance(self):
        """At MAP setpoint, sympathetic ≈ parasympathetic ≈ 0.5."""
        from vpjax.cardiac.baroreceptor import sympathetic_fraction, parasympathetic_fraction, BaroreflexParams
        p = BaroreflexParams()
        eta_s = sympathetic_fraction(p.map_setpoint, p)
        eta_p = parasympathetic_fraction(p.map_setpoint, p)
        assert abs(float(eta_s) - 0.5) < 0.05
        assert abs(float(eta_p) - 0.5) < 0.05

    def test_baroreflex_ode_steady_state(self):
        """At setpoint MAP, ODE derivatives should be near zero."""
        from vpjax.cardiac.baroreceptor import BaroreflexState, BaroreflexODE, BaroreflexParams
        p = BaroreflexParams()
        model = BaroreflexODE(params=p)
        y0 = BaroreflexState.steady_state()
        dy = model(jnp.array(0.0), y0, p.map_setpoint)
        assert jnp.allclose(dy.heart_rate, 0.0, atol=0.1)
        assert jnp.allclose(dy.resistance, 0.0, atol=0.01)

    def test_hypotension_response(self):
        """Low MAP should increase HR and resistance (sympathetic activation)."""
        from vpjax.cardiac.baroreceptor import BaroreflexState, BaroreflexODE, BaroreflexParams
        model = BaroreflexODE(params=BaroreflexParams())
        y = BaroreflexState.steady_state()
        # Simulate low MAP for several steps
        dt = 0.02
        for _ in range(500):
            dy = model(jnp.array(0.0), y, jnp.array(60.0))
            y = jax.tree.map(lambda yi, dyi: yi + dt * dyi, y, dy)
        # HR should increase, resistance should increase
        assert float(y.heart_rate) > 72.0  # above baseline 72
        assert float(y.resistance) > 1.0   # above baseline 1.0

    def test_hypertension_response(self):
        """High MAP should decrease HR (parasympathetic activation)."""
        from vpjax.cardiac.baroreceptor import BaroreflexState, BaroreflexODE, BaroreflexParams
        model = BaroreflexODE(params=BaroreflexParams())
        y = BaroreflexState.steady_state()
        dt = 0.02
        for _ in range(500):
            dy = model(jnp.array(0.0), y, jnp.array(140.0))
            y = jax.tree.map(lambda yi, dyi: yi + dt * dyi, y, dy)
        assert float(y.heart_rate) < 72.0

    def test_differentiable(self):
        from vpjax.cardiac.baroreceptor import sympathetic_fraction
        g = jax.grad(sympathetic_fraction)(jnp.array(90.0))
        assert jnp.isfinite(g)
