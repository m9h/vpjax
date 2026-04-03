"""Tests for the Riera neurovascular coupling model."""

import jax
import jax.numpy as jnp
import pytest

from vpjax.hemodynamics.riera import RieraNVC, RieraParams, RieraState, riera_total_cbv, riera_to_balloon


class TestRieraSteadyState:
    def test_rhs_at_rest(self):
        """RHS should be ~zero at baseline with zero stimulus."""
        model = RieraNVC(params=RieraParams())
        y0 = RieraState.steady_state()
        dy = model(jnp.array(0.0), y0, jnp.array(0.0))
        assert jnp.allclose(dy.s_no, 0.0, atol=1e-6)
        assert jnp.allclose(dy.s_ade, 0.0, atol=1e-6)
        assert jnp.allclose(dy.f_a, 0.0, atol=1e-6)
        assert jnp.allclose(dy.v_a, 0.0, atol=1e-6)
        assert jnp.allclose(dy.v_v, 0.0, atol=1e-6)
        assert jnp.allclose(dy.q_v, 0.0, atol=1e-6)

    def test_total_cbv_baseline(self):
        """Total CBV at rest = 1.0."""
        y0 = RieraState.steady_state()
        cbv = riera_total_cbv(y0)
        assert jnp.allclose(cbv, 1.0, atol=1e-6)

    def test_to_balloon_baseline(self):
        """Balloon-equivalent should be (1, 1) at rest."""
        y0 = RieraState.steady_state()
        v, q = riera_to_balloon(y0)
        assert jnp.allclose(v, 1.0, atol=1e-6)
        assert jnp.allclose(q, 1.0, atol=1e-6)


class TestRieraStimulus:
    def test_no_signal_drives_flow(self):
        """NO vasodilatory signal should respond to stimulus."""
        model = RieraNVC(params=RieraParams())
        y0 = RieraState.steady_state()
        dy = model(jnp.array(0.0), y0, jnp.array(1.0))
        # NO signal derivative should be positive with positive stimulus
        assert float(dy.s_no) > 0.0

    def test_metabolic_response(self):
        """CMRO2 should move toward elevated state with stimulus."""
        model = RieraNVC(params=RieraParams())
        y0 = RieraState.steady_state()
        dy = model(jnp.array(0.0), y0, jnp.array(1.0))
        assert float(dy.cmro2) > 0.0

    def test_differentiable(self):
        """Gradients should flow through the model."""
        model = RieraNVC(params=RieraParams())
        y0 = RieraState.steady_state()
        def loss(c_no):
            p = RieraParams(c_no=c_no)
            m = RieraNVC(params=p)
            dy = m(jnp.array(0.0), y0, jnp.array(1.0))
            return dy.s_no
        g = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0
