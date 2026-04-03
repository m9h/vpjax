"""Tests for the integrators package (Local Linearization)."""

import jax
import jax.numpy as jnp
import pytest

from vpjax.integrators.local_linearization import ll_step, ll_solve


class TestLocalLinearization:
    def test_linear_ode_exact(self):
        """LL should solve linear ODEs exactly (one step)."""
        # dx/dt = -x  →  x(t) = x0 * exp(-t)
        def f(x):
            return -x

        x0 = jnp.array([1.0])
        dt = 0.5
        x1 = ll_step(f, x0, dt)
        expected = x0 * jnp.exp(-dt)
        assert jnp.allclose(x1, expected, atol=1e-4)

    def test_2d_linear(self):
        """LL on a 2D linear system."""
        # dx/dt = Ax where A = [[-1, 0], [0, -2]]
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        def f(x):
            return A @ x

        x0 = jnp.array([1.0, 1.0])
        dt = 0.1
        x1 = ll_step(f, x0, dt)
        expected = jnp.array([jnp.exp(-0.1), jnp.exp(-0.2)])
        assert jnp.allclose(x1, expected, atol=1e-3)

    def test_solve_trajectory(self):
        """ll_solve should produce correct trajectory length."""
        def f(x):
            return -x

        x0 = jnp.array([1.0])
        ts = jnp.linspace(0, 1, 11)
        ys = ll_solve(f, x0, ts)
        assert ys.shape == (11, 1)
        # Initial condition preserved
        assert jnp.allclose(ys[0], x0, atol=1e-6)
        # Should decay
        assert float(ys[-1, 0]) < float(ys[0, 0])

    def test_exponential_decay(self):
        """ll_solve for dx/dt = -x should match exp(-t)."""
        def f(x):
            return -x

        x0 = jnp.array([1.0])
        ts = jnp.linspace(0, 2, 21)
        ys = ll_solve(f, x0, ts)
        expected = jnp.exp(-ts)[:, None]
        assert jnp.allclose(ys, expected, atol=1e-3)

    def test_nonlinear_ode(self):
        """LL should handle nonlinear ODEs reasonably."""
        # dx/dt = -x^3 (stiff nonlinear)
        def f(x):
            return -x ** 3

        x0 = jnp.array([1.0])
        ts = jnp.linspace(0, 1, 51)
        ys = ll_solve(f, x0, ts)
        # Solution should decay and stay positive
        assert jnp.all(ys > 0)
        assert float(ys[-1, 0]) < float(ys[0, 0])

    def test_differentiable(self):
        """Gradients should flow through LL solver."""
        def f(x):
            return -x

        def loss(x0_val):
            x0 = jnp.array([x0_val])
            ts = jnp.linspace(0, 1, 11)
            ys = ll_solve(f, x0, ts)
            return ys[-1, 0]

        g = jax.grad(loss)(1.0)
        assert jnp.isfinite(g)
