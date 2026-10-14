"""The gnf-jax E/I block as a state-space latent: consistency and filter checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

gnfjax = pytest.importorskip("gnfjax")
from gnfjax import GNFParams, simulate_linearised  # noqa: E402

from vpjax.neural import ENTRY_POINTS, band_power, linear_ei_block, statespace_spec  # noqa: E402
from vpjax.statespace import ll_filter  # noqa: E402

# Stable, oscillatory, single admissible steady state; resonance near 9 Hz.
PARAMS = GNFParams(alpha_ee=11.0, alpha_ie=10.0, alpha_ei=10.0, alpha_ii=2.0, p=-1.0, q=-8.0,
                   tau_e=0.01, tau_i=0.02, noise_e=0.05, noise_i=0.05)


@pytest.fixture(scope="module")
def single():
    return linear_ei_block(PARAMS)


@pytest.fixture(scope="module")
def multi():
    return linear_ei_block(PARAMS, eigs=jnp.array([0.0, -1.0, -4.0]))


def test_steady_state_found_and_stable(single):
    assert int(single.n_valid) == 1
    assert float(single.distance_to_instability()) > 0
    assert 5.0 < float(single.resonance_hz()[0]) < 15.0


def test_lyapunov_and_jacobian(multi):
    J, P0, Q = multi.jacobian(), multi.stationary_covariance(), multi.Q
    assert J.shape == (6, 6)
    np.testing.assert_allclose(np.asarray(J @ P0 + P0 @ J.T + Q), 0.0, atol=1e-4 * float(jnp.abs(Q).max()))
    x = jnp.arange(6.0) * 0.1
    np.testing.assert_allclose(np.asarray(jax.jacobian(multi.drift)(x)), np.asarray(J), rtol=1e-6)


def test_spectrum_integrates_to_variance(multi):
    f = jnp.linspace(0.0, 2000.0, 400_001)
    s_e, s_i = multi.spectrum(f)
    var = jnp.diag(multi.stationary_covariance()).reshape(3, 2)
    np.testing.assert_allclose(np.asarray(jnp.trapezoid(s_e, f, axis=0)), np.asarray(var[:, 0]), rtol=1e-2)
    np.testing.assert_allclose(np.asarray(jnp.trapezoid(s_i, f, axis=0)), np.asarray(var[:, 1]), rtol=1e-2)


def test_input_enters_the_constant_mode_only(multi):
    x = jnp.zeros(6)
    p = multi.params
    a = p.d_e * multi.ess * (1.0 - p.d_e * multi.ess)
    b = p.d_i * multi.iss * (1.0 - p.d_i * multi.iss)
    dp = multi.drift(x, 1.0, "p") - multi.drift(x)
    dq = multi.drift(x, 1.0, "q") - multi.drift(x)
    np.testing.assert_allclose(np.asarray(dp), [a / p.tau_e, 0, 0, 0, 0, 0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(dq), [0, b / p.tau_i, 0, 0, 0, 0], rtol=1e-6)
    for entry in ENTRY_POINTS:
        v = multi.input_vector(entry)
        assert np.all(np.asarray(v[2:]) == 0.0)
    w = multi.input_vector("p", mode_weights=jnp.array([0.0, 1.0, 0.0]))
    assert float(w[0]) == 0.0 and float(w[2]) != 0.0
    with pytest.raises(ValueError):
        multi.input_vector("drug")


def test_band_power_is_a_derived_differentiable_observable():
    def alpha_band_e(log_aii):
        prm = PARAMS._replace(alpha_ii=jnp.exp(log_aii))
        return band_power(linear_ei_block(prm), 8.0, 13.0)[0][0]

    g = jax.grad(alpha_band_e)(jnp.log(2.0))
    assert np.isfinite(float(g)) and float(g) != 0.0
    lo, hi = alpha_band_e(jnp.log(1.0)), alpha_band_e(jnp.log(4.0))
    assert lo != hi


def test_filter_prefers_true_parameters():
    dt, n_steps, down = 2e-4, 100_000, 10
    b = linear_ei_block(PARAMS)
    e, i = simulate_linearised(PARAMS, b.eigs, b.ess, b.iss, jax.random.PRNGKey(0),
                               dt=dt, n_steps=n_steps, downsample=down)
    t = jnp.arange(e.shape[0]) * dt * down
    noise_sd = 0.02
    y = e[:, 0:1] + noise_sd * jax.random.normal(jax.random.PRNGKey(1), (e.shape[0], 1))
    mask = jnp.ones_like(y, dtype=bool)

    def loglik(tau_e):
        blk = linear_ei_block(PARAMS._replace(tau_e=tau_e), steady=(b.ess, b.iss))
        spec = statespace_spec(blk)
        h = lambda t_, x, args: x[0:1]
        return ll_filter(spec["f"], h, t, y, mask, spec["Q"], jnp.array([noise_sd**2]),
                         spec["m0"], spec["P0"]).log_likelihood

    ll = {k: float(loglik(k * PARAMS.tau_e)) for k in (0.5, 1.0, 2.0)}
    assert all(np.isfinite(v) for v in ll.values())
    assert ll[1.0] > ll[0.5] and ll[1.0] > ll[2.0]
    g = jax.grad(loglik)(PARAMS.tau_e)
    assert np.isfinite(float(g))
