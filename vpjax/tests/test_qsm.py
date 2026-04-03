"""Tests for the QSM package."""

import jax
import jax.numpy as jnp
import pytest

from vpjax.qsm.susceptibility import SusceptibilityParams, susceptibility_from_sources, oef_from_susceptibility
from vpjax.qsm.r2star_fitting import fit_r2star_loglinear, fit_r2star_volume, multi_echo_combine
from vpjax.qsm.phase import temporal_unwrap, phase_to_frequency, echo_combination_weights


class TestSusceptibility:
    def test_iron_positive(self):
        """Iron should produce positive (paramagnetic) susceptibility."""
        chi = susceptibility_from_sources(jnp.array(1.0), jnp.array(0.0))
        assert float(chi) > 0.0

    def test_myelin_negative(self):
        """Myelin should produce negative (diamagnetic) susceptibility."""
        chi = susceptibility_from_sources(jnp.array(0.0), jnp.array(1.0))
        assert float(chi) < 0.0

    def test_zero_sources(self):
        """No iron/myelin should give reference susceptibility."""
        params = SusceptibilityParams()
        chi = susceptibility_from_sources(jnp.array(0.0), jnp.array(0.0), params=params)
        assert jnp.allclose(chi, params.chi_ref, atol=1e-6)

    def test_oef_from_chi_roundtrip(self):
        """Susceptibility → OEF should recover original OEF."""
        params = SusceptibilityParams()
        iron = jnp.array(0.5)
        myelin = jnp.array(0.3)
        oef = jnp.array(0.34)
        dbv = jnp.array(0.03)
        chi = susceptibility_from_sources(iron, myelin, oef, dbv, params)
        oef_back = oef_from_susceptibility(chi, iron, myelin, dbv, params)
        assert jnp.allclose(oef_back, oef, atol=0.02)

    def test_differentiable(self):
        g = jax.grad(lambda iron: susceptibility_from_sources(iron, jnp.array(0.3)))(jnp.array(0.5))
        assert jnp.isfinite(g)


class TestR2StarFitting:
    def test_loglinear_mono_exponential(self):
        """Should recover R2* from a perfect monoexponential."""
        te = jnp.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035])
        r2_true = 25.0
        s0_true = 1000.0
        signal = s0_true * jnp.exp(-r2_true * te)
        r2_fit, s0_fit = fit_r2star_loglinear(signal[None, :], te)
        assert jnp.allclose(r2_fit[0], r2_true, atol=0.5)
        assert jnp.allclose(s0_fit[0], s0_true, rtol=0.01)

    def test_volume_fit(self):
        """fit_r2star_volume should handle multiple voxels."""
        te = jnp.array([0.005, 0.015, 0.025, 0.035])
        r2_vals = jnp.array([20.0, 30.0, 40.0])
        signal = 1000.0 * jnp.exp(-r2_vals[:, None] * te[None, :])
        r2_fit, s0_fit = fit_r2star_volume(signal, te)
        assert r2_fit.shape == (3,)
        assert jnp.allclose(r2_fit, r2_vals, atol=1.0)

    def test_combine_mean(self):
        """Mean combination should give the average."""
        signal = jnp.array([100.0, 80.0, 60.0])
        te = jnp.array([0.01, 0.02, 0.03])
        combined = multi_echo_combine(signal, te, method="mean")
        assert jnp.allclose(combined, 80.0, atol=0.1)

    def test_combine_first(self):
        """First echo combination should return first echo."""
        signal = jnp.array([100.0, 80.0, 60.0])
        te = jnp.array([0.01, 0.02, 0.03])
        combined = multi_echo_combine(signal, te, method="first")
        assert jnp.allclose(combined, 100.0, atol=1e-6)


class TestPhase:
    def test_temporal_unwrap_no_wraps(self):
        """No wraps → output should match input."""
        te = jnp.array([0.005, 0.010, 0.015])
        phase = jnp.array([0.1, 0.2, 0.3])
        unwrapped = temporal_unwrap(phase, te)
        assert jnp.allclose(unwrapped, phase, atol=0.1)

    def test_phase_to_frequency(self):
        """Should recover frequency from linear phase ramp."""
        te = jnp.array([0.005, 0.010, 0.015, 0.020])
        freq_true = 10.0  # Hz
        phase = 2.0 * jnp.pi * freq_true * te
        freq_est = phase_to_frequency(phase, te)
        assert jnp.allclose(freq_est, freq_true, atol=0.5)

    def test_echo_weights_sum_to_one(self):
        """Combination weights should sum to 1."""
        te = jnp.array([0.005, 0.015, 0.025, 0.035])
        weights = echo_combination_weights(te)
        assert jnp.allclose(jnp.sum(weights), 1.0, atol=1e-5)

    def test_echo_weights_positive(self):
        """All weights should be positive."""
        te = jnp.array([0.005, 0.015, 0.025, 0.035])
        weights = echo_combination_weights(te)
        assert jnp.all(weights > 0)
