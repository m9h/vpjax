"""Tests for perfusion extensions (kinetic, trust, calibration)."""

import jax
import jax.numpy as jnp
import pytest

from vpjax.perfusion.kinetic import ASLKineticParams, asl_kinetic_signal, quantify_cbf
from vpjax.perfusion.trust import TRUSTParams, t2_to_svo2, svo2_to_t2, trust_oef
from vpjax.perfusion.calibration import blood_t1, m0_from_proton_density, labeling_efficiency


class TestASLKinetic:
    def test_no_signal_before_arrival(self):
        """No ASL signal before bolus arrives."""
        params = ASLKineticParams(delta=jnp.array(1.0))
        t = jnp.array([0.0, 0.5])
        signal = asl_kinetic_signal(t, jnp.array(50.0), params)
        assert jnp.allclose(signal, 0.0, atol=1e-8)

    def test_signal_positive_after_arrival(self):
        """ASL signal should be positive after bolus arrival."""
        params = ASLKineticParams(delta=jnp.array(1.0), tau=jnp.array(1.8))
        t = jnp.array([2.0, 3.0])
        signal = asl_kinetic_signal(t, jnp.array(50.0), params)
        assert jnp.all(signal > 0)

    def test_higher_cbf_more_signal(self):
        """Higher CBF should produce more ASL signal."""
        t = jnp.array([2.5])
        sig_low = asl_kinetic_signal(t, jnp.array(30.0))
        sig_high = asl_kinetic_signal(t, jnp.array(70.0))
        assert float(sig_high[0]) > float(sig_low[0])

    def test_quantify_cbf_roundtrip(self):
        """quantify_cbf should approximately recover CBF."""
        params = ASLKineticParams()
        cbf_true = jnp.array(50.0)
        pld = jnp.array(1.8)
        t = params.delta + pld
        delta_m = asl_kinetic_signal(jnp.array([float(t)]), cbf_true, params)
        cbf_est = quantify_cbf(delta_m[0], pld, params)
        # Should be in the right ballpark (not exact due to model simplification)
        assert 20.0 < float(cbf_est) < 100.0

    def test_differentiable(self):
        t = jnp.array([2.5])
        g = jax.grad(lambda cbf: jnp.sum(asl_kinetic_signal(t, cbf)))(jnp.array(50.0))
        assert jnp.isfinite(g)


class TestTRUST:
    def test_svo2_roundtrip(self):
        """SvO2 → T2 → SvO2 should roundtrip."""
        svo2 = jnp.array(0.65)
        t2 = svo2_to_t2(svo2)
        svo2_back = t2_to_svo2(t2)
        assert jnp.allclose(svo2_back, svo2, atol=0.01)

    def test_oef_physiological(self):
        """TRUST OEF for typical T2 should be in physiological range."""
        # Typical venous T2 at 3T: ~40-70ms
        t2 = jnp.array(0.055)  # 55ms
        oef = trust_oef(t2)
        assert 0.15 < float(oef) < 0.60

    def test_higher_t2_lower_oef(self):
        """Higher T2 → higher SvO2 → lower OEF."""
        oef_low_t2 = trust_oef(jnp.array(0.040))
        oef_high_t2 = trust_oef(jnp.array(0.070))
        assert float(oef_high_t2) < float(oef_low_t2)


class TestCalibration:
    def test_blood_t1_3t(self):
        """Blood T1 at 3T should be ~1.6s."""
        t1 = blood_t1()
        assert 1.4 < float(t1) < 2.0

    def test_m0_from_pd(self):
        """M0 from proton density should scale by 1/lambda."""
        m0_tissue = jnp.array(1000.0)
        m0b = m0_from_proton_density(m0_tissue)
        assert float(m0b) > float(m0_tissue)  # 1/0.9 > 1

    def test_labeling_efficiency(self):
        """Perfect B1 should give nominal efficiency."""
        alpha = labeling_efficiency(jnp.array(1.0))
        assert jnp.allclose(alpha, 0.85, atol=0.01)
