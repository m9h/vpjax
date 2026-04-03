"""Tests for the qBOLD package."""

import jax
import jax.numpy as jnp
import pytest

from vpjax.qbold.signal_model import QBOLDParams, qbold_signal, compute_r2prime, compute_r2star, characteristic_frequency
from vpjax.qbold.dbv import dbv_from_r2prime, dbv_from_cbv, DBVParams
from vpjax.qbold.calibrated import CalibratedBOLDParams, davis_model, estimate_M_from_r2prime, estimate_cmro2_change


class TestQBOLDSignal:
    def test_signal_at_te_zero(self):
        """Signal at TE=0 should be ~S0."""
        te = jnp.array([0.0, 0.01, 0.02])
        signal = qbold_signal(te, jnp.array(0.34), jnp.array(0.03))
        assert jnp.allclose(signal[0], 1.0, atol=0.01)

    def test_signal_decays(self):
        """Signal should decay with TE."""
        te = jnp.array([0.005, 0.015, 0.025, 0.035])
        signal = qbold_signal(te, jnp.array(0.34), jnp.array(0.03))
        # Should monotonically decrease
        assert jnp.all(jnp.diff(signal) < 0)

    def test_higher_oef_faster_decay(self):
        """Higher OEF should cause faster signal decay."""
        te = jnp.array([0.0, 0.020, 0.040])
        sig_low = qbold_signal(te, jnp.array(0.20), jnp.array(0.03))
        sig_high = qbold_signal(te, jnp.array(0.50), jnp.array(0.03))
        # At long TE, high OEF signal should be lower
        assert float(sig_high[-1]) < float(sig_low[-1])

    def test_r2prime_positive(self):
        r2p = compute_r2prime(jnp.array(0.34), jnp.array(0.03))
        assert float(r2p) > 0.0

    def test_r2star_greater_than_r2(self):
        params = QBOLDParams()
        r2star = compute_r2star(jnp.array(0.34), jnp.array(0.03), params)
        assert float(r2star) > float(params.R2t)

    def test_differentiable(self):
        te = jnp.array([0.01, 0.02, 0.03])
        g = jax.grad(lambda oef: jnp.sum(qbold_signal(te, oef, jnp.array(0.03))))(jnp.array(0.34))
        assert jnp.isfinite(g)


class TestDBV:
    def test_dbv_roundtrip(self):
        """DBV from R2' should recover original DBV."""
        oef = jnp.array(0.34)
        dbv_orig = jnp.array(0.03)
        r2p = compute_r2prime(oef, dbv_orig)
        dbv_recovered = dbv_from_r2prime(r2p, oef)
        assert jnp.allclose(dbv_recovered, dbv_orig, atol=1e-4)

    def test_dbv_from_cbv(self):
        """DBV from CBV should be reasonable."""
        cbv = jnp.array(0.04)
        dbv = dbv_from_cbv(cbv)
        assert 0.0 < float(dbv) < float(cbv)


class TestCalibratedBOLD:
    def test_davis_baseline(self):
        """Davis model at baseline (CBF=CMRO2=1) should give 0."""
        bold = davis_model(jnp.array(1.0), jnp.array(1.0), jnp.array(0.08))
        assert jnp.allclose(bold, 0.0, atol=1e-6)

    def test_davis_activation(self):
        """CBF increase > CMRO2 increase → positive BOLD."""
        bold = davis_model(jnp.array(1.5), jnp.array(1.2), jnp.array(0.08))
        assert float(bold) > 0.0

    def test_M_from_r2prime(self):
        """M should be positive and reasonable."""
        M = estimate_M_from_r2prime(jnp.array(3.0))
        assert 0.01 < float(M) < 0.5

    def test_cmro2_roundtrip(self):
        """Estimate CMRO2 change then verify via Davis model."""
        M = jnp.array(0.08)
        cbf_r = jnp.array(1.5)
        cmro2_r = jnp.array(1.2)
        bold = davis_model(cbf_r, cmro2_r, M)
        cmro2_back = estimate_cmro2_change(bold, cbf_r, M)
        assert jnp.allclose(cmro2_back, cmro2_r, atol=0.01)
