"""Tests for the metabolism package (cmro2, oef, fick)."""

import jax
import jax.numpy as jnp
import pytest

from vpjax.metabolism.cmro2 import CMRO2Params, compute_cmro2, compute_cmro2_absolute, compute_cmro2_from_cbf_oef
from vpjax.metabolism.oef import OEFParams, compute_oef, extraction_fraction, oef_from_coupled_ratio
from vpjax.metabolism.fick import FickParams, compute_cao2, fick_cmro2, fick_oef, fick_cbf


class TestCMRO2:
    def test_baseline(self):
        """CMRO2 ratio = 1.0 at zero activity."""
        assert float(compute_cmro2(jnp.array(0.0))) == 1.0

    def test_positive_activity(self):
        """CMRO2 increases with activity."""
        ratio = compute_cmro2(jnp.array(1.0))
        assert float(ratio) > 1.0

    def test_absolute_baseline(self):
        """Absolute CMRO2 at rest = baseline value."""
        params = CMRO2Params()
        val = compute_cmro2_absolute(jnp.array(0.0), params)
        assert jnp.allclose(val, 150.0, atol=0.1)

    def test_fick_based(self):
        """CMRO2 from CBF × OEF × CaO2."""
        cbf = jnp.array(50.0)  # mL/100g/min
        oef = jnp.array(0.34)
        cao2 = jnp.array(8.3)  # µmol/mL
        cmro2 = compute_cmro2_from_cbf_oef(cbf, oef, cao2)
        assert float(cmro2) == pytest.approx(50.0 * 0.34 * 8.3, rel=0.01)

    def test_differentiable(self):
        g = jax.grad(lambda a: compute_cmro2(a))(jnp.array(1.0))
        assert jnp.isfinite(g)


class TestOEF:
    def test_baseline(self):
        """OEF = E0 at baseline (CBF=1, CMRO2=1)."""
        oef = compute_oef(jnp.array(1.0), jnp.array(1.0))
        assert float(oef) == pytest.approx(0.34, abs=0.01)

    def test_oef_decreases_with_flow(self):
        """OEF decreases when flow increases more than metabolism."""
        oef_rest = compute_oef(jnp.array(1.0), jnp.array(1.0))
        oef_active = compute_oef(jnp.array(1.5), jnp.array(1.2))
        assert float(oef_active) < float(oef_rest)

    def test_extraction_at_rest(self):
        """Buxton extraction at baseline flow = E0."""
        E0 = jnp.array(0.34)
        E = extraction_fraction(jnp.array(1.0), E0)
        assert jnp.allclose(E, E0, atol=1e-6)

    def test_extraction_decreases_with_flow(self):
        E_low = extraction_fraction(jnp.array(1.0))
        E_high = extraction_fraction(jnp.array(2.0))
        assert float(E_high) < float(E_low)

    def test_coupled_ratio(self):
        """With coupling ratio n, OEF should decrease during activation."""
        oef_rest = oef_from_coupled_ratio(jnp.array(1.0))
        oef_active = oef_from_coupled_ratio(jnp.array(1.5))
        assert float(oef_active) < float(oef_rest)


class TestFick:
    def test_cao2_physiological(self):
        """CaO2 should be in physiological range (~7-9 µmol/mL)."""
        cao2 = compute_cao2()
        assert 7.0 < float(cao2) < 10.0

    def test_fick_roundtrip(self):
        """CMRO2 → OEF → CMRO2 roundtrip."""
        cbf = jnp.array(50.0)
        oef = jnp.array(0.34)
        cao2 = compute_cao2()
        cmro2 = fick_cmro2(cbf, oef, cao2)
        oef_back = fick_oef(cmro2, cbf, cao2)
        assert jnp.allclose(oef_back, oef, atol=1e-4)

    def test_fick_cbf(self):
        """Recover CBF from CMRO2 and OEF."""
        cbf = jnp.array(50.0)
        oef = jnp.array(0.34)
        cao2 = compute_cao2()
        cmro2 = fick_cmro2(cbf, oef, cao2)
        cbf_back = fick_cbf(cmro2, oef, cao2)
        assert jnp.allclose(cbf_back, cbf, atol=0.1)

    def test_differentiable(self):
        g = jax.grad(lambda o: fick_cmro2(jnp.array(50.0), o))(jnp.array(0.34))
        assert jnp.isfinite(g)
