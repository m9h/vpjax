"""Tests for the vascular package."""

import jax
import jax.numpy as jnp
import pytest

from vpjax.vascular.compliance import ComplianceParams, grubb_cbv, pressure_to_volume, volume_to_pressure, vessel_resistance, transit_time
from vpjax.vascular.autoregulation import AutoregParams, static_autoregulation, autoregulation_index
from vpjax.vascular.geometry import VascularParams, blood_volume_fraction, total_cbv, deoxygenation_along_capillary


class TestCompliance:
    def test_grubb_baseline(self):
        """CBV/CBV0 = 1 when CBF/CBF0 = 1."""
        v = grubb_cbv(jnp.array(1.0))
        assert jnp.allclose(v, 1.0, atol=1e-6)

    def test_grubb_increases(self):
        """CBV increases with CBF."""
        v = grubb_cbv(jnp.array(1.5))
        assert float(v) > 1.0

    def test_grubb_sublinear(self):
        """CBV increase is sublinear (alpha < 1)."""
        v = grubb_cbv(jnp.array(2.0))
        assert float(v) < 2.0

    def test_pressure_volume_roundtrip(self):
        """P → V → P roundtrip."""
        p = jnp.array(30.0)
        v = pressure_to_volume(p)
        p_back = volume_to_pressure(v)
        assert jnp.allclose(p_back, p, atol=0.1)

    def test_resistance_decreases_with_volume(self):
        """Resistance decreases when volume (vessel diameter) increases."""
        r_base = vessel_resistance(jnp.array(1.0))
        r_dilated = vessel_resistance(jnp.array(1.5))
        assert float(r_dilated) < float(r_base)

    def test_transit_time_baseline(self):
        """Transit time at baseline = tau0."""
        tt = transit_time(jnp.array(1.0), jnp.array(1.0), jnp.array(2.0))
        assert jnp.allclose(tt, 2.0, atol=1e-6)

    def test_differentiable(self):
        g = jax.grad(lambda f: grubb_cbv(f))(jnp.array(1.5))
        assert jnp.isfinite(g)


class TestAutoregulation:
    def test_baseline(self):
        """CBF ~1.0 at target CPP."""
        params = AutoregParams()
        cbf = static_autoregulation(params.cpp_target, params)
        assert jnp.allclose(cbf, 1.0, atol=0.1)

    def test_plateau(self):
        """CBF should be ~constant within autoregulatory range."""
        params = AutoregParams()
        cbf_70 = static_autoregulation(jnp.array(70.0), params)
        cbf_120 = static_autoregulation(jnp.array(120.0), params)
        # Should be close (within 20% of each other)
        ratio = float(cbf_120 / cbf_70)
        assert 0.8 < ratio < 1.2

    def test_ari_perfect(self):
        """ARI = 1 when CBF doesn't change."""
        ari = autoregulation_index(jnp.array(0.0), jnp.array(0.2))
        assert jnp.allclose(ari, 1.0, atol=0.01)

    def test_ari_passive(self):
        """ARI = 0 when CBF changes proportionally to CPP."""
        ari = autoregulation_index(jnp.array(0.2), jnp.array(0.2))
        assert jnp.allclose(ari, 0.0, atol=0.01)


class TestGeometry:
    def test_bvf_positive(self):
        """Blood volume fraction should be positive."""
        bvf = blood_volume_fraction(
            jnp.array(3000.0), jnp.array(3.0), jnp.array(100.0)
        )
        assert float(bvf) > 0.0

    def test_total_cbv_positive(self):
        """Total CBV should be positive and finite."""
        cbv = total_cbv()
        assert float(cbv) > 0.0
        assert jnp.isfinite(cbv)

    def test_deoxygenation_monotonic(self):
        """OEF profile should monotonically increase along capillary."""
        profile = deoxygenation_along_capillary(
            jnp.array(100.0), jnp.array(0.34)
        )
        assert jnp.all(jnp.diff(profile) >= 0)

    def test_deoxygenation_endpoints(self):
        """OEF profile: 0 at inlet, ~total OEF at outlet."""
        profile = deoxygenation_along_capillary(
            jnp.array(100.0), jnp.array(0.34)
        )
        assert jnp.allclose(profile[0], 0.0, atol=0.05)
        assert jnp.allclose(profile[-1], 0.34, atol=0.05)
