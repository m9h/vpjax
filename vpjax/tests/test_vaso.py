"""Tests for the VASO package."""

import jax
import jax.numpy as jnp
import pytest

from vpjax.vaso.signal_model import VASOParams, vaso_signal, vaso_signal_change, cbv_from_vaso_signal, blood_nulling_ti
from vpjax.vaso.boco import bold_correction, delta_cbv_from_boco
from vpjax.vaso.devein import DeveinParams, build_drainage_matrix, apply_vein_contamination, devein
from vpjax.vaso.cbv_mapping import relative_cbv_change, balloon_cbv_ratio


class TestVASOSignalModel:
    def test_baseline_signal(self):
        """VASO signal at baseline CBV should be 1.0."""
        params = VASOParams()
        s = vaso_signal(params.CBV0, params)
        assert jnp.allclose(s, 1.0, atol=1e-6)

    def test_signal_decreases_with_cbv(self):
        """VASO signal decreases when CBV increases."""
        params = VASOParams()
        s_rest = vaso_signal(jnp.array(0.05), params)
        s_active = vaso_signal(jnp.array(0.07), params)
        assert float(s_active) < float(s_rest)

    def test_signal_change_baseline(self):
        """Signal change should be 0 at baseline (v=1)."""
        ds = vaso_signal_change(jnp.array(1.0))
        assert jnp.allclose(ds, 0.0, atol=1e-6)

    def test_signal_change_negative_activation(self):
        """VASO signal change should be negative during activation (CBV up)."""
        ds = vaso_signal_change(jnp.array(1.3))  # 30% CBV increase
        assert float(ds) < 0.0

    def test_cbv_recovery(self):
        """cbv_from_vaso_signal should invert vaso_signal_change."""
        params = VASOParams()
        v = jnp.array(1.2)
        ds = vaso_signal_change(v, params)
        delta_cbv = cbv_from_vaso_signal(ds, params)
        expected_delta_cbv = params.CBV0 * (v - 1.0)
        assert jnp.allclose(delta_cbv, expected_delta_cbv, atol=1e-4)

    def test_blood_nulling_ti(self):
        """TI should be positive and reasonable."""
        ti = blood_nulling_ti(jnp.array(2.09))
        assert 1.0 < float(ti) < 2.0

    def test_differentiable(self):
        g = jax.grad(lambda v: vaso_signal_change(v))(jnp.array(1.2))
        assert jnp.isfinite(g)


class TestBOCO:
    def test_bold_correction_unity(self):
        """BOCO with equal null/not-null should give ~1."""
        s = jnp.array(100.0)
        corrected = bold_correction(s, s)
        assert jnp.allclose(corrected, 1.0, atol=1e-6)

    def test_delta_cbv_from_boco(self):
        """Baseline period should give ~0 CBV change."""
        # Simulate constant baseline
        s_corrected = jnp.ones((20, 4, 4))
        dcbv = delta_cbv_from_boco(s_corrected, baseline_start=0, baseline_end=10)
        assert jnp.allclose(dcbv, 0.0, atol=1e-6)


class TestDevein:
    def test_drainage_matrix_diagonal(self):
        """Diagonal of drainage matrix should be 1."""
        D = build_drainage_matrix()
        for i in range(3):
            assert jnp.allclose(D[i, i], 1.0, atol=1e-6)

    def test_devein_roundtrip(self):
        """Contaminate then devein should recover original signal."""
        local = jnp.array([1.0, 0.5, 0.2])
        contaminated = apply_vein_contamination(local)
        recovered = devein(contaminated)
        assert jnp.allclose(recovered, local, atol=0.05)

    def test_contamination_increases_superficial(self):
        """Ascending veins should add signal to superficial layers."""
        local = jnp.array([1.0, 0.0, 0.0])  # only deep layer
        contaminated = apply_vein_contamination(local)
        assert float(contaminated[2]) > 0.0  # superficial contaminated


class TestCBVMapping:
    def test_relative_cbv_baseline(self):
        """Relative CBV change at baseline should be ~0."""
        s = jnp.array(100.0)
        dcbv = relative_cbv_change(s, s)
        assert jnp.allclose(dcbv, 0.0, atol=1e-6)

    def test_balloon_cbv_ratio_baseline(self):
        """Balloon ratio at 0 change = 1."""
        v = balloon_cbv_ratio(jnp.array(0.0))
        assert jnp.allclose(v, 1.0, atol=1e-6)

    def test_balloon_cbv_ratio_positive(self):
        """Positive ΔCBV/CBV₀ → v > 1."""
        v = balloon_cbv_ratio(jnp.array(0.2))
        assert float(v) == pytest.approx(1.2)
