"""Tests for vpjax.pet — SUVR, BPnd, and joint amyloid–CMRO2 models."""

import jax.numpy as jnp
import numpy as np
import pytest

from vpjax.metabolism.fick import compute_cao2
from vpjax.pet import (
    BindingPotentialParams,
    JointAmyloidParams,
    PETPVCParams,
    compute_suvr,
    fit_bpnd_per_region,
    fit_joint_amyloid_cmro2,
    forward_joint_amyloid_cmro2,
    forward_static_suvr,
    muller_gartner_correction,
    regional_suvr,
)


# ---------------------------------------------------------------------------
# SUVR computation
# ---------------------------------------------------------------------------

class TestComputeSuvr:

    def test_uniform_pet_yields_unit_suvr(self):
        pet = np.full((4, 4, 4), 5.0, dtype=np.float32)
        ref = np.zeros_like(pet, dtype=np.float32)
        ref[1:3, 1:3, 1:3] = 1.0
        suvr, ref_mean = compute_suvr(pet, ref)
        assert np.isclose(ref_mean, 5.0)
        assert np.allclose(suvr, 1.0)

    def test_target_twice_reference(self):
        pet = np.zeros((4, 4, 4), dtype=np.float32)
        # reference uptake = 1, target uptake = 2 (SUVR = 2)
        pet[..., 0:2] = 1.0
        pet[..., 2:4] = 2.0
        ref = np.zeros_like(pet)
        ref[..., 0:2] = 1.0
        suvr, _ = compute_suvr(pet, ref)
        assert np.allclose(suvr[..., 2:4], 2.0)

    def test_empty_reference_raises(self):
        pet = np.ones((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            compute_suvr(pet, np.zeros_like(pet))


class TestRegionalSuvr:

    def test_per_region_means(self):
        pet = np.zeros((4, 4, 2), dtype=np.float32)
        ref = np.zeros_like(pet)
        ref[0, 0, 0] = ref[0, 1, 0] = ref[1, 0, 0] = 1.0
        pet[ref > 0] = 1.0  # reference uptake 1.0

        labels = np.zeros((4, 4, 2), dtype=np.int32)
        labels[2, :, :] = 10
        labels[3, :, :] = 20
        pet[labels == 10] = 1.5  # SUVR 1.5
        pet[labels == 20] = 2.0  # SUVR 2.0

        suvr_vals, ids = regional_suvr(
            pet, ref, labels, min_voxels=1,
        )
        order = np.argsort(ids)
        assert list(ids[order]) == [10, 20]
        assert np.allclose(suvr_vals[order], [1.5, 2.0])


class TestMullerGartner:

    def test_pure_gm_unchanged(self):
        pet = np.full((4, 4, 4), 1.5, dtype=np.float32)
        gm = np.ones_like(pet)
        wm = np.zeros_like(pet)
        out = muller_gartner_correction(pet, gm, wm)
        assert np.allclose(out, 1.5, atol=1e-5)

    def test_recovers_gm_signal_from_mixed_voxels(self):
        # observed = 0.5*GM_true + 0.5*WM_assumed; with WM = 0.6, observed=1
        # → GM_true = 1.4 / 0.5 - 0.6 = 1.4 (since 0.5*GM + 0.5*0.6 = 1.0
        # gives GM = (1.0 - 0.3) / 0.5 = 1.4)
        pet = np.full((4, 4, 4), 1.0, dtype=np.float32)
        gm = np.full_like(pet, 0.5)
        wm = np.full_like(pet, 0.5)
        params = PETPVCParams(
            gm_threshold=jnp.array(0.3),
            wm_assumed_uptake=jnp.array(0.6),
        )
        out = muller_gartner_correction(pet, gm, wm, params=params)
        assert np.allclose(out, 1.4, atol=1e-5)


# ---------------------------------------------------------------------------
# Static SUVR / BPnd forward + inverse
# ---------------------------------------------------------------------------

class TestForwardStaticSuvr:

    def test_no_flow_correction(self):
        bpnd = jnp.array([0.0, 0.5, 1.0])
        suvr = forward_static_suvr(bpnd)
        assert jnp.allclose(suvr, jnp.array([1.0, 1.5, 2.0]))

    def test_flow_correction_above_baseline(self):
        bpnd = jnp.array([0.5])
        cbf = jnp.array([60.0])
        suvr = forward_static_suvr(
            bpnd, cbf=cbf, cbf_ref=50.0, beta_flow=0.1,
        )
        # 1 + 0.5 + 0.1 * (60/50 - 1) = 1.5 + 0.02 = 1.52
        assert jnp.allclose(suvr, jnp.array([1.52]))


class TestFitBpndPerRegion:

    def test_recovers_bpnd_no_flow(self):
        true_bpnd = np.array([0.0, 0.4, 0.8, 1.2])
        suvr_obs = 1.0 + true_bpnd
        out = fit_bpnd_per_region(suvr_obs, n_steps=200, learning_rate=0.1)
        assert np.allclose(out["bpnd"], true_bpnd, atol=1e-3)
        assert float(out["loss"]) < 1e-6

    def test_recovers_bpnd_with_flow(self):
        true_bpnd = np.array([0.0, 0.5, 1.0, 1.5])
        true_beta = 0.15
        cbf_ref = 50.0
        cbf = np.array([45.0, 55.0, 60.0, 40.0])
        suvr_obs = (
            1.0 + true_bpnd + true_beta * (cbf / cbf_ref - 1.0)
        ).astype(np.float32)

        out = fit_bpnd_per_region(
            suvr_obs, cbf_obs=cbf, cbf_ref=cbf_ref,
            fit_flow=True, n_steps=600, learning_rate=0.05,
        )
        assert np.allclose(out["bpnd"], true_bpnd, atol=5e-2)
        assert float(out["loss"]) < 1e-3


# ---------------------------------------------------------------------------
# Joint amyloid–CMRO2
# ---------------------------------------------------------------------------

class TestJointAmyloidCMRO2:

    def test_forward_returns_three_quantities(self):
        bpnd = jnp.array([0.0, 0.5, 1.0])
        oef = jnp.array([0.40, 0.40, 0.40])
        cbf = jnp.array([50.0, 50.0, 50.0])
        params = JointAmyloidParams()
        out = forward_joint_amyloid_cmro2(
            bpnd, oef, cbf, cbf_ref=50.0, params=params,
        )
        assert "suvr" in out and "cmro2" in out and "cmro2_coupling" in out
        # CMRO2 baseline ~160; for bpnd=0 should equal 160; with α=-0.10:
        # bpnd=1 → 160·(1 - 0.10) = 144
        assert jnp.allclose(out["cmro2_coupling"][0], 160.0)
        assert jnp.allclose(out["cmro2_coupling"][2], 144.0)

    def test_oef_pulled_toward_prior(self):
        """With strong λ_oef and weak coupling, OEF stays at prior."""
        rng = np.random.default_rng(0)
        true_bpnd = rng.uniform(0.0, 1.0, size=8).astype(np.float32)
        cbf = rng.uniform(40, 60, size=8).astype(np.float32)
        suvr = 1.0 + true_bpnd

        out = fit_joint_amyloid_cmro2(
            suvr, cbf, cbf_ref=50.0,
            lambda_couple=0.0, lambda_oef=10.0,
            n_steps=300,
        )
        assert np.allclose(out["oef"], 0.40, atol=1e-2)
        assert np.allclose(out["bpnd"], true_bpnd, atol=1e-2)

    def test_coupling_reduces_oef_in_high_amyloid(self):
        """With dominant coupling prior, OEF must adjust so Fick·CBF
        matches CMRO2_baseline·(1 + α·BPnd) — i.e. OEF must drop where
        BPnd is high (since baseline CMRO2 is fixed and CBF is similar).
        """
        cao2 = float(compute_cao2())
        cbf = np.full(4, 50.0, dtype=np.float32)
        true_bpnd = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
        suvr = 1.0 + true_bpnd

        out = fit_joint_amyloid_cmro2(
            suvr, cbf, cbf_ref=50.0,
            lambda_couple=10.0, lambda_oef=1e-3,
            n_steps=600, learning_rate=0.02,
        )

        # Higher BPnd → lower implied OEF (alpha_amy = -0.10 default).
        assert out["oef"][0] > out["oef"][3]
        # Implied CMRO2 should stay in physiological range
        assert np.all(out["cmro2_fick"] > 50)
        assert np.all(out["cmro2_fick"] < 250)
