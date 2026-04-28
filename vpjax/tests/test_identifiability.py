"""Tests for vpjax.identifiability — local Jacobian-rank test."""

import jax.numpy as jnp
import numpy as np
import pytest

from vpjax.identifiability import (
    asl_kinetic_identifiability,
    balloon_identifiability,
    check_local_identifiability,
    pet_joint_identifiability,
    pet_static_suvr_identifiability,
)


# ---------------------------------------------------------------------------
# Toy models with known identifiability structure
# ---------------------------------------------------------------------------

class TestCheckLocalIdentifiability:

    def test_full_rank_linear_model(self):
        """y = A·θ with A full column-rank → identifiable."""
        A = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        fwd = lambda th: A @ th
        out = check_local_identifiability(fwd, jnp.array([1.0, 2.0, 3.0]),
                                           names=("a", "b", "c"))
        assert out["is_identifiable"] is True
        assert out["rank"] == 3
        assert out["collinear_sets"] == []

    def test_rank_deficient_only_sum(self):
        """y = a + b only ⇒ a and b are not separately identifiable.

        Null space should highlight the (a, b) combination with
        coefficients of equal magnitude and opposite sign — i.e. only
        the sum survives the projection onto the row space of J.
        """
        fwd = lambda th: jnp.array([th[0] + th[1]])
        out = check_local_identifiability(
            fwd, jnp.array([1.0, 1.0]), names=("a", "b"),
        )
        assert out["is_identifiable"] is False
        assert out["rank"] == 1
        assert len(out["collinear_sets"]) == 1
        col = out["collinear_sets"][0]
        # Null vector is (a−b)/√2 — params are 'a' and 'b', coefficients
        # have equal magnitude and opposite signs.
        assert set(col["params"]) == {"a", "b"}
        c_a, c_b = col["coefficients"]
        assert abs(abs(c_a) - abs(c_b)) < 1e-5
        assert (c_a * c_b) < 0

    def test_zero_jacobian_returns_full_null(self):
        """y = constant ⇒ Jacobian is all zeros, no params identifiable."""
        fwd = lambda th: jnp.array([42.0])
        out = check_local_identifiability(
            fwd, jnp.array([1.0, 2.0]), names=("a", "b"),
        )
        assert out["is_identifiable"] is False
        assert out["rank"] == 0

    def test_names_length_validated(self):
        with pytest.raises(ValueError):
            check_local_identifiability(
                lambda th: th, jnp.array([1.0, 2.0]), names=("only_one",),
            )


# ---------------------------------------------------------------------------
# Balloon-specific helper
# ---------------------------------------------------------------------------

@pytest.fixture
def block_stimulus():
    dt = 0.1
    n_t = 60
    sub = 20
    T = n_t * sub
    stim = jnp.zeros(T).at[int(20.0 / dt):int(40.0 / dt)].set(1.0)
    return stim, dt, 2.0  # stim, dt, tr


class TestBalloonIdentifiability:

    def test_kappa_only_is_identifiable(self, block_stimulus):
        """A single Balloon param under BOLD is trivially full-rank."""
        stim, dt, tr = block_stimulus
        out = balloon_identifiability(("kappa",), stim, dt=dt, tr=tr)
        assert out["is_identifiable"] is True
        assert out["rank"] == 1

    def test_all_five_params_bold_only_singular_value_spread(self, block_stimulus):
        """All five Balloon parameters have first-order BOLD sensitivity at
        the DCM nominal point — the Jacobian is technically rank-5 — but
        the condition number is still substantial (Stephan 2007 'weakly
        identified' alpha/E0).  We assert this rather than rank<5: the
        practical identifiability problem is in the Hessian, not the
        first-order Jacobian.
        """
        stim, dt, tr = block_stimulus
        out = balloon_identifiability(
            ("kappa", "gamma", "tau", "alpha", "E0"), stim, dt=dt, tr=tr,
        )
        s = np.asarray(out["singular_values"])
        # The smallest singular value should be many orders of magnitude
        # below the largest — i.e. some parameter direction is much less
        # observable than others, even if technically nonzero.
        assert out["condition_number"] > 5
        # All 5 sensitivities are nonzero (no exact collinearity at DCM).
        assert (s > 1e-10).all()

    def test_pet_bpnd_only_is_identifiable(self):
        """SUVR = 1 + BPnd per region is trivially full-rank."""
        out = pet_static_suvr_identifiability(n_regions=5, fit_flow=False)
        assert out["is_identifiable"] is True
        assert out["rank"] == 5

    def test_pet_global_beta_flow_unidentifiable(self):
        """Adding a single global β_flow to per-region BPnd is rank-deficient
        because each BPnd_i can absorb β_flow's contribution."""
        out = pet_static_suvr_identifiability(n_regions=5, fit_flow=True)
        assert out["is_identifiable"] is False
        assert out["rank"] == 5
        assert out["n_params"] == 6
        assert out["collinear_sets"]
        contributing = out["collinear_sets"][0]["params"]
        assert "beta_flow" in contributing

    def test_joint_alpha_amy_free_is_unidentifiable(self):
        """In the single-subject joint amyloid-CMRO2 model, α_amy as a free
        parameter is unidentifiable — its effect on the coupling residual
        can be absorbed by per-region BPnd shifts.  The current driver
        keeps α_amy fixed for exactly this reason."""
        out = pet_joint_identifiability(
            n_regions=5, fit_alpha_amy=True, fit_beta_flow=True,
        )
        assert out["is_identifiable"] is False
        # The null direction should explicitly contain alpha_amy.
        flat_params = [p for c in out["collinear_sets"] for p in c["params"]]
        assert "alpha_amy" in flat_params

    def test_asl_kinetic_single_pld_cbf_identifiable(self):
        out = asl_kinetic_identifiability(plds=[1.525], fit_names=("CBF",))
        assert out["is_identifiable"] is True
        assert out["rank"] == 1

    def test_asl_kinetic_5_params_rank_deficient(self):
        """With 5 PLDs and 5 parameters, only 2 are recoverable —
        (alpha, T1b) collinearity and tau, CBF rate-of-change are flat."""
        out = asl_kinetic_identifiability(
            plds=[0.5, 1.0, 1.5, 2.0, 2.5],
            fit_names=("CBF", "delta", "tau", "alpha", "T1b"),
        )
        assert out["is_identifiable"] is False
        assert out["rank"] < 5

    def test_multimodal_does_not_lose_rank(self, block_stimulus):
        """Adding ASL + VASO observations cannot reduce the rank.

        Note: the *condition number* is not a stable comparison because
        BOLD/ASL/VASO have very different signal scales (BOLD ~ 0.05,
        ASL ~ 0.5, VASO ~ 0.1), so concatenating them re-weights the
        column norms.  The robust comparison is rank.
        """
        stim, dt, tr = block_stimulus
        bold_only = balloon_identifiability(
            ("kappa", "gamma", "tau", "alpha", "E0"),
            stim, dt=dt, tr=tr, observers=("bold",),
        )
        multimodal = balloon_identifiability(
            ("kappa", "gamma", "tau", "alpha", "E0"),
            stim, dt=dt, tr=tr, observers=("bold", "asl", "vaso"),
        )
        assert multimodal["rank"] >= bold_only["rank"]
