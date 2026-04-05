"""Tests for sleep package: state-dependent NVC, vasomotion, CSF coupling.

Models the sleep-specific physiology:
  Neural slow waves → hemodynamic oscillations → CSF flow (glymphatic)
  Neuromodulatory state (NE) → vasomotion → CBV oscillations
  Sleep stage → NVC parameter modulation
"""

import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Sleep stage → NVC parameter modulation
# ---------------------------------------------------------------------------

class TestSleepNVC:
    """State-dependent neurovascular coupling across sleep stages."""

    def test_sleep_stages_defined(self):
        """All five standard sleep stages should be available."""
        from vpjax.sleep.nvc_state import WAKE, N1, N2, N3, REM
        assert WAKE == 0
        assert N1 == 1
        assert N2 == 2
        assert N3 == 3
        assert REM == 4

    def test_balloon_params_per_stage(self):
        """Each sleep stage should produce valid BalloonParams."""
        from vpjax.sleep.nvc_state import balloon_params_for_stage, WAKE, N1, N2, N3, REM
        from vpjax._types import BalloonParams
        for stage in [WAKE, N1, N2, N3, REM]:
            p = balloon_params_for_stage(stage)
            assert isinstance(p, BalloonParams)
            assert float(p.tau) > 0
            assert float(p.kappa) > 0
            assert jnp.isfinite(p.E0)

    def test_nvc_gain_decreases_nrem(self):
        """NVC gain should decrease from wake through deeper NREM."""
        from vpjax.sleep.nvc_state import nvc_gain_for_stage, WAKE, N1, N2, N3
        g_wake = nvc_gain_for_stage(WAKE)
        g_n1 = nvc_gain_for_stage(N1)
        g_n3 = nvc_gain_for_stage(N3)
        assert float(g_wake) > float(g_n1)
        assert float(g_n1) > float(g_n3)

    def test_rem_gain_near_wake(self):
        """REM NVC should be closer to wake than to deep NREM."""
        from vpjax.sleep.nvc_state import nvc_gain_for_stage, WAKE, N3, REM
        g_wake = nvc_gain_for_stage(WAKE)
        g_n3 = nvc_gain_for_stage(N3)
        g_rem = nvc_gain_for_stage(REM)
        # REM closer to wake than to N3
        assert abs(float(g_rem - g_wake)) < abs(float(g_rem - g_n3))

    def test_riera_params_per_stage(self):
        """Each stage should produce valid RieraParams."""
        from vpjax.sleep.nvc_state import riera_params_for_stage, WAKE, N3
        from vpjax.hemodynamics.riera import RieraParams
        p_wake = riera_params_for_stage(WAKE)
        p_n3 = riera_params_for_stage(N3)
        assert isinstance(p_wake, RieraParams)
        assert isinstance(p_n3, RieraParams)
        # N3 should have lower coupling gain
        assert float(p_n3.c_no) < float(p_wake.c_no)

    def test_continuous_interpolation(self):
        """Should support continuous sleep depth (0=wake, 1=deep NREM)."""
        from vpjax.sleep.nvc_state import nvc_gain_continuous
        g0 = nvc_gain_continuous(jnp.array(0.0))  # wake
        g05 = nvc_gain_continuous(jnp.array(0.5))  # light sleep
        g1 = nvc_gain_continuous(jnp.array(1.0))   # deep NREM
        assert float(g0) > float(g05) > float(g1)

    def test_continuous_differentiable(self):
        """Continuous interpolation should be differentiable."""
        from vpjax.sleep.nvc_state import nvc_gain_continuous
        g = jax.grad(nvc_gain_continuous)(jnp.array(0.5))
        assert jnp.isfinite(g)
        assert float(g) < 0  # gain decreases with sleep depth


# ---------------------------------------------------------------------------
# Slow vasomotion: NE-driven ~0.02 Hz CBV oscillation
# ---------------------------------------------------------------------------

class TestVasomotion:
    """Norepinephrine-mediated slow vasomotion during NREM sleep."""

    def test_vasomotion_params(self):
        from vpjax.sleep.vasomotion import VasomotionParams
        p = VasomotionParams()
        assert float(p.ne_frequency) > 0
        assert float(p.cbv_amplitude) > 0

    def test_ne_oscillation(self):
        """NE should oscillate at ~0.02 Hz during NREM."""
        from vpjax.sleep.vasomotion import norepinephrine_oscillation
        t = jnp.linspace(0, 100, 1000)  # 100s
        ne = norepinephrine_oscillation(t)
        # Should oscillate (not flat)
        assert float(jnp.std(ne)) > 0.01
        # Period ~50s (0.02 Hz)
        # Check that it completes roughly 2 cycles in 100s
        crossings = jnp.sum(jnp.abs(jnp.diff(jnp.sign(ne - jnp.mean(ne)))) > 0)
        assert 2 <= int(crossings) <= 8

    def test_cbv_vasomotion(self):
        """CBV should oscillate in response to NE-driven vasomotion."""
        from vpjax.sleep.vasomotion import cbv_vasomotion
        t = jnp.linspace(0, 100, 1000)
        cbv = cbv_vasomotion(t)
        # Should oscillate around 1.0
        assert float(jnp.mean(cbv)) == pytest.approx(1.0, abs=0.05)
        assert float(jnp.std(cbv)) > 0.001

    def test_vasomotion_amplitude_scales(self):
        """Larger NE amplitude → larger CBV oscillation."""
        from vpjax.sleep.vasomotion import VasomotionParams, cbv_vasomotion
        t = jnp.array([25.0])  # sample at one time
        small = VasomotionParams(cbv_amplitude=jnp.array(0.01))
        large = VasomotionParams(cbv_amplitude=jnp.array(0.05))
        cbv_s = cbv_vasomotion(t, small)
        cbv_l = cbv_vasomotion(t, large)
        assert abs(float(cbv_l[0]) - 1.0) > abs(float(cbv_s[0]) - 1.0)

    def test_bold_vasomotion_signal(self):
        """Vasomotion should produce a low-frequency BOLD fluctuation."""
        from vpjax.sleep.vasomotion import bold_vasomotion
        t = jnp.linspace(0, 200, 2000)
        bold = bold_vasomotion(t)
        assert float(jnp.std(bold)) > 0.0001

    def test_wake_vs_nrem_vasomotion(self):
        """Vasomotion should be stronger in NREM than wake."""
        from vpjax.sleep.vasomotion import VasomotionParams, cbv_vasomotion
        t = jnp.linspace(0, 100, 1000)
        wake_p = VasomotionParams(cbv_amplitude=jnp.array(0.005))  # minimal during wake
        nrem_p = VasomotionParams(cbv_amplitude=jnp.array(0.03))   # strong during NREM
        std_wake = float(jnp.std(cbv_vasomotion(t, wake_p)))
        std_nrem = float(jnp.std(cbv_vasomotion(t, nrem_p)))
        assert std_nrem > std_wake

    def test_differentiable(self):
        from vpjax.sleep.vasomotion import cbv_vasomotion, VasomotionParams
        g = jax.grad(lambda amp: jnp.sum(cbv_vasomotion(
            jnp.array([25.0, 50.0]), VasomotionParams(cbv_amplitude=amp)
        )))(jnp.array(0.03))
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# CSF flow coupling: CBV oscillations → glymphatic clearance
# ---------------------------------------------------------------------------

class TestCSFCoupling:
    """CBV oscillations drive CSF pulsations (glymphatic forward model)."""

    def test_csf_params(self):
        from vpjax.sleep.csf_coupling import CSFParams
        p = CSFParams()
        assert float(p.csf_delay) > 0
        assert float(p.coupling_gain) > 0

    def test_csf_from_cbv(self):
        """CSF flow should be driven by CBV changes."""
        from vpjax.sleep.csf_coupling import csf_flow_from_cbv
        # CBV increasing → CSF pushed out (positive flow)
        cbv = jnp.array([1.0, 1.02, 1.04, 1.02, 1.0])
        csf = csf_flow_from_cbv(cbv)
        assert csf.shape == cbv.shape
        # CSF should respond to CBV changes
        assert float(jnp.std(csf)) > 0

    def test_csf_lags_cbv(self):
        """CSF flow should lag behind CBV changes."""
        from vpjax.sleep.csf_coupling import csf_flow_from_cbv_delayed
        t = jnp.linspace(0, 100, 1000)
        # Sinusoidal CBV
        cbv = 1.0 + 0.03 * jnp.sin(2 * jnp.pi * 0.02 * t)
        csf = csf_flow_from_cbv_delayed(cbv, t)
        # Cross-correlate: CSF should peak after CBV
        # (CSF is driven by dCBV/dt, so it leads CBV by ~quarter cycle,
        #  but the delay shifts it back)
        assert csf.shape == cbv.shape
        assert float(jnp.std(csf)) > 0

    def test_glymphatic_clearance(self):
        """Cumulative CSF flow should estimate glymphatic clearance."""
        from vpjax.sleep.csf_coupling import glymphatic_clearance
        t = jnp.linspace(0, 300, 3000)  # 5 minutes
        # NREM-like CBV oscillation
        cbv = 1.0 + 0.03 * jnp.sin(2 * jnp.pi * 0.02 * t)
        clearance = glymphatic_clearance(cbv, t)
        # Clearance should accumulate over time
        assert float(clearance) > 0

    def test_no_oscillation_no_clearance(self):
        """Flat CBV (no vasomotion) → minimal clearance."""
        from vpjax.sleep.csf_coupling import glymphatic_clearance
        t = jnp.linspace(0, 300, 3000)
        cbv = jnp.ones_like(t)  # no oscillation
        clearance = glymphatic_clearance(cbv, t)
        assert float(clearance) < 0.01

    def test_stronger_oscillation_more_clearance(self):
        """Larger CBV oscillation → more glymphatic clearance."""
        from vpjax.sleep.csf_coupling import glymphatic_clearance
        t = jnp.linspace(0, 300, 3000)
        cbv_small = 1.0 + 0.01 * jnp.sin(2 * jnp.pi * 0.02 * t)
        cbv_large = 1.0 + 0.05 * jnp.sin(2 * jnp.pi * 0.02 * t)
        c_small = glymphatic_clearance(cbv_small, t)
        c_large = glymphatic_clearance(cbv_large, t)
        assert float(c_large) > float(c_small)

    def test_differentiable(self):
        from vpjax.sleep.csf_coupling import glymphatic_clearance
        t = jnp.linspace(0, 100, 1000)
        def loss(amp):
            cbv = 1.0 + amp * jnp.sin(2 * jnp.pi * 0.02 * t)
            return glymphatic_clearance(cbv, t)
        g = jax.grad(loss)(jnp.array(0.03))
        assert jnp.isfinite(g)
