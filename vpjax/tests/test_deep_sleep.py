"""Tests for improved deep sleep (N3) models.

Addresses 5 gaps in the original N3 model:
1. Sawtooth NE waveform (sharp drops, slow recovery)
2. Stage-dependent NVC delay (longer in N3)
3. Global BOLD waves (spatially correlated, volume-level)
4. CSF-BOLD signal contribution
5. Locus coeruleus firing pattern
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Sawtooth NE waveform
# ---------------------------------------------------------------------------

class TestSawtoothNE:
    """NE oscillation in N3 is asymmetric: sharp drop, slow recovery."""

    def test_sawtooth_not_sinusoidal(self):
        """Sawtooth should have different rise and fall times."""
        from vpjax.sleep.vasomotion import ne_sawtooth_oscillation
        t = jnp.linspace(0, 100, 2000)
        ne = ne_sawtooth_oscillation(t)
        # Should have skewness (asymmetric)
        skew = float(jnp.mean((ne - jnp.mean(ne)) ** 3) / jnp.std(ne) ** 3)
        assert abs(skew) > 0.1  # significantly non-symmetric

    def test_sawtooth_sharp_drops(self):
        """NE should have fast negative-going transients."""
        from vpjax.sleep.vasomotion import ne_sawtooth_oscillation
        t = jnp.linspace(0, 100, 2000)
        ne = ne_sawtooth_oscillation(t)
        dne = jnp.diff(ne)
        # Max negative derivative should be larger in magnitude than max positive
        assert float(jnp.abs(jnp.min(dne))) > float(jnp.max(dne))

    def test_sawtooth_frequency(self):
        """Should still oscillate at ~0.02 Hz."""
        from vpjax.sleep.vasomotion import ne_sawtooth_oscillation
        t = jnp.linspace(0, 200, 4000)
        ne = ne_sawtooth_oscillation(t)
        # Count zero crossings
        crossings = jnp.sum(jnp.abs(jnp.diff(jnp.sign(ne - jnp.mean(ne)))) > 0)
        # ~0.02 Hz over 200s = 4 cycles = ~8 crossings
        assert 4 <= int(crossings) <= 20

    def test_differentiable(self):
        from vpjax.sleep.vasomotion import ne_sawtooth_oscillation, VasomotionParams
        g = jax.grad(lambda amp: jnp.sum(ne_sawtooth_oscillation(
            jnp.array([25.0, 50.0]), VasomotionParams(ne_amplitude=amp)
        )))(jnp.array(1.0))
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# 2. Stage-dependent NVC delay
# ---------------------------------------------------------------------------

class TestNVCDelay:
    """Hemodynamic impulse response delay should increase in deeper sleep."""

    def test_delay_increases_with_depth(self):
        """N3 impulse response should peak later than Wake."""
        from vpjax.sleep.nvc_state import hrf_peak_time, WAKE, N1, N2, N3
        t_wake = hrf_peak_time(WAKE)
        t_n1 = hrf_peak_time(N1)
        t_n3 = hrf_peak_time(N3)
        assert float(t_n3) > float(t_n1) > float(t_wake)

    def test_n3_peak_around_8_to_10s(self):
        """N3 HRF should peak around 8-10s (vs ~5-6s for wake)."""
        from vpjax.sleep.nvc_state import hrf_peak_time, N3
        t = hrf_peak_time(N3)
        assert 7.0 < float(t) < 12.0

    def test_wake_peak_around_5_to_6s(self):
        """Wake HRF should peak around 5-6s (standard)."""
        from vpjax.sleep.nvc_state import hrf_peak_time, WAKE
        t = hrf_peak_time(WAKE)
        assert 4.0 < float(t) < 7.0


# ---------------------------------------------------------------------------
# 3. Global BOLD waves
# ---------------------------------------------------------------------------

class TestGlobalBOLDWaves:
    """N3 has massive spatially correlated BOLD fluctuations."""

    def test_global_bold_amplitude_increases_n3(self):
        """Global BOLD wave amplitude should be larger in N3 than wake."""
        from vpjax.sleep.global_waves import global_bold_amplitude, WAKE, N3
        amp_wake = global_bold_amplitude(WAKE)
        amp_n3 = global_bold_amplitude(N3)
        assert float(amp_n3) > float(amp_wake)

    def test_global_bold_wave_signal(self):
        """Should generate a time course of global BOLD fluctuations."""
        from vpjax.sleep.global_waves import global_bold_wave
        t = jnp.linspace(0, 300, 3000)
        signal = global_bold_wave(t, stage=3)
        assert signal.shape == t.shape
        assert float(jnp.std(signal)) > 0.001

    def test_global_wave_dominated_by_low_freq(self):
        """Global BOLD waves should be dominated by <0.1 Hz power."""
        from vpjax.sleep.global_waves import global_bold_wave
        t = jnp.linspace(0, 300, 3000)
        signal = np.array(global_bold_wave(t, stage=3))
        # PSD
        from scipy.signal import welch
        f, psd = welch(signal, fs=10.0, nperseg=512)
        low_power = np.sum(psd[f < 0.1])
        high_power = np.sum(psd[f >= 0.1])
        assert low_power > high_power * 5  # dominated by low freq


# ---------------------------------------------------------------------------
# 4. CSF-BOLD signal contribution
# ---------------------------------------------------------------------------

class TestCSFBOLDContribution:
    """CSF displacement creates additional BOLD signal changes in N3."""

    def test_csf_bold_nonzero_n3(self):
        """CSF contribution to BOLD should be nonzero during N3."""
        from vpjax.sleep.csf_coupling import csf_bold_contribution
        t = jnp.linspace(0, 100, 1000)
        cbv = 1.0 + 0.03 * jnp.sin(2 * jnp.pi * 0.02 * t)  # N3 vasomotion
        csf_bold = csf_bold_contribution(cbv, t, bold_sensitivity=1.0)
        assert float(jnp.std(csf_bold)) > 0.00001

    def test_csf_bold_zero_flat_cbv(self):
        """No CBV oscillation → no CSF-BOLD contribution."""
        from vpjax.sleep.csf_coupling import csf_bold_contribution
        t = jnp.linspace(0, 100, 1000)
        cbv = jnp.ones_like(t)
        csf_bold = csf_bold_contribution(cbv, t)
        assert float(jnp.std(csf_bold)) < 0.001


# ---------------------------------------------------------------------------
# 5. Locus coeruleus firing pattern
# ---------------------------------------------------------------------------

class TestLocusCoeruleus:
    """LC firing model drives NE oscillation during NREM."""

    def test_lc_params(self):
        from vpjax.sleep.locus_coeruleus import LCParams
        p = LCParams()
        assert float(p.firing_rate_wake) > 0
        assert float(p.firing_rate_n3) >= 0

    def test_lc_firing_decreases_nrem(self):
        """LC firing rate should decrease from wake through N3."""
        from vpjax.sleep.locus_coeruleus import lc_firing_rate, WAKE, N1, N2, N3
        fr_wake = lc_firing_rate(WAKE)
        fr_n1 = lc_firing_rate(N1)
        fr_n3 = lc_firing_rate(N3)
        assert float(fr_wake) > float(fr_n1) > float(fr_n3)

    def test_lc_to_ne(self):
        """LC firing should produce NE concentration time course."""
        from vpjax.sleep.locus_coeruleus import lc_to_norepinephrine
        t = jnp.linspace(0, 100, 1000)
        ne = lc_to_norepinephrine(t, stage=3)
        assert ne.shape == t.shape
        # NE should oscillate during N3
        assert float(jnp.std(ne)) > 0.01

    def test_lc_ne_drives_vasomotion(self):
        """LC→NE should connect to vasomotion model."""
        from vpjax.sleep.locus_coeruleus import lc_to_norepinephrine
        from vpjax.sleep.vasomotion import cbv_from_ne
        t = jnp.linspace(0, 100, 1000)
        ne = lc_to_norepinephrine(t, stage=3)
        cbv = cbv_from_ne(ne)
        # CBV should oscillate in anti-phase with NE
        assert float(jnp.std(cbv)) > 0.001
        # Correlation should be negative (NE up → vasoconstriction → CBV down)
        r = float(jnp.corrcoef(ne, cbv)[0, 1])
        assert r < -0.3

    def test_differentiable(self):
        from vpjax.sleep.locus_coeruleus import lc_to_norepinephrine, LCParams
        g = jax.grad(lambda rate: jnp.sum(lc_to_norepinephrine(
            jnp.array([25.0, 50.0]), stage=3,
            params=LCParams(firing_rate_n3=rate)
        )))(jnp.array(0.5))
        assert jnp.isfinite(g)
