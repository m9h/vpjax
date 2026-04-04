"""Tests for cardiac package: heart-brain coupling models.

Models the bidirectional cardiac-brain interaction:
  Brain → Heart: frontal-vagal pathway (TMS → HR deceleration)
  Heart → Brain: baroreceptor-mediated cortical inhibition
  Heart → BOLD:  cardiac pulsatility confound
"""

import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Vagal pathway: TMS → heart rate coupling (NCG-TMS model)
# ---------------------------------------------------------------------------

class TestVagalPathway:
    """Frontal-vagal model: DLPFC stimulation → vagal tone → HR change."""

    def test_vagal_params_defaults(self):
        from vpjax.cardiac.vagal import VagalParams
        p = VagalParams()
        assert float(p.tau_vagal) > 0
        assert float(p.gain) > 0

    def test_no_stimulus_no_hr_change(self):
        """Zero neural input → no heart rate change."""
        from vpjax.cardiac.vagal import vagal_hr_response
        hr_change = vagal_hr_response(jnp.array(0.0))
        assert jnp.allclose(hr_change, 0.0, atol=1e-6)

    def test_stimulus_decelerates_hr(self):
        """DLPFC stimulation should produce HR deceleration (negative change)."""
        from vpjax.cardiac.vagal import vagal_hr_response
        hr_change = vagal_hr_response(jnp.array(1.0))
        assert float(hr_change) < 0.0

    def test_vagal_ode_steady_state(self):
        """Vagal ODE at rest should have zero derivative."""
        from vpjax.cardiac.vagal import VagalState, VagalODE, VagalParams
        model = VagalODE(params=VagalParams())
        y0 = VagalState.steady_state()
        dy = model(jnp.array(0.0), y0, jnp.array(0.0))
        assert jnp.allclose(dy.vagal_tone, 0.0, atol=1e-6)
        assert jnp.allclose(dy.hr_deviation, 0.0, atol=1e-6)

    def test_tms_entrainment(self):
        """Pulsed TMS should produce oscillatory HR at the train frequency."""
        from vpjax.cardiac.vagal import VagalState, VagalODE, VagalParams
        import jax.numpy as jnp

        model = VagalODE(params=VagalParams())
        y = VagalState.steady_state()
        dt = 0.01

        # iTBS-like: 2s on, 8s off → 0.1 Hz
        hr_trace = []
        for i in range(3000):  # 30s
            t = i * dt
            # 2s on, 8s off cycle
            stim = jnp.where((t % 10.0) < 2.0, 1.0, 0.0)
            dy = model(jnp.array(t), y, stim)
            y = jax.tree.map(lambda yi, dyi: yi + dt * dyi, y, dy)
            hr_trace.append(float(y.hr_deviation))

        hr = jnp.array(hr_trace)
        # HR should oscillate (not be flat)
        assert float(jnp.std(hr[1000:])) > 0.01
        # HR should have negative deviations (deceleration)
        assert float(jnp.min(hr)) < 0.0

    def test_rr_interval_from_hr(self):
        """Should convert HR deviation to RR interval change."""
        from vpjax.cardiac.vagal import hr_to_rr_interval
        # Baseline HR ~70 bpm → RR ~857ms
        # HR deceleration → longer RR interval
        rr_base = hr_to_rr_interval(jnp.array(0.0))
        rr_decel = hr_to_rr_interval(jnp.array(-5.0))  # 5 bpm slower
        assert float(rr_decel) > float(rr_base)

    def test_differentiable(self):
        from vpjax.cardiac.vagal import vagal_hr_response, VagalParams
        g = jax.grad(lambda gain: vagal_hr_response(
            jnp.array(1.0), VagalParams(gain=gain)
        ))(jnp.array(0.5))
        assert jnp.isfinite(g)
        assert float(g) != 0.0


# ---------------------------------------------------------------------------
# Baroreceptor: cardiac phase → cortical excitability
# ---------------------------------------------------------------------------

class TestBaroreceptor:
    """Baroreceptor model: cardiac systole/diastole modulates brain excitability."""

    def test_baroreceptor_params(self):
        from vpjax.cardiac.baroreceptor import BaroreceptorParams
        p = BaroreceptorParams()
        assert float(p.inhibition_strength) > 0

    def test_systole_inhibits(self):
        """Cortical excitability should be lower during peak systole."""
        from vpjax.cardiac.baroreceptor import cortical_excitability
        # Peak systole ~0.3π (150ms after R-peak), diastole ~π
        exc_systole = cortical_excitability(jnp.array(0.3 * jnp.pi))
        exc_diastole = cortical_excitability(jnp.array(jnp.pi))
        assert float(exc_systole) < float(exc_diastole)

    def test_excitability_periodic(self):
        """Excitability should be periodic with the cardiac cycle."""
        from vpjax.cardiac.baroreceptor import cortical_excitability
        exc_0 = cortical_excitability(jnp.array(0.0))
        exc_2pi = cortical_excitability(jnp.array(2 * jnp.pi))
        assert jnp.allclose(exc_0, exc_2pi, atol=1e-4)

    def test_excitability_modulates_bold(self):
        """Baroreceptor should modulate the effective neural drive for BOLD."""
        from vpjax.cardiac.baroreceptor import modulate_neural_drive
        drive = jnp.array(1.0)  # constant neural input
        # Peak systole (~0.3π) should reduce effective drive vs diastole (π)
        mod_systole = modulate_neural_drive(drive, cardiac_phase=jnp.array(0.3 * jnp.pi))
        mod_diastole = modulate_neural_drive(drive, cardiac_phase=jnp.array(jnp.pi))
        assert float(mod_systole) < float(mod_diastole)

    def test_blood_pressure_waveform(self):
        """Should generate a realistic arterial BP waveform from cardiac phase."""
        from vpjax.cardiac.baroreceptor import arterial_pressure
        phases = jnp.linspace(0, 2 * jnp.pi, 100)
        bp = jax.vmap(arterial_pressure)(phases)
        # Systolic peak should be higher than diastolic trough
        assert float(jnp.max(bp)) > float(jnp.min(bp))
        # Should be in physiological range (60-140 mmHg)
        assert float(jnp.min(bp)) > 40
        assert float(jnp.max(bp)) < 180

    def test_differentiable(self):
        from vpjax.cardiac.baroreceptor import cortical_excitability
        g = jax.grad(cortical_excitability)(jnp.array(0.5))
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# Cardiac pulsatility: arterial pressure waves → BOLD/ASL
# ---------------------------------------------------------------------------

class TestPulsatility:
    """Cardiac pulsatility model: BP waves → vessel volume → BOLD confound."""

    def test_pulsatility_params(self):
        from vpjax.cardiac.pulsatility import PulsatilityParams
        p = PulsatilityParams()
        assert float(p.hr_bpm) > 0
        assert float(p.pulse_pressure) > 0

    def test_cbv_pulsation(self):
        """CBV should oscillate with the cardiac cycle."""
        from vpjax.cardiac.pulsatility import cbv_pulsation
        phases = jnp.linspace(0, 2 * jnp.pi, 100)
        cbv = jax.vmap(cbv_pulsation)(phases)
        # Should oscillate around baseline
        assert float(jnp.max(cbv)) > 1.0
        assert float(jnp.min(cbv)) < 1.0

    def test_bold_pulsation(self):
        """Cardiac pulsation should produce ~1Hz BOLD fluctuation."""
        from vpjax.cardiac.pulsatility import bold_cardiac_confound
        # Simulate 10s at 1ms resolution
        t = jnp.linspace(0, 10, 10000)
        bold_confound = bold_cardiac_confound(t)
        # Should have nonzero variance
        assert float(jnp.std(bold_confound)) > 0.0001
        # Should be zero-mean (confound, not signal)
        assert abs(float(jnp.mean(bold_confound))) < 0.01

    def test_asl_pulsation(self):
        """ASL signal should also be affected by cardiac pulsatility."""
        from vpjax.cardiac.pulsatility import asl_cardiac_confound
        t = jnp.linspace(0, 10, 10000)
        asl_confound = asl_cardiac_confound(t)
        assert float(jnp.std(asl_confound)) > 0.0001

    def test_pulsatility_scales_with_compliance(self):
        """Higher vascular compliance → larger pulsatile volume change."""
        from vpjax.cardiac.pulsatility import PulsatilityParams, cbv_pulsation
        low_comp = PulsatilityParams(compliance=jnp.array(0.01))
        high_comp = PulsatilityParams(compliance=jnp.array(0.05))
        amp_low = float(cbv_pulsation(jnp.array(0.0), low_comp) - 1.0)
        amp_high = float(cbv_pulsation(jnp.array(0.0), high_comp) - 1.0)
        assert abs(amp_high) > abs(amp_low)

    def test_differentiable(self):
        from vpjax.cardiac.pulsatility import bold_cardiac_confound, PulsatilityParams
        g = jax.grad(lambda hr: bold_cardiac_confound(
            jnp.array([1.0]), PulsatilityParams(hr_bpm=hr)
        )[0])(jnp.array(70.0))
        assert jnp.isfinite(g)
