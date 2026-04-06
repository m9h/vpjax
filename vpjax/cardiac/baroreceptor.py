"""Baroreceptor-mediated cortical excitability modulation.

Arterial baroreceptors fire during cardiac systole (when blood
pressure peaks), sending afferent signals via the vagus/glossopharyngeal
nerves to the brainstem → thalamus → cortex.  This produces
phasic inhibition of cortical excitability tied to the cardiac cycle.

TMS studies show that motor evoked potentials (MEPs) are smaller
during systole than diastole, confirming the cardiac gating of
cortical excitability.

Model:
    cardiac_phase → arterial_pressure → baroreceptor_firing
    → cortical_inhibition → modulated neural drive

References
----------
Critchley HD, Garfinkel SN (2018) Nat Rev Neurosci 19:7-18
    "The influence of physiological signals on cognition"
Al E et al. (2024) PNAS 121:e2312685121
    "Cardiac phase modulates short-interval intracortical inhibition"
Lacey BC, Lacey JI (1978) "Two-way communication between the heart
    and the brain"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


# Physiological defaults
_SYSTOLIC_BP = 120.0   # mmHg
_DIASTOLIC_BP = 80.0   # mmHg
_MEAN_BP = 93.0        # mmHg


class BaroreceptorParams(eqx.Module):
    """Parameters for the baroreceptor-cortical inhibition model.

    Attributes
    ----------
    inhibition_strength : fractional reduction in excitability at peak systole
    systolic_bp : systolic blood pressure (mmHg)
    diastolic_bp : diastolic blood pressure (mmHg)
    systole_phase_width : width of the systolic pulse (radians)
    """
    inhibition_strength: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.15)
    )
    systolic_bp: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(_SYSTOLIC_BP)
    )
    diastolic_bp: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(_DIASTOLIC_BP)
    )
    systole_phase_width: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(1.2)
    )


def arterial_pressure(
    cardiac_phase: Float[Array, "..."],
    params: BaroreceptorParams | None = None,
) -> Float[Array, "..."]:
    """Generate arterial blood pressure waveform from cardiac phase.

    Simplified model: systolic peak near phase=0 (R-peak),
    diastolic trough near phase=π, with a dicrotic notch.

    Parameters
    ----------
    cardiac_phase : phase in radians (0 = R-peak, 2π = next R-peak)
    params : BaroreceptorParams

    Returns
    -------
    Arterial pressure (mmHg)
    """
    if params is None:
        params = BaroreceptorParams()

    pp = params.systolic_bp - params.diastolic_bp  # pulse pressure
    mean_bp = (params.systolic_bp + params.diastolic_bp) / 2.0

    # Systolic peak: sharp rise after R-peak, wider exponential decay
    # Use a skewed waveform centered near phase=0.3π (peak systole ~150ms after R)
    peak_phase = 0.3 * jnp.pi
    width = params.systole_phase_width

    # Wrapped distance from peak
    dp = cardiac_phase - peak_phase
    dp_wrapped = jnp.angle(jnp.exp(1j * dp))

    # Asymmetric pulse: fast rise, slow decay
    pulse = jnp.exp(-0.5 * (dp_wrapped / (width * 0.4)) ** 2)

    # Dicrotic notch (small secondary bump at ~π)
    notch_phase = jnp.pi
    dn = jnp.angle(jnp.exp(1j * (cardiac_phase - notch_phase)))
    notch = 0.15 * jnp.exp(-0.5 * (dn / 0.3) ** 2)

    bp = params.diastolic_bp + pp * (pulse + notch)
    return bp


def cortical_excitability(
    cardiac_phase: Float[Array, "..."],
    params: BaroreceptorParams | None = None,
) -> Float[Array, "..."]:
    """Compute cortical excitability as a function of cardiac phase.

    Excitability is reduced during systole (baroreceptor-mediated
    inhibition) and maximal during diastole.

    excitability = 1 - inhibition_strength × baroreceptor_activity(phase)

    Parameters
    ----------
    cardiac_phase : radians (0 = R-peak)
    params : BaroreceptorParams

    Returns
    -------
    Relative cortical excitability (0-1, 1 = maximal)
    """
    if params is None:
        params = BaroreceptorParams()

    bp = arterial_pressure(cardiac_phase, params)
    # Baroreceptor firing rate proportional to BP above diastolic
    pp = params.systolic_bp - params.diastolic_bp
    pp_safe = jnp.where(pp > 1.0, pp, 1.0)
    baro_activity = (bp - params.diastolic_bp) / pp_safe
    baro_activity = jnp.clip(baro_activity, 0.0, 1.0)

    excitability = 1.0 - params.inhibition_strength * baro_activity
    return excitability


def modulate_neural_drive(
    neural_drive: Float[Array, "..."],
    cardiac_phase: Float[Array, "..."],
    params: BaroreceptorParams | None = None,
) -> Float[Array, "..."]:
    """Modulate effective neural drive by cardiac phase.

    The baroreceptor-mediated inhibition gates the neural input
    to the hemodynamic model, so BOLD response amplitude depends
    on when (in the cardiac cycle) the stimulus arrives.

    Parameters
    ----------
    neural_drive : stimulus intensity (a.u.)
    cardiac_phase : radians (0 = R-peak)
    params : BaroreceptorParams

    Returns
    -------
    Effective neural drive (reduced during systole)
    """
    exc = cortical_excitability(cardiac_phase, params)
    return neural_drive * exc


# ---------------------------------------------------------------------------
# Pulse-style baroreflex ODE model (Kitware Pulse methodology)
# ---------------------------------------------------------------------------

class BaroreflexParams(eqx.Module):
    """Parameters for the Pulse-style baroreflex ODE model.

    Sigmoid sympathetic/parasympathetic fractions with 4 effector ODEs.

    Attributes
    ----------
    map_setpoint : mean arterial pressure setpoint (mmHg)
    nu           : sigmoid steepness parameter
    tau_hr       : HR adjustment time constant (s)
    tau_resistance : resistance adjustment time constant (s)
    tau_compliance : compliance adjustment time constant (s)
    tau_elastance : elastance adjustment time constant (s)
    hr_range     : [min_hr, max_hr] bpm
    hr_baseline  : resting HR (bpm)

    References
    ----------
    Kitware Pulse Physiology Engine, Nervous Methodology
    """
    map_setpoint: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(93.0))
    nu: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(4.0))
    tau_hr: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(2.0))
    tau_resistance: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(5.0))
    tau_compliance: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(10.0))
    tau_elastance: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(3.0))
    hr_baseline: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(72.0))
    hr_sympathetic_max: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(180.0))
    hr_parasympathetic_min: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(40.0))


class BaroreflexState(eqx.Module):
    """State of the baroreflex ODE.

    Attributes
    ----------
    heart_rate  : current HR (bpm)
    resistance  : systemic vascular resistance ratio (1.0 = baseline)
    compliance  : venous compliance ratio (1.0 = baseline)
    elastance   : cardiac elastance ratio (1.0 = baseline)
    """
    heart_rate: Float[Array, "..."]
    resistance: Float[Array, "..."]
    compliance: Float[Array, "..."]
    elastance: Float[Array, "..."]

    @staticmethod
    def steady_state(shape: tuple[int, ...] = ()) -> BaroreflexState:
        return BaroreflexState(
            heart_rate=jnp.full(shape, 72.0),
            resistance=jnp.ones(shape),
            compliance=jnp.ones(shape),
            elastance=jnp.ones(shape),
        )


def sympathetic_fraction(
    map_pressure: Float[Array, "..."],
    params: BaroreflexParams | None = None,
) -> Float[Array, "..."]:
    """Sympathetic response fraction (Pulse sigmoid model).

    η_s(Pa) = [1 + (Pa / Pa_setpoint)^ν]^(-1)

    High MAP → low sympathetic (vagal dominance).
    """
    if params is None:
        params = BaroreflexParams()
    ratio = map_pressure / params.map_setpoint
    return 1.0 / (1.0 + jnp.power(ratio, params.nu))


def parasympathetic_fraction(
    map_pressure: Float[Array, "..."],
    params: BaroreflexParams | None = None,
) -> Float[Array, "..."]:
    """Parasympathetic response fraction (Pulse sigmoid model).

    η_p(Pa) = [1 + (Pa / Pa_setpoint)^(-ν)]^(-1)

    High MAP → high parasympathetic.
    """
    if params is None:
        params = BaroreflexParams()
    ratio = map_pressure / params.map_setpoint
    return 1.0 / (1.0 + jnp.power(ratio, -params.nu))


class BaroreflexODE(eqx.Module):
    """Baroreflex ODE: MAP → sympathetic/parasympathetic → CV adjustment.

    Four effector ODEs driven by the sympathetic/parasympathetic balance:
    - Heart rate: ↑sympathetic → ↑HR, ↑parasympathetic → ↓HR
    - Resistance: ↑sympathetic → ↑resistance (vasoconstriction)
    - Compliance: ↑sympathetic → ↓compliance (venoconstriction)
    - Elastance: ↑sympathetic → ↑elastance (stronger contraction)
    """

    params: BaroreflexParams

    def __call__(
        self,
        t: Float[Array, ""],
        y: BaroreflexState,
        args: Float[Array, "..."],
    ) -> BaroreflexState:
        """Evaluate RHS. args = current mean arterial pressure (mmHg)."""
        map_pressure = args
        p = self.params

        eta_s = sympathetic_fraction(map_pressure, p)
        eta_p = parasympathetic_fraction(map_pressure, p)

        # Target values driven by autonomic balance
        # Net autonomic drive: positive = sympathetic dominant, negative = parasympathetic
        # At setpoint: eta_s ≈ eta_p ≈ 0.5 → net ≈ 0 → targets at baseline
        net_sympathetic = eta_s - eta_p
        hr_target = (
            p.hr_baseline
            + (p.hr_sympathetic_max - p.hr_baseline) * jnp.clip(net_sympathetic, 0.0, None)
            + (p.hr_parasympathetic_min - p.hr_baseline) * jnp.clip(-net_sympathetic, 0.0, None)
        )
        resistance_target = 1.0 + 0.5 * net_sympathetic
        compliance_target = 1.0 - 0.3 * net_sympathetic
        elastance_target = 1.0 + 0.4 * net_sympathetic

        # First-order approach to target
        d_hr = (hr_target - y.heart_rate) / p.tau_hr
        d_resistance = (resistance_target - y.resistance) / p.tau_resistance
        d_compliance = (compliance_target - y.compliance) / p.tau_compliance
        d_elastance = (elastance_target - y.elastance) / p.tau_elastance

        return BaroreflexState(
            heart_rate=d_hr,
            resistance=d_resistance,
            compliance=d_compliance,
            elastance=d_elastance,
        )
