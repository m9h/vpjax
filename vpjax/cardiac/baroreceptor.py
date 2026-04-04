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
