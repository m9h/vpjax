"""Cardiac pulsatility: arterial pressure waves → BOLD/ASL confound.

The heartbeat produces ~1 Hz pressure waves that cause pulsatile
changes in cerebral blood volume and flow.  These appear as a
physiological confound in BOLD and ASL fMRI, especially at short
TRs where cardiac aliasing occurs.

Models:
    BP waveform → vessel compliance → ΔCBV → ΔBOLD (via BOLD signal model)
    BP waveform → velocity change → ΔCBF → ΔASL

References
----------
Dagli MS et al. (1999) MRM 41:296-295
    "Localization of cardiac-induced signal change in fMRI"
Glover GH et al. (2000) MRM 44:162-167
    "Image-based method for retrospective correction of physiological
    motion effects in fMRI: RETROICOR"
Birn RM et al. (2006) NeuroImage 31:1536-1548
Chang C, Glover GH (2009) NeuroImage 47:1448-1459
    "Effects of model-based physiological noise correction on default
    mode network anti-correlations and correlations"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class PulsatilityParams(eqx.Module):
    """Parameters for cardiac pulsatility model.

    Attributes
    ----------
    hr_bpm       : heart rate (beats per minute)
    pulse_pressure : systolic - diastolic pressure (mmHg)
    compliance   : vascular compliance (fractional CBV change per mmHg)
    V0           : resting venous blood volume fraction
    bold_sensitivity : BOLD signal change per unit CBV change
    """
    hr_bpm: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(70.0))
    pulse_pressure: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(40.0))
    compliance: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.02))
    V0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.04))
    bold_sensitivity: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.5))


def _cardiac_phase(
    t: Float[Array, "..."],
    params: PulsatilityParams | None = None,
) -> Float[Array, "..."]:
    """Convert time to cardiac phase.

    phase = 2π × (t × hr/60) mod 2π

    Parameters
    ----------
    t : time in seconds
    params : PulsatilityParams

    Returns
    -------
    Cardiac phase (radians, 0-2π)
    """
    if params is None:
        params = PulsatilityParams()

    freq = params.hr_bpm / 60.0  # Hz
    phase = 2.0 * jnp.pi * freq * t
    return phase % (2.0 * jnp.pi)


def cbv_pulsation(
    cardiac_phase: Float[Array, "..."],
    params: PulsatilityParams | None = None,
) -> Float[Array, "..."]:
    """Compute pulsatile CBV change from cardiac phase.

    ΔCBV/CBV₀ = compliance × ΔP(phase)

    The pressure waveform is modeled as a truncated Fourier series
    (first two harmonics) for differentiability.

    Parameters
    ----------
    cardiac_phase : radians (0 = R-peak)
    params : PulsatilityParams

    Returns
    -------
    CBV/CBV₀ (1.0 at mean, oscillates with cardiac cycle)
    """
    if params is None:
        params = PulsatilityParams()

    # Pressure fluctuation: fundamental + 2nd harmonic
    # Peaked near systole (phase ~ 0.3π)
    dp = (
        jnp.cos(cardiac_phase - 0.3 * jnp.pi)
        + 0.3 * jnp.cos(2 * cardiac_phase - 0.6 * jnp.pi)
    )

    # Scale by pulse pressure and compliance
    delta_cbv = params.compliance * params.pulse_pressure * dp

    return 1.0 + delta_cbv


def bold_cardiac_confound(
    t: Float[Array, "T"],
    params: PulsatilityParams | None = None,
) -> Float[Array, "T"]:
    """Generate BOLD signal confound from cardiac pulsatility.

    Models the RETROICOR-like cardiac confound:
        ΔBOLD ∝ V₀ × sensitivity × ΔCBV(t)

    Parameters
    ----------
    t : time points (s)
    params : PulsatilityParams

    Returns
    -------
    BOLD confound signal (fractional, zero-mean)
    """
    if params is None:
        params = PulsatilityParams()

    phase = _cardiac_phase(t, params)
    cbv = cbv_pulsation(phase, params)

    # BOLD confound from CBV pulsation
    confound = params.V0 * params.bold_sensitivity * (cbv - 1.0)

    # Remove mean (confound should be zero-mean)
    return confound - jnp.mean(confound)


def asl_cardiac_confound(
    t: Float[Array, "T"],
    params: PulsatilityParams | None = None,
) -> Float[Array, "T"]:
    """Generate ASL signal confound from cardiac pulsatility.

    ASL is affected by velocity changes during the cardiac cycle
    (arrival time modulation).  The confound is approximately the
    derivative of the pressure waveform (velocity ∝ dP/dt).

    Parameters
    ----------
    t : time points (s)
    params : PulsatilityParams

    Returns
    -------
    ASL confound signal (fractional, zero-mean)
    """
    if params is None:
        params = PulsatilityParams()

    phase = _cardiac_phase(t, params)

    # Velocity confound ∝ dP/dt (derivative of pressure waveform)
    dp_dt = (
        -jnp.sin(phase - 0.3 * jnp.pi)
        - 0.6 * jnp.sin(2 * phase - 0.6 * jnp.pi)
    )

    confound = params.compliance * params.pulse_pressure * 0.1 * dp_dt
    return confound - jnp.mean(confound)
