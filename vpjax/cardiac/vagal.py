"""Frontal-vagal pathway: brain stimulation → heart rate coupling.

Models the top-down pathway from prefrontal cortex (DLPFC) through
the vagus nerve to the heart.  TMS at the DLPFC activates this
pathway, producing heart rate deceleration that can entrain to the
TMS pulse train frequency.

This is the physiological model underlying Neuro-Cardiac-Guided TMS
(NCG-TMS, Brainclinics): the cardiac response verifies that TMS
engaged the frontal-vagal network.

Model:
    DLPFC stimulation → vagal tone increase → HR deceleration
    d(vagal_tone)/dt = gain × stimulus - vagal_tone / τ_vagal
    d(hr_deviation)/dt = -coupling × vagal_tone - hr_deviation / τ_hr

References
----------
Iseger TA et al. (2023) Biol Psychiatry Global Open Science 3:283-292
    "TMS-induced heart-brain coupling"
Iseger TA et al. (2020) Brain Stimulation 13:1664-1672
    "High- and low-frequency NCG-TMS"
Jiao Y et al. (2024) Psychophysiology 61:e14631
    "NCG-TMS modulatory effects on heart rate"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


# Baseline heart rate
_HR_BASELINE_BPM = 70.0


class VagalParams(eqx.Module):
    """Parameters for the frontal-vagal pathway model.

    Attributes
    ----------
    tau_vagal  : vagal tone time constant (s). ~1-3s for fast vagal response.
    tau_hr     : heart rate recovery time constant (s). ~5-10s.
    gain       : coupling from neural stimulus to vagal tone (a.u.)
    coupling   : vagal tone → HR deceleration strength (bpm per unit tone)
    hr_baseline : resting heart rate (bpm)
    """
    tau_vagal: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(2.0))
    tau_hr: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(8.0))
    gain: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.5))
    coupling: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(5.0))
    hr_baseline: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(_HR_BASELINE_BPM)
    )


class VagalState(eqx.Module):
    """State of the frontal-vagal pathway.

    Attributes
    ----------
    vagal_tone   : current vagal tone (0 at rest, increases with stimulation)
    hr_deviation : heart rate deviation from baseline (bpm, negative = deceleration)
    """
    vagal_tone: Float[Array, "..."]
    hr_deviation: Float[Array, "..."]

    @staticmethod
    def steady_state(shape: tuple[int, ...] = ()) -> VagalState:
        return VagalState(
            vagal_tone=jnp.zeros(shape),
            hr_deviation=jnp.zeros(shape),
        )


class VagalODE(eqx.Module):
    """Frontal-vagal pathway ODE.

    d(vagal_tone)/dt = gain × stimulus - vagal_tone / τ_vagal
    d(hr_deviation)/dt = -coupling × vagal_tone - hr_deviation / τ_hr
    """

    params: VagalParams

    def __call__(
        self,
        t: Float[Array, ""],
        y: VagalState,
        args: Float[Array, "..."],
    ) -> VagalState:
        """Evaluate RHS.

        Parameters
        ----------
        t : time
        y : current VagalState
        args : neural stimulus (e.g., TMS intensity, 0-1)
        """
        u = args
        p = self.params

        d_vagal = p.gain * u - y.vagal_tone / p.tau_vagal
        d_hr = -p.coupling * y.vagal_tone - y.hr_deviation / p.tau_hr

        return VagalState(vagal_tone=d_vagal, hr_deviation=d_hr)


def vagal_hr_response(
    stimulus: Float[Array, "..."],
    params: VagalParams | None = None,
) -> Float[Array, "..."]:
    """Instantaneous steady-state HR response to vagal stimulation.

    At steady state: vagal_tone = gain × stimulus × τ_vagal
    HR deviation = -coupling × vagal_tone × τ_hr

    Parameters
    ----------
    stimulus : neural drive (0 = rest, 1 = full stimulation)
    params : VagalParams

    Returns
    -------
    HR deviation (bpm), negative = deceleration
    """
    if params is None:
        params = VagalParams()

    vagal_ss = params.gain * stimulus * params.tau_vagal
    hr_ss = -params.coupling * vagal_ss
    return hr_ss


def hr_to_rr_interval(
    hr_deviation: Float[Array, "..."],
    hr_baseline: float = _HR_BASELINE_BPM,
) -> Float[Array, "..."]:
    """Convert HR deviation (bpm) to RR interval (ms).

    RR = 60000 / (HR_baseline + hr_deviation)

    Parameters
    ----------
    hr_deviation : HR change from baseline (bpm)
    hr_baseline : resting HR (bpm)

    Returns
    -------
    RR interval (ms)
    """
    hr = hr_baseline + hr_deviation
    hr_safe = jnp.where(hr > 10.0, hr, 10.0)
    return 60000.0 / hr_safe
