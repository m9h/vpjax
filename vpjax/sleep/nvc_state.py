"""Sleep-stage-dependent neurovascular coupling parameters.

NVC properties change across sleep stages:
- Wake: standard NVC, strong coupling
- N1/N2: reduced NVC gain, longer transit times
- N3 (SWS): substantially reduced NVC, increased vascular compliance
- REM: near-wake NVC with different neuromodulatory balance

These changes affect how neural activity maps to BOLD/ASL signals
and must be accounted for in sleep fMRI forward models.

References
----------
Horovitz SG et al. (2008) HBM 29:671-682
    "Decoupling of brain hemodynamics and neural activity during sleep"
Fukunaga M et al. (2006) MRM 56:1479-1484
    "Large-amplitude, spatially correlated fluctuations in BOLD fMRI"
Fultz NE et al. (2019) Science 366:628-631
    "Coupled oscillations in human sleep"
Mitra A et al. (2015) PNAS 112:E2235-E2244
    "Lag structure in resting-state fMRI"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax._types import BalloonParams
from vpjax.hemodynamics.riera import RieraParams


# Sleep stage constants
WAKE = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4

# NVC gain per stage (relative to wake = 1.0)
# Based on Horovitz 2008, Fukunaga 2006: NREM shows reduced NVC
_GAIN_TABLE = {
    WAKE: 1.0,
    N1: 0.75,
    N2: 0.55,
    N3: 0.35,
    REM: 0.85,
}

# Transit time scaling (N3 has increased vascular compliance → longer transit)
_TAU_SCALE = {
    WAKE: 1.0,
    N1: 1.1,
    N2: 1.2,
    N3: 1.4,
    REM: 1.05,
}

# Signal decay rate scaling (slower signal decay in deeper sleep)
_KAPPA_SCALE = {
    WAKE: 1.0,
    N1: 0.9,
    N2: 0.8,
    N3: 0.65,
    REM: 0.95,
}


def nvc_gain_for_stage(stage: int) -> float:
    """Get NVC gain for a discrete sleep stage.

    Parameters
    ----------
    stage : WAKE=0, N1=1, N2=2, N3=3, REM=4

    Returns
    -------
    Relative NVC gain (1.0 = wake)
    """
    return _GAIN_TABLE.get(stage, 1.0)


def nvc_gain_continuous(
    sleep_depth: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Continuous NVC gain as a function of sleep depth.

    Smooth interpolation from wake (0) to deep NREM (1).
    Differentiable for use in optimization.

    gain(d) = gain_wake × (1 - d) + gain_N3 × d
            = 1.0 - 0.65 × d

    Parameters
    ----------
    sleep_depth : 0.0 = wake, 1.0 = deep NREM (N3)

    Returns
    -------
    NVC gain (relative)
    """
    d = jnp.clip(sleep_depth, 0.0, 1.0)
    gain_wake = _GAIN_TABLE[WAKE]
    gain_n3 = _GAIN_TABLE[N3]
    return gain_wake + (gain_n3 - gain_wake) * d


def balloon_params_for_stage(stage: int) -> BalloonParams:
    """Get stage-specific BalloonParams.

    Modifies transit time (tau) and signal decay (kappa) based on
    sleep stage.  Other parameters use wake defaults.

    Parameters
    ----------
    stage : sleep stage (WAKE, N1, N2, N3, REM)

    Returns
    -------
    BalloonParams with stage-appropriate values
    """
    base = BalloonParams()
    tau_scale = _TAU_SCALE.get(stage, 1.0)
    kappa_scale = _KAPPA_SCALE.get(stage, 1.0)

    return BalloonParams(
        kappa=base.kappa * kappa_scale,
        gamma=base.gamma,
        tau=base.tau * tau_scale,
        alpha=base.alpha,
        E0=base.E0,
    )


def riera_params_for_stage(stage: int) -> RieraParams:
    """Get stage-specific RieraParams.

    Modifies coupling gains and transit times based on sleep stage.

    Parameters
    ----------
    stage : sleep stage

    Returns
    -------
    RieraParams with stage-appropriate values
    """
    base = RieraParams()
    gain = nvc_gain_for_stage(stage)
    tau_s = _TAU_SCALE.get(stage, 1.0)
    kappa_s = _KAPPA_SCALE.get(stage, 1.0)

    return RieraParams(
        kappa_no=base.kappa_no * kappa_s,
        kappa_ade=base.kappa_ade * kappa_s,
        gamma_no=base.gamma_no,
        gamma_ade=base.gamma_ade,
        c_no=base.c_no * gain,
        c_ade=base.c_ade * gain,
        tau_a=base.tau_a * tau_s,
        tau_c=base.tau_c * tau_s,
        tau_v=base.tau_v * tau_s,
        alpha_a=base.alpha_a,
        alpha_c=base.alpha_c,
        alpha_v=base.alpha_v,
        E0=base.E0,
        phi=base.phi * gain,
        tau_m=base.tau_m,
    )
