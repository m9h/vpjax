"""Global BOLD waves during NREM sleep.

During N3, massive spatially correlated BOLD fluctuations occur that
are not driven by local neurovascular coupling but by volume-level
vasomotion and autonomic state changes. These are the dominant signal
in deep sleep fMRI.

The global wave combines:
- Slow vasomotion (~0.02 Hz, NE-driven)
- Infraslow fluctuations (~0.01-0.05 Hz)
- Reduced local NVC (gain ↓)

References
----------
Fukunaga M et al. (2006) MRM 56:1479-1484
Mitra A et al. (2015) PNAS 112:E2235-E2244
Fultz NE et al. (2019) Science 366:628-631
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax.sleep.nvc_state import WAKE, N1, N2, N3, REM

# Global BOLD wave amplitude per stage (fractional signal change)
_GLOBAL_AMPLITUDE = {
    WAKE: 0.002,
    N1: 0.005,
    N2: 0.010,
    N3: 0.025,
    REM: 0.003,
}

# Dominant frequency per stage (Hz)
_GLOBAL_FREQ = {
    WAKE: 0.04,
    N1: 0.03,
    N2: 0.025,
    N3: 0.02,
    REM: 0.035,
}


def global_bold_amplitude(stage: int) -> float:
    """Get global BOLD wave amplitude for a sleep stage.

    Parameters
    ----------
    stage : WAKE=0, N1=1, N2=2, N3=3, REM=4

    Returns
    -------
    Amplitude of global BOLD fluctuation (fractional)
    """
    return _GLOBAL_AMPLITUDE.get(stage, 0.002)


def global_bold_wave(
    t: Float[Array, "T"],
    stage: int = N3,
) -> Float[Array, "T"]:
    """Generate global BOLD wave time course for a sleep stage.

    Combines the dominant infraslow oscillation with harmonics
    to produce a realistic global BOLD fluctuation.

    Parameters
    ----------
    t : time points (s)
    stage : sleep stage

    Returns
    -------
    Global BOLD signal (fractional change, zero-mean)
    """
    amp = _GLOBAL_AMPLITUDE.get(stage, 0.002)
    freq = _GLOBAL_FREQ.get(stage, 0.02)

    # Multi-component infraslow oscillation
    signal = (
        amp * jnp.sin(2 * jnp.pi * freq * t)
        + amp * 0.5 * jnp.sin(2 * jnp.pi * freq * 0.5 * t + 0.7)
        + amp * 0.3 * jnp.sin(2 * jnp.pi * freq * 1.5 * t + 1.3)
    )

    return signal - jnp.mean(signal)
