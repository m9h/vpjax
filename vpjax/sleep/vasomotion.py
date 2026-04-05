"""Norepinephrine-mediated slow vasomotion during NREM sleep.

During NREM sleep, norepinephrine (NE) exhibits stereotypic ~0.02 Hz
oscillations that drive coordinated blood volume changes.  These slow
vasomotor waves are the primary driver of glymphatic clearance.

The causal chain:
    NE oscillation (~0.02 Hz)
    → arteriolar smooth muscle relaxation/contraction
    → CBV oscillation
    → CSF displacement (glymphatic pump)

This is distinct from cardiac pulsatility (~1 Hz) and respiratory
oscillations (~0.25 Hz).

References
----------
Fultz NE et al. (2019) Science 366:628-631
    "Coupled electrophysiological, hemodynamic, and CSF oscillations
    in human sleep"
Hauglund NL et al. (2025) Cell 188:1-17
    "Norepinephrine-mediated slow vasomotion drives glymphatic
    clearance during sleep"
Mitra A et al. (2015) PNAS 112:E2235-E2244
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class VasomotionParams(eqx.Module):
    """Parameters for NE-driven slow vasomotion.

    Attributes
    ----------
    ne_frequency   : NE oscillation frequency (Hz), ~0.02 in NREM
    ne_amplitude   : NE oscillation amplitude (a.u.)
    cbv_amplitude  : fractional CBV change amplitude (e.g., 0.03 = 3%)
    cbv_delay      : delay from NE to CBV response (s)
    V0             : resting venous blood volume fraction
    bold_sensitivity : BOLD signal per unit CBV change
    """
    ne_frequency: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.02)
    )
    ne_amplitude: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(1.0)
    )
    cbv_amplitude: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.03)
    )
    cbv_delay: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(2.0)
    )
    V0: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.04)
    )
    bold_sensitivity: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.5)
    )


def norepinephrine_oscillation(
    t: Float[Array, "T"],
    params: VasomotionParams | None = None,
) -> Float[Array, "T"]:
    """Generate NE oscillation time course during NREM sleep.

    NE(t) = amplitude × sin(2π × freq × t)

    Parameters
    ----------
    t : time points (s)
    params : VasomotionParams

    Returns
    -------
    NE level (a.u., zero-mean oscillation)
    """
    if params is None:
        params = VasomotionParams()

    return params.ne_amplitude * jnp.sin(2.0 * jnp.pi * params.ne_frequency * t)


def cbv_vasomotion(
    t: Float[Array, "T"],
    params: VasomotionParams | None = None,
) -> Float[Array, "T"]:
    """Compute CBV oscillation from NE-driven vasomotion.

    CBV(t) = 1 + amplitude × sin(2π × freq × (t - delay))

    NE increase → arteriolar constriction → CBV decrease
    (inverted relationship: NE up → CBV down)

    Parameters
    ----------
    t : time points (s)
    params : VasomotionParams

    Returns
    -------
    CBV/CBV₀ (oscillates around 1.0)
    """
    if params is None:
        params = VasomotionParams()

    # NE-to-CBV: inverted (NE constricts), delayed
    phase = 2.0 * jnp.pi * params.ne_frequency * (t - params.cbv_delay)
    return 1.0 - params.cbv_amplitude * jnp.sin(phase)


def bold_vasomotion(
    t: Float[Array, "T"],
    params: VasomotionParams | None = None,
) -> Float[Array, "T"]:
    """Generate BOLD signal fluctuation from slow vasomotion.

    ΔBOLD ∝ V₀ × sensitivity × (CBV - 1)

    This produces the ~0.02 Hz BOLD oscillation observed in NREM.

    Parameters
    ----------
    t : time points (s)
    params : VasomotionParams

    Returns
    -------
    BOLD fractional signal change (zero-mean)
    """
    if params is None:
        params = VasomotionParams()

    cbv = cbv_vasomotion(t, params)
    bold = params.V0 * params.bold_sensitivity * (cbv - 1.0)
    return bold - jnp.mean(bold)
