"""CMRO₂ (cerebral metabolic rate of oxygen) from neural activity.

Models the relationship between neural activity and oxygen consumption.
CMRO₂ increases during neural activation, but typically less than CBF
(the flow-metabolism coupling ratio n ≈ 2-3).

References
----------
Hoge RD et al. (1999) MRM 42:849-863
    "Linear coupling between cerebral blood flow and oxygen consumption"
Davis TL et al. (1998) PNAS 95:1834-1839
    "Calibrated functional MRI"
Buxton RB (2010) Reports on Progress in Physics 73:1-29
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class CMRO2Params(eqx.Module):
    """Parameters for CMRO₂ estimation.

    Attributes
    ----------
    baseline : baseline CMRO₂ (µmol O₂/100g/min), default ~150
    coupling_gain : fractional CMRO₂ increase per unit neural activity.
        Default 0.5 (i.e., 50% CMRO₂ increase for unit stimulus).
        This is typically less than the CBF gain (~1.0), reflecting
        the flow-metabolism coupling ratio n ≈ 2.
    """
    baseline: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(150.0)
    )
    coupling_gain: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.5)
    )


def compute_cmro2(
    activity: Float[Array, "..."],
    params: CMRO2Params | None = None,
) -> Float[Array, "..."]:
    """Compute fractional CMRO₂ change from neural activity.

    CMRO₂/CMRO₂₀ = 1 + gain × activity

    Parameters
    ----------
    activity : neural activity (0 at baseline, arbitrary units)
    params : CMRO2Params

    Returns
    -------
    CMRO₂ ratio (CMRO₂/CMRO₂₀), 1.0 at baseline.
    """
    if params is None:
        params = CMRO2Params()

    return 1.0 + params.coupling_gain * activity


def compute_cmro2_absolute(
    activity: Float[Array, "..."],
    params: CMRO2Params | None = None,
) -> Float[Array, "..."]:
    """Compute absolute CMRO₂ from neural activity.

    CMRO₂ = baseline × (1 + gain × activity)

    Parameters
    ----------
    activity : neural activity
    params : CMRO2Params

    Returns
    -------
    CMRO₂ in µmol O₂/100g/min
    """
    if params is None:
        params = CMRO2Params()

    ratio = compute_cmro2(activity, params)
    return params.baseline * ratio


def compute_cmro2_from_cbf_oef(
    cbf: Float[Array, "..."],
    oef: Float[Array, "..."],
    cao2: Float[Array, "..."] | None = None,
) -> Float[Array, "..."]:
    """Compute CMRO₂ from CBF, OEF, and CaO₂ via Fick's principle.

    CMRO₂ = CBF × OEF × CaO₂

    Parameters
    ----------
    cbf : cerebral blood flow (mL/100g/min)
    oef : oxygen extraction fraction
    cao2 : arterial O₂ content (µmol O₂/mL), default ~8.3

    Returns
    -------
    CMRO₂ in µmol O₂/100g/min
    """
    if cao2 is None:
        cao2 = jnp.array(8.3)

    return cbf * oef * cao2
