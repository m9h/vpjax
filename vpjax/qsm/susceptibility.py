"""Susceptibility models for QSM.

Relates tissue susceptibility (χ) to its microstructural sources
(iron, myelin, blood oxygenation) and provides forward/inverse
models for susceptibility-weighted imaging.

QSM reconstructs χ from MRI phase data (external tools like QSMxT).
vpjax provides the biophysical interpretation layer: mapping χ to
iron, myelin, and oxygenation.

Key relationships:
    χ_tissue = χ_iron + χ_myelin + χ_blood(OEF)
    χ_iron > 0 (paramagnetic)
    χ_myelin < 0 (diamagnetic)
    χ_blood ∝ Hct × (1-Y) × Δχ_oxy  (depends on oxygenation)

References
----------
Schweser F et al. (2011) NeuroImage 54:2789-2807
Langkammer C et al. (2012) NeuroImage 62:1593-1599
Stueber C et al. (2014) NeuroImage 93:95-106
Liu C et al. (2015) MRM 74:1125-1135
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


# Susceptibility constants (SI, ppm)
DELTA_CHI_OXY = 0.264  # ppm, susceptibility difference between oxy- and deoxy-Hb (SI)
CHI_WATER = -9.05      # ppm, diamagnetic susceptibility of water


class SusceptibilityParams(eqx.Module):
    """Parameters for tissue susceptibility model.

    Attributes
    ----------
    chi_iron_per_mg : susceptibility per mg/g iron (ppm/(mg/g))
    chi_myelin_per_frac : susceptibility per myelin volume fraction (ppm)
    Hct : hematocrit
    chi_ref : reference susceptibility (CSF/water, ppm)
    """
    chi_iron_per_mg: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.045)
    )
    chi_myelin_per_frac: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(-0.030)
    )
    Hct: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.40))
    chi_ref: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.0))


def susceptibility_from_sources(
    iron: Float[Array, "..."],
    myelin: Float[Array, "..."],
    oef: Float[Array, "..."] | None = None,
    dbv: Float[Array, "..."] | None = None,
    params: SusceptibilityParams | None = None,
) -> Float[Array, "..."]:
    """Compute tissue susceptibility from microstructural sources.

    χ = χ_ref + χ_iron·[iron] + χ_myelin·[myelin] + χ_blood(OEF, DBV)

    Parameters
    ----------
    iron : iron concentration (mg/g or a.u.)
    myelin : myelin volume fraction
    oef : oxygen extraction fraction (for blood contribution). Optional.
    dbv : deoxygenated blood volume fraction. Optional.
    params : SusceptibilityParams

    Returns
    -------
    Tissue susceptibility (ppm, relative to reference)
    """
    if params is None:
        params = SusceptibilityParams()

    chi = (
        params.chi_ref
        + params.chi_iron_per_mg * iron
        + params.chi_myelin_per_frac * myelin
    )

    # Blood contribution (if OEF and DBV provided)
    if oef is not None and dbv is not None:
        chi_blood = DELTA_CHI_OXY * params.Hct * oef * dbv
        chi = chi + chi_blood

    return chi


def oef_from_susceptibility(
    chi: Float[Array, "..."],
    iron: Float[Array, "..."],
    myelin: Float[Array, "..."],
    dbv: Float[Array, "..."],
    params: SusceptibilityParams | None = None,
) -> Float[Array, "..."]:
    """Estimate OEF from susceptibility given iron, myelin, and DBV.

    Rearranging the susceptibility model:
        OEF = (χ - χ_ref - χ_iron·[iron] - χ_myelin·[myelin]) / (Δχ_oxy·Hct·DBV)

    Parameters
    ----------
    chi : measured susceptibility (ppm)
    iron : iron concentration
    myelin : myelin fraction
    dbv : deoxygenated blood volume fraction
    params : SusceptibilityParams

    Returns
    -------
    OEF estimate, clipped to [0, 1]
    """
    if params is None:
        params = SusceptibilityParams()

    chi_tissue = (
        params.chi_ref
        + params.chi_iron_per_mg * iron
        + params.chi_myelin_per_frac * myelin
    )

    chi_blood = chi - chi_tissue

    denom = DELTA_CHI_OXY * params.Hct * dbv
    denom_safe = jnp.where(jnp.abs(denom) > 1e-10, denom, 1e-10)

    oef = chi_blood / denom_safe
    return jnp.clip(oef, 0.0, 1.0)
