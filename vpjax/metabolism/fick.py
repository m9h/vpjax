"""Fick's principle for cerebral oxygen metabolism.

Fick's principle relates oxygen consumption to blood flow and oxygen
content difference across the vascular bed:

    CMRO₂ = CBF × (CaO₂ - CvO₂) = CBF × OEF × CaO₂

where:
- CaO₂ = arterial O₂ content = 1.34 × [Hb] × SaO₂ (bound) + 0.003 × PaO₂ (dissolved)
- CvO₂ = venous O₂ content (lower due to tissue extraction)
- OEF = (CaO₂ - CvO₂) / CaO₂

References
----------
Fick A (1870) Über die Messung des Blutquantums in den Herzventrikeln
Kety SS, Schmidt CF (1948) J Clin Invest 27:476-483
    "The nitrous oxide method for the quantitative determination of
    cerebral blood flow in man"
Mintun MA et al. (1984) JCBFM 4:163-172
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


# Hüfner's constant: mL O₂ per gram Hb at full saturation
HUFNER_CONSTANT = 1.34  # mL O₂/g Hb

# Conversion: 1 mL O₂ (STPD) = 44.66 µmol O₂
ML_O2_TO_UMOL = 44.66


class FickParams(eqx.Module):
    """Parameters for Fick's principle calculations.

    Attributes
    ----------
    Hb   : hemoglobin concentration (g/dL)
    SaO2 : arterial oxygen saturation (fraction, 0-1)
    PaO2 : arterial partial pressure of O₂ (mmHg), for dissolved O₂
    """
    Hb: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(15.0))
    SaO2: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.98))
    PaO2: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(100.0))


def compute_cao2(
    params: FickParams | None = None,
) -> Float[Array, "..."]:
    """Compute arterial oxygen content CaO₂.

    CaO₂ = 1.34 × [Hb] × SaO₂ + 0.003 × PaO₂  (mL O₂/dL blood)

    Converted to µmol O₂/mL for compatibility with CBF in mL/100g/min.

    Parameters
    ----------
    params : FickParams

    Returns
    -------
    CaO₂ in µmol O₂/mL blood
    """
    if params is None:
        params = FickParams()

    # Bound O₂ (mL O₂/dL blood)
    bound = HUFNER_CONSTANT * params.Hb * params.SaO2

    # Dissolved O₂ (mL O₂/dL blood) — Henry's law
    dissolved = 0.003 * params.PaO2

    # Total in mL O₂/dL → convert to µmol O₂/mL
    # mL O₂/dL × (1 dL / 100 mL) × 44.66 µmol/mL O₂
    cao2_ml_per_dl = bound + dissolved
    cao2_umol_per_ml = cao2_ml_per_dl * ML_O2_TO_UMOL / 100.0

    return cao2_umol_per_ml


def fick_cmro2(
    cbf: Float[Array, "..."],
    oef: Float[Array, "..."],
    cao2: Float[Array, "..."] | None = None,
) -> Float[Array, "..."]:
    """Compute CMRO₂ via Fick's principle.

    CMRO₂ = CBF × OEF × CaO₂

    Parameters
    ----------
    cbf : cerebral blood flow (mL/100g/min)
    oef : oxygen extraction fraction (0-1)
    cao2 : arterial O₂ content (µmol/mL). If None, computed from defaults.

    Returns
    -------
    CMRO₂ in µmol O₂/100g/min
    """
    if cao2 is None:
        cao2 = compute_cao2()

    return cbf * oef * cao2


def fick_oef(
    cmro2: Float[Array, "..."],
    cbf: Float[Array, "..."],
    cao2: Float[Array, "..."] | None = None,
) -> Float[Array, "..."]:
    """Compute OEF from CMRO₂ and CBF via Fick's principle.

    OEF = CMRO₂ / (CBF × CaO₂)

    Parameters
    ----------
    cmro2 : cerebral metabolic rate of oxygen (µmol/100g/min)
    cbf : cerebral blood flow (mL/100g/min)
    cao2 : arterial O₂ content (µmol/mL)

    Returns
    -------
    OEF (oxygen extraction fraction), clipped to [0, 1].
    """
    if cao2 is None:
        cao2 = compute_cao2()

    denom = cbf * cao2
    denom_safe = jnp.where(denom > 1e-10, denom, 1e-10)
    return jnp.clip(cmro2 / denom_safe, 0.0, 1.0)


def fick_cbf(
    cmro2: Float[Array, "..."],
    oef: Float[Array, "..."],
    cao2: Float[Array, "..."] | None = None,
) -> Float[Array, "..."]:
    """Compute CBF from CMRO₂ and OEF via Fick's principle.

    CBF = CMRO₂ / (OEF × CaO₂)

    Parameters
    ----------
    cmro2 : cerebral metabolic rate of oxygen (µmol/100g/min)
    oef : oxygen extraction fraction
    cao2 : arterial O₂ content (µmol/mL)

    Returns
    -------
    CBF in mL/100g/min
    """
    if cao2 is None:
        cao2 = compute_cao2()

    denom = oef * cao2
    denom_safe = jnp.where(denom > 1e-10, denom, 1e-10)
    return cmro2 / denom_safe
