"""Oxygen extraction fraction (OEF) dynamics.

Models the relationship between CBF, CMRO₂, and OEF. During neural
activation, CBF increases more than CMRO₂ (flow-metabolism coupling
ratio n ≈ 2-3), causing OEF to decrease — the basis of the BOLD effect.

Key relationship:
    OEF = CMRO₂ / (CBF × CaO₂)

Or in fractional terms:
    OEF/OEF₀ = (CMRO₂/CMRO₂₀) / (CBF/CBF₀)

The Buxton extraction curve models the nonlinear relationship between
flow and extraction at the capillary level.

References
----------
Buxton RB et al. (1998) MRM 40:163-174
Germuska M, Bulte DP (2014) NeuroImage 102:789-798
Blockley NP et al. (2015) NeuroImage 112:225-234
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class OEFParams(eqx.Module):
    """Parameters for OEF computation.

    Attributes
    ----------
    E0      : resting oxygen extraction fraction (~0.30-0.40)
    n_ratio : flow-metabolism coupling ratio. Typical ~2-3, meaning
              CBF increases 2-3× more than CMRO₂ during activation.
    """
    E0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.34))
    n_ratio: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(2.5))


def compute_oef(
    cbf_ratio: Float[Array, "..."],
    cmro2_ratio: Float[Array, "..."],
    E0: Float[Array, "..."] | None = None,
) -> Float[Array, "..."]:
    """Compute OEF given fractional changes in CBF and CMRO₂.

    OEF = E0 × (CMRO₂/CMRO₂₀) / (CBF/CBF₀)

    Parameters
    ----------
    cbf_ratio : CBF/CBF₀ (1.0 at baseline)
    cmro2_ratio : CMRO₂/CMRO₂₀ (1.0 at baseline)
    E0 : resting OEF (default 0.34)

    Returns
    -------
    OEF (oxygen extraction fraction), clipped to [0, 1].
    """
    if E0 is None:
        E0 = jnp.array(0.34)

    cbf_safe = jnp.where(cbf_ratio > 1e-6, cbf_ratio, 1e-6)
    oef = E0 * cmro2_ratio / cbf_safe
    return jnp.clip(oef, 0.0, 1.0)


def extraction_fraction(
    cbf_ratio: Float[Array, "..."],
    E0: Float[Array, "..."] | None = None,
) -> Float[Array, "..."]:
    """Buxton oxygen extraction-flow curve.

    E(f) = 1 - (1 - E0)^(1/f)

    This models oxygen extraction at the capillary level as a function
    of blood flow: faster flow → less time for O₂ exchange → lower
    extraction per unit blood.

    Parameters
    ----------
    cbf_ratio : CBF/CBF₀ (1.0 at baseline)
    E0 : resting extraction fraction

    Returns
    -------
    E(f) : extraction fraction at the given flow.
    """
    if E0 is None:
        E0 = jnp.array(0.34)

    cbf_safe = jnp.where(cbf_ratio > 1e-6, cbf_ratio, 1e-6)
    return 1.0 - jnp.power(1.0 - E0, 1.0 / cbf_safe)


def oef_from_coupled_ratio(
    cbf_ratio: Float[Array, "..."],
    params: OEFParams | None = None,
) -> Float[Array, "..."]:
    """Compute OEF assuming a fixed flow-metabolism coupling ratio.

    If CBF increases by factor f, CMRO₂ increases by f^(1/n):
        CMRO₂/CMRO₂₀ = (CBF/CBF₀)^(1/n)

    Then: OEF = E0 × (CBF/CBF₀)^(1/n - 1)

    Parameters
    ----------
    cbf_ratio : CBF/CBF₀
    params : OEFParams (provides E0 and n_ratio)

    Returns
    -------
    OEF at the given flow level.
    """
    if params is None:
        params = OEFParams()

    cmro2_ratio = jnp.power(cbf_ratio, 1.0 / params.n_ratio)
    return compute_oef(cbf_ratio, cmro2_ratio, params.E0)
