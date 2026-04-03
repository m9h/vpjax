"""Vessel compliance: pressure-volume relationships.

Models the relationship between transmural pressure and blood volume
for cerebral vessels.  Compliance varies by vessel type (arterioles
are most compliant, capillaries least).

The Mandeville (1999) model uses a power-law:
    CBV/CBV₀ = (CBF/CBF₀)^α  (Grubb relationship)

The underlying mechanism:
    v = (P/P₀)^(1/α_iv)  (pressure-volume compliance)
    R = R₀ × v^(-2/α)    (Poiseuille resistance scaling)

References
----------
Grubb RL et al. (1974) Stroke 5:630-639
    "The effects of changes in PaCO₂ on CBV, CBF, and vascular
    mean transit time"
Mandeville JB et al. (1999) JCBFM 19:679-689
    "Evidence of a cerebrovascular postarteriole Windkessel"
Buxton RB et al. (1998) MRM 40:163-174
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class ComplianceParams(eqx.Module):
    """Parameters for vessel compliance model.

    Attributes
    ----------
    alpha   : Grubb exponent (CBF-CBV power law), default 0.38
              Classic Grubb: 0.38. DCM: 0.32.
    P0      : reference transmural pressure (mmHg)
    alpha_iv : intracranial volume-pressure elastance
    """
    alpha: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.38))
    P0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(25.0))
    alpha_iv: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.11))


def grubb_cbv(
    cbf_ratio: Float[Array, "..."],
    params: ComplianceParams | None = None,
) -> Float[Array, "..."]:
    """Compute CBV/CBV₀ from CBF/CBF₀ using Grubb's power law.

    CBV/CBV₀ = (CBF/CBF₀)^α

    Parameters
    ----------
    cbf_ratio : CBF/CBF₀ (1.0 at baseline)
    params : ComplianceParams

    Returns
    -------
    CBV/CBV₀ (volume ratio)
    """
    if params is None:
        params = ComplianceParams()

    return jnp.power(jnp.clip(cbf_ratio, 1e-6, None), params.alpha)


def pressure_to_volume(
    pressure: Float[Array, "..."],
    params: ComplianceParams | None = None,
) -> Float[Array, "..."]:
    """Compute volume ratio from transmural pressure.

    v/v₀ = (P/P₀)^(1/α_iv)

    Parameters
    ----------
    pressure : transmural pressure (mmHg)
    params : ComplianceParams

    Returns
    -------
    Volume ratio CBV/CBV₀
    """
    if params is None:
        params = ComplianceParams()

    p_ratio = pressure / params.P0
    p_ratio = jnp.clip(p_ratio, 1e-6, None)
    return jnp.power(p_ratio, 1.0 / params.alpha_iv)


def volume_to_pressure(
    volume_ratio: Float[Array, "..."],
    params: ComplianceParams | None = None,
) -> Float[Array, "..."]:
    """Compute transmural pressure from volume ratio.

    P = P₀ × (v/v₀)^α_iv

    Parameters
    ----------
    volume_ratio : CBV/CBV₀
    params : ComplianceParams

    Returns
    -------
    Transmural pressure (mmHg)
    """
    if params is None:
        params = ComplianceParams()

    v = jnp.clip(volume_ratio, 1e-6, None)
    return params.P0 * jnp.power(v, params.alpha_iv)


def vessel_resistance(
    volume_ratio: Float[Array, "..."],
    params: ComplianceParams | None = None,
) -> Float[Array, "..."]:
    """Compute resistance ratio from volume change (Poiseuille scaling).

    For a cylindrical vessel, R ∝ 1/r⁴. If volume (πr²L) changes
    while length is constant, r ∝ √v, so R ∝ v^(-2).

    In the Grubb framework: R/R₀ = (v/v₀)^(-2/α)

    Parameters
    ----------
    volume_ratio : CBV/CBV₀
    params : ComplianceParams (α is the Grubb exponent)

    Returns
    -------
    Resistance ratio R/R₀
    """
    if params is None:
        params = ComplianceParams()

    v = jnp.clip(volume_ratio, 1e-6, None)
    return jnp.power(v, -2.0 / params.alpha)


def transit_time(
    volume_ratio: Float[Array, "..."],
    cbf_ratio: Float[Array, "..."],
    tau0: Float[Array, "..."] | None = None,
) -> Float[Array, "..."]:
    """Compute mean transit time from volume and flow.

    MTT = CBV / CBF  (central volume principle)

    In fractional terms: MTT/MTT₀ = (CBV/CBV₀) / (CBF/CBF₀)

    Parameters
    ----------
    volume_ratio : CBV/CBV₀
    cbf_ratio : CBF/CBF₀
    tau0 : baseline transit time (s), default 2.0

    Returns
    -------
    Transit time (s)
    """
    if tau0 is None:
        tau0 = jnp.array(2.0)

    cbf_safe = jnp.where(cbf_ratio > 1e-6, cbf_ratio, 1e-6)
    return tau0 * volume_ratio / cbf_safe
