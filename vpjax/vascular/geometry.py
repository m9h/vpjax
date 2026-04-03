"""Vascular tree geometry.

Models cerebral vascular morphometry for estimating blood volume
fractions, transit times, and oxygen transport along the capillary bed.
These parameters vary across brain regions and cortical layers.

References
----------
Duvernoy HM et al. (1981) Brain Research Bulletin 7:519-579
    "Cortical blood vessels of the human brain"
Cassot F et al. (2006) Microcirculation 13:1-18
    "A novel three-dimensional computer-assisted method for a
    quantitative study of microvascular networks"
Lorthois S, Bhogalia N (2010) NeuroImage 53:1355-1361
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class VesselSegment(eqx.Module):
    """Properties of a single vessel segment.

    Attributes
    ----------
    radius     : vessel radius (µm)
    length     : vessel length (µm)
    wall_thickness : vessel wall thickness (µm)
    vessel_type : 0=artery/arteriole, 1=capillary, 2=venule/vein
    """
    radius: Float[Array, "..."]
    length: Float[Array, "..."]
    wall_thickness: Float[Array, "..."]
    vessel_type: int = eqx.field(static=True, default=1)


class VascularParams(eqx.Module):
    """Regional vascular morphometry parameters.

    Default values represent average cortical gray matter.

    Attributes
    ----------
    density_art : arteriolar density (vessels/mm³)
    density_cap : capillary density (vessels/mm³)
    density_ven : venular density (vessels/mm³)
    radius_art  : mean arteriolar radius (µm)
    radius_cap  : mean capillary radius (µm)
    radius_ven  : mean venular radius (µm)
    length_art  : mean arteriolar segment length (µm)
    length_cap  : mean capillary segment length (µm)
    length_ven  : mean venular segment length (µm)
    """
    density_art: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(400.0)
    )
    density_cap: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(3000.0)
    )
    density_ven: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(200.0)
    )
    radius_art: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(15.0)
    )
    radius_cap: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(3.0)
    )
    radius_ven: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(25.0)
    )
    length_art: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(200.0)
    )
    length_cap: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(100.0)
    )
    length_ven: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(250.0)
    )


def blood_volume_fraction(
    density: Float[Array, "..."],
    radius: Float[Array, "..."],
    length: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Estimate blood volume fraction from vascular morphometry.

    BVF = π × r² × L × density / V_tissue

    where V_tissue = 1 mm³ = 10⁹ µm³.

    Parameters
    ----------
    density : vessel density (vessels/mm³)
    radius : vessel radius (µm)
    length : vessel segment length (µm)

    Returns
    -------
    Blood volume fraction (dimensionless, typically 0.01-0.05)
    """
    # Volume per vessel = π r² L (in µm³)
    vol_per_vessel = jnp.pi * radius ** 2 * length

    # Tissue volume = 1 mm³ = 10⁹ µm³
    tissue_vol = 1e9

    return density * vol_per_vessel / tissue_vol


def total_cbv(
    params: VascularParams | None = None,
) -> Float[Array, "..."]:
    """Compute total CBV from all vessel compartments.

    Parameters
    ----------
    params : VascularParams

    Returns
    -------
    Total blood volume fraction
    """
    if params is None:
        params = VascularParams()

    bvf_art = blood_volume_fraction(params.density_art, params.radius_art, params.length_art)
    bvf_cap = blood_volume_fraction(params.density_cap, params.radius_cap, params.length_cap)
    bvf_ven = blood_volume_fraction(params.density_ven, params.radius_ven, params.length_ven)

    return bvf_art + bvf_cap + bvf_ven


def mean_transit_time(
    length: Float[Array, "..."],
    velocity: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Compute transit time through a vessel segment.

    MTT = L / v

    Parameters
    ----------
    length : vessel length (µm)
    velocity : blood velocity (µm/s)

    Returns
    -------
    Transit time (s)
    """
    v_safe = jnp.where(velocity > 1e-6, velocity, 1e-6)
    return length / v_safe


def deoxygenation_along_capillary(
    length: Float[Array, "..."],
    oef: Float[Array, "..."],
    n_segments: int = 20,
) -> Float[Array, "... N"]:
    """Model oxygen extraction along capillary length.

    Oxygen saturation decreases exponentially along the capillary:
        SvO₂(x) = SaO₂ × exp(-k × x/L)

    where k is determined by the total OEF:
        OEF = 1 - exp(-k)  →  k = -ln(1 - OEF)

    Parameters
    ----------
    length : capillary length (µm), shape (...)
    oef : total oxygen extraction fraction across the capillary
    n_segments : number of segments along the capillary

    Returns
    -------
    OEF profile along the capillary, shape (..., n_segments).
    Each value is the cumulative extraction up to that point.
    """
    # Extraction rate constant
    oef_clipped = jnp.clip(oef, 1e-6, 1.0 - 1e-6)
    k = -jnp.log(1.0 - oef_clipped)

    # Positions along the capillary (fractional, 0 to 1)
    x = jnp.linspace(0.0, 1.0, n_segments)

    # Cumulative extraction at each position
    # E(x) = 1 - exp(-k * x)
    cumulative_oef = 1.0 - jnp.exp(-k[..., None] * x)

    return cumulative_oef
