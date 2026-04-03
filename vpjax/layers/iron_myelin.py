"""Iron-myelin separation from multi-contrast MRI.

Separates paramagnetic (iron) and diamagnetic (myelin) contributions
to the MRI signal using combinations of R2*, QSM, and BPF (from QMT).

Key relationships:
- R2* = R2*_tissue + R2*_iron + R2*_myelin
  Both iron and myelin increase R2* (both shorten T2*)
- QSM: χ = χ_iron (+, paramagnetic) + χ_myelin (−, diamagnetic)
  QSM separates them by sign
- BPF (from QMT): direct myelin water fraction

The combination R2* + QSM separates iron from myelin:
  iron ∝ R2* + β·χ    (both positive for iron)
  myelin ∝ R2* − β·χ  (R2* positive, χ negative for myelin → both terms add)

Layer-resolved: combining with LAYNII depth maps reveals the
laminar distribution of iron and myelin independently.

References
----------
Stüber C et al. (2014) NeuroImage 93:95-106
    "Myelin and iron concentration in the human brain"
Langkammer C et al. (2012) NeuroImage 62:1593-1599
Schweser F et al. (2011) NeuroImage 54:2789-2807
Stueber C et al. (2014) NeuroImage 93:95-106
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class IronMyelinParams(eqx.Module):
    """Parameters for iron-myelin decomposition.

    Attributes
    ----------
    beta_r2star_iron   : R2* relaxivity of iron (s⁻¹ per mg/g)
    beta_r2star_myelin : R2* relaxivity of myelin (s⁻¹ per fraction)
    beta_chi_iron      : susceptibility per iron concentration (ppm per mg/g)
    beta_chi_myelin    : susceptibility per myelin fraction (ppm per fraction)
    R2star_tissue      : baseline R2* without iron/myelin (s⁻¹)
    chi_tissue         : baseline susceptibility without iron/myelin (ppm)
    """
    beta_r2star_iron: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(7.0)
    )
    beta_r2star_myelin: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(15.0)
    )
    beta_chi_iron: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.045)
    )
    beta_chi_myelin: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(-0.030)
    )
    R2star_tissue: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(12.0)
    )
    chi_tissue: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(-0.01)
    )


def decompose_r2star_qsm(
    r2star: Float[Array, "..."],
    chi: Float[Array, "..."],
    params: IronMyelinParams | None = None,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Separate iron and myelin contributions using R2* and QSM.

    Solves the 2×2 linear system:
        R2* - R2*_tissue = β_R2*_iron · [iron] + β_R2*_myelin · [myelin]
        χ - χ_tissue     = β_χ_iron · [iron]   + β_χ_myelin · [myelin]

    Parameters
    ----------
    r2star : R2* map (s⁻¹)
    chi : susceptibility map (ppm)
    params : IronMyelinParams

    Returns
    -------
    iron : iron concentration proxy (mg/g or a.u.)
    myelin : myelin fraction proxy (fraction or a.u.)
    """
    if params is None:
        params = IronMyelinParams()

    # Residuals after removing baseline
    dr2 = r2star - params.R2star_tissue
    dchi = chi - params.chi_tissue

    # Solve: [β_r2_iron  β_r2_myelin] [iron  ]   [dr2 ]
    #        [β_chi_iron β_chi_myelin] [myelin] = [dchi]
    det = (
        params.beta_r2star_iron * params.beta_chi_myelin
        - params.beta_r2star_myelin * params.beta_chi_iron
    )
    det_safe = jnp.where(jnp.abs(det) > 1e-10, det, 1e-10)

    iron = (params.beta_chi_myelin * dr2 - params.beta_r2star_myelin * dchi) / det_safe
    myelin = (params.beta_r2star_iron * dchi - params.beta_chi_iron * dr2) / det_safe

    # Clamp to non-negative (physical constraint)
    iron = jnp.clip(iron, 0.0, None)
    myelin = jnp.clip(myelin, 0.0, None)

    return iron, myelin


def decompose_with_bpf(
    r2star: Float[Array, "..."],
    chi: Float[Array, "..."],
    bpf: Float[Array, "..."],
    params: IronMyelinParams | None = None,
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Iron-myelin separation using R2*, QSM, and BPF (overconstrained).

    BPF (bound pool fraction from QMT) provides a direct myelin
    measurement, allowing a more robust decomposition.

    Strategy: use BPF as the myelin estimate, then extract iron from
    R2* or QSM residual.

    Parameters
    ----------
    r2star : R2* map (s⁻¹)
    chi : susceptibility map (ppm)
    bpf : bound pool fraction from quantitative magnetization transfer

    Returns
    -------
    iron : iron concentration proxy
    myelin_qsm : myelin from R2*/QSM decomposition
    myelin_bpf : myelin from BPF (direct measurement, used as reference)
    """
    if params is None:
        params = IronMyelinParams()

    # Method 1: direct from R2*/QSM
    iron_qsm, myelin_qsm = decompose_r2star_qsm(r2star, chi, params)

    # Method 2: use BPF as myelin, extract iron from R2*
    dr2 = r2star - params.R2star_tissue
    iron_from_r2 = (dr2 - params.beta_r2star_myelin * bpf) / jnp.where(
        jnp.abs(params.beta_r2star_iron) > 1e-10,
        params.beta_r2star_iron,
        1e-10,
    )
    iron_from_r2 = jnp.clip(iron_from_r2, 0.0, None)

    # Weighted average of iron estimates (BPF-constrained is more reliable)
    iron = 0.3 * iron_qsm + 0.7 * iron_from_r2

    return iron, myelin_qsm, bpf


def r2star_from_iron_myelin(
    iron: Float[Array, "..."],
    myelin: Float[Array, "..."],
    params: IronMyelinParams | None = None,
) -> Float[Array, "..."]:
    """Predict R2* from iron and myelin concentrations (forward model).

    R2* = R2*_tissue + β_R2*_iron · [iron] + β_R2*_myelin · [myelin]

    Parameters
    ----------
    iron : iron concentration
    myelin : myelin fraction
    params : IronMyelinParams

    Returns
    -------
    Predicted R2* (s⁻¹)
    """
    if params is None:
        params = IronMyelinParams()

    return (
        params.R2star_tissue
        + params.beta_r2star_iron * iron
        + params.beta_r2star_myelin * myelin
    )


def chi_from_iron_myelin(
    iron: Float[Array, "..."],
    myelin: Float[Array, "..."],
    params: IronMyelinParams | None = None,
) -> Float[Array, "..."]:
    """Predict susceptibility χ from iron and myelin (forward model).

    χ = χ_tissue + β_χ_iron · [iron] + β_χ_myelin · [myelin]

    Parameters
    ----------
    iron : iron concentration
    myelin : myelin fraction
    params : IronMyelinParams

    Returns
    -------
    Predicted susceptibility (ppm)
    """
    if params is None:
        params = IronMyelinParams()

    return (
        params.chi_tissue
        + params.beta_chi_iron * iron
        + params.beta_chi_myelin * myelin
    )
