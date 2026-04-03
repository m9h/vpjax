"""ASL calibration: M₀ and blood T₁ estimation.

Calibration is needed to convert ASL difference signals to absolute
CBF. The key quantities are:
- M₀b: equilibrium magnetization of blood
- T₁b: longitudinal relaxation of blood (depends on Hct, field)
- Labeling efficiency α

References
----------
Alsop DC et al. (2015) MRM 73:102-116
    "Recommended implementation of ASL for clinical applications"
Zhang X et al. (2013) MRM 70:1125-1136
    "In vivo blood T1 measurements at 1.5T, 3T, and 7T"
Herscovitch P, Raichle ME (1985) JCBFM 5:65-69
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class CalibrationParams(eqx.Module):
    """Parameters for ASL calibration.

    Attributes
    ----------
    Hct    : hematocrit fraction (venous blood)
    Hct_ratio : arterial/venous Hct ratio (~0.85)
    B0     : field strength (T)
    lambda_p : blood-brain partition coefficient (mL/g)
    T2star_b : blood T2* (s), for signal correction
    TE     : echo time of the ASL acquisition (s)
    """
    Hct: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.42))
    Hct_ratio: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.85))
    B0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(3.0))
    lambda_p: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.90))
    T2star_b: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.050))
    TE: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.012))


def blood_t1(
    hct: Float[Array, "..."] | None = None,
    B0: Float[Array, "..."] | None = None,
) -> Float[Array, "..."]:
    """Estimate blood T₁ from hematocrit and field strength.

    Uses the empirical model from Zhang et al. (2013) and Lu et al. (2004):

    At 3T: T1b ≈ 1 / (0.52 · Hct + 0.38)  (simplified)

    More generally (Zhang et al. 2013):
        1/T1b = 0.50 · Hct + R1_plasma(B0)

    Parameters
    ----------
    hct : hematocrit fraction (default 0.42)
    B0 : field strength in T (default 3.0)

    Returns
    -------
    T1b in seconds
    """
    if hct is None:
        hct = jnp.array(0.42)
    if B0 is None:
        B0 = jnp.array(3.0)

    # Plasma R1 depends on field (Zhang et al. 2013)
    # At 3T: ~0.38 s⁻¹, at 7T: ~0.28 s⁻¹, at 1.5T: ~0.47 s⁻¹
    r1_plasma = 0.52 - 0.02 * B0

    r1_blood = 0.50 * hct + r1_plasma
    return 1.0 / r1_blood


def m0_from_proton_density(
    m0_tissue: Float[Array, "..."],
    params: CalibrationParams | None = None,
) -> Float[Array, "..."]:
    """Estimate M₀ of blood from tissue M₀ (proton density image).

    M₀b = M₀_tissue / λ

    where λ is the blood-brain partition coefficient.

    Parameters
    ----------
    m0_tissue : proton density-weighted tissue signal
    params : CalibrationParams

    Returns
    -------
    M₀b : equilibrium magnetization of blood
    """
    if params is None:
        params = CalibrationParams()

    return m0_tissue / params.lambda_p


def m0_csf_correction(
    m0_csf: Float[Array, "..."],
    params: CalibrationParams | None = None,
) -> Float[Array, "..."]:
    """Estimate M₀ of blood from CSF M₀ (reference region method).

    CSF is used as a reference because it has known T₁, T₂*:
        M₀b = M₀_csf × (1/λ_csf) × T2*_correction

    Parameters
    ----------
    m0_csf : M₀ signal from CSF voxels
    params : CalibrationParams

    Returns
    -------
    M₀b : equilibrium magnetization of blood
    """
    if params is None:
        params = CalibrationParams()

    # CSF partition coefficient ~ 1.0 (it is water)
    # T2* correction: account for different T2* of blood vs CSF
    # CSF T2* at 3T ~ 400ms (very long), blood T2* ~ 50ms
    T2star_csf = jnp.array(0.400)

    t2star_correction = jnp.exp(-params.TE / params.T2star_b) / jnp.exp(
        -params.TE / T2star_csf
    )

    return m0_csf * t2star_correction


def labeling_efficiency(
    b1_ratio: Float[Array, "..."],
    alpha_nominal: float = 0.85,
) -> Float[Array, "..."]:
    """Estimate actual labeling efficiency from B₁ field map.

    pCASL labeling efficiency depends on the B₁ field at the
    labeling plane. Approximate correction:

        α_actual ≈ α_nominal × (1 - 0.5 × (1 - B₁_ratio)²)

    Parameters
    ----------
    b1_ratio : B₁ actual / B₁ nominal (from B₁ map), shape (...)
    alpha_nominal : nominal labeling efficiency (default 0.85)

    Returns
    -------
    Corrected labeling efficiency
    """
    correction = 1.0 - 0.5 * (1.0 - b1_ratio) ** 2
    return jnp.clip(alpha_nominal * correction, 0.0, 1.0)
