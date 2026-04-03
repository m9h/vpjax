"""TRUST MRI: T2 Relaxation Under Spin Tagging.

TRUST measures venous blood T2 in the sagittal sinus, which is then
converted to venous oxygen saturation (SvO₂) via a calibration curve.
Combined with arterial saturation (SaO₂ ≈ 0.98), this gives global OEF:

    OEF = (SaO₂ - SvO₂) / SaO₂

References
----------
Lu H, Ge Y (2008) MRM 60:357-363
    "Quantitative evaluation of oxygenation in venous vessels using
    T2-Relaxation-Under-Spin-Tagging MRI"
Lu H et al. (2012) MRM 67:42-49
    "Calibration and validation of TRUST MRI for the estimation of
    cerebral blood oxygenation"
Bush A et al. (2018) MRM 80:2065-2075
    "Calibration of T2 to oxygen saturation"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class TRUSTParams(eqx.Module):
    """Parameters for TRUST T2-to-SvO₂ calibration.

    The T2-Y relationship is modeled as (Luz-Meiboom):
        1/T2 = 1/T2_0 + K₁·(1-Y)·Hct + K₂·((1-Y)·Hct)²

    where Y is blood oxygen saturation and Hct is hematocrit.
    Calibration constants are field-strength dependent.

    Attributes
    ----------
    K1    : linear calibration constant (s⁻¹)
    K2    : quadratic calibration constant (s⁻¹)
    T2_0  : T2 of fully oxygenated blood (s)
    Hct   : hematocrit fraction
    SaO2  : arterial oxygen saturation
    tau_cpmg : CPMG inter-echo spacing (s), affects calibration
    """
    K1: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(6.56))
    K2: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(188.2))
    T2_0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.220))
    Hct: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.42))
    SaO2: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.98))
    tau_cpmg: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.010))


def t2_to_svo2(
    t2: Float[Array, "..."],
    params: TRUSTParams | None = None,
) -> Float[Array, "..."]:
    """Convert venous blood T2 to oxygen saturation SvO₂.

    Inverts the Luz-Meiboom calibration:
        R2 = R2_0 + K₁·(1-Y)·Hct + K₂·((1-Y)·Hct)²

    Solving the quadratic for (1-Y)·Hct, then extracting Y.

    Parameters
    ----------
    t2 : venous blood T2 (s)
    params : TRUSTParams (default: 3T calibration)

    Returns
    -------
    SvO₂ : venous oxygen saturation (0-1)
    """
    if params is None:
        params = TRUSTParams()

    r2 = 1.0 / t2
    r2_0 = 1.0 / params.T2_0

    # Quadratic in x = (1-Y)*Hct:  K2*x² + K1*x + (R2_0 - R2) = 0
    # Solving: x = (-K1 + sqrt(K1² - 4*K2*(R2_0 - R2))) / (2*K2)
    a = params.K2
    b = params.K1
    c = r2_0 - r2

    discriminant = b ** 2 - 4.0 * a * c
    discriminant = jnp.clip(discriminant, 0.0, None)

    x = (-b + jnp.sqrt(discriminant)) / (2.0 * a)
    x = jnp.clip(x, 0.0, None)

    # x = (1 - Y) * Hct → Y = 1 - x/Hct
    hct_safe = jnp.where(params.Hct > 1e-6, params.Hct, 1e-6)
    svo2 = 1.0 - x / hct_safe
    return jnp.clip(svo2, 0.0, 1.0)


def svo2_to_t2(
    svo2: Float[Array, "..."],
    params: TRUSTParams | None = None,
) -> Float[Array, "..."]:
    """Convert oxygen saturation SvO₂ to blood T2.

    Forward model: R2 = R2_0 + K₁·(1-Y)·Hct + K₂·((1-Y)·Hct)²

    Parameters
    ----------
    svo2 : venous oxygen saturation (0-1)
    params : TRUSTParams

    Returns
    -------
    T2 in seconds
    """
    if params is None:
        params = TRUSTParams()

    x = (1.0 - svo2) * params.Hct
    r2 = 1.0 / params.T2_0 + params.K1 * x + params.K2 * x ** 2
    return 1.0 / r2


def trust_oef(
    t2: Float[Array, "..."],
    params: TRUSTParams | None = None,
) -> Float[Array, "..."]:
    """Compute OEF from TRUST-measured venous T2.

    OEF = (SaO₂ - SvO₂) / SaO₂

    Parameters
    ----------
    t2 : venous blood T2 from TRUST (s)
    params : TRUSTParams

    Returns
    -------
    Global OEF
    """
    if params is None:
        params = TRUSTParams()

    svo2 = t2_to_svo2(t2, params)
    oef = (params.SaO2 - svo2) / params.SaO2
    return jnp.clip(oef, 0.0, 1.0)


def trust_global_cmro2(
    t2: Float[Array, "..."],
    cbf_global: Float[Array, "..."],
    cao2: Float[Array, "..."] | None = None,
    params: TRUSTParams | None = None,
) -> Float[Array, "..."]:
    """Compute global CMRO₂ from TRUST + global CBF.

    CMRO₂ = CBF × OEF × CaO₂

    This is "Level 1" in the CMRO₂ hierarchy (global OEF from TRUST).

    Parameters
    ----------
    t2 : venous blood T2 from TRUST (s)
    cbf_global : global CBF (mL/100g/min)
    cao2 : arterial O₂ content (µmol/mL), default ~8.3
    params : TRUSTParams

    Returns
    -------
    CMRO₂ in µmol/100g/min
    """
    if params is None:
        params = TRUSTParams()
    if cao2 is None:
        cao2 = jnp.array(8.3)

    oef = trust_oef(t2, params)
    return cbf_global * oef * cao2
