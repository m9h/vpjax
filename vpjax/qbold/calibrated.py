"""Gas-free calibrated BOLD (Bulte et al.).

Calibrated fMRI estimates the calibration parameter M, which relates
BOLD signal change to underlying physiological changes (CBF, CMRO₂).
The Davis model:

    ΔBOLD/BOLD₀ = M · (1 - (CBF/CBF₀)^(α-β) · (CMRO₂/CMRO₂₀)^β)

Traditional calibration uses gas challenges (hypercapnia/hyperoxia).
Bulte's gas-free approach estimates M from R₂' (qBOLD) instead.

References
----------
Davis TL et al. (1998) PNAS 95:1834-1839 (original calibrated fMRI)
Hoge RD et al. (1999) MRM 42:849-863
Bulte DP et al. (2012) NeuroImage 60:279-289 (gas-free calibration)
Blockley NP et al. (2015) NeuroImage 112:225-234
Germuska M, Bulte DP (2014) NeuroImage 102:789-798 (dual-calibrated)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class CalibratedBOLDParams(eqx.Module):
    """Parameters for calibrated BOLD / Davis model.

    Attributes
    ----------
    alpha : Grubb exponent (CBF-CBV coupling), ~0.18-0.38
    beta  : field-strength-dependent parameter (~1.3 at 3T)
    TE    : echo time (s)
    """
    alpha: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.18))
    beta: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.3))
    TE: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.030))


def estimate_M_from_r2prime(
    r2prime: Float[Array, "..."],
    params: CalibratedBOLDParams | None = None,
) -> Float[Array, "..."]:
    """Estimate the calibration parameter M from R₂' (gas-free method).

    Bulte et al. (2012): M = TE × R₂'

    This avoids the need for gas challenges by using the resting-state
    R₂' (from qBOLD) as a proxy for the maximum possible BOLD signal
    change.

    Parameters
    ----------
    r2prime : reversible relaxation rate (s⁻¹) from qBOLD
    params : CalibratedBOLDParams

    Returns
    -------
    M : BOLD calibration parameter (dimensionless, typically 0.03-0.15)
    """
    if params is None:
        params = CalibratedBOLDParams()

    return params.TE * r2prime


def estimate_M_hypercapnia(
    delta_bold: Float[Array, "..."],
    delta_cbf: Float[Array, "..."],
    params: CalibratedBOLDParams | None = None,
) -> Float[Array, "..."]:
    """Estimate M from a hypercapnia challenge (traditional method).

    During hypercapnia, CMRO₂ is assumed unchanged (ΔCMRO₂ ≈ 0), so:
        ΔBOLD/BOLD₀ = M × (1 - (CBF/CBF₀)^(α-β))

    Solving for M:
        M = (ΔBOLD/BOLD₀) / (1 - (1 + ΔCBF/CBF₀)^(α-β))

    Parameters
    ----------
    delta_bold : fractional BOLD change during hypercapnia
    delta_cbf : fractional CBF change during hypercapnia
    params : CalibratedBOLDParams

    Returns
    -------
    M : calibration parameter
    """
    if params is None:
        params = CalibratedBOLDParams()

    cbf_ratio = 1.0 + delta_cbf
    denominator = 1.0 - jnp.power(cbf_ratio, params.alpha - params.beta)
    # Avoid division by zero
    denominator = jnp.where(jnp.abs(denominator) > 1e-8, denominator, 1e-8)

    return delta_bold / denominator


def davis_model(
    cbf_ratio: Float[Array, "..."],
    cmro2_ratio: Float[Array, "..."],
    M: Float[Array, "..."],
    params: CalibratedBOLDParams | None = None,
) -> Float[Array, "..."]:
    """Predict BOLD signal change using the Davis model.

    ΔBOLD/BOLD₀ = M · (1 - cbf_ratio^(α-β) · cmro2_ratio^β)

    Parameters
    ----------
    cbf_ratio : CBF/CBF₀ (fractional change, 1.0 at baseline)
    cmro2_ratio : CMRO₂/CMRO₂₀ (fractional change, 1.0 at baseline)
    M : calibration parameter (from gas challenge or qBOLD)
    params : CalibratedBOLDParams

    Returns
    -------
    Fractional BOLD signal change (0 at baseline).
    """
    if params is None:
        params = CalibratedBOLDParams()

    bold = M * (
        1.0
        - jnp.power(cbf_ratio, params.alpha - params.beta)
        * jnp.power(cmro2_ratio, params.beta)
    )
    return bold


def estimate_cmro2_change(
    delta_bold: Float[Array, "..."],
    cbf_ratio: Float[Array, "..."],
    M: Float[Array, "..."],
    params: CalibratedBOLDParams | None = None,
) -> Float[Array, "..."]:
    """Estimate CMRO₂ change from BOLD and CBF changes (inverse Davis model).

    Rearranging the Davis model:
        cmro2_ratio = ((1 - ΔBOLD/M) / cbf_ratio^(α-β))^(1/β)

    Parameters
    ----------
    delta_bold : fractional BOLD signal change
    cbf_ratio : CBF/CBF₀
    M : calibration parameter
    params : CalibratedBOLDParams

    Returns
    -------
    cmro2_ratio : CMRO₂/CMRO₂₀
    """
    if params is None:
        params = CalibratedBOLDParams()

    # Avoid division by zero in M
    M_safe = jnp.where(jnp.abs(M) > 1e-8, M, 1e-8)

    flow_term = jnp.power(cbf_ratio, params.alpha - params.beta)
    # Avoid division by zero in flow_term
    flow_safe = jnp.where(jnp.abs(flow_term) > 1e-8, flow_term, 1e-8)

    inner = (1.0 - delta_bold / M_safe) / flow_safe
    # Clip to avoid negative values before power
    inner = jnp.clip(inner, 1e-8, None)

    cmro2_ratio = jnp.power(inner, 1.0 / params.beta)
    return cmro2_ratio
