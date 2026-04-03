"""SS-SI-VASO signal model.

Slice-Saturation Slab-Inversion VASO (SS-SI-VASO) measures cerebral
blood volume (CBV) by nulling blood signal with an inversion pulse.
At the inversion null point, tissue signal is present but blood signal
is suppressed, so:

    S_VASO ∝ (1 - CBV)

The fractional signal change during activation:
    ΔS/S₀ ≈ -ΔCBV / (1 - CBV₀)

VASO provides the missing Balloon-Windkessel observable: CBV (v).

References
----------
Lu H et al. (2003) MRM 50:263-274
    "Functional MRI based on changes in vascular space occupancy"
Huber L et al. (2014) NeuroImage 101:1-12
    "Slab-selective, BOLD-corrected VASO at 7T"
Huber L et al. (2021) NeuroImage 237:118091
    "LAYNII"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class VASOParams(eqx.Module):
    """Parameters for the VASO signal model.

    Attributes
    ----------
    CBV0     : resting cerebral blood volume fraction (~0.05)
    T1b      : blood T1 (s), field-dependent
    T1t      : tissue T1 (s), field-dependent
    TI       : inversion time (s), tuned to null blood
    TR       : repetition time (s)
    M0_ratio : ratio of blood to tissue equilibrium magnetization (~1.0)
    """
    CBV0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.05))
    T1b: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(2.09))
    T1t: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.80))
    TI: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.10))
    TR: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(3.00))
    M0_ratio: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))


def blood_nulling_ti(
    T1b: Float[Array, "..."],
    TR: Float[Array, "..."] | None = None,
) -> Float[Array, "..."]:
    """Compute the optimal inversion time to null blood signal.

    For an inversion recovery: M(TI) = M₀(1 - 2·exp(-TI/T₁))
    Null point: TI = T₁ · ln(2)

    With finite TR (incomplete recovery before next pulse):
    TI = T1b · ln(2 / (1 + exp(-TR/T1b)))

    Parameters
    ----------
    T1b : blood T1 (s)
    TR : repetition time (s). If None, assumes TR >> T1b.

    Returns
    -------
    Optimal TI (s)
    """
    if TR is None:
        return T1b * jnp.log(2.0)

    return T1b * jnp.log(2.0 / (1.0 + jnp.exp(-TR / T1b)))


def vaso_signal(
    cbv: Float[Array, "..."],
    params: VASOParams | None = None,
) -> Float[Array, "..."]:
    """Compute the VASO signal (tissue-only, blood nulled).

    At the blood null point:
        S_VASO = M₀_tissue · (1 - CBV) · R_tissue(TI)

    where R_tissue(TI) accounts for tissue T1 recovery.

    Normalized to baseline: S/S₀ = (1 - CBV) / (1 - CBV₀)

    Parameters
    ----------
    cbv : cerebral blood volume fraction (absolute, e.g. 0.05)
    params : VASOParams

    Returns
    -------
    VASO signal normalized to baseline (1.0 at rest).
    """
    if params is None:
        params = VASOParams()

    return (1.0 - cbv) / (1.0 - params.CBV0)


def vaso_signal_change(
    cbv_ratio: Float[Array, "..."],
    params: VASOParams | None = None,
) -> Float[Array, "..."]:
    """Compute VASO fractional signal change from Balloon model v.

    Given v = CBV/CBV₀ from the Balloon model:
        CBV = CBV₀ · v
        ΔS/S₀ = (1 - CBV₀·v)/(1 - CBV₀) - 1
               = -CBV₀·(v - 1) / (1 - CBV₀)

    Parameters
    ----------
    cbv_ratio : CBV/CBV₀ from Balloon model (1.0 at baseline)
    params : VASOParams

    Returns
    -------
    Fractional VASO signal change (0 at baseline, negative during activation).
    """
    if params is None:
        params = VASOParams()

    return -params.CBV0 * (cbv_ratio - 1.0) / (1.0 - params.CBV0)


def cbv_from_vaso_signal(
    delta_s: Float[Array, "..."],
    params: VASOParams | None = None,
) -> Float[Array, "..."]:
    """Recover absolute ΔCBV from VASO signal change.

    Inverting: ΔS/S₀ = -ΔCBV / (1 - CBV₀)
    → ΔCBV = -ΔS/S₀ × (1 - CBV₀)

    Parameters
    ----------
    delta_s : fractional VASO signal change (ΔS/S₀)
    params : VASOParams

    Returns
    -------
    ΔCBV (absolute change in blood volume fraction)
    """
    if params is None:
        params = VASOParams()

    return -delta_s * (1.0 - params.CBV0)
