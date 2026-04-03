"""BOLD contamination correction (BOCO) for VASO.

VASO images acquired with GRE readout are contaminated by BOLD-weighted
T₂* effects.  The BOCO procedure (Huber et al. 2014) divides the
not-nulled (BOLD) image by the nulled (VASO) image to remove the
T₂* component, isolating the CBV-weighted signal.

The interleaved SS-SI-VASO acquisition provides alternating:
  - Nulled images (VASO-weighted, CBV contrast)
  - Not-nulled images (BOLD-weighted, T₂* contrast)

BOCO: S_corrected = S_null / S_not_null
This removes shared T₂* weighting, leaving pure CBV contrast.

References
----------
Huber L et al. (2014) NeuroImage 101:1-12
    "Slab-selective, BOLD-corrected VASO at 7T"
Huber L et al. (2021) NeuroImage 237:118091
    "LAYNII: LN_BOCO"
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def bold_correction(
    s_null: Float[Array, "..."],
    s_not_null: Float[Array, "..."],
    eps: float = 1e-6,
) -> Float[Array, "..."]:
    """Apply BOLD contamination correction to VASO images.

    S_corrected = S_null / S_not_null

    This removes T₂* weighting that contaminates the VASO signal.

    Parameters
    ----------
    s_null : nulled (VASO) image
    s_not_null : not-nulled (BOLD) image
    eps : small value to prevent division by zero

    Returns
    -------
    BOLD-corrected VASO signal.
    """
    denom = jnp.where(jnp.abs(s_not_null) > eps, s_not_null, eps)
    return s_null / denom


def bold_correction_timeseries(
    s_null: Float[Array, "T ..."],
    s_not_null: Float[Array, "T ..."],
    eps: float = 1e-6,
) -> Float[Array, "T ..."]:
    """Apply BOCO to interleaved VASO time series.

    If null and not-null volumes are acquired alternately, they may
    need temporal interpolation to align.  This function assumes
    they are already paired (same number of time points).

    Parameters
    ----------
    s_null : nulled time series, shape (T, ...)
    s_not_null : not-nulled time series, shape (T, ...)
    eps : division floor

    Returns
    -------
    BOLD-corrected VASO time series, shape (T, ...)
    """
    denom = jnp.where(jnp.abs(s_not_null) > eps, s_not_null, eps)
    return s_null / denom


def delta_cbv_from_boco(
    s_corrected: Float[Array, "T ..."],
    baseline_start: int = 0,
    baseline_end: int = 10,
) -> Float[Array, "T ..."]:
    """Compute fractional CBV change from BOCO-corrected time series.

    ΔCBV/CBV₀ ≈ -(S(t) - S₀) / S₀

    where S₀ is the mean baseline signal.

    Parameters
    ----------
    s_corrected : BOCO-corrected VASO time series, shape (T, ...)
    baseline_start : first volume index for baseline averaging
    baseline_end : last volume index for baseline averaging

    Returns
    -------
    Fractional CBV change time series (0 at baseline, positive during
    activation = CBV increase).
    """
    s0 = jnp.mean(s_corrected[baseline_start:baseline_end], axis=0)
    s0_safe = jnp.where(jnp.abs(s0) > 1e-6, s0, 1e-6)

    # VASO signal decreases with CBV increase, so negate
    return -(s_corrected - s0) / s0_safe
