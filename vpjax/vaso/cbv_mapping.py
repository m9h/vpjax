"""CBV mapping from VASO data.

Estimates absolute and relative CBV from VASO acquisitions.
Combines BOCO-corrected VASO signal with the signal model to
produce quantitative CBV maps.

The key relationships:
    S_VASO ∝ (1 - CBV)
    ΔCBV/CBV₀ = -ΔS_VASO / (S₀_VASO × CBV₀/(1-CBV₀))

For layer-specific CBV, applies deveining before mapping.

References
----------
Lu H et al. (2003) MRM 50:263-274
Huber L et al. (2014) NeuroImage 101:1-12
Huber L et al. (2017) Neuron 96:1253-1263
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax.vaso.signal_model import VASOParams


def relative_cbv_change(
    s_vaso: Float[Array, "..."],
    s_vaso_baseline: Float[Array, "..."],
    params: VASOParams | None = None,
) -> Float[Array, "..."]:
    """Compute relative CBV change from VASO signal.

    ΔCBV/CBV₀ = -(S - S₀)/S₀ × (1 - CBV₀)/CBV₀

    Parameters
    ----------
    s_vaso : VASO signal (BOCO-corrected)
    s_vaso_baseline : baseline VASO signal
    params : VASOParams

    Returns
    -------
    ΔCBV/CBV₀ (fractional CBV change, positive = volume increase)
    """
    if params is None:
        params = VASOParams()

    s0_safe = jnp.where(jnp.abs(s_vaso_baseline) > 1e-6, s_vaso_baseline, 1e-6)
    delta_s = (s_vaso - s_vaso_baseline) / s0_safe

    cbv0_safe = jnp.where(params.CBV0 > 1e-6, params.CBV0, 1e-6)
    return -delta_s * (1.0 - params.CBV0) / cbv0_safe


def absolute_cbv(
    s_vaso: Float[Array, "..."],
    m0: Float[Array, "..."],
    params: VASOParams | None = None,
) -> Float[Array, "..."]:
    """Estimate absolute CBV from VASO signal and M₀.

    CBV = 1 - S_VASO / (M₀ × R_tissue)

    where R_tissue accounts for tissue T1 recovery at the inversion
    time.  Simplified:

    CBV ≈ 1 - S_VASO / S_max

    Parameters
    ----------
    s_vaso : VASO signal
    m0 : proton density / M₀ reference signal
    params : VASOParams

    Returns
    -------
    Absolute CBV fraction (typically 0.03-0.06 for gray matter)
    """
    if params is None:
        params = VASOParams()

    # Tissue signal recovery factor at TI
    r_tissue = 1.0 - 2.0 * jnp.exp(-params.TI / params.T1t) + jnp.exp(-params.TR / params.T1t)
    s_max = m0 * jnp.clip(r_tissue, 0.01, None)

    s_max_safe = jnp.where(jnp.abs(s_max) > 1e-6, s_max, 1e-6)
    cbv = 1.0 - s_vaso / s_max_safe
    return jnp.clip(cbv, 0.0, 1.0)


def balloon_cbv_ratio(
    delta_cbv_cbv0: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Convert ΔCBV/CBV₀ to Balloon model v = CBV/CBV₀.

    v = 1 + ΔCBV/CBV₀

    This connects VASO-measured CBV to the Balloon-Windkessel
    state variable v, enabling direct observation of this
    previously hidden model variable.

    Parameters
    ----------
    delta_cbv_cbv0 : fractional CBV change from VASO

    Returns
    -------
    v = CBV/CBV₀ (Balloon model convention, 1.0 at baseline)
    """
    return 1.0 + delta_cbv_cbv0


def layer_cbv_profile(
    s_vaso_layers: Float[Array, "... L"],
    s_vaso_baseline_layers: Float[Array, "... L"],
    params: VASOParams | None = None,
) -> Float[Array, "... L"]:
    """Compute layer-resolved relative CBV change.

    Applies the CBV mapping independently to each layer.
    For deveined signals, call devein() first before this function.

    Parameters
    ----------
    s_vaso_layers : VASO signal per layer, shape (..., L)
    s_vaso_baseline_layers : baseline VASO signal per layer, shape (..., L)
    params : VASOParams

    Returns
    -------
    ΔCBV/CBV₀ per layer, shape (..., L)
    """
    return relative_cbv_change(s_vaso_layers, s_vaso_baseline_layers, params)
