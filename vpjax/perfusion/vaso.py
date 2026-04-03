"""VASO (Vascular Space Occupancy) observation function.

SS-SI-VASO signal is sensitive to cerebral blood volume (CBV).
At baseline: S ∝ (1 - CBV). The fractional signal change is
approximately -ΔCBV/CBV0, i.e. inversely related to volume change.

References
----------
Lu H et al. (2003) MRM 50:263-274 (original VASO)
Huber L et al. (2014) NeuroImage 101:1-12 (SS-SI-VASO)
Huber L et al. (2021) NeuroImage (LAYNII)
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax._types import BalloonState


def observe_vaso(state: BalloonState) -> Float[Array, "..."]:
    """Compute VASO signal change from hemodynamic state.

    Parameters
    ----------
    state : BalloonState with volume (v).

    Returns
    -------
    VASO fractional signal change (0 at baseline, negative with activation).
    """
    # v is CBV/CBV0, so Δv = v - 1
    # VASO signal ∝ -(v - 1) = 1 - v
    return 1.0 - state.v
