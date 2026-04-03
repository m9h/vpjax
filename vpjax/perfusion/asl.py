"""ASL (Arterial Spin Labeling) observation function.

ASL signal is proportional to cerebral blood flow (CBF).
The simplest model: Delta_M ∝ f(t) - 1, i.e. the fractional CBF change.

For the full Buxton general kinetic model (pCASL signal with
labeling duration, transit delay, T1 decay), see kinetic.py (Phase 5).

References
----------
Buxton RB et al. (1998) MRM 40:383-396 (general kinetic model)
Alsop DC et al. (2015) MRM 73:102-116 (ASL consensus paper)
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax._types import BalloonState


def observe_asl(state: BalloonState) -> Float[Array, "..."]:
    """Compute ASL perfusion signal from hemodynamic state.

    Parameters
    ----------
    state : BalloonState with flow (f).

    Returns
    -------
    Fractional CBF change (0 at baseline).
    """
    return state.f - 1.0
