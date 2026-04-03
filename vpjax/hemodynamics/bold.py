"""BOLD signal observation function.

Maps Balloon-Windkessel state (v, q) to the BOLD signal change,
following the Stephan et al. (2007) DCM formulation:

    y_BOLD = V0 * (k1*(1-q) + k2*(1-q/v) + k3*(1-v))

where V0 is the resting venous blood volume fraction, and k1,k2,k3
are constants that depend on field strength, echo time, and E0.

References
----------
Buxton RB et al. (1998) MRM 40:163-174
Stephan KJ et al. (2007) NeuroImage 38:387-401
Obata T et al. (2004) NeuroImage 21:144-153
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax._types import BalloonState


class BOLDParams(eqx.Module):
    """Parameters for the BOLD signal model.

    Default values for 3T, TE=30ms (standard fMRI).

    Attributes
    ----------
    V0   : resting venous blood volume fraction (~0.04)
    k1   : intravascular contribution weight
    k2   : intravascular induction effect weight
    k3   : extravascular contribution weight
    TE   : echo time (s)
    r0   : slope of intravascular relaxation rate vs. OEF (s^-1), field-dependent
    nu0  : frequency offset at 100% extraction (s^-1), field-dependent
    """
    V0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.04))
    k1: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(3.72))
    k2: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.53))
    k3: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.53))


def observe_bold(
    state: BalloonState,
    params: BOLDParams | None = None,
) -> Float[Array, "..."]:
    """Compute BOLD signal change from Balloon-Windkessel state.

    Parameters
    ----------
    state : BalloonState with volume (v) and deoxyhemoglobin (q).
    params : BOLDParams (default: 3T standard).

    Returns
    -------
    BOLD signal fractional change (0 at baseline).
    """
    if params is None:
        params = BOLDParams()

    v = state.v
    q = state.q

    bold = params.V0 * (
        params.k1 * (1.0 - q)
        + params.k2 * (1.0 - q / v)
        + params.k3 * (1.0 - v)
    )
    return bold
