"""Core types for vpjax cerebrovascular models.

HemodynamicState holds the full ODE state vector.
BalloonState holds the minimal Balloon-Windkessel state.
VascularParams holds physiological parameters.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# Balloon-Windkessel state (Phase 1: single-compartment)
# ---------------------------------------------------------------------------

class BalloonState(eqx.Module):
    """State of the single-compartment Balloon-Windkessel model.

    All variables are fractional changes relative to steady-state baseline
    (i.e. baseline value = 1.0).

    Attributes
    ----------
    s : vasodilatory signal (dimensionless)
    f : blood inflow, CBF / CBF_0
    v : blood volume, CBV / CBV_0
    q : deoxyhemoglobin content, dHb / dHb_0
    """
    s: Float[Array, "..."]
    f: Float[Array, "..."]
    v: Float[Array, "..."]
    q: Float[Array, "..."]

    @staticmethod
    def steady_state(shape: tuple[int, ...] = ()) -> BalloonState:
        """Return the resting steady-state (all variables = baseline)."""
        return BalloonState(
            s=jnp.zeros(shape),
            f=jnp.ones(shape),
            v=jnp.ones(shape),
            q=jnp.ones(shape),
        )


# ---------------------------------------------------------------------------
# Balloon-Windkessel parameters
# ---------------------------------------------------------------------------

class BalloonParams(eqx.Module):
    """Parameters for the Balloon-Windkessel model.

    Default values from Friston et al. (2000) / Stephan et al. (2007) DCM.

    Attributes
    ----------
    kappa : signal decay rate (s^-1)
    gamma : flow-dependent elimination rate (s^-1)
    tau   : hemodynamic transit time, mean (s)
    alpha : Grubb exponent (CBV-CBF power law)
    E0    : resting oxygen extraction fraction
    """
    kappa: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.65))
    gamma: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.41))
    tau: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.98))
    alpha: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.32))
    E0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.34))


# ---------------------------------------------------------------------------
# Full multi-compartment hemodynamic state (Phase 2+, placeholder structure)
# ---------------------------------------------------------------------------

class HemodynamicState(eqx.Module):
    """Full multi-compartment hemodynamic state (SimCVR).

    This extends BalloonState with compartmental volumes, gas exchange,
    and metabolic variables.  Populated in later phases.

    Attributes
    ----------
    cbf   : cerebral blood flow (fractional)
    cbv_a : arterial blood volume (fractional)
    cbv_c : capillary blood volume (fractional)
    cbv_v : venous blood volume (fractional)
    hbo   : oxyhemoglobin concentration (mM)
    hbr   : deoxyhemoglobin concentration (mM)
    oef   : oxygen extraction fraction
    cmro2 : cerebral metabolic rate of oxygen (fractional)
    """
    cbf: Float[Array, "..."]
    cbv_a: Float[Array, "..."]
    cbv_c: Float[Array, "..."]
    cbv_v: Float[Array, "..."]
    hbo: Float[Array, "..."]
    hbr: Float[Array, "..."]
    oef: Float[Array, "..."]
    cmro2: Float[Array, "..."]

    @staticmethod
    def steady_state(shape: tuple[int, ...] = ()) -> HemodynamicState:
        """Return resting steady-state."""
        return HemodynamicState(
            cbf=jnp.ones(shape),
            cbv_a=jnp.ones(shape),
            cbv_c=jnp.ones(shape),
            cbv_v=jnp.ones(shape),
            hbo=jnp.ones(shape),
            hbr=jnp.ones(shape),
            oef=jnp.full(shape, 0.34),
            cmro2=jnp.ones(shape),
        )
