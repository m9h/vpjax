"""Multi-compartment glymphatic solute transport model.

Inspired by the SIMULA MPET (Multiple-Network Poroelasticity) framework
(Mardal, Rognes et al.), simplified to a lumped-parameter ODE for use in
vpjax forward models.

Three fluid compartments:
  1. Arterial perivascular space (PVS_art): CSF inflow along arteries
  2. Extracellular space (ECS): brain parenchyma, waste solutes
  3. Venous perivascular space (PVS_ven): clearance outflow along veins

Solute transport:
  PVS_art → ECS (diffusion + advection from arterial pulsation)
  ECS → PVS_ven (diffusion + vasomotion-enhanced advection)
  PVS_ven → clearance (drainage to cervical lymph)

Vasomotion drives advective transport: NE oscillations pump CSF through
the perivascular network, enhancing clearance during NREM sleep.

References
----------
Mardal KA, Rognes ME et al. (2023) PLOS ONE 18:e0280501
    "Multi-compartmental model of glymphatic clearance"
Mardal KA et al. (2024) bioRxiv
    "Modeling CSF circulation and the glymphatic system during infusion"
Iliff JJ et al. (2012) Sci Transl Med 4:147ra111
Hauglund NL et al. (2025) Cell 188:606-622
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class GlymphaticParams(eqx.Module):
    """Parameters for multi-compartment glymphatic transport.

    Attributes
    ----------
    kappa_pvs_art   : hydraulic conductivity, arterial PVS (mm²/s)
    kappa_ecs       : hydraulic conductivity, ECS (mm²/s)
    kappa_pvs_ven   : hydraulic conductivity, venous PVS (mm²/s)
    gamma_art_ecs   : transfer coefficient, PVS_art → ECS (1/s)
    gamma_ecs_ven   : transfer coefficient, ECS → PVS_ven (1/s)
    gamma_ven_out   : clearance rate from venous PVS (1/s)
    D_eff           : effective diffusion coefficient in ECS (mm²/s)
    vaso_gain       : vasomotion enhancement of advective transport
    phi_ecs         : ECS volume fraction (porosity)
    phi_pvs         : PVS volume fraction
    """
    kappa_pvs_art: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1e-4))
    kappa_ecs: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(2e-5))
    kappa_pvs_ven: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1e-4))
    gamma_art_ecs: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(5e-4))
    gamma_ecs_ven: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(3e-4))
    gamma_ven_out: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(2e-4))
    D_eff: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1e-5))
    vaso_gain: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(5.0))
    phi_ecs: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.14))
    phi_pvs: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.01))


class GlymphaticState(eqx.Module):
    """Solute concentrations in each compartment.

    Attributes
    ----------
    c_pvs_art : solute concentration in arterial PVS (a.u.)
    c_ecs     : solute concentration in ECS (a.u.)
    c_pvs_ven : solute concentration in venous PVS (a.u.)
    """
    c_pvs_art: Float[Array, "..."]
    c_ecs: Float[Array, "..."]
    c_pvs_ven: Float[Array, "..."]

    @staticmethod
    def with_solute(
        c_pvs_art: float = 0.0,
        c_ecs: float = 1.0,
        c_pvs_ven: float = 0.0,
    ) -> GlymphaticState:
        return GlymphaticState(
            c_pvs_art=jnp.array(c_pvs_art),
            c_ecs=jnp.array(c_ecs),
            c_pvs_ven=jnp.array(c_pvs_ven),
        )


class GlymphaticODE(eqx.Module):
    """Multi-compartment glymphatic transport ODE.

    Solute transport between compartments via diffusion and
    vasomotion-enhanced advection.
    """

    params: GlymphaticParams

    def __call__(
        self,
        state: GlymphaticState,
        vasomotion_drive: Float[Array, "..."] = jnp.array(0.0),
    ) -> GlymphaticState:
        """Evaluate RHS of the glymphatic transport equations.

        Parameters
        ----------
        state : current concentrations
        vasomotion_drive : NE-driven vasomotion amplitude (0=none, 1=strong NREM)

        Returns
        -------
        GlymphaticState with concentration derivatives
        """
        p = self.params

        # Effective transfer rates (enhanced by vasomotion)
        vaso_factor = 1.0 + p.vaso_gain * jnp.abs(vasomotion_drive)
        gamma_ae = p.gamma_art_ecs * vaso_factor
        gamma_ev = p.gamma_ecs_ven * vaso_factor
        gamma_vo = p.gamma_ven_out

        # Concentration-driven transport (diffusion-like)
        # Plus advective component from vasomotion pumping
        flux_art_ecs = gamma_ae * (state.c_pvs_art - state.c_ecs)
        flux_ecs_ven = gamma_ev * (state.c_ecs - state.c_pvs_ven)
        flux_ven_out = gamma_vo * state.c_pvs_ven

        # Diffusive contribution within ECS
        # In the lumped model, this appears as a decay term
        # (solute spreads out and encounters boundaries)
        diff_decay = p.D_eff * state.c_ecs * 0.01  # scale factor for lumped model

        # Rate equations
        dc_pvs_art = -flux_art_ecs / p.phi_pvs
        dc_ecs = (flux_art_ecs - flux_ecs_ven - diff_decay) / p.phi_ecs
        dc_pvs_ven = (flux_ecs_ven - flux_ven_out) / p.phi_pvs

        return GlymphaticState(
            c_pvs_art=dc_pvs_art,
            c_ecs=dc_ecs,
            c_pvs_ven=dc_pvs_ven,
        )


@dataclass
class ClearanceResult:
    """Result of a glymphatic clearance simulation."""
    c_pvs_art: jnp.ndarray  # (T,)
    c_ecs: jnp.ndarray      # (T,)
    c_pvs_ven: jnp.ndarray  # (T,)
    t: jnp.ndarray           # (T,)


def simulate_clearance(
    c_ecs_init: float,
    t: Float[Array, "T"],
    vasomotion_drive: Float[Array, "T"],
    params: GlymphaticParams | None = None,
) -> ClearanceResult:
    """Simulate glymphatic solute clearance over time.

    Parameters
    ----------
    c_ecs_init : initial ECS solute concentration
    t : time points (s)
    vasomotion_drive : vasomotion amplitude at each time point
    params : GlymphaticParams

    Returns
    -------
    ClearanceResult with concentration time courses
    """
    if params is None:
        params = GlymphaticParams()

    model = GlymphaticODE(params=params)
    state = GlymphaticState.with_solute(c_pvs_art=0.0, c_ecs=c_ecs_init, c_pvs_ven=0.0)

    def scan_fn(state, inputs):
        t_curr, t_next, vaso = inputs
        dt = t_next - t_curr
        dy = model(state, vaso)
        new_state = GlymphaticState(
            c_pvs_art=jnp.clip(state.c_pvs_art + dt * dy.c_pvs_art, 0.0, None),
            c_ecs=jnp.clip(state.c_ecs + dt * dy.c_ecs, 0.0, None),
            c_pvs_ven=jnp.clip(state.c_pvs_ven + dt * dy.c_pvs_ven, 0.0, None),
        )
        return new_state, (new_state.c_pvs_art, new_state.c_ecs, new_state.c_pvs_ven)

    inputs = (t[:-1], t[1:], vasomotion_drive[:-1])
    _, (c_art, c_ecs, c_ven) = jax.lax.scan(scan_fn, state, inputs)

    # Prepend initial values
    c_art = jnp.concatenate([jnp.array([0.0]), c_art])
    c_ecs = jnp.concatenate([jnp.array([c_ecs_init]), c_ecs])
    c_ven = jnp.concatenate([jnp.array([0.0]), c_ven])

    return ClearanceResult(c_pvs_art=c_art, c_ecs=c_ecs, c_pvs_ven=c_ven, t=t)
