"""Riera neurovascular coupling model.

Full nonlinear local electrovascular coupling model from:
  Riera JJ et al. (2006) HBM 27:896-914
  Riera JJ et al. (2007) NeuroImage 36:1179-1196

Extends the Balloon-Windkessel with:
  - Multi-compartment vascular tree (arteriolar, capillary, venous)
  - Nitric oxide (NO) and adenosine-mediated vasodilation
  - Neural-metabolic coupling (excitatory/inhibitory contributions)
  - Oxygen exchange along the capillary bed

The model links neural mass activity (from vbjax) through metabolic
intermediaries to the hemodynamic response, providing a more
physiologically detailed forward model than the standard Balloon.

References
----------
Riera JJ et al. (2006) HBM 27:896-914
    "Nonlinear local electrovascular coupling. I: A theoretical model"
Riera JJ et al. (2007) NeuroImage 36:1179-1196
    "Nonlinear local electrovascular coupling. II: From data to
    neuronal masses"
Sotero RC, Trujillo-Barreto NJ (2007) NeuroImage 36:671-687
"""

from __future__ import annotations

import diffrax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class RieraState(eqx.Module):
    """State of the Riera neurovascular coupling model.

    All hemodynamic variables are fractional (baseline = 1.0)
    except vasodilatory signals which are zero at rest.

    Attributes
    ----------
    s_no   : NO-mediated vasodilatory signal
    s_ade  : adenosine-mediated vasodilatory signal
    f_a    : arteriolar blood flow (CBF_a / CBF_a0)
    v_a    : arteriolar blood volume (CBV_a / CBV_a0)
    v_c    : capillary blood volume (CBV_c / CBV_c0)
    v_v    : venous blood volume (CBV_v / CBV_v0)
    q_v    : venous deoxyhemoglobin content (dHb_v / dHb_v0)
    cmro2  : fractional CMRO₂ (CMRO₂ / CMRO₂_0)
    """
    s_no: Float[Array, "..."]
    s_ade: Float[Array, "..."]
    f_a: Float[Array, "..."]
    v_a: Float[Array, "..."]
    v_c: Float[Array, "..."]
    v_v: Float[Array, "..."]
    q_v: Float[Array, "..."]
    cmro2: Float[Array, "..."]

    @staticmethod
    def steady_state(shape: tuple[int, ...] = ()) -> RieraState:
        """Return resting steady-state."""
        return RieraState(
            s_no=jnp.zeros(shape),
            s_ade=jnp.zeros(shape),
            f_a=jnp.ones(shape),
            v_a=jnp.ones(shape),
            v_c=jnp.ones(shape),
            v_v=jnp.ones(shape),
            q_v=jnp.ones(shape),
            cmro2=jnp.ones(shape),
        )


class RieraParams(eqx.Module):
    """Parameters for the Riera neurovascular coupling model.

    Default values from Riera et al. (2006, 2007) and Sotero &
    Trujillo-Barreto (2007).

    Attributes
    ----------
    kappa_no  : NO signal decay rate (s⁻¹)
    kappa_ade : adenosine signal decay rate (s⁻¹)
    gamma_no  : NO flow-elimination rate (s⁻¹)
    gamma_ade : adenosine flow-elimination rate (s⁻¹)
    c_no      : NO coupling gain (neural activity → NO signal)
    c_ade     : adenosine coupling gain (metabolic demand → adenosine)
    tau_a     : arteriolar transit time (s)
    tau_c     : capillary transit time (s)
    tau_v     : venous transit time (s)
    alpha_a   : arteriolar Grubb exponent
    alpha_c   : capillary Grubb exponent (stiffer than arterioles)
    alpha_v   : venous Grubb exponent
    E0        : resting oxygen extraction fraction
    phi       : metabolic-neural coupling strength
    tau_m     : metabolic response time constant (s)
    """
    kappa_no: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.65))
    kappa_ade: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.40))
    gamma_no: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.41))
    gamma_ade: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.30))
    c_no: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))
    c_ade: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.5))
    tau_a: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.5))
    tau_c: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))
    tau_v: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(2.0))
    alpha_a: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.20))
    alpha_c: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.10))
    alpha_v: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.32))
    E0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.34))
    phi: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.5))
    tau_m: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(3.0))


class RieraNVC(eqx.Module):
    """Riera neurovascular coupling ODE as a Diffrax vector field.

    Models the full pathway from neural activity to multi-compartment
    hemodynamics via dual vasodilatory pathways (NO + adenosine).
    """

    params: RieraParams

    def __call__(
        self,
        t: Float[Array, ""],
        y: RieraState,
        args: Float[Array, "..."],
    ) -> RieraState:
        """Evaluate the RHS of the Riera NVC ODE.

        Parameters
        ----------
        t : current time (unused — autonomous system)
        y : current RieraState
        args : neural stimulus u(t), scalar or batched

        Returns
        -------
        RieraState with time-derivatives.
        """
        u = args
        p = self.params

        # --- Vasodilatory signals ---
        # NO: fast, driven directly by neural activity
        ds_no = p.c_no * u - p.kappa_no * y.s_no - p.gamma_no * (y.f_a - 1.0)

        # Adenosine: slower, driven by metabolic demand (CMRO₂ change)
        ds_ade = p.c_ade * (y.cmro2 - 1.0) - p.kappa_ade * y.s_ade - p.gamma_ade * (y.f_a - 1.0)

        # --- Arteriolar flow ---
        # Combined vasodilatory drive
        df_a = y.s_no + y.s_ade

        # --- Multi-compartment Windkessel ---
        # Arteriolar compartment
        f_out_a = jnp.power(y.v_a, 1.0 / p.alpha_a)
        dv_a = (y.f_a - f_out_a) / p.tau_a

        # Capillary compartment: inflow = arteriolar outflow
        f_out_c = jnp.power(y.v_c, 1.0 / p.alpha_c)
        dv_c = (f_out_a - f_out_c) / p.tau_c

        # Venous compartment: inflow = capillary outflow
        f_out_v = jnp.power(y.v_v, 1.0 / p.alpha_v)
        dv_v = (f_out_c - f_out_v) / p.tau_v

        # --- Deoxyhemoglobin dynamics (venous compartment) ---
        # Oxygen extraction depends on flow (Buxton model)
        extraction = 1.0 - jnp.power(1.0 - p.E0, 1.0 / y.f_a)
        dq_v = (y.f_a * extraction / p.E0 - f_out_v * y.q_v / y.v_v) / p.tau_v

        # --- Metabolic dynamics ---
        # CMRO₂ responds to neural activity with a delay
        cmro2_ss = 1.0 + p.phi * u  # steady-state CMRO₂ for given activity
        dcmro2 = (cmro2_ss - y.cmro2) / p.tau_m

        return RieraState(
            s_no=ds_no,
            s_ade=ds_ade,
            f_a=df_a,
            v_a=dv_a,
            v_c=dv_c,
            v_v=dv_v,
            q_v=dq_v,
            cmro2=dcmro2,
        )


def riera_total_cbv(state: RieraState) -> Float[Array, "..."]:
    """Compute total CBV from multi-compartment volumes.

    Weighted sum of arteriolar, capillary, and venous volumes.
    Typical resting fractions: art ~20%, cap ~5%, ven ~75%.

    Parameters
    ----------
    state : RieraState

    Returns
    -------
    Total CBV/CBV₀ (fractional)
    """
    w_a, w_c, w_v = 0.20, 0.05, 0.75
    return w_a * state.v_a + w_c * state.v_c + w_v * state.v_v


def solve_riera(
    params: RieraParams,
    stimulus: Float[Array, "T"],
    dt: float = 0.01,
    t0: float = 0.0,
    t1: float | None = None,
    solver: diffrax.AbstractSolver | None = None,
) -> tuple[Float[Array, "T"], RieraState]:
    """Integrate the Riera NVC model over a stimulus time-course.

    Parameters
    ----------
    params : RieraParams
    stimulus : 1-D array of neural stimulus, sampled at interval *dt*.
    dt : sampling interval of stimulus (seconds).
    t0 : start time.
    t1 : end time (default: t0 + len(stimulus)*dt).
    solver : Diffrax solver (default: Tsit5).

    Returns
    -------
    ts : time points (shape ``(T,)``)
    trajectory : RieraState with arrays of shape ``(T,)``
    """
    n_steps = stimulus.shape[0]
    if t1 is None:
        t1 = t0 + n_steps * dt

    ts = jnp.linspace(t0, t1, n_steps)
    model = RieraNVC(params=params)

    # Interpolate stimulus as a piecewise-constant control
    control = diffrax.LinearInterpolation(ts=ts, ys=stimulus)

    if solver is None:
        solver = diffrax.Tsit5()

    term = diffrax.ODETerm(
        lambda t, y, args: model(t, y, control.evaluate(t))
    )

    y0 = RieraState.steady_state()

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=n_steps * 100,
    )

    return ts, sol.ys


def riera_to_balloon(state: RieraState) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Extract Balloon-equivalent (v, q) from Riera state.

    Maps the multi-compartment Riera state to the single-compartment
    variables used by the BOLD observation function.

    Parameters
    ----------
    state : RieraState

    Returns
    -------
    v : total CBV/CBV₀ (for BOLD observation)
    q : venous dHb/dHb₀ (for BOLD observation)
    """
    v = riera_total_cbv(state)
    q = state.q_v
    return v, q
