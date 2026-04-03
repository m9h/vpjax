"""Buxton general kinetic model for ASL signal.

Predicts the ASL difference signal ΔM for pCASL (pseudo-continuous ASL)
given labeling duration, post-label delay, transit time, CBF, and
relaxation parameters.

The general kinetic model:
    ΔM(t) = 2 · M₀_b · CBF · T₁' · α · q(t)

where q(t) encodes the delivery and clearance of labeled blood,
accounting for transit delay (δ), label duration (τ), and T₁ decay.

References
----------
Buxton RB et al. (1998) MRM 40:383-396
    "A general kinetic model for quantitative perfusion imaging with
    arterial spin labeling"
Alsop DC et al. (2015) MRM 73:102-116
    "Recommended implementation of ASL for clinical applications"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class ASLKineticParams(eqx.Module):
    """Parameters for the Buxton general kinetic model.

    Attributes
    ----------
    M0b       : equilibrium magnetization of blood (a.u.)
    T1b       : longitudinal relaxation time of blood (s)
    T1t       : longitudinal relaxation time of tissue (s)
    alpha     : labeling efficiency (0-1)
    tau       : labeling duration (s)
    delta     : arterial transit time / bolus arrival time (s)
    lambda_p  : blood-brain partition coefficient (mL/g)
    """
    M0b: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))
    T1b: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.65))
    T1t: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.30))
    alpha: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.85))
    tau: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.80))
    delta: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.20))
    lambda_p: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.90))


def asl_kinetic_signal(
    t: Float[Array, "T"],
    cbf: Float[Array, "..."],
    params: ASLKineticParams | None = None,
) -> Float[Array, "... T"]:
    """Compute the ASL difference signal ΔM(t) using the general kinetic model.

    Three temporal phases:
    1. t < δ           : no signal (label hasn't arrived)
    2. δ ≤ t < δ + τ   : label arriving, signal builds up
    3. t ≥ δ + τ       : label ended, signal decays

    Parameters
    ----------
    t : time points after labeling onset (s), shape (T,)
    cbf : cerebral blood flow (mL/100g/min), shape (...)
    params : ASLKineticParams

    Returns
    -------
    ΔM(t), shape (..., T)
    """
    if params is None:
        params = ASLKineticParams()

    # Convert CBF from mL/100g/min to mL/g/s
    f = cbf[..., None] / 6000.0  # (..., 1)

    # Effective T1 for tissue (accounts for clearance by flow)
    # 1/T1' = 1/T1t + f/λ
    T1_eff = 1.0 / (1.0 / params.T1t[..., None] + f / params.lambda_p[..., None])

    delta_ = params.delta[..., None]  # (..., 1)
    tau_ = params.tau[..., None]      # (..., 1)
    t_ = t                            # (T,)

    # q(t) function (delivery/clearance)
    # Phase 1: t < delta → q = 0
    # Phase 2: delta <= t < delta + tau → rising
    #   q = T1' * (1 - exp(-(t-delta)/T1'))
    # Phase 3: t >= delta + tau → decaying
    #   q = T1' * exp(-(t-delta-tau)/T1') * (1 - exp(-tau/T1'))

    q_rising = T1_eff * (1.0 - jnp.exp(-(t_ - delta_) / T1_eff))
    q_decay = T1_eff * jnp.exp(-(t_ - delta_ - tau_) / T1_eff) * (
        1.0 - jnp.exp(-tau_ / T1_eff)
    )

    # T1b decay of the label during transit
    label_decay = jnp.exp(-delta_ / params.T1b[..., None])

    q = jnp.where(
        t_ < delta_,
        0.0,
        jnp.where(t_ < delta_ + tau_, q_rising, q_decay),
    )

    # ΔM = 2 * M0b * f * alpha * label_decay * q
    delta_m = (
        2.0
        * params.M0b[..., None]
        * f
        * params.alpha[..., None]
        * label_decay
        * q
    )

    return delta_m


def quantify_cbf(
    delta_m: Float[Array, "..."],
    pld: Float[Array, "..."],
    params: ASLKineticParams | None = None,
) -> Float[Array, "..."]:
    """Quantify CBF from a single-PLD pCASL difference image.

    Uses the simplified single-compartment model (Alsop et al. 2015):

        CBF = (6000 · λ · ΔM · exp(PLD/T1b)) / (2 · α · T1b · M0b · (1-exp(-τ/T1b)))

    Parameters
    ----------
    delta_m : ASL difference signal (control - label)
    pld : post-labeling delay (s)
    params : ASLKineticParams

    Returns
    -------
    CBF in mL/100g/min
    """
    if params is None:
        params = ASLKineticParams()

    numerator = 6000.0 * params.lambda_p * delta_m * jnp.exp(pld / params.T1b)
    denominator = (
        2.0
        * params.alpha
        * params.T1b
        * params.M0b
        * (1.0 - jnp.exp(-params.tau / params.T1b))
    )

    # Avoid division by zero
    denominator = jnp.where(jnp.abs(denominator) > 1e-10, denominator, 1e-10)

    return numerator / denominator
