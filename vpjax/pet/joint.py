"""Joint amyloid–CMRO2 forward model.

Couples three observables on a per-region basis:

* PET SUVR (target / reference)
* ASL CBF (mL/100g/min)
* Oxygen extraction fraction (latent; weak prior at age-typical 0.40)

The coupling is the Vaishnavi 2010 / Sperling 2009 / Drzezga 2011
empirical finding: regions with higher amyloid burden show lower
oxidative metabolism in healthy aging::

    CMRO2(region) = CMRO2_baseline · (1 + α_amy · BPnd)

with α_amy negative.  Combined with Fick (``CMRO2 = CBF · OEF · CaO2``)
this lets us estimate OEF jointly with BPnd: a region whose ASL CBF
is reduced *and* whose amyloid is elevated must have a corresponding
OEF response, otherwise the implied CMRO2 violates the coupling prior.

This is the part no existing tool does — niftypet does dynamic kinetic
modelling, PETSurfer does PVC + SRTM, and FSL ``oxford_asl`` does
Bayesian-Buxton ASL.  None of them fit BPnd, OEF, and CMRO2 jointly
under a Vaishnavi-style metabolic coupling.

References
----------
Vaishnavi SN et al. (2010) PNAS 107:17757-17762
Sperling RA et al. (2009) Neuron 63:178-188
Drzezga A et al. (2011) Brain 134:1635-1646
Mintun MA et al. (1984) JCBFM 4:163-172
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from vpjax.metabolism.fick import compute_cao2
from vpjax.pet.binding import forward_static_suvr


class JointAmyloidParams(eqx.Module):
    """Parameters of the joint amyloid–CMRO2 model.

    Attributes
    ----------
    cmro2_baseline : per-region baseline CMRO2 in healthy non-amyloid
        tissue (µmol O₂/100g/min).  Default 160 — Vaishnavi 2010
        whole-cortex mean for older adults.
    alpha_amy : amyloid–metabolism coupling coefficient.  Negative.
        Default -0.10 → 10% CMRO2 drop per unit BPnd, consistent with
        Vaishnavi 2010 figure 3 fits across cortical regions.
    beta_flow : SUVR flow-bias coefficient (shared with BPnd model)
    oef_prior : weak prior on OEF; typical adult is 0.40
    cao2 : arterial O₂ content in µmol/mL (computed from default Fick
        params if None)
    """
    cmro2_baseline: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(160.0)
    )
    alpha_amy: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(-0.10)
    )
    beta_flow: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.0)
    )
    oef_prior: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.40)
    )
    cao2: Float[Array, "..."] | None = None


def forward_joint_amyloid_cmro2(
    bpnd: Float[Array, "R"],
    oef: Float[Array, "R"],
    cbf: Float[Array, "R"],
    cbf_ref: Float[Array, ""] | float,
    params: JointAmyloidParams,
) -> dict[str, Float[Array, "R"]]:
    """Forward-predict SUVR and CMRO2 from latent BPnd, OEF, observed CBF.

    Returns
    -------
    Dict with keys ``suvr``, ``cmro2``, ``cmro2_coupling`` (the
    Vaishnavi-implied CMRO2 from amyloid alone — used as the prior
    target during inversion).
    """
    cao2 = params.cao2 if params.cao2 is not None else compute_cao2()

    suvr = forward_static_suvr(
        bpnd, cbf=cbf, cbf_ref=cbf_ref, beta_flow=params.beta_flow,
    )
    cmro2_fick = cbf * oef * cao2
    cmro2_coupling = params.cmro2_baseline * (1.0 + params.alpha_amy * bpnd)

    return {
        "suvr": suvr,
        "cmro2": cmro2_fick,
        "cmro2_coupling": cmro2_coupling,
    }


def fit_joint_amyloid_cmro2(
    suvr_obs: Float[np.ndarray, "R"],
    cbf_obs: Float[np.ndarray, "R"],
    cbf_ref: float,
    params: JointAmyloidParams | None = None,
    fit_alpha_amy: bool = False,
    fit_beta_flow: bool = True,
    lambda_couple: float = 0.5,
    lambda_oef: float = 1.0,
    n_steps: int = 500,
    learning_rate: float = 0.02,
) -> dict[str, np.ndarray]:
    """Recover (BPnd_r, OEF_r, β_flow, α_amy) given SUVR and CBF.

    Loss::

        L = MSE(SUVR_pred, SUVR_obs)
          + λ_couple · MSE(CMRO2_Fick, CMRO2_coupling)
          + λ_oef    · MSE(OEF, OEF_prior)

    The first term enforces consistency with the PET observation; the
    second is the Vaishnavi metabolic-coupling prior; the third keeps
    OEF in physiological range when no other constraint identifies it.

    The coupling residual is internally normalised by the CMRO2
    baseline (CMRO2_residual / CMRO2_baseline) so its squared scale is
    O(1), matching the SUVR residual.  ``λ_couple = 1`` therefore
    weights the two priors equally; setting it to 0 falls back to
    independent BPnd-only fitting.

    Parameters
    ----------
    suvr_obs, cbf_obs : per-region observations, shape (R,)
    cbf_ref : reference-region mean CBF (mL/100g/min)
    params : :class:`JointAmyloidParams`
    fit_alpha_amy : if True, learn the amyloid–CMRO2 slope rather than
        fix it at the literature default
    fit_beta_flow : if True, learn the SUVR flow-bias coefficient
    lambda_couple, lambda_oef : loss weights
    n_steps, learning_rate : optimiser controls

    Returns
    -------
    Dict::

        bpnd : (R,)
        oef : (R,)
        beta_flow : scalar
        alpha_amy : scalar
        suvr_predicted : (R,)
        cmro2_fick : (R,)        – µmol O₂/100g/min via Fick
        cmro2_coupling : (R,)    – Vaishnavi-implied CMRO2
        loss : scalar
    """
    if params is None:
        params = JointAmyloidParams()
    cao2 = params.cao2 if params.cao2 is not None else compute_cao2()

    suvr = jnp.asarray(suvr_obs)
    cbf = jnp.asarray(cbf_obs)
    R = suvr.shape[0]

    bpnd_init = suvr - 1.0
    oef_init = jnp.full((R,), float(params.oef_prior))
    beta_init = jnp.array(float(params.beta_flow))
    alpha_init = jnp.array(float(params.alpha_amy))

    # Theta layout: [bpnd_R, oef_R, beta_flow, alpha_amy]
    theta = jnp.concatenate([
        bpnd_init, oef_init, beta_init[None], alpha_init[None]
    ])

    def unpack(theta):
        bpnd = theta[:R]
        oef = jnp.clip(theta[R:2 * R], 0.05, 0.85)
        beta = theta[2 * R] if fit_beta_flow else jnp.array(float(params.beta_flow))
        alpha = theta[2 * R + 1] if fit_alpha_amy else jnp.array(float(params.alpha_amy))
        return bpnd, oef, beta, alpha

    oef_target = jnp.full((R,), float(params.oef_prior))

    def loss_fn(theta):
        bpnd, oef, beta, alpha = unpack(theta)

        suvr_pred = 1.0 + bpnd + beta * (cbf / cbf_ref - 1.0)
        cmro2_fick = cbf * oef * cao2
        cmro2_coupling = params.cmro2_baseline * (1.0 + alpha * bpnd)

        l_suvr = jnp.mean((suvr_pred - suvr) ** 2)
        # Normalise CMRO2 residual by baseline so squared scale is O(1).
        cmro2_resid = (cmro2_fick - cmro2_coupling) / params.cmro2_baseline
        l_couple = jnp.mean(cmro2_resid ** 2)
        l_oef = jnp.mean((oef - oef_target) ** 2)

        return l_suvr + lambda_couple * l_couple + lambda_oef * l_oef

    @jax.jit
    def step(theta):
        g = jax.grad(loss_fn)(theta)
        safe = jnp.all(jnp.isfinite(g))
        return jnp.where(safe, theta - learning_rate * g, theta)

    for _ in range(n_steps):
        theta = step(theta)

    bpnd_f, oef_f, beta_f, alpha_f = unpack(theta)
    fwd = forward_joint_amyloid_cmro2(
        bpnd_f, oef_f, cbf, cbf_ref,
        eqx.tree_at(
            lambda p: (p.beta_flow, p.alpha_amy),
            params,
            (beta_f, alpha_f),
        ),
    )
    final_loss = loss_fn(theta)

    return {
        "bpnd": np.asarray(bpnd_f),
        "oef": np.asarray(oef_f),
        "beta_flow": np.asarray(beta_f),
        "alpha_amy": np.asarray(alpha_f),
        "suvr_predicted": np.asarray(fwd["suvr"]),
        "cmro2_fick": np.asarray(fwd["cmro2"]),
        "cmro2_coupling": np.asarray(fwd["cmro2_coupling"]),
        "loss": np.asarray(final_loss),
    }
