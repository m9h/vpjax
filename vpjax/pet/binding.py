"""Binding-potential models for static-frame PET.

For dynamic PET the Logan reference-tissue model gives::

    DVR = SUVR / (1 + 1/(k2_ref · t)) → 1 + BPnd  as t → ∞

For a single late frame acquired in equilibrium, this collapses to::

    SUVR ≈ 1 + BPnd                        (no flow bias)

The simple SUVR ≈ 1 + BPnd identity ignores a known source of bias:
when the target region's blood flow differs substantially from the
reference region, the late-frame uptake is shifted by the relative
delivery rate, so apparent SUVR confounds binding with perfusion.
The flow-corrected forward model used here is::

    SUVR = 1 + BPnd + β_flow · (CBF / CBF_ref - 1)

with ``CBF_ref`` taken as the mean CBF across voxels in the reference
region.  ``β_flow`` is small (~0.1) for amyloid PET and is one of the
parameters fit per region.

The fitter is a JAX gradient descent — the model is linear in
(BPnd, β_flow) so this converges to the OLS solution, but writing it
as a differentiable forward model lets us later swap in nonlinear
joint-modality terms (see :mod:`vpjax.pet.joint`).

References
----------
Logan J et al. (1996) JCBFM 16:834-840
Chen Y et al. (2015) JCBFM 35:1104-1110
    "Fast quantification of amyloid PET ... flow-bias correction"
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


class BindingPotentialParams(eqx.Module):
    """Parameters for the static-SUVR binding-potential model.

    Attributes
    ----------
    bpnd : non-displaceable binding potential (per region)
    beta_flow : SUVR sensitivity to relative CBF deviation
        (Chen 2015 reports |β_flow| ~ 0.05-0.2 for amyloid)
    """
    bpnd: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.0))
    beta_flow: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.0))


def forward_static_suvr(
    bpnd: Float[Array, "..."],
    cbf: Float[Array, "..."] | None = None,
    cbf_ref: Float[Array, "..."] | float = 50.0,
    beta_flow: Float[Array, "..."] | float = 0.0,
) -> Float[Array, "..."]:
    """Static-frame SUVR forward model with optional flow correction.

    Parameters
    ----------
    bpnd : binding potential (any shape)
    cbf : matching CBF (mL/100g/min); if None, the flow term is skipped
    cbf_ref : reference-region mean CBF (mL/100g/min)
    beta_flow : flow-bias coefficient

    Returns
    -------
    SUVR predicted by the static model, same shape as ``bpnd``.
    """
    suvr = 1.0 + bpnd
    if cbf is not None:
        suvr = suvr + beta_flow * (cbf / cbf_ref - 1.0)
    return suvr


def fit_bpnd_per_region(
    suvr_obs: Float[np.ndarray, "R"],
    cbf_obs: Float[np.ndarray, "R"] | None = None,
    cbf_ref: float = 50.0,
    fit_flow: bool = True,
    n_steps: int = 300,
    learning_rate: float = 0.05,
) -> dict[str, np.ndarray]:
    """Recover (BPnd, β_flow) per region from observed SUVR and CBF.

    With ``fit_flow=False`` this is just ``BPnd = SUVR - 1``; the
    differentiable path exists so the same machinery composes with the
    joint amyloid–CMRO2 model.

    The β_flow coefficient is a *single* scalar shared across regions
    (it is a property of the tracer, not the tissue), so we fit one
    global β_flow plus per-region BPnd.

    Parameters
    ----------
    suvr_obs : observed SUVR per region, shape (R,)
    cbf_obs : optional observed CBF per region, shape (R,)
    cbf_ref : reference CBF (mL/100g/min) — typically cerebellum mean
    fit_flow : if True and cbf_obs given, fit β_flow jointly
    n_steps : gradient descent iterations
    learning_rate : step size

    Returns
    -------
    Dict with::

        bpnd : np.ndarray (R,)
        beta_flow : np.ndarray scalar
        suvr_predicted : np.ndarray (R,)
        loss : np.ndarray scalar
    """
    suvr = jnp.asarray(suvr_obs)
    use_flow = cbf_obs is not None and fit_flow
    cbf = jnp.asarray(cbf_obs) if cbf_obs is not None else None
    R = suvr.shape[0]

    bpnd_init = suvr - 1.0
    beta_init = jnp.array(0.0)
    theta = jnp.concatenate([bpnd_init, beta_init[None]])

    def unpack(theta):
        return theta[:R], theta[R]

    def loss_fn(theta):
        bpnd, beta = unpack(theta)
        if use_flow:
            pred = forward_static_suvr(
                bpnd, cbf=cbf, cbf_ref=cbf_ref, beta_flow=beta,
            )
        else:
            pred = 1.0 + bpnd
        return jnp.mean((pred - suvr) ** 2)

    @jax.jit
    def step(theta):
        g = jax.grad(loss_fn)(theta)
        return theta - learning_rate * g

    for _ in range(n_steps):
        theta = step(theta)

    bpnd_final, beta_final = unpack(theta)
    if use_flow:
        suvr_pred = forward_static_suvr(
            bpnd_final, cbf=cbf, cbf_ref=cbf_ref, beta_flow=beta_final,
        )
    else:
        suvr_pred = 1.0 + bpnd_final
    final_loss = jnp.mean((suvr_pred - suvr) ** 2)

    return {
        "bpnd": np.asarray(bpnd_final),
        "beta_flow": np.asarray(beta_final),
        "suvr_predicted": np.asarray(suvr_pred),
        "loss": np.asarray(final_loss),
    }
