"""Residual error models for concentration and effect data.

Concentration measurements are not homoscedastic: assay error grows
roughly in proportion to the concentration over most of the range, with
an additive floor near the limit of quantification.  The combined model

    y = f(theta) + sqrt(sigma_add^2 + (sigma_prop f(theta))^2) * eps,
    eps ~ N(0, 1)

is therefore the default in pharmacometrics, and getting it wrong is
not a cosmetic matter — a plain additive assumption lets the highest
concentrations dominate the fit and biases clearance.

Two alternatives are provided because the literature uses both: fitting
on the log scale, which is appropriate when the error really is
multiplicative and the data span orders of magnitude, and a pure
proportional model.

References
----------
Beal SL (2001) J Pharmacokinet Pharmacodyn 28:481-504
    "Ways to fit a PK model with some data below the quantification limit"
Mould DR, Upton RN (2012) CPT Pharmacometrics Syst Pharmacol 1:e6
    "Basic concepts in population modeling, simulation, and model-based drug
    development"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class ResidualErrorParams(eqx.Module):
    """Combined additive-plus-proportional residual error.

    Attributes
    ----------
    sigma_add  : additive standard deviation, in the units of the data
    sigma_prop : proportional standard deviation, as a fraction
    """

    sigma_add: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.01)
    )
    sigma_prop: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.1)
    )


def residual_sd(
    pred: Float[Array, "N"],
    params: ResidualErrorParams | None = None,
) -> Float[Array, "N"]:
    """Predicted standard deviation at each observation."""
    if params is None:
        params = ResidualErrorParams()
    return jnp.sqrt(params.sigma_add**2 + (params.sigma_prop * pred) ** 2)


def weighted_residuals(
    pred: Float[Array, "N"],
    obs: Float[Array, "N"],
    params: ResidualErrorParams | None = None,
) -> Float[Array, "N"]:
    """Residuals scaled by their predicted standard deviation.

    This is the vector a least-squares solver should minimise: with the
    combined error model, unweighted residuals correspond to the wrong
    likelihood.  The scaling uses the *predicted* value rather than the
    observed one, which is the standard choice and keeps the weights
    differentiable in the parameters.
    """
    return (obs - pred) / residual_sd(pred, params)


def log_likelihood(
    pred: Float[Array, "N"],
    obs: Float[Array, "N"],
    params: ResidualErrorParams | None = None,
) -> Float[Array, ""]:
    """Gaussian log-likelihood under the combined error model."""
    sd = residual_sd(pred, params)
    z = (obs - pred) / sd
    return jnp.sum(-0.5 * z**2 - jnp.log(sd) - 0.5 * jnp.log(2.0 * jnp.pi))


def log_scale_residuals(
    pred: Float[Array, "N"],
    obs: Float[Array, "N"],
    sigma: Float[Array, ""] | float = 0.1,
    floor: float = 1e-12,
) -> Float[Array, "N"]:
    """Residuals of ``log(obs)`` against ``log(pred)``.

    Appropriate when the data span several orders of magnitude and the
    error is believed to be multiplicative throughout.  Observations at
    or below zero cannot be represented on this scale; *floor* keeps the
    logarithm finite rather than silently dropping them, so check for
    non-positive values before using this model.
    """
    return (jnp.log(jnp.maximum(obs, floor)) - jnp.log(jnp.maximum(pred, floor))) / sigma
