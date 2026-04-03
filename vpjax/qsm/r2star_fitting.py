"""R2* fitting from multi-echo GRE magnitude data.

Fits the mono-exponential decay model:
    S(TE) = S₀ · exp(-R₂* · TE)

to multi-echo gradient-echo magnitude data.  R₂* = 1/T₂* maps are
used in iron/myelin quantification and as input to the qBOLD model.

Methods:
1. Log-linear (fast, closed-form)
2. Nonlinear least squares (more accurate, differentiable)

References
----------
Pei M et al. (2015) MRM 73:2190-2197
Weiskopf N et al. (2014) Front Neurosci 8:278
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def fit_r2star_loglinear(
    signal: Float[Array, "... T"],
    te: Float[Array, "T"],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Fit R₂* using log-linear regression (fast, closed-form).

    log(S(TE)) = log(S₀) - R₂* · TE

    This is a simple linear regression in log-space.

    Parameters
    ----------
    signal : multi-echo magnitude data, shape (..., T)
    te : echo times in seconds, shape (T,)

    Returns
    -------
    r2star : R₂* map (s⁻¹), shape (...)
    s0 : S₀ map, shape (...)
    """
    # Clamp to avoid log(0)
    log_s = jnp.log(jnp.clip(signal, 1e-10, None))

    # Linear regression: y = a + b*x where y=log(S), x=TE, b=-R2*
    n = te.shape[0]
    te_mean = jnp.mean(te)
    log_s_mean = jnp.mean(log_s, axis=-1)

    # Covariance and variance
    te_centered = te - te_mean  # (T,)
    log_s_centered = log_s - log_s_mean[..., None]  # (..., T)

    cov = jnp.sum(log_s_centered * te_centered, axis=-1)  # (...)
    var_te = jnp.sum(te_centered ** 2)  # scalar

    slope = cov / jnp.where(var_te > 1e-12, var_te, 1e-12)
    intercept = log_s_mean - slope * te_mean

    r2star = -slope
    s0 = jnp.exp(intercept)

    # Clamp R2* to physical range
    r2star = jnp.clip(r2star, 0.0, 500.0)

    return r2star, s0


def fit_r2star_nonlinear(
    signal: Float[Array, "T"],
    te: Float[Array, "T"],
    n_steps: int = 100,
    learning_rate: float = 0.01,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Fit R₂* using gradient-based nonlinear optimization (single voxel).

    Minimizes: Σ_t (S(t) - S₀·exp(-R₂*·TE_t))²

    Initialized from log-linear estimate for fast convergence.

    Parameters
    ----------
    signal : multi-echo magnitude, shape (T,)
    te : echo times, shape (T,)
    n_steps : optimization steps
    learning_rate : step size

    Returns
    -------
    r2star : R₂* (s⁻¹)
    s0 : S₀
    """
    # Initialize from log-linear
    r2_init, s0_init = fit_r2star_loglinear(signal[None, :], te)
    theta = jnp.array([jnp.log(jnp.clip(s0_init[0], 1e-6, None)), r2_init[0]])

    def loss(theta):
        s0 = jnp.exp(theta[0])
        r2 = jnp.clip(theta[1], 0.0, 500.0)
        predicted = s0 * jnp.exp(-r2 * te)
        return jnp.sum((signal - predicted) ** 2)

    grad_fn = jax.grad(loss)

    def step(theta, _):
        g = grad_fn(theta)
        theta = theta - learning_rate * g
        return theta, None

    theta, _ = jax.lax.scan(step, theta, None, length=n_steps)

    s0 = jnp.exp(theta[0])
    r2star = jnp.clip(theta[1], 0.0, 500.0)
    return r2star, s0


def fit_r2star_volume(
    signal: Float[Array, "N T"],
    te: Float[Array, "T"],
) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
    """Fit R₂* for a volume of voxels using log-linear method (vmapped).

    Parameters
    ----------
    signal : multi-echo data, shape (N, T)
    te : echo times, shape (T,)

    Returns
    -------
    r2star : R₂* per voxel, shape (N,)
    s0 : S₀ per voxel, shape (N,)
    """
    return fit_r2star_loglinear(signal, te)


def multi_echo_combine(
    signal: Float[Array, "... T"],
    te: Float[Array, "T"],
    method: str = "t2star_weighted",
) -> Float[Array, "..."]:
    """Combine multi-echo data into a single image.

    Parameters
    ----------
    signal : multi-echo magnitude, shape (..., T)
    te : echo times, shape (T,)
    method : combination method
        - 'mean': simple average
        - 'first': first echo only
        - 't2star_weighted': optimal T₂*-weighted combination

    Returns
    -------
    Combined image, shape (...)
    """
    if method == "mean":
        return jnp.mean(signal, axis=-1)
    elif method == "first":
        return signal[..., 0]
    elif method == "t2star_weighted":
        # Weight by TE (longer TEs have more T2* contrast but lower SNR)
        # Optimal: weight by TE × S(TE)
        weights = te * signal
        weights_sum = jnp.sum(weights, axis=-1)
        signal_sum = jnp.sum(signal * weights, axis=-1)
        total = jnp.where(jnp.abs(weights_sum) > 1e-10, weights_sum, 1e-10)
        return signal_sum / total
    else:
        raise ValueError(f"Unknown combination method: {method}")
