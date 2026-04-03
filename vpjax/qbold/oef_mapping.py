"""Per-voxel OEF mapping from multi-echo GRE data.

Fits the qBOLD signal model to multi-echo magnitude data to estimate
OEF and DBV per voxel.  Uses JAX-differentiable optimization so that
OEF maps can be embedded in larger differentiable pipelines.

The key relationship:
    S(TE) = S₀ · exp(-R₂·TE) · exp(-DBV · g(δω·TE))

Given data at multiple TEs, we fit {S₀, R₂, OEF, DBV} per voxel.

References
----------
He X, Yablonskiy DA (2007) MRM 57:115-126
An H, Lin W (2003) MRM 50:708-716
Christen T et al. (2012) NeuroImage 60:582-591
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax.qbold.signal_model import QBOLDParams, qbold_signal


def log_signal_residuals(
    theta: Float[Array, "4"],
    data: Float[Array, "T"],
    te: Float[Array, "T"],
    params: QBOLDParams,
) -> Float[Array, ""]:
    """Compute sum-of-squares residuals in log domain for one voxel.

    Parameters
    ----------
    theta : [log(S0), R2, OEF, DBV] — parameter vector
    data : measured signal at each TE
    te : echo times (s)
    params : QBOLDParams (B0, Hct held fixed)

    Returns
    -------
    Sum of squared log-signal residuals.
    """
    log_s0 = theta[0]
    r2 = theta[1]
    oef = jnp.clip(theta[2], 0.01, 0.99)
    dbv = jnp.clip(theta[3], 0.001, 0.15)

    p = QBOLDParams(B0=params.B0, Hct=params.Hct, R2t=r2, S0=jnp.exp(log_s0))
    model = qbold_signal(te, oef, dbv, p)

    # Residuals in log domain (more numerically stable for exponential decay)
    log_model = jnp.log(jnp.clip(model, 1e-10, None))
    log_data = jnp.log(jnp.clip(data, 1e-10, None))

    return jnp.sum((log_model - log_data) ** 2)


def fit_oef_voxel(
    data: Float[Array, "T"],
    te: Float[Array, "T"],
    params: QBOLDParams | None = None,
    n_steps: int = 200,
    learning_rate: float = 1e-3,
) -> dict[str, Float[Array, ""]]:
    """Fit OEF and DBV for a single voxel using gradient descent.

    Uses a simple differentiable optimization loop (Adam-like)
    to fit the qBOLD model parameters.

    Parameters
    ----------
    data : measured multi-echo signal, shape (T,)
    te : echo times in seconds, shape (T,)
    params : QBOLDParams (default: 3T)
    n_steps : number of optimization steps
    learning_rate : step size

    Returns
    -------
    Dict with keys: 's0', 'r2', 'oef', 'dbv', 'r2prime', 'loss'
    """
    if params is None:
        params = QBOLDParams()

    # Initialize from data: S0 ~ data[0], R2 from log-linear fit
    s0_init = data[0]
    # Simple log-linear R2* estimate from first and last echo
    r2star_init = -jnp.log(jnp.clip(data[-1] / data[0], 1e-6, None)) / (te[-1] - te[0])

    theta = jnp.array([
        jnp.log(jnp.clip(s0_init, 1e-6, None)),  # log(S0)
        jnp.clip(r2star_init * 0.8, 5.0, 50.0),    # R2 (tissue, ~80% of R2*)
        0.30,                                        # OEF initial
        0.03,                                        # DBV initial
    ])

    grad_fn = jax.grad(log_signal_residuals)

    # Simple gradient descent with momentum
    velocity = jnp.zeros_like(theta)
    beta = 0.9

    def step(carry, _):
        theta, velocity = carry
        g = grad_fn(theta, data, te, params)
        velocity = beta * velocity + (1 - beta) * g
        theta = theta - learning_rate * velocity
        return (theta, velocity), None

    (theta, _), _ = jax.lax.scan(step, (theta, velocity), None, length=n_steps)

    oef = jnp.clip(theta[2], 0.01, 0.99)
    dbv = jnp.clip(theta[3], 0.001, 0.15)
    r2 = theta[1]
    s0 = jnp.exp(theta[0])
    loss = log_signal_residuals(theta, data, te, params)

    from vpjax.qbold.signal_model import compute_r2prime
    r2p = compute_r2prime(oef, dbv, params)

    return {
        "s0": s0,
        "r2": r2,
        "oef": oef,
        "dbv": dbv,
        "r2prime": r2p,
        "loss": loss,
    }


def fit_oef_volume(
    data: Float[Array, "N T"],
    te: Float[Array, "T"],
    params: QBOLDParams | None = None,
    n_steps: int = 200,
    learning_rate: float = 1e-3,
) -> dict[str, Float[Array, "N"]]:
    """Fit OEF and DBV for a volume of voxels using vmap.

    Parameters
    ----------
    data : multi-echo signal, shape (N, T) where N is number of voxels
    te : echo times in seconds, shape (T,)
    params : QBOLDParams (default: 3T)
    n_steps : optimization steps per voxel
    learning_rate : step size

    Returns
    -------
    Dict with keys: 's0', 'r2', 'oef', 'dbv', 'r2prime', 'loss'
    Each value has shape (N,).
    """
    fit_fn = jax.vmap(
        lambda d: fit_oef_voxel(d, te, params, n_steps, learning_rate)
    )
    return fit_fn(data)
