"""Fokker-Planck evolution of hemodynamic population density.

Under the Laplace assumption (Marreiros et al. 2009), the population
density of hemodynamic states is Gaussian, described by mean μ and
covariance Σ:

    dμ/dt = f(μ, u, θ)                    (mean follows ODE)
    dΣ/dt = J·Σ + Σ·Jᵀ + D               (covariance: Lyapunov equation)

where J = ∂f/∂x|_{x=μ} is the Jacobian and D is the diffusion matrix.

This gives the evolution of uncertainty in hemodynamic states, which:
- Encodes information about neural noise structure
- Can distinguish direct from indirect connectivity (stochastic DCM)
- Models state transitions in sleep (covariance grows near bifurcations)

References
----------
Marreiros AC et al. (2009) NeuroImage 44:701-714
    "Population dynamics under the Laplace assumption"
Friston KJ et al. (2011) NeuroImage 54:1218-1229
    "Generalised filtering and stochastic DCM for fMRI"
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax._types import BalloonParams, BalloonState
from vpjax.hemodynamics.balloon import BalloonWindkessel


class FPParams(eqx.Module):
    """Parameters for Fokker-Planck hemodynamic evolution.

    Attributes
    ----------
    balloon : underlying deterministic Balloon parameters
    D_neural : diffusion coefficient for neural state (s)
    D_hemo : diffusion coefficient for hemodynamic states (v, q)
    """
    balloon: BalloonParams = eqx.field(default_factory=BalloonParams)
    D_neural: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.01))
    D_hemo: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.001))


class FPState(eqx.Module):
    """Fokker-Planck state: mean + covariance of Balloon states.

    State vector: [s, f, v, q] (4D)

    Attributes
    ----------
    mean : mean state vector (4,)
    cov : covariance matrix (4, 4)
    """
    mean: Float[Array, "4"]
    cov: Float[Array, "4 4"]

    @staticmethod
    def from_balloon_steady_state() -> FPState:
        """Initialize at Balloon steady state with small covariance."""
        mean = jnp.array([0.0, 1.0, 1.0, 1.0])  # s, f, v, q
        cov = jnp.eye(4) * 1e-4  # small initial uncertainty
        return FPState(mean=mean, cov=cov)


def _balloon_rhs_flat(x: Float[Array, "4"], u: Float[Array, ""], params: BalloonParams) -> Float[Array, "4"]:
    """Balloon RHS operating on flat state vector."""
    state = BalloonState(s=x[0], f=x[1], v=x[2], q=x[3])
    model = BalloonWindkessel(params=params)
    dy = model(jnp.array(0.0), state, u)
    return jnp.array([dy.s, dy.f, dy.v, dy.q])


def fp_step(
    state: FPState,
    stimulus: Float[Array, ""],
    params: FPParams,
    dt: float = 0.1,
) -> FPState:
    """Single Fokker-Planck step: propagate mean and covariance.

    dμ/dt = f(μ, u)
    dΣ/dt = J·Σ + Σ·Jᵀ + D

    Parameters
    ----------
    state : current FPState (mean, covariance)
    stimulus : neural input
    params : FPParams
    dt : time step

    Returns
    -------
    Updated FPState
    """
    # Mean evolution (deterministic ODE)
    f_mean = _balloon_rhs_flat(state.mean, stimulus, params.balloon)
    new_mean = state.mean + dt * f_mean

    # Jacobian at current mean
    J = jax.jacobian(lambda x: _balloon_rhs_flat(x, stimulus, params.balloon))(state.mean)

    # Diffusion matrix
    D = jnp.diag(jnp.array([
        params.D_neural,       # s: neural noise
        params.D_neural * 0.1, # f: flow noise (smaller)
        params.D_hemo,         # v: volume noise
        params.D_hemo,         # q: dHb noise
    ]))

    # Covariance evolution: dΣ/dt = JΣ + ΣJᵀ + D
    dSigma = J @ state.cov + state.cov @ J.T + D
    new_cov = state.cov + dt * dSigma

    # Symmetrize (numerical stability)
    new_cov = (new_cov + new_cov.T) / 2.0

    # Ensure positive semi-definite (clip eigenvalues)
    eigvals, eigvecs = jnp.linalg.eigh(new_cov)
    eigvals = jnp.clip(eigvals, 1e-8, None)
    new_cov = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

    return FPState(mean=new_mean, cov=new_cov)
