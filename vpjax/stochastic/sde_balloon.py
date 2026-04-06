"""Stochastic Balloon-Windkessel model (SDE formulation).

Extends the deterministic Balloon ODE with additive state noise:

    dx = f(x, u, θ) dt + σ dW

where dW is a Wiener process. The neural and hemodynamic state
variables receive independent noise:
- σ_neural: noise on the vasodilatory signal s (neural variability)
- σ_hemo: noise on v, q (hemodynamic fluctuations)

This is the foundation of stochastic DCM (Friston 2011,
Li et al. 2011, Daunizeau et al. 2013).

References
----------
Friston KJ et al. (2011) NeuroImage 54:1218-1229
    "Generalised filtering and stochastic DCM for fMRI"
Li B et al. (2011) NeuroImage 58:339-349
    "Stochastic dynamic causal modelling of fMRI data"
Daunizeau J et al. (2013) NeuroImage 75:244-254
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax._types import BalloonParams, BalloonState
from vpjax.hemodynamics.balloon import BalloonWindkessel


class SDEBalloonParams(eqx.Module):
    """Parameters for the stochastic Balloon model.

    Inherits deterministic params and adds noise amplitudes.

    Attributes
    ----------
    kappa, gamma, tau, alpha, E0 : standard Balloon parameters
    sigma_neural : noise amplitude on vasodilatory signal s (a.u.)
    sigma_hemo : noise amplitude on hemodynamic states v, q (a.u.)
    """
    kappa: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.65))
    gamma: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.41))
    tau: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.98))
    alpha: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.32))
    E0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.34))
    sigma_neural: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.1))
    sigma_hemo: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.01))

    def to_balloon_params(self) -> BalloonParams:
        return BalloonParams(
            kappa=self.kappa, gamma=self.gamma,
            tau=self.tau, alpha=self.alpha, E0=self.E0,
        )


def sde_balloon_step(
    state: BalloonState,
    stimulus: Float[Array, "..."],
    params: SDEBalloonParams,
    key: jax.random.PRNGKey,
    dt: float = 0.01,
) -> BalloonState:
    """Single Euler-Maruyama step for the stochastic Balloon model.

    x_{n+1} = x_n + f(x_n, u) dt + σ √dt ξ

    Parameters
    ----------
    state : current BalloonState
    stimulus : neural input
    params : SDEBalloonParams
    key : JAX PRNG key
    dt : time step

    Returns
    -------
    Next BalloonState
    """
    model = BalloonWindkessel(params=params.to_balloon_params())
    dy = model(jnp.array(0.0), state, stimulus)

    # Generate noise
    k1, k2, k3, k4 = jax.random.split(key, 4)
    sqrt_dt = jnp.sqrt(dt)

    noise_s = params.sigma_neural * sqrt_dt * jax.random.normal(k1, shape=state.s.shape)
    noise_f = params.sigma_neural * sqrt_dt * jax.random.normal(k2, shape=state.f.shape) * 0.1
    noise_v = params.sigma_hemo * sqrt_dt * jax.random.normal(k3, shape=state.v.shape)
    noise_q = params.sigma_hemo * sqrt_dt * jax.random.normal(k4, shape=state.q.shape)

    return BalloonState(
        s=state.s + dt * dy.s + noise_s,
        f=jnp.clip(state.f + dt * dy.f + noise_f, 0.01, None),
        v=jnp.clip(state.v + dt * dy.v + noise_v, 0.01, None),
        q=jnp.clip(state.q + dt * dy.q + noise_q, 0.01, None),
    )


def sde_balloon_solve(
    params: SDEBalloonParams,
    stimulus: Float[Array, "T"],
    dt: float = 0.1,
    key: jax.random.PRNGKey = None,
) -> BalloonState:
    """Integrate the stochastic Balloon model over a stimulus.

    Parameters
    ----------
    params : SDEBalloonParams
    stimulus : neural stimulus (one per dt step)
    dt : time step
    key : PRNG key

    Returns
    -------
    BalloonState trajectory (arrays of shape (T,))
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n_steps = stimulus.shape[0]

    def scan_fn(carry, inputs):
        state, key = carry
        stim, = inputs
        key, subkey = jax.random.split(key)
        new_state = sde_balloon_step(state, stim, params, subkey, dt)
        return (new_state, key), (new_state.s, new_state.f, new_state.v, new_state.q)

    y0 = BalloonState.steady_state()
    _, (s_arr, f_arr, v_arr, q_arr) = jax.lax.scan(
        scan_fn, (y0, key), (stimulus,)
    )

    return BalloonState(s=s_arr, f=f_arr, v=v_arr, q=q_arr)
