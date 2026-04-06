"""Noise-driven sleep state transitions.

Models sleep as a stochastic process in a multi-well potential
landscape, where noise drives transitions between attractor states
(Wake, NREM, REM).

The dynamics follow a Langevin equation:
    dx = -∂V/∂x dt + σ dW

where V(x) is the sleep potential energy landscape with minima at:
    x ≈ 0: Wake
    x ≈ 1: NREM (deep sleep)
    x ≈ 2: REM

Transition rates follow Kramers' escape rate:
    r = (ω_min × ω_max / 2π) × exp(-ΔV / σ²)

This connects the noise amplitude (modulated by circadian/homeostatic
drive) to the probability of sleep stage transitions, producing
realistic hypnogram-like dynamics.

References
----------
Lo CC et al. (2013) PNAS 110:10467-10472
    "Common scale-invariant patterns of sleep-wake transitions"
Fultz NE et al. (2019) Science 366:628-631
Phillips AJK, Robinson PA (2007) J Biological Rhythms 22:167-179
    "A quantitative model of sleep-wake dynamics based on the
    physiology of the brainstem ascending arousal system"
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def sleep_potential(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Multi-well potential landscape for sleep states.

    Three local minima:
        x ≈ 0: Wake (shallowest well — easiest to leave)
        x ≈ 1: NREM (deepest well — most stable during night)
        x ≈ 2: REM (moderate well)

    V(x) = product of quadratic wells centered at 0, 1, 2

    Parameters
    ----------
    x : state variable (continuous sleep depth)

    Returns
    -------
    Potential energy V(x)
    """
    # Three-well potential: minima at x=0, x=1, x=2
    # V(x) = (x)(x-1)(x-2) squared, scaled
    # This gives zeros (minima) at exactly 0, 1, 2
    well = x * (x - 1.0) * (x - 2.0)
    # Add slight asymmetry: wake well shallower, NREM deepest
    asymmetry = -0.05 * (x - 1.0)
    return well ** 2 + asymmetry


def kramers_transition_rate(
    barrier_height: Float[Array, "..."],
    noise: Float[Array, "..."],
    omega_prefactor: float = 1.0,
) -> Float[Array, "..."]:
    """Kramers escape rate from a potential well.

    r = ω × exp(-��V / σ²)

    Parameters
    ----------
    barrier_height : energy barrier ΔV
    noise : noise amplitude σ
    omega_prefactor : attempt frequency

    Returns
    -------
    Transition rate (1/s)
    """
    noise_sq = jnp.clip(noise ** 2, 1e-10, None)
    return omega_prefactor * jnp.exp(-barrier_height / noise_sq)


def simulate_sleep_states(
    t: Float[Array, "T"],
    key: jax.random.PRNGKey,
    noise: float = 0.4,
    x0: float = 0.0,
) -> Float[Array, "T"]:
    """Simulate sleep state trajectory via overdamped Langevin dynamics.

    dx = -∂V/��x dt + σ dW

    Parameters
    ----------
    t : time points (s)
    key : PRNG key
    noise : noise amplitude σ
    x0 : initial state (0=wake)

    Returns
    -------
    State trajectory x(t), continuous values where
    ~0=Wake, ~1=NREM, ~2=REM
    """
    dt_arr = jnp.diff(t)

    def scan_fn(carry, inputs):
        x, key = carry
        dt, = inputs
        key, subkey = jax.random.split(key)

        # Gradient of potential (force)
        dVdx = jax.grad(sleep_potential)(x)

        # Langevin step
        sqrt_dt = jnp.sqrt(dt)
        xi = jax.random.normal(subkey)
        x_new = x - dVdx * dt + noise * sqrt_dt * xi

        # Soft boundaries (reflect at 0 and 2.5)
        x_new = jnp.clip(x_new, -0.5, 2.5)

        return (x_new, key), x_new

    (_, _), trajectory = jax.lax.scan(scan_fn, (jnp.array(x0), key), (dt_arr,))

    # Prepend initial state
    return jnp.concatenate([jnp.array([x0]), trajectory])
