"""Cerebral autoregulation.

Cerebral autoregulation maintains nearly constant CBF despite changes
in cerebral perfusion pressure (CPP = MAP - ICP).  The classic Lassen
curve shows a plateau from ~60-150 mmHg MAP.

Static autoregulation: the steady-state CBF-CPP curve.
Dynamic autoregulation: the time-domain response (first-order lag).

References
----------
Lassen NA (1959) Physiol Rev 39:183-238
    "Cerebral blood flow and oxygen consumption in man"
Aaslid R et al. (1989) Stroke 20:45-52
    "Cerebral autoregulation dynamics in humans"
Tiecks FP et al. (1995) Stroke 26:1014-1019
    "Comparison of static and dynamic cerebral autoregulation measurements"
Panerai RB (2008) Cardiovascular Engineering 8:42-59
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class AutoregParams(eqx.Module):
    """Parameters for cerebral autoregulation model.

    Attributes
    ----------
    tau        : autoregulatory time constant (s), default ~15
    gain       : strength of autoregulation (0=none, 1=perfect)
    cpp_lower  : lower limit of autoregulation plateau (mmHg)
    cpp_upper  : upper limit of autoregulation plateau (mmHg)
    cpp_target : target CPP (mmHg), center of plateau
    cbf0       : baseline CBF (mL/100g/min)
    """
    tau: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(15.0))
    gain: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.9))
    cpp_lower: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(60.0))
    cpp_upper: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(150.0))
    cpp_target: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(80.0))
    cbf0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(50.0))


def static_autoregulation(
    cpp: Float[Array, "..."],
    params: AutoregParams | None = None,
) -> Float[Array, "..."]:
    """Compute static autoregulation curve (Lassen curve).

    Within the autoregulatory range [cpp_lower, cpp_upper], CBF
    remains approximately constant.  Outside this range, CBF
    changes linearly (passively) with pressure.

    A smooth sigmoid approximation is used instead of hard cutoffs
    to maintain differentiability.

    Parameters
    ----------
    cpp : cerebral perfusion pressure (mmHg)
    params : AutoregParams

    Returns
    -------
    CBF/CBF₀ (fractional, 1.0 at target CPP)
    """
    if params is None:
        params = AutoregParams()

    # Passive flow: linear with pressure
    passive_ratio = cpp / params.cpp_target

    # Regulated flow: ~1.0 within plateau
    # Use a smooth blend: CBF = CBF₀ within range, linear outside
    # Sigmoid-based smooth transition at boundaries
    k = 0.15  # sharpness of transition (1/mmHg)

    # Below lower limit: passive
    below = jax_sigmoid((params.cpp_lower - cpp) * k)
    # Above upper limit: passive
    above = jax_sigmoid((cpp - params.cpp_upper) * k)

    # Within plateau: regulated at 1.0
    # Outside: passive flow
    regulated = 1.0
    cbf_ratio = (
        below * (passive_ratio * params.cpp_target / params.cpp_lower)
        + above * (passive_ratio * params.cpp_target / params.cpp_upper)
        + (1.0 - below - above) * regulated
    )

    # Blend with gain (0=no autoregulation, 1=perfect)
    cbf_ratio = params.gain * cbf_ratio + (1.0 - params.gain) * passive_ratio

    return cbf_ratio


def dynamic_autoreg(
    cpp: Float[Array, "..."],
    x_autoreg: Float[Array, "..."],
    params: AutoregParams | None = None,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Dynamic autoregulation as a first-order system.

    The autoregulatory response has a latency (~5-15s) modeled as:
        dx/dt = (target - x) / τ

    where target is the steady-state autoregulatory response.

    Parameters
    ----------
    cpp : current cerebral perfusion pressure (mmHg)
    x_autoreg : current autoregulatory state (CBF ratio)
    params : AutoregParams

    Returns
    -------
    dx_autoreg : time derivative of autoregulatory state
    cbf_ratio : current regulated CBF/CBF₀
    """
    if params is None:
        params = AutoregParams()

    target = static_autoregulation(cpp, params)
    dx = (target - x_autoreg) / params.tau

    return dx, x_autoreg


def autoregulation_index(
    cbf_change: Float[Array, "..."],
    cpp_change: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Compute the static autoregulation index (sARI).

    sARI = 1 - (ΔCBF/CBF₀) / (ΔCPP/CPP₀)

    sARI = 0: no autoregulation (passive flow)
    sARI = 1: perfect autoregulation (constant CBF)

    Parameters
    ----------
    cbf_change : fractional CBF change (ΔCBF/CBF₀)
    cpp_change : fractional CPP change (ΔCPP/CPP₀)

    Returns
    -------
    Autoregulation index (0-1)
    """
    cpp_safe = jnp.where(jnp.abs(cpp_change) > 1e-6, cpp_change, 1e-6)
    ari = 1.0 - cbf_change / cpp_safe
    return jnp.clip(ari, 0.0, 1.0)


def jax_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Numerically stable sigmoid."""
    return jnp.where(
        x >= 0,
        1.0 / (1.0 + jnp.exp(-x)),
        jnp.exp(x) / (1.0 + jnp.exp(x)),
    )
