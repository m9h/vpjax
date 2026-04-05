"""Locus coeruleus (LC) firing model for NREM sleep.

The LC is the primary source of norepinephrine in the brain. During
NREM sleep, LC firing shows stereotypic ~0.02 Hz oscillations:
periods of near-silence alternating with brief bursts. This drives
the NE oscillation that powers vasomotion and glymphatic clearance.

Firing rate hierarchy:
    Wake: ~3 Hz (tonic)
    N1: ~1.5 Hz (reduced)
    N2: ~0.8 Hz (further reduced, beginning to oscillate)
    N3: ~0.5 Hz mean, oscillating 0-1.5 Hz at ~0.02 Hz
    REM: ~0 Hz (nearly silent, except phasic bursts)

References
----------
Aston-Jones G, Cohen JD (2005) Annu Rev Neurosci 28:403-450
    "An integrative theory of locus coeruleus-norepinephrine function"
Hauglund NL et al. (2025) Cell 188:1-17
    "Norepinephrine-mediated slow vasomotion drives glymphatic clearance"
Takahashi K et al. (2010) Neuroscience 169:1115-1126
    "Locus coeruleus neuronal activity during the sleep-waking cycle"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax.sleep.nvc_state import WAKE, N1, N2, N3, REM


class LCParams(eqx.Module):
    """Parameters for locus coeruleus firing model.

    Attributes
    ----------
    firing_rate_wake : tonic LC firing rate during wake (Hz)
    firing_rate_n1   : LC rate during N1 (Hz)
    firing_rate_n2   : LC rate during N2 (Hz)
    firing_rate_n3   : mean LC rate during N3 (Hz), oscillates around this
    firing_rate_rem  : LC rate during REM (Hz), near zero
    oscillation_freq : frequency of LC burst-silence oscillation in N3 (Hz)
    oscillation_depth : modulation depth of the oscillation (0-1)
    ne_gain          : NE release per unit firing rate (a.u.)
    ne_tau           : NE clearance time constant (s)
    """
    firing_rate_wake: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(3.0))
    firing_rate_n1: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.5))
    firing_rate_n2: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.8))
    firing_rate_n3: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.5))
    firing_rate_rem: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.05))
    oscillation_freq: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.02))
    oscillation_depth: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.9))
    ne_gain: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))
    ne_tau: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(5.0))


# Firing rate lookup
_FIRING_RATE_ATTR = {
    WAKE: "firing_rate_wake",
    N1: "firing_rate_n1",
    N2: "firing_rate_n2",
    N3: "firing_rate_n3",
    REM: "firing_rate_rem",
}


def lc_firing_rate(
    stage: int,
    params: LCParams | None = None,
) -> float:
    """Get mean LC firing rate for a sleep stage.

    Parameters
    ----------
    stage : sleep stage
    params : LCParams

    Returns
    -------
    Firing rate (Hz)
    """
    if params is None:
        params = LCParams()
    attr = _FIRING_RATE_ATTR.get(stage, "firing_rate_wake")
    return float(getattr(params, attr))


def lc_firing_timecourse(
    t: Float[Array, "T"],
    stage: int = N3,
    params: LCParams | None = None,
) -> Float[Array, "T"]:
    """Generate LC firing rate time course.

    During N3, LC firing oscillates: near-silence alternating with
    brief bursts at ~0.02 Hz. In wake, it's tonic (constant).

    Parameters
    ----------
    t : time points (s)
    stage : sleep stage
    params : LCParams

    Returns
    -------
    LC firing rate (Hz) over time
    """
    if params is None:
        params = LCParams()

    attr = _FIRING_RATE_ATTR.get(stage, "firing_rate_wake")
    mean_rate = getattr(params, attr)

    # Oscillation depth depends on stage (strongest in N3)
    depth_scale = {WAKE: 0.05, N1: 0.2, N2: 0.5, N3: 1.0, REM: 0.1}
    depth = params.oscillation_depth * depth_scale.get(stage, 0.0)

    # Asymmetric oscillation: brief bursts (high rate) with longer silent periods
    phase = 2 * jnp.pi * params.oscillation_freq * t
    # Use a rectified sinusoid for burst-silence pattern
    burst = jnp.clip(jnp.sin(phase), 0.0, None) ** 0.5  # sharp bursts
    modulation = 1.0 - depth + depth * 2.0 * burst

    firing = mean_rate * modulation
    return jnp.clip(firing, 0.0, None)


def lc_to_norepinephrine(
    t: Float[Array, "T"],
    stage: int = N3,
    params: LCParams | None = None,
) -> Float[Array, "T"]:
    """Convert LC firing to NE concentration via first-order kinetics.

    d[NE]/dt = gain × firing_rate - [NE] / τ_NE

    Computed via convolution with exponential kernel for efficiency.

    Parameters
    ----------
    t : time points (s)
    stage : sleep stage
    params : LCParams

    Returns
    -------
    NE concentration (a.u., centered around stage mean)
    """
    if params is None:
        params = LCParams()

    firing = lc_firing_timecourse(t, stage, params)

    # Exponential smoothing (first-order filter)
    dt = t[1] - t[0]
    alpha = 1.0 - jnp.exp(-dt / params.ne_tau)

    import jax

    def scan_fn(ne_prev, firing_i):
        ne_drive = params.ne_gain * firing_i
        ne = ne_prev + alpha * (ne_drive - ne_prev)
        return ne, ne

    ne_init = params.ne_gain * getattr(params, _FIRING_RATE_ATTR.get(stage, "firing_rate_wake"))
    _, ne_trace = jax.lax.scan(scan_fn, ne_init, firing)

    # Center around mean
    return ne_trace - jnp.mean(ne_trace)
