"""CSF flow coupling: CBV oscillations → glymphatic clearance.

Cerebral blood volume changes displace CSF through the perivascular
spaces (Virchow-Robin spaces), driving glymphatic clearance.  During
NREM sleep, the coordinated ~0.02 Hz vasomotion produces rhythmic
CSF pulsations that flush metabolic waste (including amyloid-β).

Model:
    dCBV/dt → CSF displacement (volume conservation)
    Cumulative |CSF flow| → glymphatic clearance metric

The coupling is approximately:
    CSF_flow ∝ -dCBV/dt  (volume conservation: blood in → CSF out)

References
----------
Fultz NE et al. (2019) Science 366:628-631
Hauglund NL et al. (2025) Cell 188:1-17
Iliff JJ et al. (2012) Science Translational Medicine 4:147ra111
    "A paravascular pathway facilitates CSF flow through the brain"
Nedergaard M (2013) Science 340:1529-1530
    "Garbage truck of the brain"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class CSFParams(eqx.Module):
    """Parameters for CSF flow coupling model.

    Attributes
    ----------
    coupling_gain : CSF flow per unit dCBV/dt (a.u.)
    csf_delay     : delay from CBV change to CSF response (s)
    tau_csf       : CSF flow time constant (s)
    clearance_efficiency : fraction of CSF flow contributing to clearance
    """
    coupling_gain: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(1.0)
    )
    csf_delay: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(1.0)
    )
    tau_csf: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(3.0)
    )
    clearance_efficiency: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.5)
    )


def csf_flow_from_cbv(
    cbv: Float[Array, "T"],
    params: CSFParams | None = None,
) -> Float[Array, "T"]:
    """Compute CSF flow from CBV time series.

    CSF flow ∝ -dCBV/dt (volume conservation)

    Blood volume increase → CSF pushed out (positive flow).

    Parameters
    ----------
    cbv : CBV/CBV₀ time series
    params : CSFParams

    Returns
    -------
    CSF flow (a.u., positive = outward)
    """
    if params is None:
        params = CSFParams()

    # Numerical derivative (forward difference, padded)
    dcbv = jnp.diff(cbv, prepend=cbv[0])
    csf = -params.coupling_gain * dcbv

    return csf


def csf_flow_from_cbv_delayed(
    cbv: Float[Array, "T"],
    t: Float[Array, "T"],
    params: CSFParams | None = None,
) -> Float[Array, "T"]:
    """Compute CSF flow with temporal delay and smoothing.

    Applies a first-order lag to the CBV-driven CSF flow:
        τ × d(csf)/dt = -coupling × dCBV/dt - csf

    Implemented as a convolution with exponential kernel.

    Parameters
    ----------
    cbv : CBV/CBV₀ time series
    t : time points (s)
    params : CSFParams

    Returns
    -------
    Delayed and smoothed CSF flow
    """
    if params is None:
        params = CSFParams()

    # Instantaneous CSF drive
    dcbv = jnp.diff(cbv, prepend=cbv[0])
    drive = -params.coupling_gain * dcbv

    # First-order exponential smoothing (causal filter)
    dt = jnp.diff(t, prepend=t[0] - (t[1] - t[0]))
    alpha = 1.0 - jnp.exp(-dt / params.tau_csf)

    def scan_fn(csf_prev, inputs):
        drive_i, alpha_i = inputs
        csf_i = csf_prev + alpha_i * (drive_i - csf_prev)
        return csf_i, csf_i

    _, csf = jax_scan(scan_fn, jnp.array(0.0), (drive, alpha))
    return csf


def glymphatic_clearance(
    cbv: Float[Array, "T"],
    t: Float[Array, "T"],
    params: CSFParams | None = None,
) -> Float[Array, ""]:
    """Estimate cumulative glymphatic clearance from CBV oscillations.

    Clearance ∝ cumulative |CSF flow| × efficiency

    Larger CBV oscillations → more CSF displacement → more clearance.

    Parameters
    ----------
    cbv : CBV/CBV₀ time series
    t : time points (s)
    params : CSFParams

    Returns
    -------
    Cumulative clearance metric (a.u., higher = more clearance)
    """
    if params is None:
        params = CSFParams()

    csf = csf_flow_from_cbv(cbv, params)

    # Duration
    T = t[-1] - t[0]
    T_safe = jnp.where(T > 0, T, 1.0)

    # Time-averaged absolute CSF flow × efficiency × duration
    mean_abs_flow = jnp.mean(jnp.abs(csf))
    clearance = mean_abs_flow * params.clearance_efficiency * T_safe

    return clearance


# JAX lax.scan wrapper
import jax

def jax_scan(f, init, xs):
    return jax.lax.scan(f, init, xs)
