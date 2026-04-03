"""Ascending vein removal for layer-specific VASO.

In laminar VASO fMRI, ascending (pial) veins draining deep cortical
layers contaminate the signal in superficial layers.  Deveining
removes this contamination to recover layer-specific CBV changes.

LAYNII provides LN2_DEVEIN for this purpose.  vpjax implements a
differentiable version of the linear deveining model for use in
forward models and gradient-based fitting.

The model: observed signal at each layer is a mixture of local signal
and ascending vein drainage from deeper layers:

    S_obs[L] = S_local[L] + Σ_{l<L} w(l→L) × S_local[l]

Deveining inverts this mixing to recover S_local.

References
----------
Huber L et al. (2017) Neuron 96:1253-1263
Markuerkiaga I et al. (2016) NeuroImage 132:491-498
Huber L et al. (2021) NeuroImage 237:118091
    "LAYNII: LN2_DEVEIN"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class DeveinParams(eqx.Module):
    """Parameters for ascending vein deveining model.

    Attributes
    ----------
    n_layers     : number of cortical layers
    drain_frac   : fraction of each layer's venous blood that drains
                   to the next superficial layer (0-1), shape (L,)
    """
    n_layers: int = eqx.field(static=True, default=3)
    drain_frac: Float[Array, "L"] = eqx.field(
        default_factory=lambda: jnp.array([0.3, 0.3, 0.3])
    )


def build_drainage_matrix(
    params: DeveinParams | None = None,
) -> Float[Array, "L L"]:
    """Build the vein drainage mixing matrix.

    D[i,j] represents how much of layer j's signal appears in layer i.
    - Diagonal: 1.0 (local contribution)
    - Below diagonal: 0.0 (deeper layers don't receive from shallower)
    - Above diagonal: drain_frac[j] for adjacent layers, attenuated
      for non-adjacent

    Parameters
    ----------
    params : DeveinParams

    Returns
    -------
    Drainage matrix D, shape (L, L). Lower triangular + diagonal.
    """
    if params is None:
        params = DeveinParams()

    n = params.n_layers

    def _row(i):
        # Layer i receives drainage from layers 0..i-1
        # Contribution from layer j to layer i:
        #   drain_frac[j] × (attenuation for distance)
        # Attenuation: each intermediate layer absorbs some drainage
        j = jnp.arange(n, dtype=jnp.float32)
        distance = i - j  # >0 for deeper layers

        # Only layers below (j < i) contribute drainage
        is_below = (j < i).astype(jnp.float32)

        # Cumulative drainage: each passing layer retains (1-drain) fraction
        # Amount reaching layer i from layer j:
        #   drain_frac[j] × prod_{k=j+1}^{i-1} (1 - drain_frac[k])
        # Simplified: drain_frac[j] × (1-drain_mean)^(distance-1) for distance>0
        drain_mean = jnp.mean(params.drain_frac)
        contribution = params.drain_frac * jnp.where(
            distance > 1,
            jnp.power(1.0 - drain_mean, distance - 1),
            1.0,
        )

        row = is_below * contribution
        # Diagonal = 1.0
        row = row.at[i].set(1.0)
        return row

    import jax
    D = jax.vmap(_row)(jnp.arange(n))
    return D


def apply_vein_contamination(
    local_signal: Float[Array, "... L"],
    params: DeveinParams | None = None,
) -> Float[Array, "... L"]:
    """Simulate ascending vein contamination of laminar signal.

    Parameters
    ----------
    local_signal : true local signal per layer, shape (..., L)
    params : DeveinParams

    Returns
    -------
    Observed (contaminated) signal per layer, shape (..., L)
    """
    D = build_drainage_matrix(params)
    return jnp.einsum("...l,ml->...m", local_signal, D)


def devein(
    observed_signal: Float[Array, "... L"],
    params: DeveinParams | None = None,
) -> Float[Array, "... L"]:
    """Remove ascending vein contamination (deveining).

    Inverts the drainage mixing matrix to recover local layer signals.

    Parameters
    ----------
    observed_signal : observed signal with vein contamination, shape (..., L)
    params : DeveinParams

    Returns
    -------
    Estimated local signal per layer, shape (..., L)
    """
    D = build_drainage_matrix(params)
    D_inv = jnp.linalg.inv(D)
    return jnp.einsum("...l,ml->...m", observed_signal, D_inv)
