"""Layer-specific neurovascular coupling (NVC).

Cortical layers differ in their neurovascular coupling properties:
- Deep layers (V-VI): primarily feedforward connections,
  higher excitatory drive, stronger vasodilatory response
- Superficial layers (I-III): primarily feedback connections,
  more inhibitory modulation
- Layer IV: thalamocortical input (granular layer)

This maps onto Valdes-Sosa's ξ-αNET feedforward/feedback hierarchy
and connects to the geodesic cortical flow directions from Liu et al.

Layer-specific NVC parameters allow different BOLD/VASO responses
per cortical depth, which is testable with sub-millimeter fMRI.

References
----------
Huber L et al. (2017) Neuron 96:1253-1263
    "High-resolution CBV-fMRI allows mapping of laminar activity"
Markuerkiaga I et al. (2016) NeuroImage 132:491-498
    "A cortical vascular model for examining the specificity of the
    laminar BOLD signal"
Heinzle J et al. (2016) NeuroImage 125:680-691
    "A hemodynamic model for layered BOLD signals"
Valdes-Sosa PA et al. (2009) NeuroImage 47:1007-1023
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class LayerNVCParams(eqx.Module):
    """Layer-specific neurovascular coupling parameters.

    Default values approximate a 3-layer model:
    - Layer 0 (deep, V-VI): strong feedforward NVC
    - Layer 1 (middle, IV): thalamocortical input
    - Layer 2 (superficial, I-III): feedback NVC

    Attributes
    ----------
    n_layers       : number of cortical layers
    kappa          : signal decay per layer (s⁻¹), shape (L,)
    gamma          : flow-elimination per layer (s⁻¹), shape (L,)
    coupling_gain  : NVC gain per layer (a.u.), shape (L,)
    vein_drain     : ascending vein drainage fraction per layer, shape (L,)
    tau            : transit time per layer (s), shape (L,)
    alpha          : Grubb exponent per layer, shape (L,)
    """
    n_layers: int = eqx.field(static=True, default=3)
    kappa: Float[Array, "L"] = eqx.field(
        default_factory=lambda: jnp.array([0.65, 0.65, 0.65])
    )
    gamma: Float[Array, "L"] = eqx.field(
        default_factory=lambda: jnp.array([0.41, 0.41, 0.41])
    )
    coupling_gain: Float[Array, "L"] = eqx.field(
        default_factory=lambda: jnp.array([1.0, 0.8, 0.6])
    )
    vein_drain: Float[Array, "L"] = eqx.field(
        default_factory=lambda: jnp.array([0.1, 0.3, 0.6])
    )
    tau: Float[Array, "L"] = eqx.field(
        default_factory=lambda: jnp.array([0.8, 1.0, 1.2])
    )
    alpha: Float[Array, "L"] = eqx.field(
        default_factory=lambda: jnp.array([0.32, 0.32, 0.32])
    )


def layer_stimulus(
    stimulus: Float[Array, "..."],
    feedforward_frac: Float[Array, "..."] | None = None,
    feedback_frac: Float[Array, "..."] | None = None,
    params: LayerNVCParams | None = None,
) -> Float[Array, "... L"]:
    """Distribute neural stimulus across cortical layers.

    Deep layers receive feedforward drive, superficial layers receive
    feedback. Middle layers receive a mix (thalamocortical input).

    Parameters
    ----------
    stimulus : total neural stimulus, shape (...)
    feedforward_frac : fraction that is feedforward (0-1). Default 0.7.
    feedback_frac : fraction that is feedback. Default 0.3.
    params : LayerNVCParams

    Returns
    -------
    Per-layer stimulus, shape (..., L).
    """
    if params is None:
        params = LayerNVCParams()

    if feedforward_frac is None:
        feedforward_frac = jnp.array(0.7)
    if feedback_frac is None:
        feedback_frac = jnp.array(0.3)

    n = params.n_layers
    # Default distribution: deep gets FF, superficial gets FB, middle gets both
    # Weights: [FF_deep, 0.5*(FF+FB)_middle, FB_superficial]
    if n == 3:
        weights = jnp.array([
            feedforward_frac,
            0.5 * (feedforward_frac + feedback_frac),
            feedback_frac,
        ])
    else:
        # Linear interpolation from FF (deep) to FB (superficial)
        t = jnp.linspace(0.0, 1.0, n)
        weights = feedforward_frac * (1.0 - t) + feedback_frac * t

    # Apply coupling gains
    weights = weights * params.coupling_gain

    return stimulus[..., None] * weights


def ascending_vein_contamination(
    layer_bold: Float[Array, "... L"],
    params: LayerNVCParams | None = None,
) -> Float[Array, "... L"]:
    """Model ascending vein contamination of laminar BOLD signal.

    Ascending veins drain blood from deep layers toward the pial
    surface, causing superficial layers to contain signal from deeper
    layers. This is the primary confound in laminar BOLD fMRI.

    The drainage model: each layer's BOLD signal contains contributions
    from deeper layers via ascending veins.

    observed_BOLD[L] = local_BOLD[L] + Σ_{l<L} drain[l→L] × BOLD[l]

    Parameters
    ----------
    layer_bold : local BOLD signal per layer, shape (..., L)
    params : LayerNVCParams (vein_drain field controls mixing)

    Returns
    -------
    Observed BOLD signal per layer (with vein contamination), shape (..., L).
    """
    if params is None:
        params = LayerNVCParams()

    n = params.n_layers

    # Build drainage matrix: D[i,j] = fraction of layer j's signal
    # that appears in layer i (due to ascending veins)
    # D is lower-triangular: only deeper layers contaminate shallower
    def _build_row(i):
        # Layer i receives contamination from layers 0..i-1
        drain = params.vein_drain
        # Amount draining from layer j to layer i scales with
        # the drain fraction of layer j and distance
        row = jnp.where(
            jnp.arange(n) < i,
            drain * (1.0 / jnp.clip(i - jnp.arange(n, dtype=jnp.float32), 1.0, None)),
            0.0,
        )
        # Diagonal: local contribution
        row = row.at[i].set(1.0)
        return row

    import jax
    D = jax.vmap(_build_row)(jnp.arange(n))  # (L, L)

    # Apply: observed = layer_bold @ D^T
    observed = jnp.einsum("...l,ml->...m", layer_bold, D)

    return observed


def devein_bold(
    observed_bold: Float[Array, "... L"],
    params: LayerNVCParams | None = None,
) -> Float[Array, "... L"]:
    """Remove ascending vein contamination (deveining).

    Inverts the drainage model to recover local BOLD responses
    per layer from the observed (contaminated) signals.

    Parameters
    ----------
    observed_bold : observed BOLD per layer (with vein contamination)
    params : LayerNVCParams

    Returns
    -------
    Estimated local BOLD per layer (deveined), shape (..., L).
    """
    if params is None:
        params = LayerNVCParams()

    n = params.n_layers

    def _build_row(i):
        drain = params.vein_drain
        row = jnp.where(
            jnp.arange(n) < i,
            drain * (1.0 / jnp.clip(i - jnp.arange(n, dtype=jnp.float32), 1.0, None)),
            0.0,
        )
        row = row.at[i].set(1.0)
        return row

    import jax
    D = jax.vmap(_build_row)(jnp.arange(n))  # (L, L)

    # Invert D to get deveining matrix
    D_inv = jnp.linalg.inv(D)

    # Apply: local = observed @ D_inv^T
    local = jnp.einsum("...l,ml->...m", observed_bold, D_inv)

    return local
