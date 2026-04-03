"""Equivolume cortical layer definition.

Defines cortical layers using the equivolume principle (Waehnert et al. 2014),
which accounts for cortical curvature to produce layers of equal volume.
This is the standard approach used by LAYNII (Huber et al. 2021) and
Nighres (Huntenburg et al. 2018).

vpjax does not call LAYNII/Nighres directly.  Instead, it provides
functions that:
1. Compute equivolume depth fractions from curvature
2. Assign voxels to layers given a depth map
3. Create layer masks for profile extraction

The actual depth maps (metric_equivol) come from LAYNII's LN2_LAYERS
or Nighres' laminar_analysis.

References
----------
Waehnert MD et al. (2014) NeuroImage 93:210-220
    "Anatomically motivated modeling of cortical laminae"
Huber L et al. (2021) NeuroImage 237:118091
    "LAYNII: A software suite for cortical depth-dependent analysis"
Huntenburg JM et al. (2018) GigaScience 7:giy082
    "Nighres: cortical surface and laminar analysis"
Bok ST (1929) Zeitschrift für die gesamte Neurologie und Psychiatrie
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def equivolume_depths(
    n_layers: int,
    curvature: Float[Array, "..."] | None = None,
) -> Float[Array, "... L+1"]:
    """Compute equivolume layer boundaries accounting for curvature.

    For flat cortex (curvature=0), equivolume = equidistant.
    For curved cortex, layer boundaries shift to maintain equal volume.

    The Bok (1929) model: cortical thickness varies with curvature as
        T(curv) = T₀ / (1 + curv · d)
    where d is depth from the outer surface.

    Equivolume boundaries satisfy:
        ∫[d_i to d_{i+1}] (1 + curv·d)^2 dd = const for each layer

    Parameters
    ----------
    n_layers : number of cortical layers
    curvature : mean curvature at each voxel (1/mm). Positive = sulcus,
        negative = gyrus. If None, assumes flat cortex (equidistant layers).

    Returns
    -------
    Layer boundaries as depth fractions in [0, 1], shape (..., n_layers+1).
    0 = WM/GM boundary, 1 = pial surface.
    """
    if curvature is None:
        # Equidistant (flat cortex case)
        return jnp.linspace(0.0, 1.0, n_layers + 1)

    # Equivolume computation (Waehnert et al. 2014)
    # Volume element at depth d with curvature κ: dV ∝ (1 + κ·d)² dd
    # Integrate: V(d) = d + κ·d² + (κ²·d³)/3
    # Normalized total: V(1) = 1 + κ + κ²/3
    kappa = curvature[..., None]  # (..., 1)

    # Target volume per layer
    V_total = 1.0 + kappa + kappa ** 2 / 3.0
    V_per_layer = V_total / n_layers

    # Find boundaries by inverting V(d) for each layer
    # V(d) = d + κd² + κ²d³/3
    # We solve numerically using a fine grid and interpolation
    d_fine = jnp.linspace(0.0, 1.0, 1000)
    V_fine = d_fine + kappa * d_fine ** 2 + kappa ** 2 * d_fine ** 3 / 3.0

    # Target cumulative volumes
    layer_idx = jnp.arange(n_layers + 1)
    V_targets = layer_idx * V_per_layer  # (..., n_layers+1)

    # Interpolate to find depth at each target volume
    # For each voxel, interpolate V_fine -> d_fine
    def _interp_one(v_fine_1d, v_target_1d):
        return jnp.interp(v_target_1d, v_fine_1d, d_fine)

    # Handle batched case: flatten, interpolate, reshape
    orig_shape = curvature.shape
    V_fine_flat = V_fine.reshape(-1, 1000)  # (N, 1000)
    V_targets_flat = V_targets.reshape(-1, n_layers + 1)  # (N, L+1)

    import jax
    boundaries_flat = jax.vmap(_interp_one)(V_fine_flat, V_targets_flat)
    boundaries = boundaries_flat.reshape(*orig_shape, n_layers + 1)

    return boundaries


def assign_layers(
    depth: Float[Array, "..."],
    n_layers: int,
    boundaries: Float[Array, "L+1"] | None = None,
) -> Int[Array, "..."]:
    """Assign voxels to cortical layers based on depth.

    Parameters
    ----------
    depth : cortical depth map (0 = WM, 1 = pial), shape (...)
    n_layers : number of layers
    boundaries : layer boundaries (default: equidistant). Shape (L+1,).

    Returns
    -------
    Layer index (0 = deepest/WM, n_layers-1 = superficial/pial).
    Values outside [0,1] are clipped.
    """
    if boundaries is None:
        boundaries = jnp.linspace(0.0, 1.0, n_layers + 1)

    depth_clipped = jnp.clip(depth, 0.0, 1.0)

    # Digitize: find which bin each depth falls into
    # boundaries[i] <= depth < boundaries[i+1] → layer i
    layer = jnp.searchsorted(boundaries, depth_clipped, side="right") - 1
    layer = jnp.clip(layer, 0, n_layers - 1)
    return layer.astype(jnp.int32)


def layer_mask(
    depth: Float[Array, "..."],
    layer_idx: int,
    n_layers: int,
    boundaries: Float[Array, "L+1"] | None = None,
) -> Float[Array, "..."]:
    """Create a soft mask for a specific cortical layer.

    Returns a continuous mask (0-1) that can be used for weighted
    averaging, maintaining differentiability.

    Parameters
    ----------
    depth : cortical depth map (0 = WM, 1 = pial)
    layer_idx : which layer (0 = deepest)
    n_layers : total number of layers
    boundaries : layer boundaries (default: equidistant)

    Returns
    -------
    Soft mask with values in [0, 1].
    """
    if boundaries is None:
        boundaries = jnp.linspace(0.0, 1.0, n_layers + 1)

    lower = boundaries[layer_idx]
    upper = boundaries[layer_idx + 1]
    width = upper - lower

    # Smooth step function (differentiable)
    # Sigmoid-based soft boundaries with sharpness proportional to layer width
    sharpness = 20.0 / jnp.clip(width, 0.01, None)

    mask_lower = jax_sigmoid((depth - lower) * sharpness)
    mask_upper = jax_sigmoid((upper - depth) * sharpness)

    return mask_lower * mask_upper


def jax_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Numerically stable sigmoid."""
    return jnp.where(x >= 0, 1.0 / (1.0 + jnp.exp(-x)), jnp.exp(x) / (1.0 + jnp.exp(x)))
