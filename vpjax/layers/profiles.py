"""Depth-dependent profile sampling.

Extracts cortical depth profiles from volumetric maps (T1, R2*, QSM, etc.)
using cortical depth maps from LAYNII/Nighres.

A depth profile samples any volumetric quantity at multiple cortical depths,
producing a curve from white matter to pial surface. These profiles reveal
laminar structure (e.g., T1 myelination gradient, R2* iron content).

References
----------
Huber L et al. (2021) NeuroImage 237:118091
Waehnert MD et al. (2014) NeuroImage 93:210-220
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def sample_profile(
    volume: Float[Array, "..."],
    depth: Float[Array, "..."],
    n_depths: int = 20,
    roi_mask: Float[Array, "..."] | None = None,
) -> tuple[Float[Array, "D"], Float[Array, "D"]]:
    """Sample a depth profile from a volumetric map.

    Bins voxels by cortical depth and computes the mean value at each
    depth, producing a 1-D profile from WM to pial surface.

    Parameters
    ----------
    volume : the volumetric map to sample (e.g., T1, R2*, QSM)
    depth : cortical depth map (0 = WM, 1 = pial), same shape as volume
    n_depths : number of depth bins
    roi_mask : optional binary mask to restrict to a region of interest

    Returns
    -------
    depth_centers : centers of depth bins, shape (n_depths,)
    profile : mean value at each depth, shape (n_depths,)
    """
    edges = jnp.linspace(0.0, 1.0, n_depths + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    depth_flat = depth.ravel()
    vol_flat = volume.ravel()

    if roi_mask is not None:
        mask_flat = roi_mask.ravel()
    else:
        mask_flat = jnp.ones_like(depth_flat)

    # Valid voxels: within cortex (0 <= depth <= 1) and in ROI
    valid = (depth_flat >= 0.0) & (depth_flat <= 1.0) & (mask_flat > 0.5)

    def _bin_mean(i):
        lower = edges[i]
        upper = edges[i + 1]
        in_bin = valid & (depth_flat >= lower) & (depth_flat < upper)
        weights = in_bin.astype(jnp.float32)
        total = jnp.sum(weights)
        total_safe = jnp.where(total > 0, total, 1.0)
        mean_val = jnp.sum(vol_flat * weights) / total_safe
        # Return 0 if bin is empty
        return jnp.where(total > 0, mean_val, 0.0)

    profile = jax.vmap(_bin_mean)(jnp.arange(n_depths))

    return centers, profile


def sample_profile_weighted(
    volume: Float[Array, "..."],
    depth: Float[Array, "..."],
    n_depths: int = 20,
    weights: Float[Array, "..."] | None = None,
) -> tuple[Float[Array, "D"], Float[Array, "D"], Float[Array, "D"]]:
    """Sample a weighted depth profile with uncertainty.

    Like sample_profile but supports voxel weights (e.g., from
    signal quality maps) and returns standard deviation.

    Parameters
    ----------
    volume : volumetric map to sample
    depth : cortical depth map (0 = WM, 1 = pial)
    n_depths : number of depth bins
    weights : per-voxel weights (e.g., SNR map). Default: uniform.

    Returns
    -------
    depth_centers : shape (n_depths,)
    profile_mean : weighted mean at each depth, shape (n_depths,)
    profile_std : weighted std at each depth, shape (n_depths,)
    """
    edges = jnp.linspace(0.0, 1.0, n_depths + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    depth_flat = depth.ravel()
    vol_flat = volume.ravel()

    if weights is not None:
        w_flat = weights.ravel()
    else:
        w_flat = jnp.ones_like(depth_flat)

    valid = (depth_flat >= 0.0) & (depth_flat <= 1.0)

    def _bin_stats(i):
        lower = edges[i]
        upper = edges[i + 1]
        in_bin = valid & (depth_flat >= lower) & (depth_flat < upper)
        bin_w = w_flat * in_bin.astype(jnp.float32)
        total_w = jnp.sum(bin_w)
        total_safe = jnp.where(total_w > 0, total_w, 1.0)

        mean_val = jnp.sum(vol_flat * bin_w) / total_safe
        var_val = jnp.sum(bin_w * (vol_flat - mean_val) ** 2) / total_safe
        std_val = jnp.sqrt(jnp.clip(var_val, 0.0, None))

        return jnp.where(total_w > 0, mean_val, 0.0), jnp.where(total_w > 0, std_val, 0.0)

    means, stds = jax.vmap(_bin_stats)(jnp.arange(n_depths))

    return centers, means, stds


def normalize_profile(
    profile: Float[Array, "D"],
    method: str = "minmax",
) -> Float[Array, "D"]:
    """Normalize a depth profile.

    Parameters
    ----------
    profile : raw depth profile, shape (D,)
    method : 'minmax' (scale to [0,1]) or 'zscore' (zero-mean, unit-var)

    Returns
    -------
    Normalized profile, shape (D,)
    """
    if method == "minmax":
        pmin = jnp.min(profile)
        pmax = jnp.max(profile)
        denom = jnp.where(pmax - pmin > 1e-10, pmax - pmin, 1.0)
        return (profile - pmin) / denom
    elif method == "zscore":
        mu = jnp.mean(profile)
        sigma = jnp.std(profile)
        sigma = jnp.where(sigma > 1e-10, sigma, 1.0)
        return (profile - mu) / sigma
    else:
        raise ValueError(f"Unknown normalization method: {method}")
