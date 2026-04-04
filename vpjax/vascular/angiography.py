"""Vessel tree analysis from TOF angiography → vpjax model parameters.

Converts vascular morphometry extracted from TOF-MRA into
subject-specific parameters for vpjax forward models:

    TOF segmentation → VesselTree → {BalloonParams, RieraParams,
                                      VascularParams, DBV, CBV₀}

This replaces literature defaults with per-subject measurements,
enabling individually constrained hemodynamic models.

Connections to vpjax models:
    - VascularParams: density, radius, length per compartment
    - BalloonParams.tau: transit time from measured vessel lengths
    - RieraParams.tau_a/c/v: per-compartment transit times
    - qBOLD DBV: venous blood volume from morphometry
    - VASOParams.CBV0: baseline CBV prior

Uses scikit-image skeletonize + scipy distance transform as the
default backend.  VMTK provides higher-quality centerlines when
available (brew install mhough/neuro/vmtk).

References
----------
Antiga L et al. (2008) Medical Image Analysis 12:514-526 (VMTK)
Duvernoy HM et al. (1981) Brain Research Bulletin 7:519-579
Cassot F et al. (2006) Microcirculation 13:1-18
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# Vessel classification thresholds (µm)
# ---------------------------------------------------------------------------

_ART_RADIUS_THRESH = 10.0   # radius >= this → artery/arteriole
_CAP_RADIUS_THRESH = 5.0    # radius >= this (and < art) → capillary
# below cap threshold → venule/vein (large veins also caught by radius > art,
# but TOF primarily shows arteries so most large vessels are arterial)

# Typical blood velocities for transit time estimation (µm/s)
_VELOCITY_ART = 10000.0   # ~10 mm/s in arterioles
_VELOCITY_CAP = 1000.0    # ~1 mm/s in capillaries
_VELOCITY_VEN = 5000.0    # ~5 mm/s in venules


# ---------------------------------------------------------------------------
# VesselTree
# ---------------------------------------------------------------------------

@dataclass
class VesselTree:
    """Vessel tree from TOF angiography segmentation.

    Attributes
    ----------
    points : centerline coordinates (N, 3), voxel or world space (µm)
    radii : vessel radius at each point (N,), in µm
    branch_ids : branch label per point (N,), integer
    """
    points: np.ndarray
    radii: np.ndarray
    branch_ids: np.ndarray

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def n_branches(self) -> int:
        return len(np.unique(self.branch_ids))

    def branch_lengths(self) -> list[float]:
        """Arc length of each branch (same units as points)."""
        lengths = []
        for bid in np.unique(self.branch_ids):
            pts = self.points[self.branch_ids == bid]
            if len(pts) < 2:
                lengths.append(0.0)
                continue
            lengths.append(float(np.sum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)))))
        return lengths

    def branch_mean_radii(self) -> list[float]:
        """Mean radius per branch."""
        return [
            float(np.mean(self.radii[self.branch_ids == bid]))
            for bid in np.unique(self.branch_ids)
        ]

    def _classify_branches(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify branches into arterial, capillary, venous by radius.

        Returns masks (boolean arrays over unique branch IDs).
        """
        mr = np.array(self.branch_mean_radii())
        art = mr >= _ART_RADIUS_THRESH
        cap = (mr >= _CAP_RADIUS_THRESH) & (mr < _ART_RADIUS_THRESH)
        ven = mr < _CAP_RADIUS_THRESH
        return art, cap, ven

    def to_vascular_params(self, tissue_volume_mm3: float = 1.0):
        """Convert to VascularParams with measured morphometry.

        Parameters
        ----------
        tissue_volume_mm3 : tissue ROI volume for density calculation

        Returns
        -------
        VascularParams with subject-specific values
        """
        from vpjax.vascular.geometry import VascularParams

        mr = np.array(self.branch_mean_radii())
        lengths = np.array(self.branch_lengths())
        art, cap, ven = self._classify_branches()

        def _safe(arr, default):
            return float(np.mean(arr)) if len(arr) > 0 else default

        scale = 1.0 / max(tissue_volume_mm3, 1e-10)

        return VascularParams(
            density_art=jnp.array(float(np.sum(art)) * scale),
            density_cap=jnp.array(float(np.sum(cap)) * scale),
            density_ven=jnp.array(float(np.sum(ven)) * scale),
            radius_art=jnp.array(_safe(mr[art], 15.0)),
            radius_cap=jnp.array(_safe(mr[cap], 3.0)),
            radius_ven=jnp.array(_safe(mr[ven], 25.0)),
            length_art=jnp.array(_safe(lengths[art], 200.0)),
            length_cap=jnp.array(_safe(lengths[cap], 100.0)),
            length_ven=jnp.array(_safe(lengths[ven], 250.0)),
        )


# ---------------------------------------------------------------------------
# TOF → BalloonParams (subject-specific transit time)
# ---------------------------------------------------------------------------

def balloon_params_from_tree(
    tree: VesselTree,
    velocity_ven: float = _VELOCITY_VEN,
):
    """Derive BalloonParams with subject-specific transit time.

    Transit time τ is estimated from mean venous path length and
    velocity, since the Balloon model's τ represents the venous
    (Windkessel) compartment.

    Parameters
    ----------
    tree : VesselTree from TOF
    velocity_ven : venous blood velocity (µm/s)

    Returns
    -------
    BalloonParams with measured τ, defaults for other parameters
    """
    from vpjax._types import BalloonParams

    lengths = np.array(tree.branch_lengths())
    mr = np.array(tree.branch_mean_radii())
    _, _, ven = tree._classify_branches()

    if np.any(ven) and len(lengths[ven]) > 0:
        mean_ven_length = float(np.mean(lengths[ven]))
    else:
        # Fall back to all vessels if no venous branches classified
        mean_ven_length = float(np.mean(lengths)) if len(lengths) > 0 else 200.0

    # τ = L / v, convert µm to µm/s → seconds
    tau = mean_ven_length / max(velocity_ven, 1e-6)

    return BalloonParams(tau=jnp.array(tau))


# ---------------------------------------------------------------------------
# TOF → RieraParams (per-compartment transit times)
# ---------------------------------------------------------------------------

def riera_params_from_tree(
    tree: VesselTree,
    velocity_art: float = _VELOCITY_ART,
    velocity_cap: float = _VELOCITY_CAP,
    velocity_ven: float = _VELOCITY_VEN,
):
    """Derive RieraParams with per-compartment transit times from TOF.

    Parameters
    ----------
    tree : VesselTree
    velocity_art/cap/ven : compartment velocities (µm/s)

    Returns
    -------
    RieraParams with measured τ_a, τ_c, τ_v
    """
    from vpjax.hemodynamics.riera import RieraParams

    lengths = np.array(tree.branch_lengths())
    art, cap, ven = tree._classify_branches()

    def _mean_tt(mask, velocity, default_length):
        if np.any(mask):
            L = float(np.mean(lengths[mask]))
        else:
            L = default_length
        return L / max(velocity, 1e-6)

    tau_a = _mean_tt(art, velocity_art, 200.0)
    tau_c = _mean_tt(cap, velocity_cap, 100.0)
    tau_v = _mean_tt(ven, velocity_ven, 250.0)

    return RieraParams(
        tau_a=jnp.array(tau_a),
        tau_c=jnp.array(tau_c),
        tau_v=jnp.array(tau_v),
    )


# ---------------------------------------------------------------------------
# TOF → qBOLD DBV estimate
# ---------------------------------------------------------------------------

def estimate_dbv_from_tree(
    tree: VesselTree,
    tissue_volume_mm3: float = 1.0,
) -> float:
    """Estimate deoxygenated blood volume fraction from venous morphometry.

    DBV = total venous vessel volume / tissue volume

    Parameters
    ----------
    tree : VesselTree
    tissue_volume_mm3 : tissue ROI volume (mm³)

    Returns
    -------
    DBV fraction (dimensionless)
    """
    mr = np.array(tree.branch_mean_radii())
    lengths = np.array(tree.branch_lengths())
    _, cap, ven = tree._classify_branches()

    # Deoxygenated compartments: veins + capillaries
    deoxy_mask = cap | ven

    if not np.any(deoxy_mask):
        return 0.03  # literature default

    # Volume per branch: π r² L (µm³)
    volumes = np.pi * mr[deoxy_mask] ** 2 * lengths[deoxy_mask]
    total_vol_um3 = float(np.sum(volumes))

    # Convert tissue volume: 1 mm³ = 10⁹ µm³
    tissue_vol_um3 = tissue_volume_mm3 * 1e9

    return total_vol_um3 / max(tissue_vol_um3, 1e-6)


# ---------------------------------------------------------------------------
# TOF → CBV₀ baseline prior
# ---------------------------------------------------------------------------

def estimate_cbv0_from_tree(
    tree: VesselTree,
    tissue_volume_mm3: float = 1.0,
) -> float:
    """Estimate total baseline CBV fraction from vessel morphometry.

    CBV₀ = total vessel volume / tissue volume

    This provides a morphometric prior for VASOParams.CBV0 and the
    Balloon model's resting volume.

    Parameters
    ----------
    tree : VesselTree
    tissue_volume_mm3 : tissue ROI volume (mm³)

    Returns
    -------
    CBV₀ fraction
    """
    mr = np.array(tree.branch_mean_radii())
    lengths = np.array(tree.branch_lengths())

    # All vessels contribute to total CBV
    volumes = np.pi * mr ** 2 * lengths
    total_vol_um3 = float(np.sum(volumes))

    tissue_vol_um3 = tissue_volume_mm3 * 1e9
    return total_vol_um3 / max(tissue_vol_um3, 1e-6)


# ---------------------------------------------------------------------------
# Skeletonization (scikit-image fallback)
# ---------------------------------------------------------------------------

def skeletonize_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Extract skeleton from binary vessel segmentation.

    Parameters
    ----------
    segmentation : binary 3D volume (0=background, 1=vessel)

    Returns
    -------
    Skeleton voxel coordinates (N, 3)
    """
    from skimage.morphology import skeletonize

    skeleton = skeletonize((segmentation > 0).astype(np.uint8))
    return np.argwhere(skeleton > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Radius estimation from distance transform
# ---------------------------------------------------------------------------

def estimate_radii(
    segmentation: np.ndarray,
    centerline_points: np.ndarray,
) -> np.ndarray:
    """Estimate vessel radius at centerline points via distance transform.

    Parameters
    ----------
    segmentation : binary 3D volume
    centerline_points : coordinates (N, 3) in voxel space

    Returns
    -------
    Radius at each point (N,) in voxels
    """
    from scipy.ndimage import distance_transform_edt

    dt = distance_transform_edt((segmentation > 0).astype(np.uint8))
    coords = np.clip(np.round(centerline_points).astype(int), 0,
                      np.array(segmentation.shape) - 1)
    return dt[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
