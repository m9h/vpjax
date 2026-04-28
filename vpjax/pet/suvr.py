"""Standardized Uptake Value Ratio (SUVR) computation.

Late-frame SUVR is the dominant quantitative metric for amyloid
(18F-AV45 / florbetapir) and tau (18F-AV1451 / flortaucipir) PET when
dynamic data is unavailable.  It is defined as the ratio of target-
region uptake to a reference region with negligible specific binding
(typically whole cerebellum or cerebellar grey for amyloid; inferior
cerebellar grey for tau).

Optional partial-volume correction (PVC) follows the two-compartment
Müller-Gärtner method (1992): the observed PET signal is modelled as
a mixture of grey-matter, white-matter, and CSF activity weighted by
the FAST tissue partial-volume estimates, smoothed with the PET point-
spread function.  We use a closed-form approximation that doesn't
require explicit PSF convolution — adequate for static SUVR where
calibration error dominates over PVC residual.

References
----------
Müller-Gärtner HW et al. (1992) JCBFM 12:571-583
Logan J et al. (1996) JCBFM 16:834-840
    "Distribution volume ratios without blood sampling"
Vaishnavi SN et al. (2010) PNAS 107:17757-17762
    "Regional aerobic glycolysis in the human brain"
Sperling RA et al. (2009) Neuron 63:178-188
    "Amyloid deposition is associated with impaired default network
    function in older persons without dementia"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# Reference values (tracer-specific late-frame conventions)
# ---------------------------------------------------------------------------

# FreeSurfer aparc+aseg labels for common reference regions
CEREBELLUM_GM_LABELS: tuple[int, ...] = (8, 47)         # left + right cerebellar cortex
CEREBELLUM_WM_LABELS: tuple[int, ...] = (7, 46)         # cerebellar WM
WHOLE_CEREBELLUM_LABELS: tuple[int, ...] = (7, 8, 46, 47)


class PETPVCParams(eqx.Module):
    """Parameters for partial-volume correction.

    Attributes
    ----------
    gm_threshold : minimum GM PVE for inclusion in PVC denominator
    wm_assumed_uptake : white-matter uptake in same units as PET
        (relative — for amyloid usually ~0.6× whole-cerebellum)
    csf_assumed_uptake : CSF uptake (typically ~0)
    """
    gm_threshold: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.3)
    )
    wm_assumed_uptake: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.6)
    )
    csf_assumed_uptake: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.0)
    )


# ---------------------------------------------------------------------------
# SUVR computation
# ---------------------------------------------------------------------------

def compute_suvr(
    pet_3d: Float[np.ndarray, "X Y Z"],
    reference_mask: Float[np.ndarray, "X Y Z"],
) -> tuple[Float[np.ndarray, "X Y Z"], float]:
    """Voxel-wise SUVR map normalised by the reference-region mean.

    Parameters
    ----------
    pet_3d : late-frame PET volume (any units)
    reference_mask : binary mask of the reference region

    Returns
    -------
    suvr_3d : pet / reference_mean
    reference_mean : the scalar normaliser used (PET units)
    """
    pet = np.asarray(pet_3d, dtype=np.float32)
    ref = np.asarray(reference_mask) > 0
    if ref.sum() == 0:
        raise ValueError("reference_mask is empty")
    ref_mean = float(pet[ref].mean())
    if ref_mean <= 0:
        raise ValueError(f"reference region has nonpositive mean ({ref_mean})")
    return pet / ref_mean, ref_mean


def regional_suvr(
    pet_3d: Float[np.ndarray, "X Y Z"],
    reference_mask: Float[np.ndarray, "X Y Z"],
    region_volume: Float[np.ndarray, "X Y Z"],
    region_ids: np.ndarray | None = None,
    min_voxels: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-region mean SUVR using the supplied label volume.

    Parameters
    ----------
    pet_3d, reference_mask : as in :func:`compute_suvr`
    region_volume : integer label volume (e.g. ``aparc+aseg.nii.gz``)
    region_ids : optional restriction to a subset of labels
    min_voxels : drop regions with fewer than this many voxels

    Returns
    -------
    suvr_per_region : shape (R,)
    region_ids_kept : shape (R,)
    """
    suvr_3d, _ = compute_suvr(pet_3d, reference_mask)
    region_volume = np.asarray(region_volume)

    if region_ids is None:
        region_ids = np.unique(region_volume[region_volume > 0]).astype(np.int32)

    means = []
    kept = []
    for rid in region_ids:
        mask = region_volume == rid
        if int(mask.sum()) < min_voxels:
            continue
        means.append(float(suvr_3d[mask].mean()))
        kept.append(int(rid))

    if not kept:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)
    return (
        np.asarray(means, dtype=np.float32),
        np.asarray(kept, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Partial-volume correction (Müller-Gärtner two-compartment, simplified)
# ---------------------------------------------------------------------------

def muller_gartner_correction(
    pet_3d: Float[np.ndarray, "X Y Z"],
    gm_pve: Float[np.ndarray, "X Y Z"],
    wm_pve: Float[np.ndarray, "X Y Z"],
    csf_pve: Float[np.ndarray, "X Y Z"] | None = None,
    params: PETPVCParams | None = None,
) -> Float[np.ndarray, "X Y Z"]:
    """Müller-Gärtner-style two-compartment PVC for static PET.

    Models the observed PET signal as a per-voxel mixture:

        PET_obs ≈ GM_pve · GM_true + WM_pve · WM_assumed + CSF_pve · CSF_assumed

    and solves for ``GM_true`` per voxel where ``GM_pve > gm_threshold``.
    The full Müller-Gärtner method convolves WM/CSF maps with the PET
    PSF before subtracting; we omit the convolution step because it
    requires the scanner-specific PSF and (for static SUVR) its impact
    is small relative to GM PVE itself.  Voxels below threshold are
    left as the observed value (cannot be reliably corrected).

    Parameters
    ----------
    pet_3d : observed late-frame PET volume
    gm_pve, wm_pve, csf_pve : tissue partial-volume estimates from
        FAST (CSF defaults to ``1 - gm - wm`` clipped to [0,1])
    params : :class:`PETPVCParams`

    Returns
    -------
    pet_pvc : partial-volume corrected PET map
    """
    if params is None:
        params = PETPVCParams()

    pet = np.asarray(pet_3d, dtype=np.float32)
    gm = np.asarray(gm_pve, dtype=np.float32)
    wm = np.asarray(wm_pve, dtype=np.float32)
    if csf_pve is None:
        csf = np.clip(1.0 - gm - wm, 0.0, 1.0)
    else:
        csf = np.asarray(csf_pve, dtype=np.float32)

    wm_uptake = float(params.wm_assumed_uptake)
    csf_uptake = float(params.csf_assumed_uptake)
    threshold = float(params.gm_threshold)

    # If a global WM uptake is unknown, estimate from voxels where GM is
    # essentially zero (high WM-fraction voxels).  This is a one-line
    # version of the original Yang correction.
    near_pure_wm = (gm < 0.05) & (wm > 0.85)
    if near_pure_wm.sum() > 50:
        wm_uptake = float(pet[near_pure_wm].mean())

    contamination = wm * wm_uptake + csf * csf_uptake
    gm_safe = np.where(gm > threshold, gm, np.nan)
    gm_true = (pet - contamination) / gm_safe

    # Where GM is below threshold, fall back to observed value.
    gm_true = np.where(np.isfinite(gm_true), gm_true, pet)
    return gm_true.astype(np.float32)
