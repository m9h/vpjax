"""Identify brainstem ICA components from MELODIC decomposition.

Matches ICA spatial maps to brainstem nuclei ROIs using spatial
overlap metrics.  The component with the highest overlap to each
nucleus ROI is identified as that nucleus's functional signal.

This enables extraction of LC, raphe, NTS time courses from
standard resting-state fMRI MELODIC output without special
brainstem-optimized acquisitions.

References
----------
Bianciardi M et al. (2016) NeuroImage 134:53-70
    "Functional connectome of arousal and motor brainstem nuclei"
"""

from __future__ import annotations

import numpy as np


def spatial_overlap(
    ic_map: np.ndarray,
    roi: np.ndarray,
    threshold: float = 2.0,
) -> float:
    """Compute spatial overlap between an IC map and an ROI.

    Overlap = mean(|IC_map|) within ROI / mean(|IC_map|) outside ROI

    A high ratio means the IC is specifically concentrated in the ROI.
    The IC map is thresholded at `threshold` (z-score) before computing.

    Parameters
    ----------
    ic_map : ICA spatial map (z-scored), same shape as roi
    roi : binary or probabilistic ROI mask
    threshold : z-score threshold for IC map

    Returns
    -------
    Overlap score (0 = no overlap, higher = better match)
    """
    roi_mask = roi > 0.5 if roi.max() > 0.5 else roi > 0.1
    n_roi = np.sum(roi_mask)

    if n_roi == 0:
        return 0.0

    # Absolute IC values (interested in both positive and negative weights)
    ic_abs = np.abs(ic_map)

    # Mean absolute IC value within and outside ROI
    mean_in = np.mean(ic_abs[roi_mask])
    outside_mask = ~roi_mask & (ic_abs > 0)
    if np.sum(outside_mask) == 0:
        return float(mean_in) if mean_in > threshold else 0.0

    mean_out = np.mean(ic_abs[outside_mask])

    if mean_out < 1e-10:
        return float(mean_in) if mean_in > threshold else 0.0

    # Specificity ratio
    ratio = mean_in / mean_out

    # Also require the IC to have suprathreshold values in the ROI
    frac_above = np.mean(ic_abs[roi_mask] > threshold)

    return float(ratio * frac_above)


def identify_brainstem_components(
    ic_maps: np.ndarray,
    atlas: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Identify which ICA components correspond to brainstem nuclei.

    For each nucleus ROI, finds the IC component with the highest
    spatial overlap.

    Parameters
    ----------
    ic_maps : ICA spatial maps, shape (n_components, x, y, z)
    atlas : dict mapping nucleus name → ROI mask
    threshold : minimum overlap score to accept a match

    Returns
    -------
    Dict mapping nucleus name → {component_idx, overlap, name}
    Only nuclei with a match above threshold are included.
    """
    n_components = ic_maps.shape[0]
    matches = {}

    for nucleus_name, roi in atlas.items():
        best_idx = -1
        best_overlap = 0.0

        for ic_idx in range(n_components):
            overlap = spatial_overlap(ic_maps[ic_idx], roi)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = ic_idx

        if best_overlap > threshold and best_idx >= 0:
            matches[nucleus_name] = {
                "component_idx": best_idx,
                "overlap": best_overlap,
                "name": nucleus_name,
            }

    return matches


def load_melodic_ics(
    melodic_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MELODIC IC spatial maps and mixing matrix.

    Parameters
    ----------
    melodic_dir : path to MELODIC output directory

    Returns
    -------
    ic_maps : spatial maps, shape (n_components, x, y, z)
    mixing : mixing matrix, shape (n_timepoints, n_components)
    """
    import nibabel as nib
    from pathlib import Path

    mdir = Path(melodic_dir)

    # IC spatial maps
    ic_img = nib.load(str(mdir / "melodic_IC.nii.gz"))
    ic_data = ic_img.get_fdata()
    # 4D → (x, y, z, n_components) → (n_components, x, y, z)
    ic_maps = np.moveaxis(ic_data, -1, 0)

    # Mixing matrix (time courses)
    mixing = np.loadtxt(str(mdir / "melodic_mix"))

    return ic_maps, mixing
