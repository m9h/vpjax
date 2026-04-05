"""Brainstem nuclei atlas definitions.

Defines MNI coordinates and approximate extents for key brainstem
nuclei relevant to cardiorespiratory and arousal physiology.

When the Brainstem Navigator atlas (Bianciardi et al.) is available,
use load_brainstem_navigator() to load probabilistic ROIs. Otherwise,
create_synthetic_atlas() generates spherical ROIs from MNI coordinates.

Nuclei relevant to vpjax:
- LC (locus coeruleus): norepinephrine → vasomotion, arousal
- DR (dorsal raphe): serotonin → mood, sleep regulation
- MR (median raphe): serotonin → circadian, sleep
- NTS (nucleus tractus solitarius): vagal afferents → baroreceptor
- PAG (periaqueductal gray): pain, autonomic control
- VTA (ventral tegmental area): dopamine → reward, motivation

References
----------
Bianciardi M et al. (2015) NeuroImage 107:1-29
    "A probabilistic atlas of the human brainstem"
Bianciardi M et al. (2018) NITRC: Brainstem Navigator
Singh K et al. (2022) Frontiers in Neuroimaging 1:1009399
    "A structural connectivity atlas of limbic brainstem nuclei"
"""

from __future__ import annotations

import numpy as np


# MNI coordinates (mm) and approximate radius for each nucleus
# From Bianciardi et al. (2015) and Brainstem Navigator v0.9
NUCLEI = {
    "LC": {
        "name": "Locus Coeruleus",
        "mni_x": 6.0, "mni_y": -37.0, "mni_z": -21.0,
        "radius_mm": 3.0,
        "bilateral": True,
        "neurotransmitter": "norepinephrine",
    },
    "DR": {
        "name": "Dorsal Raphe",
        "mni_x": 0.0, "mni_y": -30.0, "mni_z": -16.0,
        "radius_mm": 4.0,
        "bilateral": False,
        "neurotransmitter": "serotonin",
    },
    "MR": {
        "name": "Median Raphe",
        "mni_x": 0.0, "mni_y": -28.0, "mni_z": -24.0,
        "radius_mm": 3.5,
        "bilateral": False,
        "neurotransmitter": "serotonin",
    },
    "NTS": {
        "name": "Nucleus Tractus Solitarius",
        "mni_x": 4.0, "mni_y": -42.0, "mni_z": -46.0,
        "radius_mm": 3.0,
        "bilateral": True,
        "neurotransmitter": "multiple",
    },
    "PAG": {
        "name": "Periaqueductal Gray",
        "mni_x": 0.0, "mni_y": -32.0, "mni_z": -7.0,
        "radius_mm": 5.0,
        "bilateral": False,
        "neurotransmitter": "multiple",
    },
    "VTA": {
        "name": "Ventral Tegmental Area",
        "mni_x": 2.0, "mni_y": -16.0, "mni_z": -14.0,
        "radius_mm": 4.0,
        "bilateral": True,
        "neurotransmitter": "dopamine",
    },
}


def create_synthetic_atlas(
    shape: tuple[int, int, int] = (91, 109, 91),
    voxel_size_mm: float = 2.0,
) -> dict[str, np.ndarray]:
    """Create synthetic brainstem ROIs as spheres at MNI coordinates.

    Creates binary masks for each nucleus by placing a sphere at its
    MNI coordinate, transformed to voxel space assuming standard
    MNI152 2mm template layout.

    Parameters
    ----------
    shape : volume shape (x, y, z)
    voxel_size_mm : voxel size in mm

    Returns
    -------
    Dict mapping nucleus name → binary mask array
    """
    atlas = {}

    for name, info in NUCLEI.items():
        mask = np.zeros(shape, dtype=np.float32)

        # MNI to voxel (approximate, assuming standard MNI152 origin)
        # MNI origin is approximately at voxel (45, 63, 36) for 2mm
        origin = np.array([shape[0] // 2, shape[1] // 2 + 8, shape[2] // 2 - 9])
        vx = int(origin[0] + info["mni_x"] / voxel_size_mm)
        vy = int(origin[1] + info["mni_y"] / voxel_size_mm)
        vz = int(origin[2] + info["mni_z"] / voxel_size_mm)

        radius_vox = info["radius_mm"] / voxel_size_mm

        # Create sphere
        for dx in range(-int(radius_vox) - 1, int(radius_vox) + 2):
            for dy in range(-int(radius_vox) - 1, int(radius_vox) + 2):
                for dz in range(-int(radius_vox) - 1, int(radius_vox) + 2):
                    x, y, z = vx + dx, vy + dy, vz + dz
                    if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                        dist = np.sqrt(dx**2 + dy**2 + dz**2)
                        if dist <= radius_vox:
                            mask[x, y, z] = 1.0

        # Bilateral: mirror across midline
        if info.get("bilateral", False):
            vx_mirror = int(origin[0] - info["mni_x"] / voxel_size_mm)
            for dx in range(-int(radius_vox) - 1, int(radius_vox) + 2):
                for dy in range(-int(radius_vox) - 1, int(radius_vox) + 2):
                    for dz in range(-int(radius_vox) - 1, int(radius_vox) + 2):
                        x, y, z = vx_mirror + dx, vy + dy, vz + dz
                        if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                            dist = np.sqrt(dx**2 + dy**2 + dz**2)
                            if dist <= radius_vox:
                                mask[x, y, z] = 1.0

        atlas[name] = mask

    return atlas


def load_brainstem_navigator(
    atlas_dir: str,
    shape: tuple[int, int, int] = (91, 109, 91),
) -> dict[str, np.ndarray]:
    """Load Brainstem Navigator probabilistic atlas from NITRC.

    Parameters
    ----------
    atlas_dir : path to Brainstem Navigator atlas directory
    shape : expected volume shape

    Returns
    -------
    Dict mapping nucleus name → probabilistic mask (0-1)
    """
    import nibabel as nib
    from pathlib import Path

    atlas_path = Path(atlas_dir)
    atlas = {}

    # Map our names to Brainstem Navigator file naming convention
    file_map = {
        "LC": "LC",
        "DR": "DR",
        "MR": "MnR",
        "NTS": "Sol",  # Solitary nucleus
        "PAG": "PAG",
        "VTA": "VTA",
    }

    for our_name, bn_name in file_map.items():
        # Try common naming patterns
        for pattern in [f"*{bn_name}*.nii.gz", f"*{bn_name}*.nii"]:
            matches = list(atlas_path.glob(pattern))
            if matches:
                img = nib.load(str(matches[0]))
                data = img.get_fdata()
                # Normalize to 0-1 if needed
                if data.max() > 1:
                    data = data / data.max()
                atlas[our_name] = data.astype(np.float32)
                break

    return atlas
