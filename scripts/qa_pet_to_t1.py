#!/usr/bin/env python3
"""Visual QA for PET→T1 registration of a single DLBS session.

Renders a 3×3 axial mosaic of the FreeSurfer T1 (grayscale) with the
registered PET image overlaid in a hot colormap.  Saves the mosaic PNG
next to the registered PET and prints a few sanity-check numbers
(cortex/cerebellum SUVR, brain-mean PET, rough Dice between brain mask
and PET-above-threshold).

Usage:
    python qa_pet_to_t1.py sub-1003 ses-wave1 amyloid
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

CEREBELLUM_LABELS = (7, 8, 46, 47)
CORTICAL_LABEL_MIN = 1000


def _vpjax_dir(subject: str, session: str) -> Path:
    return Path(f"/data/datasets/smri-fm-cmp/vpjax/ds004856/{subject}/{session}")


def main(subject: str, session: str, tracer: str) -> int:
    suffix = {"amyloid": "amyloid_18FAV45", "tau": "tau_18FAV1451"}[tracer]
    pet_dir = _vpjax_dir(subject, session) / "pet" / suffix
    work = _vpjax_dir(subject, session) / "anat-in-cvr" / "_work"

    t1 = nib.load(str(work / "brain.nii.gz")).get_fdata()
    pet = nib.load(str(pet_dir / "pet_in_t1.nii.gz")).get_fdata()
    aparc = nib.load(str(work / "aparc+aseg.nii.gz")).get_fdata().astype(int)

    brain = aparc > 0
    cb_mask = np.isin(aparc, CEREBELLUM_LABELS)
    cortex_mask = aparc >= CORTICAL_LABEL_MIN  # DKT cortical labels start at 1000

    cb_mean = float(pet[cb_mask].mean())
    cortex_mean = float(pet[cortex_mask].mean())
    brain_mean = float(pet[brain].mean())

    # Rough Dice: T1 brain mask vs PET above its 5th percentile within brain
    pet_brain = pet[brain]
    pet_thresh = float(np.percentile(pet_brain, 5))
    pet_mask = pet > pet_thresh
    inter = int((brain & pet_mask).sum())
    union = int(brain.sum() + pet_mask.sum())
    dice = (2.0 * inter) / max(union, 1)

    print(f"[qa_pet] cerebellum-mean PET = {cb_mean:.1f}")
    print(f"[qa_pet] cortex-mean PET     = {cortex_mean:.1f}")
    print(f"[qa_pet] cortex/cerebellum SUVR = {cortex_mean/cb_mean:.3f}")
    print(f"[qa_pet] brain-mean PET      = {brain_mean:.1f}")
    print(f"[qa_pet] rough Dice (brain ∩ PET>p5) = {dice:.3f}")

    # Pick 9 evenly-spaced axial slices through the brain.
    z_with_brain = np.where(brain.any(axis=(0, 1)))[0]
    if z_with_brain.size < 9:
        z_idx = np.linspace(0, brain.shape[2] - 1, 9).astype(int)
    else:
        lo, hi = z_with_brain[0], z_with_brain[-1]
        z_idx = np.linspace(lo + (hi - lo) * 0.15,
                            lo + (hi - lo) * 0.85, 9).astype(int)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    pet_min, pet_max = float(np.percentile(pet[brain], 5)), float(np.percentile(pet[brain], 99))
    for ax, z in zip(axes.flat, z_idx):
        t1_slice = np.rot90(t1[:, :, z])
        pet_slice = np.rot90(pet[:, :, z])
        brain_slice = np.rot90(brain[:, :, z])
        ax.imshow(t1_slice, cmap="gray", interpolation="nearest")
        # Mask PET to brain-only and to a finite range for visibility.
        masked_pet = np.where(brain_slice, pet_slice, np.nan)
        ax.imshow(masked_pet, cmap="hot", alpha=0.5,
                  vmin=pet_min, vmax=pet_max,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"z={z}", fontsize=8)
    fig.suptitle(
        f"{subject} {session} {tracer} PET → T1 (cortex/cb SUVR={cortex_mean/cb_mean:.3f}, "
        f"Dice={dice:.3f})",
        fontsize=10,
    )
    fig.tight_layout()

    out_png = pet_dir / "qa_overlay.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[qa_pet] mosaic written to {out_png}")

    # Flag obviously-bad registrations.
    if not (1.0 < (cortex_mean / cb_mean) < 1.6):
        print(f"[qa_pet] WARNING: cortex/cerebellum SUVR {cortex_mean/cb_mean:.3f} "
              "is outside [1.0, 1.6] — registration may be misaligned")
    if dice < 0.7:
        print(f"[qa_pet] WARNING: Dice {dice:.3f} is below 0.7 — likely misaligned")

    return 0


if __name__ == "__main__":
    sub, ses, trc = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.exit(main(sub, ses, trc))
