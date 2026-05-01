#!/usr/bin/env bash
# Register a session's FreeSurfer aparc+aseg + brainmask into the
# CVR / hypercapnia BOLD native space using FSL `epi_reg` (BBR cost
# function with the WM segmentation derived from FreeSurfer wmparc).
#
# Usage:
#   register_freesurfer_to_func.sh <subject> <session>
# e.g.
#   register_freesurfer_to_func.sh sub-1003 ses-wave1
#
# Outputs (under /data/datasets/smri-fm-cmp/vpjax/...):
#   anat-in-cvr/aparc+aseg.nii.gz
#   anat-in-cvr/brainmask.nii.gz
#   anat-in-cvr/epi_to_t1.mat
#   anat-in-cvr/t1_to_epi.mat
#
# Inputs assumed to exist:
#   /data/datasets/smri-fm-cmp/freesurfer/ds004856/<subject>_<session>/mri/{nu,brain,aparc+aseg,brainmask,wmparc}.mgz
#   /data/datasets/smri-fm-cmp/fsl/ds004856/cvr/<subject>_<session>_run-1/mean_func_brain.nii.gz

set -euo pipefail

SUBJECT="${1:?subject required (e.g. sub-1003)}"
SESSION="${2:?session required (e.g. ses-wave1)}"
# Optional 3rd arg picks the EPI target.  Default 'cvr' uses
# task-Hypercapnia preprocessing.  When that's absent (DLBS subjects
# without CVR), pass 'rest' — DLBS task BOLDs all share the same EPI
# grid, so the resulting aparc covers rest + task BOLDs equally.
TARGET="${3:-cvr}"

DS="ds004856"
FS_ROOT="/data/datasets/smri-fm-cmp/freesurfer/${DS}/${SUBJECT}_${SESSION}/mri"
case "${TARGET}" in
    cvr)   FUNC_DIR="/data/datasets/smri-fm-cmp/fsl/${DS}/cvr/${SUBJECT}_${SESSION}_run-1" ;;
    rest)  FUNC_DIR="/data/datasets/smri-fm-cmp/fsl/${DS}/rest/${SUBJECT}_${SESSION}_run-1" ;;
    *)     FUNC_DIR="/data/datasets/smri-fm-cmp/fsl/${DS}/task/${TARGET}/${SUBJECT}_${SESSION}_run-1" ;;
esac
CVR_DIR="${FUNC_DIR}"   # name kept for local readability of remaining script
OUT_DIR="/data/datasets/smri-fm-cmp/vpjax/${DS}/${SUBJECT}/${SESSION}/anat-in-cvr"
WORK_DIR="${OUT_DIR}/_work"

if [[ ! -d "${FS_ROOT}" ]]; then
    echo "FreeSurfer dir not found: ${FS_ROOT}" >&2
    exit 2
fi

# CVR (hypercapnia BOLD) preprocessing is optional for some subjects;
# when its output is absent we still run the FS .mgz→.nii.gz conversions
# (downstream PET / ASL stages need the T1-space aparc + brain volumes
# under _work/), and just skip the BBR registration to EPI space.
HAVE_CVR=1
if [[ ! -f "${CVR_DIR}/mean_func_brain.nii.gz" ]]; then
    echo "CVR mean_func_brain not found: ${CVR_DIR}/mean_func_brain.nii.gz" >&2
    echo "  → doing FS conversions only (no BBR registration)" >&2
    HAVE_CVR=0
fi

mkdir -p "${OUT_DIR}" "${WORK_DIR}"

# 1) Convert FreeSurfer .mgz files to .nii.gz preserving the affine.
python - <<PY
import os
import nibabel as nib
fs = "${FS_ROOT}"
work = "${WORK_DIR}"
for src, dst in [
    ("nu.mgz",          "nu.nii.gz"),
    ("brain.mgz",       "brain.nii.gz"),
    ("aparc+aseg.mgz",  "aparc+aseg.nii.gz"),
    ("brainmask.mgz",   "brainmask.nii.gz"),
    ("wmparc.mgz",      "wmparc.nii.gz"),
]:
    img = nib.load(os.path.join(fs, src))
    nib.save(nib.Nifti1Image(img.get_fdata(), img.affine, img.header),
             os.path.join(work, dst))
print("converted FreeSurfer mgz → nii.gz")
PY

# 2) Build a binary WM mask from wmparc.
#    FreeSurfer cortical WM labels: 2 (Left-Cerebral-WM), 41 (Right-Cerebral-WM),
#    plus 7/46 (cerebellar WM) and 251-255 (corpus-callosum sub-regions).
python - <<PY
import nibabel as nib
import numpy as np
img = nib.load("${WORK_DIR}/wmparc.nii.gz")
wm_labels = {2, 41, 7, 46, 251, 252, 253, 254, 255}
wm_mask = np.isin(img.get_fdata().astype(np.int32), list(wm_labels)).astype(np.float32)
nib.save(nib.Nifti1Image(wm_mask, img.affine, img.header),
         "${WORK_DIR}/wmseg.nii.gz")
print(f"wmseg voxels: {int(wm_mask.sum())}")
PY

if [[ "${HAVE_CVR}" == "0" ]]; then
    echo "done (FS conversions only) — outputs at ${OUT_DIR}/_work/"
    exit 0
fi

# 3) Run epi_reg (BBR EPI → T1).
EPI_REG_OUT="${WORK_DIR}/epi_in_t1"
echo "running epi_reg ..."
epi_reg --epi="${CVR_DIR}/mean_func_brain.nii.gz" \
        --t1="${WORK_DIR}/nu.nii.gz" \
        --t1brain="${WORK_DIR}/brain.nii.gz" \
        --wmseg="${WORK_DIR}/wmseg.nii.gz" \
        --out="${EPI_REG_OUT}"

# 4) Invert the transform to get T1 → EPI.
convert_xfm -omat "${OUT_DIR}/t1_to_epi.mat" -inverse "${EPI_REG_OUT}.mat"
cp "${EPI_REG_OUT}.mat" "${OUT_DIR}/epi_to_t1.mat"

# 5) Apply T1 → EPI transform to aparc and brainmask (nearest-neighbour).
flirt -in "${WORK_DIR}/aparc+aseg.nii.gz" \
      -ref "${CVR_DIR}/mean_func_brain.nii.gz" \
      -applyxfm -init "${OUT_DIR}/t1_to_epi.mat" \
      -interp nearestneighbour \
      -out "${OUT_DIR}/aparc+aseg.nii.gz"

flirt -in "${WORK_DIR}/brainmask.nii.gz" \
      -ref "${CVR_DIR}/mean_func_brain.nii.gz" \
      -applyxfm -init "${OUT_DIR}/t1_to_epi.mat" \
      -interp nearestneighbour \
      -out "${OUT_DIR}/brainmask.nii.gz"

echo "done — outputs at ${OUT_DIR}"
