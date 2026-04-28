#!/usr/bin/env bash
# Register a session's static PET image (amyloid 18F-AV45 or tau
# 18F-AV1451) to the FreeSurfer T1 space using FSL flirt with a
# normalized mutual-information cost function (cross-modal).
#
# We register PET → T1 (rigid 6-DOF) at T1 resolution.  The aparc+aseg
# from FreeSurfer is already in this space, so SUVR + regional means
# can be computed directly from the resampled PET-in-T1.
#
# Usage:
#   register_pet_to_t1.sh <subject> <session> <tracer>
# tracer ∈ {amyloid, tau}
# e.g.
#   register_pet_to_t1.sh sub-1003 ses-wave1 amyloid

set -euo pipefail

SUBJECT="${1:?subject required}"
SESSION="${2:?session required}"
TRACER="${3:?tracer required (amyloid|tau)}"

case "${TRACER}" in
    amyloid)  TRC="18FAV45";  TRACER_DIR="amyloid_18FAV45" ;;
    tau)      TRC="18FAV1451"; TRACER_DIR="tau_18FAV1451" ;;
    *) echo "tracer must be amyloid|tau, got ${TRACER}" >&2; exit 2 ;;
esac

DS="ds004856"
RAW_PET="/data/raw/openneuro/${DS}/${SUBJECT}/${SESSION}/pet/${SUBJECT}_${SESSION}_trc-${TRC}_run-1_pet.nii.gz"
FS_DIR="/data/datasets/smri-fm-cmp/freesurfer/${DS}/${SUBJECT}_${SESSION}/mri"
ANAT_WORK="/data/datasets/smri-fm-cmp/vpjax/${DS}/${SUBJECT}/${SESSION}/anat-in-cvr/_work"
OUT_DIR="/data/datasets/smri-fm-cmp/vpjax/${DS}/${SUBJECT}/${SESSION}/pet/${TRACER_DIR}"

if [[ ! -f "${RAW_PET}" ]]; then
    echo "raw PET not found: ${RAW_PET}" >&2
    exit 2
fi
if [[ ! -d "${FS_DIR}" ]]; then
    echo "FreeSurfer dir not found: ${FS_DIR}" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

# Reuse the FreeSurfer mgz→nii.gz conversions done by
# register_freesurfer_to_func.sh if present.  Otherwise convert here.
if [[ -f "${ANAT_WORK}/nu.nii.gz" && -f "${ANAT_WORK}/brain.nii.gz" ]]; then
    NU="${ANAT_WORK}/nu.nii.gz"
    T1_BRAIN="${ANAT_WORK}/brain.nii.gz"
else
    echo "FS nu/brain niftis not found; converting on the fly" >&2
    mkdir -p "${OUT_DIR}/_anat"
    python - <<PY
import os, nibabel as nib
fs="${FS_DIR}"
out="${OUT_DIR}/_anat"
for src,dst in [("nu.mgz","nu.nii.gz"),("brain.mgz","brain.nii.gz")]:
    img=nib.load(os.path.join(fs,src))
    nib.save(nib.Nifti1Image(img.get_fdata(),img.affine,img.header),
             os.path.join(out,dst))
PY
    NU="${OUT_DIR}/_anat/nu.nii.gz"
    T1_BRAIN="${OUT_DIR}/_anat/brain.nii.gz"
fi

# Run flirt: PET → T1.  Use brain-extracted T1 as reference; PET has
# diffuse uptake throughout the brain so a brain-only reference gives a
# cleaner mutual-information objective than the full nu image.
echo "running flirt PET → T1 (mutualinfo, 6 DOF)..."
flirt -in "${RAW_PET}" \
      -ref "${T1_BRAIN}" \
      -out "${OUT_DIR}/pet_in_t1.nii.gz" \
      -omat "${OUT_DIR}/pet_to_t1.mat" \
      -dof 6 \
      -cost normmi \
      -interp trilinear

# Save the inverse transform too (T1 → PET native).
convert_xfm -omat "${OUT_DIR}/t1_to_pet.mat" -inverse "${OUT_DIR}/pet_to_t1.mat"

echo "done — outputs at ${OUT_DIR}"
ls -la "${OUT_DIR}"
