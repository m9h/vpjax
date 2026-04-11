#!/bin/bash
# submit_wand_preprocess.sh — Submit FSL preprocessing for all WAND subjects
#
# Phase 1: oxford_asl (perfusion quantification from pCASL)
# Phase 2: vpjax pipeline (runs after preprocessing, as dependent jobs)
#
# Usage:
#   bash scripts/submit_wand_preprocess.sh          # submit all
#   bash scripts/submit_wand_preprocess.sh --dry     # preview
#   bash scripts/submit_wand_preprocess.sh --vpjax-only  # skip preprocessing
set -euo pipefail

WAND_DIR=/data/raw/wand
DERIV_DIR="${WAND_DIR}/derivatives"
VPJAX_DIR=/home/mhough/dev/vpjax
LOG_DIR="${DERIV_DIR}/logs/wand-batch"
PROCESS_SCRIPT="${VPJAX_DIR}/scripts/process_wand_vpjax.py"

# FSL setup
export FSLDIR=${FSLDIR:-/home/mhough/fsl}
export PATH="${FSLDIR}/share/fsl/bin:${FSLDIR}/bin:${PATH}"
export FSLOUTPUTTYPE=NIFTI_GZ

DRY_RUN=false
VPJAX_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --dry) DRY_RUN=true ;;
        --vpjax-only) VPJAX_ONLY=true ;;
    esac
done

mkdir -p "${LOG_DIR}"

PREPROC_SUBMITTED=0
PREPROC_SKIPPED=0
VPJAX_SUBMITTED=0
VPJAX_SKIPPED=0

for SUB_DIR in "${WAND_DIR}"/sub-*; do
    SUB=$(basename "${SUB_DIR}")

    # ---------------------------------------------------------------
    # Phase 1: oxford_asl (perfusion from pCASL)
    # ---------------------------------------------------------------
    PERF_DIR="${SUB_DIR}/ses-03/perf"
    ASL_INPUT="${PERF_DIR}/${SUB}_ses-03_acq-PCASL_cbf.nii.gz"
    M0_INPUT="${PERF_DIR}/${SUB}_ses-03_dir-AP_m0scan.nii.gz"
    ASL_OUT="${DERIV_DIR}/perfusion/${SUB}/oxford_asl"
    PREPROC_JOBID=""

    if ! $VPJAX_ONLY; then
        if [ -d "${ASL_OUT}" ]; then
            PREPROC_SKIPPED=$((PREPROC_SKIPPED + 1))
        elif [ ! -f "${ASL_INPUT}" ]; then
            echo "[SKIP] ${SUB}: no pCASL data"
        else
            # Build oxford_asl command
            ASL_CMD="export FSLDIR=${FSLDIR} && export PATH=${FSLDIR}/share/fsl/bin:${FSLDIR}/bin:\${PATH} && export FSLOUTPUTTYPE=NIFTI_GZ"
            ASL_CMD="${ASL_CMD} && mkdir -p ${DERIV_DIR}/perfusion/${SUB}"
            ASL_CMD="${ASL_CMD} && oxford_asl -i ${ASL_INPUT}"
            ASL_CMD="${ASL_CMD} --casl --bolus=1.8 --plds=2.0 --iaf=tc"

            # Add M0 calibration if available
            if [ -f "${M0_INPUT}" ]; then
                ASL_CMD="${ASL_CMD} -c ${M0_INPUT} --cmethod=voxel"
            fi

            # Add structural T1 if available
            T1_INPUT="${SUB_DIR}/ses-03/anat/${SUB}_ses-03_T1w.nii.gz"
            if [ -f "${T1_INPUT}" ]; then
                ASL_CMD="${ASL_CMD} -s ${T1_INPUT}"
            fi

            ASL_CMD="${ASL_CMD} --t1b=1.65 -o ${ASL_OUT}"

            if $DRY_RUN; then
                echo "[DRY-ASL] ${SUB}"
            else
                PREPROC_JOBID=$(sbatch --parsable \
                    --job-name="asl-${SUB}" \
                    --partition=batch \
                    --nice=100 \
                    --cpus-per-task=2 \
                    --mem=4G \
                    --time=1:00:00 \
                    --output="${LOG_DIR}/asl_${SUB}_%j.out" \
                    --error="${LOG_DIR}/asl_${SUB}_%j.err" \
                    --wrap="${ASL_CMD}")
                echo "[ASL] ${SUB} → job ${PREPROC_JOBID}"
            fi
            PREPROC_SUBMITTED=$((PREPROC_SUBMITTED + 1))
        fi
    fi

    # ---------------------------------------------------------------
    # Phase 2: vpjax pipeline (depends on preprocessing)
    # ---------------------------------------------------------------
    VPJAX_DONE="${DERIV_DIR}/vpjax/${SUB}/perfusion/perfusion_summary.json"
    if [ -f "${VPJAX_DONE}" ]; then
        VPJAX_SKIPPED=$((VPJAX_SKIPPED + 1))
        continue
    fi

    VPJAX_CMD="cd ${VPJAX_DIR} && ${HOME}/.local/bin/uv run --extra validation python ${PROCESS_SCRIPT} --subject ${SUB}"

    DEPEND_ARG=""
    if [ -n "${PREPROC_JOBID}" ]; then
        DEPEND_ARG="--dependency=afterany:${PREPROC_JOBID}"
    fi

    if $DRY_RUN; then
        echo "[DRY-VPJAX] ${SUB}${DEPEND_ARG:+ (after ${PREPROC_JOBID})}"
    else
        sbatch \
            --job-name="vpjax-${SUB}" \
            --partition=batch \
            --nice=100 \
            --cpus-per-task=2 \
            --mem=6G \
            --time=4:00:00 \
            ${DEPEND_ARG} \
            --output="${LOG_DIR}/vpjax_${SUB}_%j.out" \
            --error="${LOG_DIR}/vpjax_${SUB}_%j.err" \
            --export=ALL,XLA_FLAGS="--xla_cpu_multi_thread_eigen=false",JAX_PLATFORMS=cpu,FSLDIR=${FSLDIR} \
            --wrap="${VPJAX_CMD}"
    fi
    VPJAX_SUBMITTED=$((VPJAX_SUBMITTED + 1))
done

echo ""
echo "=== Summary ==="
echo "ASL preprocessing:  ${PREPROC_SUBMITTED} submitted, ${PREPROC_SKIPPED} skipped"
echo "vpjax pipeline:     ${VPJAX_SUBMITTED} submitted, ${VPJAX_SKIPPED} skipped"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOG_DIR}/"
