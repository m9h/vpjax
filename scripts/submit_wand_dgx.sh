#!/bin/bash
# submit_wand_dgx.sh — Submit vpjax WAND processing on DGX Spark (GPU)
#
# ASL preprocessing is already done (shared /data/raw/wand/derivatives/).
# This submits vpjax modeling jobs on the GPU partition.
#
# Usage (on DGX Spark):
#   cd ~/dev/vpjax && bash scripts/submit_wand_dgx.sh
#   bash scripts/submit_wand_dgx.sh --dry
set -euo pipefail

WAND_DIR=/data/raw/wand
VPJAX_DIR="${HOME}/dev/vpjax"
LOG_DIR="${WAND_DIR}/derivatives/logs/vpjax-dgx"
SCRIPT="${VPJAX_DIR}/scripts/process_wand_vpjax.py"

DRY_RUN=false
STAGE_ARG=""
for arg in "$@"; do
    case "$arg" in
        --dry) DRY_RUN=true ;;
        --stage=*) STAGE_ARG="--stage ${arg#*=}" ;;
    esac
done

mkdir -p "${LOG_DIR}"

SUBMITTED=0
SKIPPED=0

for SUB_DIR in "${WAND_DIR}"/sub-*; do
    SUB=$(basename "${SUB_DIR}")

    # Skip if all stages complete
    PERF_DONE="${WAND_DIR}/derivatives/vpjax/${SUB}/perfusion/perfusion_summary.json"
    HEMO_DONE="${WAND_DIR}/derivatives/vpjax/${SUB}/hemodynamics/balloon_params.json"

    if [ -z "${STAGE_ARG}" ] && [ -f "${PERF_DONE}" ] && [ -f "${HEMO_DONE}" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "[DRY] ${SUB}"
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    sbatch \
        --job-name="vpjax-${SUB}" \
        --partition=gpu \
        --nice=100 \
        --cpus-per-task=4 \
        --mem=16G \
        --gres=gpu:1 \
        --time=1:00:00 \
        --output="${LOG_DIR}/${SUB}_%j.out" \
        --error="${LOG_DIR}/${SUB}_%j.err" \
        --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false,XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 \
        --wrap="cd ${VPJAX_DIR} && uv run --extra validation --extra gpu python ${SCRIPT} --subject ${SUB} ${STAGE_ARG}"

    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "Submitted: ${SUBMITTED}  Skipped (done): ${SKIPPED}"
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOG_DIR}/"
