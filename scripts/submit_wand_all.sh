#!/bin/bash
# submit_wand_all.sh — Submit vpjax processing for all WAND subjects via SLURM
#
# Each subject is an independent low-priority job.  SLURM schedules them
# within available resources (16 CPUs, 30GB RAM on fedora-legion-nvidia).
#
# Usage:
#   bash scripts/submit_wand_all.sh          # submit all unprocessed subjects
#   bash scripts/submit_wand_all.sh --dry    # preview what would be submitted
#   bash scripts/submit_wand_all.sh --stage qbold   # only run qBOLD stage
set -euo pipefail

WAND_DIR=/data/raw/wand
VPJAX_DIR=/home/mhough/dev/vpjax
LOG_DIR="${WAND_DIR}/derivatives/logs/vpjax-batch"
SCRIPT="${VPJAX_DIR}/scripts/process_wand_vpjax.py"

DRY_RUN=false
STAGE_ARG=""
for arg in "$@"; do
    case "$arg" in
        --dry) DRY_RUN=true ;;
        --stage)  shift; STAGE_ARG="--stage $1" ;;
        --stage=*) STAGE_ARG="--stage ${arg#*=}" ;;
    esac
done

mkdir -p "${LOG_DIR}"

SUBMITTED=0
SKIPPED=0

for SUB_DIR in "${WAND_DIR}"/sub-*; do
    SUB=$(basename "${SUB_DIR}")

    # Skip if all stages complete (perfusion summary + balloon params exist)
    PERF_DONE="${WAND_DIR}/derivatives/vpjax/${SUB}/perfusion/perfusion_summary.json"
    HEMO_DONE="${WAND_DIR}/derivatives/vpjax/${SUB}/hemodynamics/balloon_params.json"

    if [ -z "${STAGE_ARG}" ] && [ -f "${PERF_DONE}" ] && [ -f "${HEMO_DONE}" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "[DRY] Would submit: ${SUB}"
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    sbatch \
        --job-name="vpjax-${SUB}" \
        --partition=batch \
        --nice=100 \
        --cpus-per-task=2 \
        --mem=6G \
        --time=4:00:00 \
        --output="${LOG_DIR}/${SUB}_%j.out" \
        --error="${LOG_DIR}/${SUB}_%j.err" \
        --export=ALL,XLA_FLAGS="--xla_cpu_multi_thread_eigen=false",JAX_PLATFORMS=cpu \
        --wrap="cd ${VPJAX_DIR} && python ${SCRIPT} --subject ${SUB} ${STAGE_ARG}"

    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "Submitted: ${SUBMITTED}  Skipped (already done): ${SKIPPED}"
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${LOG_DIR}/"
