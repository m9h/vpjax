#!/bin/bash
# submit_wand_dgx.sh — Single long-running job for all WAND vpjax processing
#
# Submits ONE job that processes all subjects sequentially.
# No preemption of other GPU work — takes the GPU once when available.
#
# Usage (on DGX Spark):
#   cd ~/dev/vpjax && bash scripts/submit_wand_dgx.sh
#   bash scripts/submit_wand_dgx.sh --dry
set -euo pipefail

VPJAX_DIR="${HOME}/dev/vpjax"
LOG_DIR="/data/raw/wand/derivatives/logs/vpjax-dgx"
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

if $DRY_RUN; then
    echo "[DRY] Would submit single job processing all subjects"
    echo "Command: uv run --extra validation --extra gpu python ${SCRIPT} --all ${STAGE_ARG}"
    exit 0
fi

JOB_ID=$(sbatch --parsable \
    --job-name="vpjax-wand-all" \
    --partition=gpu \
    --nice=1000 \
    --cpus-per-task=4 \
    --mem=16G \
    --gres=gpu:1 \
    --time=7-00:00:00 \
    --output="${LOG_DIR}/wand_all_%j.out" \
    --error="${LOG_DIR}/wand_all_%j.err" \
    --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false,XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 \
    --wrap="cd ${VPJAX_DIR} && uv run --extra validation --extra gpu python ${SCRIPT} --all ${STAGE_ARG}")

echo "Submitted job ${JOB_ID} — single job processes all subjects sequentially"
echo "Monitor: squeue -u \$USER"
echo "Progress: tail -f ${LOG_DIR}/wand_all_${JOB_ID}.err"
