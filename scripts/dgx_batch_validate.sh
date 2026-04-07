#!/bin/bash
# =============================================================
# DGX Spark batch validation: EEG-fMRI sleep models
# =============================================================
# Runs vpjax sleep validation across ds003768 subjects on GPU.
# Transfers one subject at a time to conserve disk (29GB free).
#
# Usage (from local machine):
#   ./scripts/dgx_batch_validate.sh [start_sub] [end_sub]
#
# Example:
#   ./scripts/dgx_batch_validate.sh 1 33   # all subjects
#   ./scripts/dgx_batch_validate.sh 23 23  # just sub-23
# =============================================================
set -euo pipefail

DGX="mhough@gx10-dgx-spark.local"
DGX_DIR="/home/mhough/dev/vpjax"
LOCAL_DATA="${HOME}/dev/vpjax/data/ds003768"
DGX_DATA="/data/raw/ds003768"

START_SUB="${1:-1}"
END_SUB="${2:-33}"

echo "=== DGX Spark EEG-fMRI Sleep Validation ==="
echo "Subjects: sub-${START_SUB} to sub-${END_SUB}"
echo ""

# Ensure DGX has latest code
echo ">>> Syncing code to DGX..."
ssh "${DGX}" "cd ${DGX_DIR} && git pull"

# Ensure staging data is on DGX (small, transfer once)
echo ">>> Transferring staging files..."
ssh "${DGX}" "mkdir -p ${DGX_DATA}/sourcedata"
scp -q "${LOCAL_DATA}/sourcedata/"*.tsv "${DGX}:${DGX_DATA}/sourcedata/" 2>/dev/null || true
scp -q "${LOCAL_DATA}/"*.json "${DGX}:${DGX_DATA}/" 2>/dev/null || true

# Process each subject
for sub_num in $(seq "${START_SUB}" "${END_SUB}"); do
    sub=$(printf "sub-%02d" "${sub_num}")
    echo ""
    echo ">>> Processing ${sub}..."

    # Check if subject data exists locally
    if [ ! -d "${LOCAL_DATA}/${sub}" ]; then
        echo "    Downloading ${sub} from S3..."
        aws s3 cp "s3://openneuro.org/ds003768/${sub}/" \
            "${LOCAL_DATA}/${sub}/" --recursive --no-sign-request --quiet
    fi

    # Check if staging exists for this subject
    if [ ! -f "${LOCAL_DATA}/sourcedata/${sub}-sleep-stage.tsv" ]; then
        echo "    No staging file for ${sub}, skipping"
        continue
    fi

    # Transfer subject to DGX
    echo "    Transferring to DGX..."
    ssh "${DGX}" "mkdir -p ${DGX_DATA}/${sub}"
    rsync -az --progress "${LOCAL_DATA}/${sub}/" "${DGX}:${DGX_DATA}/${sub}/"

    # Run validation on GPU
    echo "    Running validation on GPU..."
    ssh "${DGX}" "cd ${DGX_DIR} && uv run --extra gpu --extra dev \
        python -c \"
from vpjax.validation.run_all_sleep_runs import run_validation, print_results
import json, os
results = run_validation(subject=${sub_num})
print_results(results)
out = '${DGX_DATA}/${sub}_validation_results.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {out}')
\"" 2>&1 | tee "data/ds003768/${sub}_gpu_validation.log"

    # Pull results back
    echo "    Fetching results..."
    scp -q "${DGX}:${DGX_DATA}/${sub}_validation_results.json" \
        "${LOCAL_DATA}/${sub}_validation_results.json" 2>/dev/null || true

    echo "    Done with ${sub}"
done

echo ""
echo "=== Batch validation complete ==="
echo "Results in: ${LOCAL_DATA}/*_validation_results.json"
