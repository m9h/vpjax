"""Run validation across all sleep runs of a subject.

Computes predicted vs measured BOLD spectra per sleep stage
across all available runs, quantifies fit with Pearson correlation.

Usage:
    uv run python -m vpjax.validation.run_all_sleep_runs
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from vpjax.validation.sleep_eeg_fmri import (
    load_bold_global,
    load_sleep_stages,
    bold_spectrum_by_stage,
    predict_bold_spectrum_for_stage,
    predict_bold_spectrum_with_vasomotion,
)


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "ds003768"


def run_validation(
    subject: int = 23,
    use_vasomotion: bool = True,
) -> list[dict]:
    """Run validation for all sleep runs of a subject.

    Returns list of result dicts with keys:
        run, stage, n_volumes, r_value, p_value, r_vasomotion, p_vasomotion
    """
    sub_dir = DATA_DIR / f"sub-{subject:02d}"
    staging_file = DATA_DIR / "sourcedata" / f"sub-{subject:02d}-sleep-stage.tsv"

    if not sub_dir.exists():
        raise FileNotFoundError(f"Subject data not found: {sub_dir}")

    stages = load_sleep_stages(str(staging_file), subject=subject)

    # Find all sleep BOLD runs
    func_dir = sub_dir / "func"
    bold_files = sorted(func_dir.glob(f"sub-{subject:02d}_task-sleep_run-*_bold.nii.gz"))

    results = []

    for bold_path in bold_files:
        run_name = bold_path.stem.replace("_bold.nii", "").replace(f"sub-{subject:02d}_", "")
        session_key = run_name.replace("_bold", "")

        run_stages = stages.get(session_key, [])
        if not run_stages:
            print(f"  {run_name}: no staging data, skipping")
            continue

        print(f"  Loading {run_name}...")
        ts, tr = load_bold_global(str(bold_path))
        measured = bold_spectrum_by_stage(ts, tr, run_stages)

        for stage_label, (freqs, power) in measured.items():
            if len(power) < 5:
                continue

            # Balloon-only prediction
            predicted = np.array(predict_bold_spectrum_for_stage(stage_label, freqs, tr))

            # Correlation in log space (more meaningful for spectra)
            log_measured = np.log10(np.clip(power, 1e-20, None))
            log_predicted = np.log10(np.clip(predicted, 1e-20, None))

            if np.std(log_measured) < 1e-10 or np.std(log_predicted) < 1e-10:
                continue

            r, p = pearsonr(log_measured, log_predicted)

            entry = {
                "run": session_key,
                "stage": stage_label,
                "n_volumes": int(np.sum([1 for _, s in run_stages if s == stage_label]) * 30 / tr),
                "r_balloon": float(r),
                "p_balloon": float(p),
            }

            # With vasomotion
            if use_vasomotion:
                pred_vaso = np.array(predict_bold_spectrum_with_vasomotion(stage_label, freqs, tr))
                log_pred_vaso = np.log10(np.clip(pred_vaso, 1e-20, None))
                if np.std(log_pred_vaso) > 1e-10:
                    r_v, p_v = pearsonr(log_measured, log_pred_vaso)
                    entry["r_vasomotion"] = float(r_v)
                    entry["p_vasomotion"] = float(p_v)

            results.append(entry)

    return results


def print_results(results: list[dict]):
    """Print results as a formatted table."""
    print(f"\n{'Run':<25} {'Stage':>5} {'N_vol':>6} {'r(Balloon)':>11} {'r(+Vaso)':>10}")
    print("-" * 62)
    for r in results:
        r_vaso = f"{r.get('r_vasomotion', float('nan')):+.3f}" if 'r_vasomotion' in r else "   n/a"
        print(f"{r['run']:<25} {r['stage']:>5} {r['n_volumes']:>6} "
              f"{r['r_balloon']:>+11.3f} {r_vaso:>10}")


def main():
    print("Running vpjax sleep validation on sub-23...")
    results = run_validation(subject=23)

    print_results(results)

    # Save results
    out_path = DATA_DIR / "sub-23_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
