"""Batch sleep validation across all ds003768 subjects.

GPU-accelerated processing using JAX. Designed to run on DGX Spark
with one subject at a time (disk-constrained).

Usage:
    # Single subject
    uv run --extra gpu python -m vpjax.validation.batch_sleep --subject 23

    # All subjects (if data available)
    uv run --extra gpu python -m vpjax.validation.batch_sleep --all

    # Aggregate existing results
    uv run python -m vpjax.validation.batch_sleep --aggregate
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "ds003768"


def validate_subject(subject: int) -> list[dict] | None:
    """Run full sleep validation for one subject on GPU."""
    from vpjax.validation.run_all_sleep_runs import run_validation

    sub_dir = DATA_DIR / f"sub-{subject:02d}"
    staging = DATA_DIR / "sourcedata" / f"sub-{subject:02d}-sleep-stage.tsv"

    if not sub_dir.exists():
        print(f"  sub-{subject:02d}: data not found, skipping")
        return None
    if not staging.exists():
        print(f"  sub-{subject:02d}: no staging file, skipping")
        return None

    print(f"  sub-{subject:02d}: running on {jax.devices()[0]}...")

    try:
        results = run_validation(subject=subject, use_vasomotion=True)
        return results
    except Exception as e:
        print(f"  sub-{subject:02d}: ERROR - {e}")
        return None


def aggregate_results() -> dict:
    """Aggregate validation results across all subjects."""
    all_results = {}

    for f in sorted(DATA_DIR.glob("sub-*_validation_results.json")):
        sub = f.stem.replace("_validation_results", "")
        with open(f) as fh:
            all_results[sub] = json.load(fh)

    if not all_results:
        print("No results found to aggregate")
        return {}

    # Compute summary statistics per model per stage
    models = ["r_balloon", "r_vasomotion", "r_full"]
    stages = set()
    for sub_results in all_results.values():
        for r in sub_results:
            stages.add(r["stage"])

    summary = {}
    for model in models:
        summary[model] = {}
        for stage in sorted(stages):
            rs = []
            for sub_results in all_results.values():
                for r in sub_results:
                    if r["stage"] == stage and model in r:
                        rs.append(r[model])
            if rs:
                summary[model][stage] = {
                    "mean_r": float(np.mean(rs)),
                    "std_r": float(np.std(rs)),
                    "median_r": float(np.median(rs)),
                    "n_observations": len(rs),
                    "n_subjects": len(all_results),
                }

    return {
        "n_subjects": len(all_results),
        "subjects": list(all_results.keys()),
        "summary": summary,
    }


def print_aggregate(agg: dict):
    """Print formatted aggregate results."""
    if not agg:
        return

    print(f"\n=== Aggregate results across {agg['n_subjects']} subjects ===\n")

    models = ["r_balloon", "r_vasomotion", "r_full"]
    model_labels = {"r_balloon": "Balloon", "r_vasomotion": "+Vaso", "r_full": "Full"}

    # Header
    stages = sorted(set(
        stage for m in models
        if m in agg.get("summary", {})
        for stage in agg["summary"][m]
    ))

    print(f"{'Model':<12}", end="")
    for stage in stages:
        print(f" {'Stage ' + stage:>12}", end="")
    print(f" {'Overall':>12}")
    print("-" * (14 + 13 * (len(stages) + 1)))

    for model in models:
        if model not in agg.get("summary", {}):
            continue
        label = model_labels.get(model, model)
        print(f"{label:<12}", end="")
        all_rs = []
        for stage in stages:
            if stage in agg["summary"][model]:
                r = agg["summary"][model][stage]["mean_r"]
                all_rs.append(r)
                print(f" {r:>+12.3f}", end="")
            else:
                print(f" {'n/a':>12}", end="")
        if all_rs:
            print(f" {np.mean(all_rs):>+12.3f}")
        else:
            print()


def main():
    parser = argparse.ArgumentParser(description="Batch sleep validation")
    parser.add_argument("--subject", type=int, help="Single subject number")
    parser.add_argument("--all", action="store_true", help="Process all subjects")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate existing results")
    args = parser.parse_args()

    if args.aggregate:
        agg = aggregate_results()
        print_aggregate(agg)
        out = DATA_DIR / "aggregate_validation.json"
        with open(out, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nSaved to {out}")
        return

    print(f"JAX devices: {jax.devices()}")
    print(f"Data dir: {DATA_DIR}")

    subjects = []
    if args.subject:
        subjects = [args.subject]
    elif args.all:
        subjects = list(range(1, 34))
    else:
        parser.print_help()
        return

    for sub in subjects:
        results = validate_subject(sub)
        if results:
            out = DATA_DIR / f"sub-{sub:02d}_validation_results.json"
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  sub-{sub:02d}: saved {len(results)} results")


if __name__ == "__main__":
    main()
