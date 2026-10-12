"""Vary starts along constant g_bb*tau_a on identical simulated recordings.

Run from the repo: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m scripts.brain_body_initialization
Each completed fit is checkpointed; --resume skips saved cells.
SEs are log-scale. A fixed product here defines starts, not a fitted constraint.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from vpjax.validation.brain_body import simulate, fit_topology

NAMES = ("tau_z", "q_z", "tau_a", "q_a", "beta", "g_bb")



def summarize(rows):
    """Separate variation across starts of one recording from across seeds."""
    summaries = []
    for period in sorted({r["period"] for r in rows}):
        cells = [r for r in rows if r["period"] == period]
        groups = [[r for r in cells if r["seed"] == seed]
                  for seed in sorted({r["seed"] for r in cells})]
        best = [max(group, key=lambda r: r["fit"]["log_likelihood"]) for group in groups]
        def value(row, name):
            return row["product"] if name == "product" else row["fit"]["estimate"][name]
        metrics = {}
        for name in ("g_bb", "tau_a", "product"):
            vals = np.array([value(r, name) for r in best])
            metrics[name] = dict(
                max_within_seed_relative_range=max(
                    float(np.ptp([value(r, name) for r in group]) /
                          np.mean([value(r, name) for r in group])) for group in groups),
                across_seed_cv=float(np.std(vals) / abs(np.mean(vals))),
                median=float(np.median(vals)))
        product_ses = [r["product_log_se"] for r in best if r["product_log_se"] is not None]
        summaries.append(dict(period=period, n_fits=len(cells),
                         all_converged=all(r["fit"]["success"] for r in cells),
                         n_converged=sum(r["fit"]["success"] for r in cells),
                         all_positive_definite=all(r["fit"]["positive_definite"] for r in cells),
                         max_within_seed_ll_range=max(float(np.ptp(
                             [r["fit"]["log_likelihood"] for r in group])) for group in groups),
                         median_gain_log_se=float(np.median(
                             [r["fit"]["standard_error"]["g_bb"] for r in best])),
                         median_product_log_se=float(np.median(product_ses)) if product_ses else None,
                         metrics=metrics))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--periods", nargs="+", type=float, default=[32, 64])
    parser.add_argument("--factors", nargs="+", type=float, default=[0.25, 1, 4])
    parser.add_argument("--duration", type=float, default=480)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("docs/results/brain_body_initialization.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = dict(seeds=args.seeds, periods=args.periods, factors=args.factors,
                    duration=args.duration, max_steps=args.max_steps, restarts=0,
                    fit_names=NAMES, observers=("fast", "hr"), max_dt=0.25,
                    simulation_dt=0.01, snr=3, fast_dt=2, hr_dt=1,
                    drive="duty", onset=0, on_fraction=5/16,
                    q_z=0.01, q_a=0.01,
                    other_initial_values="0.5 times truth", product_start="truth product")
    settings = json.loads(json.dumps(settings))
    result = dict(settings=settings, provenance=dict(python=platform.python_version(),
                  jax=jax.__version__, backend=jax.default_backend(),
                  x64_enabled=bool(jax.config.jax_enable_x64),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_state="working tree; commit containing this artifact identifies source"), rows=[])
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text())
        if (result["settings"] != settings or
                result["provenance"].get("x64_enabled", False) != bool(jax.config.jax_enable_x64)):
            raise ValueError("saved settings differ; use a different output path")
    done = {(r["period"], r["seed"], r["factor"]) for r in result["rows"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for period in args.periods:
        for seed in args.seeds:
            sim = simulate("efferent", duration=args.duration, seed=seed,
                           drive="duty", period=period, onset=0,
                           fast_dt=2, hr_dt=1, snr=3)
            truth = sim["truth"]
            for factor in args.factors:
                if (period, seed, factor) in done:
                    continue
                start = time.monotonic()
                fit = fit_topology(sim, "efferent", ("fast", "hr"),
                                   fit_names=NAMES, restarts=0, max_steps=args.max_steps,
                                   init={"g_bb": truth["g_bb"] * factor,
                                         "tau_a": truth["tau_a"] / factor})
                i, j = NAMES.index("g_bb"), NAMES.index("tau_a")
                se_g, se_t = (fit["standard_error"][n] for n in ("g_bb", "tau_a"))
                corr = fit["correlation"][i][j]
                variance = se_g**2 + se_t**2 + 2*corr*se_g*se_t
                row = dict(period=period, seed=seed, factor=factor,
                           truth={n: truth[n] for n in NAMES}, fit=fit,
                           product=fit["estimate"]["g_bb"] * fit["estimate"]["tau_a"],
                           product_log_se=float(np.sqrt(variance)) if fit["positive_definite"] and variance >= 0 else None,
                           elapsed_s=time.monotonic()-start)
                result["rows"].append(row)
                # Refuse nonfinite JSON so a failed numerical run is visible.
                encoded = json.dumps(result, indent=2, allow_nan=False)
                temporary = args.output.with_suffix(".json.tmp")
                temporary.write_text(encoded + "\n")
                temporary.replace(args.output)
                e = fit["estimate"]
                print(f"period={period:g} seed={seed} start={factor:g}: g={e['g_bb']:.6g} tau={e['tau_a']:.6g} product={row['product']:.6g} ll={fit['log_likelihood']:.6f} success={fit['success']} logSE={se_g:.4g} corr={corr:.4g} ({row['elapsed_s']:.1f}s)", flush=True)

    result["summary"] = summarize(result["rows"])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
