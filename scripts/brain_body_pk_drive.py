"""Step 1 with a pharmacokinetic drive: the topology confusion matrix driven by an
effect-site occupancy computed from an explicit assumptions file, instead of the
Bateman shape of ``scripts/brain_body_confusion.py``.

The chain is dose -> plasma -> effect site -> Hill occupancy
(:mod:`vpjax.pharmacokinetics`), evaluated on the model's seconds axis, shifted to
``--onset`` and normalised to unit peak so ``beta`` keeps its meaning as the peak
drive; the absolute occupancy scale is recorded in the output.  Every PK/PD number
comes from the assumptions JSON, which must carry a ``source``.

    python scripts/brain_body_pk_drive.py examples/assumptions_synthetic.json \\
        --seeds 0 1 2 --output docs/results/brain_body_confusion_pk_drive.json

Nothing here is specific to any compound: the assumptions file is the only place
a compound could enter, and the example shipped is a labelled synthetic one.
"""

import argparse
import json

import jax
import numpy as np

from vpjax.pharmacokinetics import drive_series_from_pk, effect_site_drive, load_assumptions
from vpjax.validation.brain_body import CHANNELS, TOPOLOGIES, fit_topology, simulate


def main():
    jax.config.update("jax_enable_x64", True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("assumptions", help="JSON of PK/PD assumptions with a cited source")
    p.add_argument("--density", help=".npy regional receptor density (one region is used, --region)")
    p.add_argument("--region", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--duration", type=float, default=900.0)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--fast-dt", type=float, default=2.0)
    p.add_argument("--hr-dt", type=float, default=1.0)
    p.add_argument("--onset", type=float, default=120.0, help="s after run start at which the regimen begins")
    p.add_argument("--snr", type=float, default=3.0)
    p.add_argument("--q-z", type=float, default=0.01)
    p.add_argument("--q-a", type=float, default=0.01)
    p.add_argument("--loop-gain", type=float, default=0.25)
    p.add_argument("--observers", nargs="+", default=list(CHANNELS), help=f"subset of {CHANNELS}")
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--only", nargs="+", choices=sorted(TOPOLOGIES), help="generating topologies to run (rows)")
    p.add_argument("--output")
    a = p.parse_args()

    spec = load_assumptions(a.assumptions)
    density = np.load(a.density) if a.density else None
    pk = effect_site_drive(spec, density=density)
    meta = pk["metadata"]
    if pk["time_s"][-1] < a.duration - a.onset:
        print(f"warning: PK window {pk['time_s'][-1]:.0f} s is shorter than the run after onset "
              f"({a.duration - a.onset:.0f} s); the drive holds its last value beyond it")
    series = (pk["time_s"], drive_series_from_pk(pk["time_s"], pk, onset_s=0.0, region=a.region))
    t_probe = np.arange(0.0, a.duration, 1.0)
    u_probe = drive_series_from_pk(t_probe, pk, onset_s=a.onset, region=a.region)
    i_peak = int(np.argmax(u_probe))

    obs = tuple(a.observers)
    names = list(TOPOLOGIES)
    rows = a.only or names
    sim_kw = dict(duration=a.duration, tr=a.tr, fast_dt=a.fast_dt, hr_dt=a.hr_dt, onset=a.onset,
                  snr=a.snr, q_z=a.q_z, q_a=a.q_a, target_loop_gain=a.loop_gain,
                  drive="series", series=series)
    fit_kw = dict(observers=obs, restarts=a.restarts, max_steps=a.max_steps)

    print(f"PK drive from {a.assumptions}: source = {meta['source'][:90]}...")
    print(f"  peak occupancy {meta['peak_occupancy']:.3f} at {meta['time_to_peak_s']:.0f} s after dosing; "
          f"terminal t1/2 {meta['secondary']['t_half_terminal_s']:.0f} s; "
          f"on the {a.duration:.0f} s run the unit-peak drive peaks at {t_probe[i_peak]:.0f} s and ends at {u_probe[-1]:.2f}")
    print(f"channels {'+'.join(obs)} at SNR {a.snr}; {len(a.seeds)} seeds")
    print("rows are the generating topology, columns the fitted one; the entry is dAIC above the row's best (0.00 wins)")
    print()
    header = "".join(f"{n[:11]:>12s}" for n in names)
    print(f"{'data \\ fit':14s}{header}  {'truth wins':>11s}")

    out = {"settings": {k: v for k, v in vars(a).items()}, "pk_metadata": meta,
           "drive_on_run": {"time_s": t_probe.tolist(), "u": u_probe.tolist()}, "cells": [], "matrix": {}}
    for data_top in rows:
        per_fit = {n: [] for n in names}
        for seed in a.seeds:
            sim = simulate(data_top, seed=seed, **sim_kw)
            for fit_top in names:
                r = fit_topology(sim, fit_top, seed=seed + 1, **fit_kw)
                r["data_topology"] = data_top
                r["seed"] = seed
                out["cells"].append(r)
                per_fit[fit_top].append(r)
        med = {n: float(np.nanmedian([x["aic"] for x in per_fit[n]])) for n in names}
        best = min(med, key=lambda n: med[n])
        d = {n: med[n] - med[best] for n in names}
        cells = "".join((f"{d[n]:>12.2f}" if np.isfinite(d[n]) else f"{'nan':>12s}") for n in names)
        print(f"{data_top:14s}{cells}  {str(best == data_top):>11s}", flush=True)
        out["matrix"][data_top] = {
            "median_aic": med, "delta_aic": d, "best": best, "truth_wins": bool(best == data_top),
            "median_log_likelihood": {n: float(np.nanmedian([x["log_likelihood"] for x in per_fit[n]])) for n in names},
            "identifiable": {n: int(sum(x["identifiable"] for x in per_fit[n])) for n in names},
        }
        if a.output:
            json.dump(out, open(a.output, "w"), indent=1, default=float)

    print()
    print("Compare with docs/results/brain_body_confusion.json (Bateman shape, same settings):")
    print("a matrix that changes only where the drive's time course differs is the")
    print("expected outcome; a diagonal that fails here but held there would mean the")
    print("PK time course itself removes the information the shape test relied on.")
    if a.output:
        json.dump(out, open(a.output, "w"), indent=1, default=float)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
