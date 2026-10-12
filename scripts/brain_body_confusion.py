"""Step 1: can the five brain-body topologies be told apart?

Step 0 (``scripts/brain_body_topologies.py``) established that each
topology is locally identifiable **only** once a peripheral channel is
recorded, and that no topology is identifiable from BOLD alone.  That is
about each model on its own.  This script asks the next question: given
that each one *can* be estimated, can the data say **which** one it is?

Every topology is simulated and then every topology is fitted to it,
with each fit holding the gains it does not carry at **exactly zero**
rather than at the values that generated the data.  That is what a real
analysis does when it assumes a topology, and holding an omitted gain at
truth would smuggle the right answer into the wrong model.

Most pairs of topologies are **not nested** -- ``efferent`` sits inside
``loop``, but ``afferent`` and ``common`` sit inside neither -- and they
do not all carry the same number of gains, so AIC rather than a
likelihood ratio is what makes the whole matrix comparable.  Where a
pair is nested the likelihood ratio is sharper and the per-cell log
likelihoods in the JSON support it.

Read the matrix by rows: each row is one generating topology, and the
winning column should be the diagonal.  An off-diagonal win is a
topology that would be *preferred* over the truth, which is the failure
that matters for a study design.

    python scripts/brain_body_confusion.py --seeds 0 1 2 --output confusion.json

Nothing here is specific to any compound: the input is a shape applied
at a named entry point.
"""

import argparse
import json

import numpy as np

from vpjax.validation.brain_body import (
    CHANNELS,
    TOPOLOGIES,
    fit_topology,
    simulate,
)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--duration", type=float, default=900.0)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--fast-dt", type=float, default=2.0)
    p.add_argument("--hr-dt", type=float, default=1.0)
    p.add_argument("--onset", type=float, default=120.0)
    p.add_argument("--rise", type=float, default=60.0)
    p.add_argument("--decay", type=float, default=420.0)
    p.add_argument("--snr", type=float, default=3.0)
    p.add_argument("--q-z", type=float, default=0.01)
    p.add_argument("--q-a", type=float, default=0.01)
    p.add_argument("--loop-gain", type=float, default=0.25)
    p.add_argument("--observers", nargs="+", default=list(CHANNELS),
                   choices=list(CHANNELS))
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--only", nargs="+", choices=sorted(TOPOLOGIES),
                   help="restrict the generating topologies")
    p.add_argument("--output")
    a = p.parse_args()

    obs = tuple(a.observers)
    names = list(TOPOLOGIES)
    rows = a.only or names
    sim_kw = dict(
        duration=a.duration, tr=a.tr, fast_dt=a.fast_dt, hr_dt=a.hr_dt,
        onset=a.onset, rise=a.rise, decay=a.decay, snr=a.snr,
        q_z=a.q_z, q_a=a.q_a, target_loop_gain=a.loop_gain,
    )
    fit_kw = dict(observers=obs, restarts=a.restarts, max_steps=a.max_steps)

    print(f"{a.duration:.0f} s run, input peaking at 1 from {a.onset:.0f} s; "
          f"channels {'+'.join(obs)} at SNR {a.snr}; {len(a.seeds)} seeds")
    print("rows are the generating topology, columns the fitted one; "
          "the entry is dAIC above the row's best (0.00 wins)")
    print()
    header = "".join(f"{n[:11]:>12s}" for n in names)
    print(f"{'data \\ fit':14s}{header}  {'truth wins':>11s}")

    out = {"settings": vars(a), "cells": [], "matrix": {}}
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
        # Median AIC over seeds, then dAIC relative to the row's best.
        med = {n: float(np.nanmedian([x["aic"] for x in per_fit[n]])) for n in names}
        best = min(med, key=lambda n: med[n])
        d = {n: med[n] - med[best] for n in names}
        cells = "".join(
            (f"{d[n]:>12.2f}" if np.isfinite(d[n]) else f"{'nan':>12s}")
            for n in names
        )
        print(f"{data_top:14s}{cells}  {str(best == data_top):>11s}")
        out["matrix"][data_top] = {
            "median_aic": med, "delta_aic": d, "best": best,
            "truth_wins": bool(best == data_top),
            "median_log_likelihood": {
                n: float(np.nanmedian([x["log_likelihood"] for x in per_fit[n]]))
                for n in names
            },
            "identifiable": {
                n: int(sum(x["identifiable"] for x in per_fit[n])) for n in names
            },
        }

    print()
    print("dAIC is median over seeds, so a column at 0.00 is the model AIC")
    print("prefers for that row.  A difference under about 2 is not a")
    print("preference at all; 'truth wins' being False is the failure that")
    print("matters, because it names a topology a study would report instead")
    print("of the one that generated the data.  Only nested pairs admit a")
    print("likelihood ratio -- the per-cell log likelihoods are in the JSON.")

    if a.output:
        json.dump(out, open(a.output, "w"), indent=1, default=float)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
