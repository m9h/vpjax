"""Step 0: can a recording set resolve a brain-body topology's parameters at all?

Five system models -- efferent, afferent, common drive, loop and
measurement-only -- are corners of a two-gain space
(``docs/brain_body_system_models.md``).  Before asking which of them a
study can *tell apart*, there is a cheaper prerequisite: for each one,
if it is the truth, can the recording set resolve its own parameters?

A rank deficiency here is not a power problem.  It means some
combination of parameters is flat to first order no matter how long the
run, so no amount of data rescues it and the recording set has to
change.  That makes this the right first question, and it costs one
Jacobian per cell rather than a fit.

The test is the Walter & Pronzato Jacobian-rank test already in
:mod:`vpjax.identifiability`, taken in forward mode because the
Jacobian is tall and narrow.  Parameters enter as multipliers on the
nominal values and the Jacobian is evaluated at a multiplier of one, so
each column is a relative sensitivity.  Every channel is divided by its
own noise SD, set so each carries the same SNR against its own
excursion -- so a reported deficiency is a statement about structure,
not about one channel being quieter in whatever units it uses.

    python scripts/brain_body_topologies.py --output topologies.json

Nothing here is specific to any compound: the input is a shape applied
at a named entry point.
"""

import argparse
import json

from vpjax.validation.brain_body import (
    CHANNELS,
    TOPOLOGIES,
    free_names,
    loop_gain,
    topology_identifiability,
    topology_params,
)

CONFIGS = (("bold",), ("bold", "fast"), ("bold", "hr"), ("bold", "fast", "hr"))


def flat_directions(result: dict) -> list[str]:
    """Readable renderings of the parameter combinations the data cannot fix."""
    out = []
    for cs in result.get("collinear_sets", []):
        out.append(" ".join(
            f"{c:+.2f}*{n}" for n, c in zip(cs["params"], cs["coefficients"])
        ))
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--duration", type=float, default=900.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--fast-dt", type=float, default=2.0)
    p.add_argument("--hr-dt", type=float, default=1.0)
    p.add_argument("--onset", type=float, default=120.0)
    p.add_argument("--rise", type=float, default=60.0)
    p.add_argument("--decay", type=float, default=420.0)
    p.add_argument("--snr", type=float, default=3.0)
    p.add_argument("--loop-gain", type=float, default=0.25,
                   help="closed-loop gain for the loop topology; must be < 1")
    p.add_argument("--loadings", nargs="*", default=[], choices=["fast", "hr"],
                   help="channels whose observation loading is left free; a free "
                        "'hr' loading always costs one flat direction, since the "
                        "autonomic state has no scale anchor (see LOADING_NAMES)")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--only", nargs="+", choices=sorted(TOPOLOGIES),
                   help="restrict to these topologies")
    p.add_argument("--output")
    a = p.parse_args()

    topologies = a.only or list(TOPOLOGIES)
    kw = dict(
        duration=a.duration, dt=a.dt, tr=a.tr, fast_dt=a.fast_dt, hr_dt=a.hr_dt,
        onset=a.onset, rise=a.rise, decay=a.decay, snr=a.snr,
        target_loop_gain=a.loop_gain, rtol=a.rtol,
        loadings=tuple(a.loadings),
    )
    print(f"input peaks at 1 from {a.onset:.0f} s in a {a.duration:.0f} s run; "
          f"BOLD every {a.tr} s, fast channel every {a.fast_dt} s, "
          f"physiology every {a.hr_dt} s; every channel at SNR {a.snr}")
    print(f"channels: {', '.join(CHANNELS)}  (fast = drive, hr = autonomic state)")
    print(f"free loadings: {'+'.join(a.loadings) if a.loadings else 'none '
          '(gain_hr = 1 is the identifying convention, not an assumption)'}")
    print()
    print(f"{'topology':12s} {'channels':17s} {'rank/P':>7s} {'cond':>9s} "
          f"{'ident':>6s}  flat directions")

    out = {"settings": vars(a), "cells": []}
    for name in topologies:
        gains = TOPOLOGIES[name]
        for obs in CONFIGS:
            r = topology_identifiability(name, observers=obs, **kw)
            if "error" in r:
                print(f"{name:12s} {'+'.join(obs):17s}  ERROR: {r['error']}")
                out["cells"].append({"topology": name, "observers": list(obs),
                                     "error": r["error"]})
                continue
            flat = flat_directions(r)
            print(f"{name:12s} {'+'.join(obs):17s} {r['rank']:>3d}/{r['n_params']:<3d} "
                  f"{r['condition_number']:>9.2e} {str(r['is_identifiable']):>6s}  "
                  f"{'; '.join(flat) if flat else '-'}")
            out["cells"].append({
                "topology": name,
                "observers": list(obs),
                "names": list(r["names"]),
                "active_gains": list(gains),
                "rank": int(r["rank"]),
                "n_params": int(r["n_params"]),
                "condition_number": float(r["condition_number"]),
                "singular_values": [float(v) for v in r["singular_values"]],
                "is_identifiable": bool(r["is_identifiable"]),
                "flat_directions": flat,
                "loop_gain": float(r["loop_gain"]),
                "loadings": list(r["loadings"]),
                "noise_sd": {k: float(v) for k, v in r["noise_sd"].items()},
            })
        print()

    print("A flat direction is a combination the data cannot fix at any run")
    print("length: read it before the rank, because it names *what* is lost.")
    print("Coefficients near +/-0.71 on two parameters are a trade-off -- only")
    print("their product is determined.  A lone +1.00 is a parameter the")
    print("channels do not see at all.")

    if a.output:
        json.dump(out, open(a.output, "w"), indent=1, default=float)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
