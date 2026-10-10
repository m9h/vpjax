"""Does identifiability need *simultaneous* EEG, or just EEG?

Three experiments on the simulated drive-plus-Balloon model, each over
several seeds so a single realisation cannot carry the verdict:

  snr     sweep the in-scanner EEG envelope SNR downward and report
          identifiability and relative standard errors per parameter;
  fixed   the separate-session counterfactual -- BOLD only, with the
          drive's statistics (tau_z, or tau_z and q_z) held at truth,
          as if a perfect out-of-scanner recording had supplied them;
  drug    a non-stationary run with a shape-only input of the caller's
          onset/rise/decay and gain, scoring whether the gain ``beta``
          -- the drug contrast -- is identifiable from BOLD alone and
          from BOLD plus EEG.

The drug input is a shape, not a model of any compound; every time
constant comes from the command line and is recorded in the output.

    python scripts/simultaneity_case.py --seeds 0 1 2 --output case.json
    python scripts/simultaneity_case.py --only drug --onset 60 --rise 30 \\
        --decay 240 --beta 0.5 --seeds 0 1
"""

import argparse
import json

import numpy as np

from vpjax.validation.statespace_recovery import (
    _FIT_NAMES,
    drug_shaped_input,
    fit_simulation,
    simulate,
)


def summarise(fits: list[dict]) -> dict:
    """Median relative SE and estimate error across seeds, plus the identifiable count."""
    names = fits[0]["fit_names"]
    return {
        "n": len(fits),
        "identifiable": sum(f["identifiable"] for f in fits),
        "curvature_ratio_median": float(np.median([f["curvature_ratio"] for f in fits])),
        "relative_se_median": {
            n: float(np.median([f["standard_error"][n] for f in fits])) for n in names
        },
        "relative_error_median": {
            n: float(np.median([f["relative_error"][n] for f in fits])) for n in names
        },
    }


def line(tag: str, s: dict) -> str:
    se = " ".join(f"{n}={v:.3f}" for n, v in s["relative_se_median"].items())
    return (f"{tag:<36} ident {s['identifiable']}/{s['n']}  "
            f"curv {s['curvature_ratio_median']:+.2e}  SE {se}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--only", choices=["snr", "fixed", "drug"], nargs="*")
    p.add_argument("--duration", type=float, default=360.0)
    p.add_argument("--tr", type=float, default=2.0)
    p.add_argument("--eeg-dt", type=float, default=0.1)
    p.add_argument("--snr", type=float, nargs="+", default=[3.0, 1.0, 0.5, 0.25])
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--onset", type=float, help="drug input onset (s)")
    p.add_argument("--rise", type=float, help="drug input rise time scale (s)")
    p.add_argument("--decay", type=float, help="drug input decay time scale (s)")
    p.add_argument("--beta", type=float, help="drug input gain, drive units at peak")
    p.add_argument("--output")
    a = p.parse_args()
    which = set(a.only) if a.only else {"snr", "fixed", "drug"}
    if "drug" in which and None in (a.onset, a.rise, a.decay, a.beta):
        if a.only:
            p.error("--onset, --rise, --decay and --beta are required for the drug experiment")
        which.discard("drug")
        print("drug experiment skipped: no input shape given (--onset/--rise/--decay/--beta)")

    base = dict(duration=a.duration, tr=a.tr, eeg_dt=a.eeg_dt)
    kw = dict(restarts=a.restarts)
    out = {"settings": vars(a), "snr": {}, "fixed": {}, "drug": {}}

    if "snr" in which:
        fits = [fit_simulation(simulate(seed=s, **base), use_eeg=False, **kw) for s in a.seeds]
        out["snr"]["bold_only"] = summarise(fits); print(line("BOLD only", out["snr"]["bold_only"]))
        for snr in a.snr:
            fits = [fit_simulation(simulate(seed=s, eeg_snr=snr, **base), use_eeg=True, **kw)
                    for s in a.seeds]
            out["snr"][str(snr)] = summarise(fits)
            print(line(f"BOLD+EEG, envelope SNR {snr}", out["snr"][str(snr)]))

    if "fixed" in which:
        for free in [("kappa", "tau", "q_z"), ("kappa", "tau")]:
            fixed = [n for n in _FIT_NAMES if n not in free]
            fits = [fit_simulation(simulate(seed=s, **base), use_eeg=False, fit_names=free, **kw)
                    for s in a.seeds]
            key = "fixed " + ",".join(fixed)
            out["fixed"][key] = summarise(fits)
            print(line(f"BOLD only, {key} at truth", out["fixed"][key]))

    if "drug" in which:
        u = drug_shaped_input(a.onset, a.rise, a.decay)
        for use_eeg in (False, True):
            fits = [fit_simulation(simulate(seed=s, drive_input=u, beta=a.beta, **base),
                                   use_eeg=use_eeg, **kw) for s in a.seeds]
            key = "both" if use_eeg else "bold_only"
            out["drug"][key] = summarise(fits)
            print(line(f"drug input, {'BOLD+EEG' if use_eeg else 'BOLD only'}", out["drug"][key]))

    if a.output:
        json.dump(out, open(a.output, "w"), indent=1, default=float)
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
