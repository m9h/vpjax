"""BOLD-only identifiability on the public DMT EEG-fMRI release, DMT vs placebo.

The release carries parcellated, z-scored BOLD and HRF-convolved EEG
regressors (see :mod:`vpjax.validation.dmt_imaging`), so this is not the
fusion test -- it is the drug-data test of the identifiability
diagnostic: fit the drive-plus-Balloon model to one network's series per
subject per condition, and ask whether a drug effect on the drive is an
estimable quantity from fMRI alone.  The amplitude is supplied by
assumption, so ``q_z`` is in units of that assumption and only ratios
between conditions are interpretable; the time constants and the
identifiability verdicts are not affected.

    python scripts/validate_dmt_statespace.py --root ~/data/DMT_Imaging \\
        --networks global visual default --phase whole --output dmt.json
"""

import argparse
import json

import numpy as np

from vpjax.validation.dmt_imaging import (
    TR,
    bold_from_zscored,
    load_parcels,
    network_series,
    phase_mask,
    subjects,
    volume_times,
)
from vpjax.validation.eeg_fmri_statespace import fit_run


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="clone of timmer500/DMT_Imaging")
    p.add_argument("--networks", nargs="+", default=["global"])
    p.add_argument("--phase", default="whole",
                   choices=["whole", "pre", "acute", "late"])
    p.add_argument("--subjects", nargs="+", help="default: every subject with both conditions")
    p.add_argument("--assumed-fractional-sd", type=float, default=0.01)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--output")
    a = p.parse_args()

    subs = a.subjects or subjects(a.root)
    times = volume_times()
    keep = np.ones(times.size, dtype=bool) if a.phase == "whole" else phase_mask(times, a.phase)
    print(f"{len(subs)} subjects, {int(keep.sum())} volumes of {times.size} "
          f"({a.phase}), TR {TR} s, assumed fractional SD {a.assumed_fractional_sd}")

    results = {}
    for net in a.networks:
        for sub in subs:
            for cond in ("DMT", "PCB"):
                ts = network_series(load_parcels(a.root, sub, cond), net)[keep]
                bold = bold_from_zscored(ts, a.assumed_fractional_sd)
                r = fit_run(times[keep], bold, restarts=a.restarts, max_steps=a.max_steps)
                results[f"{net}/{sub}/{cond}"] = r
                print(f"{net:18s} {sub} {cond}  tau_z {r['estimate']['tau_z']:7.2f}"
                      f" +-{r['standard_error']['tau_z']:5.2f}  q_z {r['estimate']['q_z']:9.3g}"
                      f"  ident={str(r['identifiable']):5s} curv {r['curvature_ratio']:+.2e}"
                      f"  bold var {r['diagnostics']['per_channel']['bold']['variance']:.2f}"
                      f" lag1 {r['diagnostics']['per_channel']['bold']['lag1']:+.2f}", flush=True)

    # Paired within-subject contrast on the log scale, where the fit works.
    print()
    for net in a.networks:
        for name in ("tau_z", "q_z", "kappa", "tau"):
            pairs = [
                (results[f"{net}/{s}/DMT"]["estimate"][name],
                 results[f"{net}/{s}/PCB"]["estimate"][name])
                for s in subs
                if f"{net}/{s}/DMT" in results and f"{net}/{s}/PCB" in results
            ]
            d = np.array([np.log(x) - np.log(y) for x, y in pairs])
            se = d.std(ddof=1) / np.sqrt(d.size)
            print(f"{net:18s} log({name}) DMT - PCB: {d.mean():+.3f} +- {se:.3f} "
                  f"(n={d.size}, t={d.mean()/se:+.2f})")

    if a.output:
        json.dump({"settings": vars(a), "tr": TR, "subjects": subs, "fits": results},
                  open(a.output, "w"), indent=1, default=float)
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
