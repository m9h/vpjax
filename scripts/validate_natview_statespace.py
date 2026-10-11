"""State-space arms on a NATVIEW run: BOLD alone, then with EEG, pupil, respiration.

Each auxiliary channel is an observation of the shared drive through its
own loading and nuisance state; pupil is sparse (blinks, eye closure)
and enters only where valid.  Arms are fitted on the same run so the
identifiability verdicts are comparable.

    python scripts/validate_natview_statespace.py --root /data/natview --subject 01 \\
        --arms bold eeg pupil eeg+pupil --output natview_sub01.json
"""

import argparse
import json

import numpy as np

from vpjax.validation.eeg_fmri_statespace import (
    bold_fractional, fit_run, format_run, standardise_envelope,
)
from vpjax.validation.natview import load_run, run_paths
from vpjax.validation.sleep_eeg_fmri import load_bold_global


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--subject", default="01")
    p.add_argument("--task", default="rest")
    p.add_argument("--roi-file", help="npz from scripts/ds003768_rois.py (--bold/--t1 form)")
    p.add_argument("--roi", default="global")
    p.add_argument("--arms", nargs="+", default=["bold", "eeg", "pupil", "eeg+pupil"],
                   help="each arm is 'bold' or a '+'-joined set of eeg, pupil, respiration")
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--max-dt", type=float, default=0.25)
    p.add_argument("--output")
    a = p.parse_args()

    paths = run_paths(a.root, a.subject, a.task)
    run = load_run(paths)
    tr, t_vol = run["tr"], run["t_volumes"]
    if a.roi_file:
        d = np.load(a.roi_file)
        ts = d[f"bold_{a.roi}"]
    else:
        ts, tr_hdr = load_bold_global(paths["bold"])
    if ts.size != t_vol.size:
        raise ValueError(f"{t_vol.size} triggers but {ts.size} volumes")
    bold = bold_fractional(ts)
    t_bold = t_vol + tr / 2
    t0, t1 = t_bold[0] - tr / 2, t_bold[-1] + tr / 2

    streams = {}
    streams["eeg"] = standardise_envelope(run["eeg"]["values"], run["eeg"]["time_s"], t0, t1)
    if "pupil" in run:
        streams["pupil"] = standardise_envelope(
            run["pupil"]["values"], run["pupil"]["time_s"], t0, t1, valid=run["pupil"]["valid"])
    if "respiration" in run:
        streams["respiration"] = standardise_envelope(
            run["respiration"]["values"], run["respiration"]["time_s"], t0, t1)

    print(f"{run['stem']} [{a.roi}]: {bold.size} volumes, TR {tr}, BOLD fractional SD {bold.std():.4f}")
    print(f"  eeg: {streams['eeg'][0].size} samples, gradient {run['eeg']['gradient']['attenuation_db']:.1f} dB, "
          f"HR {run['eeg']['bcg']['heart_rate_bpm']:.0f} bpm")
    for k in ("pupil", "respiration"):
        if k in streams:
            info = run[k]
            print(f"  {k}: {streams[k][0].size} samples, trigger residual {info['trigger_residual_s']*1e3:.1f} ms"
                  + (f", valid {info['fraction_valid']:.2f}" if "fraction_valid" in info else ""))

    kw = dict(restarts=a.restarts, max_steps=a.max_steps, max_dt=a.max_dt)
    results = {}
    for arm in a.arms:
        chans = [] if arm == "bold" else arm.split("+")
        missing = [c for c in chans if c not in streams]
        if missing:
            print(f"skip {arm}: no {missing}"); continue
        aux = {c: streams[c] for c in chans}
        r = fit_run(t_bold, bold, aux=aux or None, **kw)
        results[arm] = r
        print(format_run(arm, r), flush=True)

    if a.output:
        json.dump({"run": run["stem"], "roi": a.roi, "tr": tr,
                   "streams": {k: int(v[0].size) for k, v in streams.items()},
                   "arms": results}, open(a.output, "w"), indent=1, default=float)
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
