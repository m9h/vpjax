"""Fit the drive-plus-Balloon state-space model to one ds003768 rest run.

BOLD alone, then BOLD plus the occipital alpha envelope, on the same
run; reports innovation whiteness and identifiability for each arm.
This is the real-data counterpart of ``validate_statespace_recovery.py``
and should be read alongside it: the simulation says what to expect
when the model is right, this says what happens on a recording.

    python scripts/validate_ds003768_statespace.py --root /data/ds003768 \\
        --subject 01 --run 1 --output sub01_rest1.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from vpjax.validation.eeg_artifacts import clean_rest_run
from vpjax.validation.eeg_fmri_statespace import (
    bold_fractional,
    fit_run,
    format_run,
    standardise_envelope,
)
from vpjax.validation.sleep_eeg_fmri import load_bold_global


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="BIDS root of ds003768")
    p.add_argument("--subject", default="01")
    p.add_argument("--task", default="rest")
    p.add_argument("--run", default="1")
    p.add_argument("--eeg-cache", help="npz from a previous cleaning pass (t_env, envelope, t_volumes)")
    p.add_argument("--roi-file", help="npz from scripts/ds003768_rois.py")
    p.add_argument("--roi", default="global",
                   help="which BOLD series to use from --roi-file (global, visual, motor)")
    p.add_argument("--drop", type=int, default=0, help="initial volumes to drop")
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--max-dt", type=float, default=0.25)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-nuisance", action="store_true",
                   help="one shared state only (no EEG-specific fast state)")
    p.add_argument("--output")
    a = p.parse_args()

    root = Path(a.root)
    stem = f"sub-{a.subject}_task-{a.task}_run-{a.run}"
    bold_path = root / f"sub-{a.subject}" / "func" / f"{stem}_bold.nii.gz"
    vhdr_path = root / f"sub-{a.subject}" / "eeg" / f"{stem}_eeg.vhdr"

    if a.roi_file:
        d = np.load(a.roi_file)
        ts, tr = d[f"bold_{a.roi}"], float(d["tr"])
    else:
        ts, tr = load_bold_global(bold_path)
    bold = bold_fractional(ts, drop=a.drop)

    if a.eeg_cache:
        d = np.load(a.eeg_cache)
        t_env_all, env_all, t_vol = d["t_env"], d["envelope"], d["t_volumes"]
        cleaning = json.loads(str(d["diagnostics"])) if "diagnostics" in d else {}
    else:
        out = clean_rest_run(vhdr_path)
        t_env_all, env_all, t_vol = out["t_env"], out["envelope"], out["t_volumes"]
        cleaning = {"gradient": out["gradient"], "bcg": out["bcg"]}

    if t_vol.size != ts.size:
        raise ValueError(f"{t_vol.size} volume triggers but {ts.size} volumes")
    # Stamp each volume at its temporal centre on the EEG clock.
    t_bold = t_vol[a.drop:] + tr / 2.0
    t_env, env = standardise_envelope(env_all, t_env_all, t_bold[0] - tr / 2, t_bold[-1] + tr / 2)

    print(f"{stem} [{a.roi}]: {bold.size} volumes at TR {tr} s, BOLD fractional SD {bold.std():.4f}; "
          f"{env.size} envelope samples")
    for k, v in cleaning.items():
        print(f"  {k}: " + ", ".join(f"{kk}={vv:.3g}" if isinstance(vv, float) else f"{kk}={vv}"
                                    for kk, vv in v.items()))

    kw = dict(max_dt=a.max_dt, restarts=a.restarts, max_steps=a.max_steps, seed=a.seed)
    bold_only = fit_run(t_bold, bold, **kw)
    print(format_run("BOLD only", bold_only))
    both = fit_run(t_bold, bold, t_env, env, nuisance=not a.no_nuisance, **kw)
    print(format_run("BOLD+EEG", both))

    if a.output:
        json.dump({"run": stem, "roi": a.roi, "tr": tr, "cleaning": cleaning,
                   "bold_only": bold_only, "both": both},
                  open(a.output, "w"), indent=1, default=float)
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
