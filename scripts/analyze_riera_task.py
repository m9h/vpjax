#!/usr/bin/env python3
"""Per-region Riera 8-state NVC fit on a single DLBS task BOLD.

Wraps :func:`vpjax.hemodynamics.inversion.fit_riera_bold` (per-region,
vmap-batched) for a single (subject, session, task, run).  Reads the
events.tsv, builds the boxcar stimulus, extracts per-region BOLD time
series via the aparc-in-CVR atlas, and fits the Riera default subset
``(c_no, kappa_no, gamma_no, tau_v, alpha_v, E0)`` per region.

Outputs land at::

    vpjax/.../task/<task>/<run_id>/riera_params.json
    vpjax/.../task/<task>/<run_id>/riera_predicted_bold.nii.gz
    vpjax/.../task/<task>/<run_id>/riera_cno_map.nii.gz   (etc.)

The vpjax identifiability tool (``vpjax.riera_identifiability``)
already documented that this 6-parameter subset is rank-6 and
moderately conditioned (cond ≈ 400) under the DLBS task design — so
the fit is on stable ground.  The full 15-parameter Riera is not
attempted (3 flat directions per the Jacobian-rank test).

Usage::

    python analyze_riera_task.py sub-1013 ses-wave1 VentralVisual run-1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force CPU for the small per-region batch — GPU launch overhead beats it.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import nibabel as nib
import numpy as np


VPJAX_ROOT = Path("/data/datasets/smri-fm-cmp/vpjax/ds004856")
FSL_ROOT = Path("/data/datasets/smri-fm-cmp/fsl/ds004856")
RAW_ROOT = Path("/data/raw/openneuro/ds004856")

DEFAULT_RIERA_FIT = ("c_no", "kappa_no", "gamma_no", "tau_v", "alpha_v", "E0")
ALL_RIERA = ("kappa_no", "kappa_ade", "gamma_no", "gamma_ade", "c_no", "c_ade",
             "tau_a", "tau_c", "tau_v", "alpha_a", "alpha_c", "alpha_v",
             "E0", "phi", "tau_m")


def _events_to_boxcar_dt(events_tsv: Path, total_dt: int, dt: float) -> np.ndarray:
    bc = np.zeros(total_dt, dtype=np.float32)
    if not events_tsv.exists():
        return bc
    with open(events_tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        if "onset" not in header or "duration" not in header:
            return bc
        ions, idur = header.index("onset"), header.index("duration")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            try:
                onset = float(parts[ions])
                dur = float(parts[idur]) if parts[idur] not in ("", "n/a") else dt
            except (ValueError, IndexError):
                continue
            i0 = int(round(onset / dt))
            i1 = int(round((onset + dur) / dt))
            i0, i1 = max(0, i0), min(total_dt, i1)
            if i1 > i0:
                bc[i0:i1] = 1.0
    return bc


def main(subject: str, session: str, task: str, run: str,
         n_steps: int = 600, learning_rate: float = 2.0,
         min_voxels: int = 20, optimizer: str = "momentum",
         prior_strength: float = 0.0) -> int:
    out_dir = VPJAX_ROOT / subject / session / "task" / task / run
    out_dir.mkdir(parents=True, exist_ok=True)
    riera_path = out_dir / "riera_params.json"

    bold_path = FSL_ROOT / "task" / task / f"{subject}_{session}_{run}" / "mc.nii.gz"
    json_path = (RAW_ROOT / subject / session / "func"
                 / f"{subject}_{session}_task-{task}_{run}_bold.json")
    events_tsv = (RAW_ROOT / subject / session / "func"
                  / f"{subject}_{session}_task-{task}_{run}_events.tsv")
    aparc_path = VPJAX_ROOT / subject / session / "anat-in-cvr" / "aparc+aseg.nii.gz"
    for p in (bold_path, json_path, events_tsv, aparc_path):
        if not p.exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 2

    bold_4d = nib.load(str(bold_path)).get_fdata(dtype=np.float32)
    aparc = nib.load(str(aparc_path)).get_fdata().astype(np.int32)
    aparc_affine = nib.load(str(aparc_path)).affine
    if aparc.shape != bold_4d.shape[:3]:
        print(f"aparc shape {aparc.shape} ≠ BOLD grid {bold_4d.shape[:3]}",
              file=sys.stderr)
        return 2
    tr = float(json.loads(json_path.read_text())["RepetitionTime"])
    n_t = bold_4d.shape[-1]
    dt = 0.1
    sub = int(round(tr / dt))
    total_dt = n_t * sub
    stim = _events_to_boxcar_dt(events_tsv, total_dt, dt)

    print(f"[riera] {subject} {session} {task} {run}: TR={tr}s, "
          f"n={n_t}, stim-on dt-samples={int(stim.sum())}/{total_dt}")

    # Per-region BOLD as fractional change.
    region_ids = np.unique(aparc[aparc > 0]).astype(np.int32)
    means: list[np.ndarray] = []
    kept: list[int] = []
    for rid in region_ids:
        m = aparc == rid
        if int(m.sum()) < min_voxels:
            continue
        ts = bold_4d[m].mean(axis=0)
        baseline = ts.mean()
        if baseline <= 0:
            continue
        means.append(((ts - baseline) / baseline).astype(np.float32))
        kept.append(int(rid))
    if not kept:
        print("no usable regions", file=sys.stderr)
        return 2
    bold_R_N = np.stack(means, axis=0)
    kept = np.asarray(kept, dtype=np.int32)
    print(f"[riera] fitting {bold_R_N.shape[0]} regions × n_steps={n_steps}, lr={learning_rate}")

    # vmap fit_riera_bold over regions.  Uses the default Riera fit_names
    # (already validated as rank-6 / cond ≈ 400 by riera_identifiability).
    from vpjax.hemodynamics.inversion import fit_riera_bold

    def _fit_one(bold_one):
        return fit_riera_bold(
            bold_one, jnp.asarray(stim), tr=tr, dt=dt,
            fit_names=DEFAULT_RIERA_FIT,
            n_steps=n_steps, learning_rate=learning_rate,
            optimizer=optimizer, prior_strength=prior_strength,
        )
    fit_batched = jax.vmap(_fit_one)

    t0 = time.time()
    out = fit_batched(jnp.asarray(bold_R_N))
    elapsed = time.time() - t0
    print(f"[riera] done in {elapsed:.1f}s")

    # Pull per-region params out of the vmap'd dict.
    payload = {
        "subject": subject,
        "session": session,
        "task": task,
        "run": run,
        "tr": tr,
        "dt": dt,
        "n_volumes": int(n_t),
        "n_regions": int(kept.shape[0]),
        "region_ids": kept.tolist(),
        "fit_names": list(DEFAULT_RIERA_FIT),
        "n_steps": n_steps,
        "learning_rate": learning_rate,
        "loss": np.asarray(out["loss"]).tolist(),
    }
    for name in ALL_RIERA:
        payload[name] = np.asarray(out[name]).tolist()

    riera_path.write_text(json.dumps(payload, indent=2))
    print(f"[riera] saved {riera_path}")

    # Voxel maps for the canonical NO subsystem + venous transit.
    for name in ("c_no", "kappa_no", "gamma_no", "tau_v"):
        arr = np.zeros(aparc.shape, dtype=np.float32)
        vec = np.asarray(out[name])
        for rid_v, val in zip(kept, vec):
            arr[aparc == int(rid_v)] = float(val)
        out_nii = out_dir / f"riera_{name}_map.nii.gz"
        nib.save(nib.Nifti1Image(arr, aparc_affine), str(out_nii))
        print(f"[riera] saved {out_nii}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject")
    parser.add_argument("session")
    parser.add_argument("task")
    parser.add_argument("run")
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=2.0)
    parser.add_argument("--optimizer", choices=("momentum", "adam"),
                        default="momentum")
    parser.add_argument("--prior-strength", type=float, default=0.0,
                        help="Gaussian prior strength on (θ−θ_lit)/σ; "
                             "0 = no prior, ~1e-3 = soft, ~1e-2 = strong")
    args = parser.parse_args()
    sys.exit(main(args.subject, args.session, args.task, args.run,
                  args.n_steps, args.learning_rate, optimizer=args.optimizer,
                  prior_strength=args.prior_strength))
