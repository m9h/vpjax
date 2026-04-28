#!/usr/bin/env python3
"""Compare PETSurfer GTM-PVC SUVR to the vpjax volumetric SUVR.

Expects both ``regional_suvr.json`` (vpjax pipeline) and
``regional_suvr_petsurfer.json`` (PETSurfer ingest) for the same
session.  Joins on FreeSurfer aparc+aseg label IDs, restricts to
cortical DKT labels (1000–1099 / 2000–2099), and produces:

  * a scatter plot (vpjax SUVR vs PETSurfer GTM SUVR) with regression line
  * Pearson + Spearman correlations
  * a JSON summary

Usage::

    python qa_petsurfer_vs_vpjax.py sub-1003 ses-wave1 amyloid
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


PET_TRACER_SUFFIX = {"amyloid": "amyloid_18FAV45", "tau": "tau_18FAV1451"}


def is_cortical_dkt(rid: int) -> bool:
    if rid in (1000, 2000, 1004, 2004):
        return False
    return (1000 <= rid < 1100) or (2000 <= rid < 2100)


def main(subject: str, session: str, tracer: str) -> int:
    pet_dir = Path(
        f"/data/datasets/smri-fm-cmp/vpjax/ds004856/{subject}/{session}"
        f"/pet/{PET_TRACER_SUFFIX[tracer]}"
    )
    vpjax_path = pet_dir / "regional_suvr.json"
    pet_path = pet_dir / "regional_suvr_petsurfer.json"
    for p in (vpjax_path, pet_path):
        if not p.exists():
            print(f"[qa] missing {p}", file=sys.stderr)
            return 2

    vp = json.loads(vpjax_path.read_text())
    ps = json.loads(pet_path.read_text())

    vp_ids = np.asarray(vp["region_ids"], dtype=int)
    ps_ids = np.asarray(ps["region_ids"], dtype=int)
    common = np.intersect1d(vp_ids, ps_ids)
    cortical = np.array(
        [r for r in common if is_cortical_dkt(int(r))], dtype=int,
    )
    print(f"[qa] regions common: {common.size}; cortical only: {cortical.size}")

    def lookup(arr_ids, arr_vals, target):
        idx = np.searchsorted(arr_ids, target)
        return np.asarray(arr_vals)[idx]

    vp_suvr = lookup(vp_ids, vp["suvr"], cortical)
    ps_suvr = lookup(ps_ids, ps["suvr"], cortical)
    ps_nopvc = lookup(ps_ids, ps["suvr_nopvc"], cortical)

    p_pearson = stats.pearsonr(vp_suvr, ps_suvr)
    p_spearman = stats.spearmanr(vp_suvr, ps_suvr)
    np_pearson = stats.pearsonr(vp_suvr, ps_nopvc)

    print(f"[qa] vpjax  vs PETSurfer GTM-PVC: r={p_pearson.statistic:+.3f} (p={p_pearson.pvalue:.2g}); "
          f"ρ={p_spearman.statistic:+.3f} (p={p_spearman.pvalue:.2g})")
    print(f"[qa] vpjax  vs PETSurfer no-PVC:  r={np_pearson.statistic:+.3f} (p={np_pearson.pvalue:.2g})")
    print(f"[qa] vpjax SUVR mean={vp_suvr.mean():.3f}, range=[{vp_suvr.min():.3f},{vp_suvr.max():.3f}]")
    print(f"[qa] PETSurfer GTM   mean={ps_suvr.mean():.3f}, range=[{ps_suvr.min():.3f},{ps_suvr.max():.3f}]")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, (label, y, pearson) in zip(
        axes,
        [("vpjax vs PETSurfer GTM-PVC SUVR", ps_suvr, p_pearson),
         ("vpjax vs PETSurfer no-PVC SUVR", ps_nopvc, np_pearson)],
    ):
        ax.scatter(vp_suvr, y, s=15, alpha=0.6)
        m, b = np.polyfit(vp_suvr, y, 1)
        xs = np.linspace(min(vp_suvr.min(), y.min()),
                         max(vp_suvr.max(), y.max()), 50)
        ax.plot(xs, xs, "k--", lw=0.8, label="y = x")
        ax.plot(xs, m * xs + b, "r-", lw=1.5, label=f"fit: y = {m:.3f}x + {b:.3f}")
        ax.set_xlabel("vpjax volumetric SUVR (Müller-Gärtner)")
        ax.set_ylabel(label.split(" vs ")[1])
        ax.set_title(f"{label}\nr={pearson.statistic:+.3f}, p={pearson.pvalue:.2g}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"{subject} {session} {tracer}: vpjax SUVR vs PETSurfer "
        f"(cortical, n={cortical.size})",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = pet_dir / "qa_vpjax_vs_petsurfer.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[qa] scatter saved to {out_png}")

    summary = {
        "subject": subject, "session": session, "tracer": tracer,
        "n_cortical": int(cortical.size),
        "vpjax_pearson_vs_gtm": {
            "r": float(p_pearson.statistic), "p": float(p_pearson.pvalue),
        },
        "vpjax_spearman_vs_gtm": {
            "rho": float(p_spearman.statistic), "p": float(p_spearman.pvalue),
        },
        "vpjax_pearson_vs_nopvc": {
            "r": float(np_pearson.statistic), "p": float(np_pearson.pvalue),
        },
        "regression_vpjax_to_gtm": {
            "slope": float(np.polyfit(vp_suvr, ps_suvr, 1)[0]),
            "intercept": float(np.polyfit(vp_suvr, ps_suvr, 1)[1]),
        },
    }
    out_json = pet_dir / "qa_vpjax_vs_petsurfer.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[qa] summary saved to {out_json}")
    return 0


if __name__ == "__main__":
    sub, ses, trc = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.exit(main(sub, ses, trc))
