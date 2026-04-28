#!/usr/bin/env python3
"""Test the Vaishnavi 2010 / Sperling 2009 / Drzezga 2011 finding.

Those papers report a NEGATIVE correlation between regional amyloid
burden (BPnd) and regional baseline CMRO2 in healthy older adults —
high-amyloid regions show metabolic suppression.  In a single subject
the effect is weak (Drzezga 2011 reports r ≈ −0.3 across cortical
regions); the SIGN is the testable thing.

Loads the existing PET, ASL, and joint outputs, restricts to FreeSurfer
DKT cortical labels (1000–1099 / 2000–2099), and computes Pearson +
Spearman correlations between BPnd and (CBF, CMRO2_fick, CMRO2_coupling).

Usage:
    python qa_vaishnavi_correlation.py sub-1003 ses-wave1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _vpjax_dir(subject: str, session: str) -> Path:
    return Path(f"/data/datasets/smri-fm-cmp/vpjax/ds004856/{subject}/{session}")


def is_cortical_dkt(rid: int) -> bool:
    """FreeSurfer DKT cortical labels: 1000-1099 (LH), 2000-2099 (RH).

    Excludes label 1000/2000 ('unknown') and 1004/2004 ('corpuscallosum').
    """
    if rid in (1000, 2000, 1004, 2004):
        return False
    return (1000 <= rid < 1100) or (2000 <= rid < 2100)


def main(subject: str, session: str) -> int:
    base = _vpjax_dir(subject, session)
    joint = json.loads((base / "metabolism/amyloid_cmro2_joint.json").read_text())

    # Use the joint output as the single source of truth — it carries
    # whichever PET source was selected (vpjax volumetric vs PETSurfer
    # GTM-PVC) and the per-region BPnd recovered by the joint fit,
    # which is consistent with the CBF used to compute CMRO2.
    j_ids = np.asarray(joint["region_ids"], dtype=int)
    j_bpnd = np.asarray(joint["bpnd"], dtype=float)
    j_suvr = np.asarray(joint["suvr"], dtype=float)
    j_cbf = np.asarray(joint["cbf"], dtype=float)
    j_oef = np.asarray(joint["oef"], dtype=float)
    j_cmro2_fick = np.asarray(joint["cmro2_fick"], dtype=float)
    j_cmro2_coup = np.asarray(joint["cmro2_coupling"], dtype=float)
    pet_source = joint.get("pet_source", "unknown")
    print(f"[vaishnavi] PET source (per joint stage): {pet_source}")

    cortical = np.array([rid for rid in j_ids if is_cortical_dkt(int(rid))],
                        dtype=int)

    def lookup(arr_ids, arr_vals, target_ids):
        idx = np.searchsorted(arr_ids, target_ids)
        return arr_vals[idx]

    bpnd = lookup(j_ids, j_bpnd, cortical)
    suvr = lookup(j_ids, j_suvr, cortical)
    cbf_v = lookup(j_ids, j_cbf, cortical)
    oef = lookup(j_ids, j_oef, cortical)
    cmro2_fick = lookup(j_ids, j_cmro2_fick, cortical)
    cmro2_coup = lookup(j_ids, j_cmro2_coup, cortical)

    print(f"[vaishnavi] cortical regions: n={cortical.size}")
    print(f"[vaishnavi] BPnd: mean={bpnd.mean():.3f}, range=[{bpnd.min():.3f}, {bpnd.max():.3f}]")
    print(f"[vaishnavi] CBF:  mean={cbf_v.mean():.2f}, range=[{cbf_v.min():.2f}, {cbf_v.max():.2f}]")
    print(f"[vaishnavi] CMRO2 (Fick): mean={cmro2_fick.mean():.1f}")

    pairs = [
        ("CBF", cbf_v),
        ("CMRO2_fick", cmro2_fick),
        ("CMRO2_coupling", cmro2_coup),
        ("OEF", oef),
    ]

    summary = {
        "subject": subject,
        "session": session,
        "pet_source": pet_source,
        "n_cortical_regions": int(cortical.size),
        "correlations": {},
    }
    for name, y in pairs:
        pr = stats.pearsonr(bpnd, y)
        sr = stats.spearmanr(bpnd, y)
        summary["correlations"][name] = {
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
        }
        sign = "−" if pr.statistic < 0 else "+"
        print(f"[vaishnavi] BPnd vs {name:14s}  Pearson r={sign}{abs(pr.statistic):.3f} (p={pr.pvalue:.3g})  "
              f"Spearman ρ={'−' if sr.statistic < 0 else '+'}{abs(sr.statistic):.3f} (p={sr.pvalue:.3g})")

    # Caveats.
    summary["caveats"] = [
        "Single-subject n; expected effect (Drzezga 2011) is r≈-0.3 across cortical regions in healthy older adults.",
        "alpha_amy in the joint model is fixed at -0.10 (default), so BPnd vs CMRO2_coupling is partly tautological.",
        "Spearman is more robust than Pearson for the small n, since BPnd has a few high-leverage points.",
    ]

    out_json = base / "metabolism" / "vaishnavi_correlation.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[vaishnavi] summary written to {out_json}")

    # Scatter plots.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, y) in zip(axes, pairs[:3]):
        ax.scatter(bpnd, y, s=15, alpha=0.6)
        m, b = np.polyfit(bpnd, y, 1)
        x_line = np.linspace(bpnd.min(), bpnd.max(), 50)
        ax.plot(x_line, m * x_line + b, "r-", lw=1.5)
        pr = stats.pearsonr(bpnd, y)
        ax.set_xlabel("BPnd")
        ax.set_ylabel(name)
        ax.set_title(f"r = {pr.statistic:+.3f}, p = {pr.pvalue:.2g}")
    fig.suptitle(f"{subject} {session} — Vaishnavi BPnd↔metabolism (cortical, n={cortical.size})",
                 fontsize=11)
    fig.tight_layout()
    out_png = base / "metabolism" / "vaishnavi_scatter.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[vaishnavi] scatter written to {out_png}")

    return 0


if __name__ == "__main__":
    sub, ses = sys.argv[1], sys.argv[2]
    sys.exit(main(sub, ses))
