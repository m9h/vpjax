#!/usr/bin/env python3
"""Fit per-region (baseline, age-slope) Balloon-Windkessel parameters across
DLBS sessions for one subject.

Uses ``vpjax.hemodynamics.longitudinal.fit_balloon_longitudinal_from_volumes``.
Each session contributes its hypercapnia BOLD + brain mask + aparc-in-CVR
volume + age.  Output is written under
``/data/datasets/smri-fm-cmp/vpjax/<ds>/<sub>/longitudinal/``.

The age-slope coefficients give an interpretable "vascular aging rate"
per region (Δ(parameter) per year), which a single-wave fit cannot
identify.

Usage::

    python analyze_longitudinal_balloon.py sub-1003
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Force CPU before importing JAX — small problem, GPU launch overhead loses.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import nibabel as nib
import numpy as np


VPJAX_ROOT = Path("/data/datasets/smri-fm-cmp/vpjax/ds004856")
FSL_CVR_ROOT = Path("/data/datasets/smri-fm-cmp/fsl/ds004856/cvr")
PARTICIPANTS_TSV = Path("/data/raw/openneuro/ds004856/participants.tsv")


def _ages_for_subject(subject: str) -> dict[str, float]:
    """Pull session-wise ages from participants.tsv (DLBS schema).

    DLBS records age_W1, age_W2, age_W3 at the time of *the wave's* MRI.
    """
    if not PARTICIPANTS_TSV.exists():
        raise FileNotFoundError(f"missing {PARTICIPANTS_TSV}")
    header: list[str] | None = None
    row_for_subject: list[str] | None = None
    with open(PARTICIPANTS_TSV) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                continue
            if parts[0] == subject:
                row_for_subject = parts
                break
    if row_for_subject is None:
        raise ValueError(f"subject {subject} not found in participants.tsv")
    h2v = dict(zip(header, row_for_subject))

    out: dict[str, float] = {}
    for ses, key in [("ses-wave1", "AgeMRI_W1"),
                     ("ses-wave2", "AgeMRI_W2"),
                     ("ses-wave3", "AgeMRI_W3")]:
        v = h2v.get(key)
        try:
            if v not in (None, "", "n/a"):
                out[ses] = float(v)
        except ValueError:
            pass
    return out


def _load_session(subject: str, session: str
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (bold_4d, brain_mask, aparc) or None if any input missing."""
    bold_path = FSL_CVR_ROOT / f"{subject}_{session}_run-1" / "mc.nii.gz"
    mask_path = FSL_CVR_ROOT / f"{subject}_{session}_run-1" / "brain_mask.nii.gz"
    aparc_path = (
        VPJAX_ROOT / subject / session / "anat-in-cvr" / "aparc+aseg.nii.gz"
    )
    for p in (bold_path, mask_path, aparc_path):
        if not p.exists():
            print(f"[long] {session}: missing {p}", file=sys.stderr)
            return None
    bold = nib.load(str(bold_path)).get_fdata(dtype=np.float32)
    mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
    aparc = nib.load(str(aparc_path)).get_fdata().astype(np.int32)
    return bold, mask, aparc


def main(subject: str, age_center: float = 60.0,
         n_steps: int = 200, learning_rate: float = 2.0) -> int:
    ages_by_session = _ages_for_subject(subject)
    if len(ages_by_session) < 2:
        print(f"[long] need ≥2 sessions for a longitudinal fit, got "
              f"{len(ages_by_session)}", file=sys.stderr)
        return 2

    sessions = sorted(ages_by_session)
    bold_per_session: list[np.ndarray] = []
    mask_per_session: list[np.ndarray] = []
    label_per_session: list[np.ndarray] = []
    ages: list[float] = []
    used_sessions: list[str] = []
    for ses in sessions:
        loaded = _load_session(subject, ses)
        if loaded is None:
            continue
        b, m, a = loaded
        bold_per_session.append(b)
        mask_per_session.append(m)
        label_per_session.append(a)
        ages.append(ages_by_session[ses])
        used_sessions.append(ses)
    if len(used_sessions) < 2:
        print("[long] fewer than 2 usable sessions after loading", file=sys.stderr)
        return 2

    print(f"[long] subject={subject}, sessions={used_sessions}, ages={ages}")

    # Lazy import — JAX init is slow.
    from vpjax.hemodynamics.longitudinal import fit_balloon_longitudinal_from_volumes

    out = fit_balloon_longitudinal_from_volumes(
        bold_4d_per_session=bold_per_session,
        brain_mask_per_session=mask_per_session,
        region_volume_per_session=label_per_session,
        ages=ages,
        tr=2.0, dt=0.1, age_center=age_center,
        fit_names=("kappa", "tau"),
        n_steps=n_steps, learning_rate=learning_rate,
        min_voxels=20,
    )
    region_ids = out["region_ids"]
    baselines = out["baselines"]    # (R, P)
    slopes = out["slopes"]          # (R, P)
    fit_names = out["fit_names"].tolist()
    print(f"[long] fit {region_ids.size} regions across {len(used_sessions)} sessions")
    for i, name in enumerate(fit_names):
        bs = baselines[:, i]
        sl = slopes[:, i]
        print(f"  {name:6s}: baseline mean={bs.mean():.3f} (range {bs.min():.3f}–{bs.max():.3f}); "
              f"slope mean={sl.mean():.4f}, range=[{sl.min():.4f},{sl.max():.4f}], "
              f"|slope|>0.005 in {int((np.abs(sl)>0.005).sum())} regions")

    out_dir = VPJAX_ROOT / subject / "longitudinal"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject": subject,
        "sessions": used_sessions,
        "ages": [float(a) for a in ages],
        "age_center": age_center,
        "fit_names": fit_names,
        "region_ids": [int(r) for r in region_ids],
        "baselines": baselines.tolist(),
        "slopes": slopes.tolist(),
        "loss": float(out["loss"]),
        "n_steps": n_steps, "learning_rate": learning_rate,
        "note": (
            "Per-region (baseline, age-slope) Balloon-Windkessel parameters "
            "fit jointly across S sessions via "
            "vpjax.hemodynamics.longitudinal.fit_balloon_longitudinal_from_volumes. "
            "Slope = Δ(parameter) per year, centred at age_center."
        ),
    }
    json_path = out_dir / "balloon_longitudinal.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[long] saved {json_path}")

    # Also project the slopes back into voxel space (using wave-1 aparc
    # for the per-region label volume) for visualization.
    template_aparc = label_per_session[0]
    affine = nib.load(str(
        VPJAX_ROOT / subject / used_sessions[0] / "anat-in-cvr" / "aparc+aseg.nii.gz"
    )).affine
    rid_to_idx = {int(r): i for i, r in enumerate(region_ids)}
    for i, name in enumerate(fit_names):
        slope_map = np.zeros(template_aparc.shape, dtype=np.float32)
        for rid, idx in rid_to_idx.items():
            slope_map[template_aparc == rid] = float(slopes[idx, i])
        out_nii = out_dir / f"slope_{name}_per_year.nii.gz"
        nib.save(nib.Nifti1Image(slope_map, affine), str(out_nii))
        print(f"[long] saved {out_nii}")

    # Companion: regression-β longitudinal fit.  Reads each session's
    # cvr/balloon_params.json (per-region OLS β of BOLD on HRF-convolved
    # stimulus) and does a per-region linear regression of β on
    # (age - age_center).  Closed-form OLS, no JAX, instant.
    fit_regression_longitudinal(
        subject, used_sessions, ages, age_center, out_dir, template_aparc, affine,
    )
    return 0


def fit_regression_longitudinal(
    subject: str,
    sessions: list[str],
    ages: list[float],
    age_center: float,
    out_dir: Path,
    template_aparc: np.ndarray,
    affine: np.ndarray,
) -> None:
    """Per-region OLS regression of regression-β on (age − age_center)."""
    per_session_betas: list[dict[int, float]] = []
    for ses in sessions:
        params_path = (
            VPJAX_ROOT / subject / ses / "cvr" / "balloon_params.json"
        )
        if not params_path.exists():
            print(f"[long-reg] missing {params_path}, skipping regression fit",
                  file=sys.stderr)
            return
        params = json.loads(params_path.read_text())
        if "cvr_regression_beta" not in params:
            print(f"[long-reg] {ses} balloon_params.json predates "
                  "regression-β output; skipping regression fit",
                  file=sys.stderr)
            return
        ids = params["region_ids"]
        betas = params["cvr_regression_beta"]
        per_session_betas.append({int(r): float(b) for r, b in zip(ids, betas)})

    common = set(per_session_betas[0])
    for d in per_session_betas[1:]:
        common &= set(d)
    common_ids = sorted(common)
    if not common_ids:
        print("[long-reg] no common regions across sessions", file=sys.stderr)
        return

    Y = np.array(
        [[d[r] for r in common_ids] for d in per_session_betas]
    )  # (S, R)
    age_offsets = np.array([a - age_center for a in ages], dtype=np.float64)

    # OLS per region: y_s = beta0 + beta1 · age_offset_s
    X = np.stack([np.ones_like(age_offsets), age_offsets], axis=1)  # (S, 2)
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ Y    # (2, R) — [intercept, slope]
    fitted = X @ coef           # (S, R)
    resid = Y - fitted
    rss = (resid ** 2).sum(axis=0)
    tss = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - rss / np.maximum(tss, 1e-30)

    payload = {
        "subject": subject,
        "sessions": sessions,
        "ages": [float(a) for a in ages],
        "age_center": age_center,
        "n_regions": len(common_ids),
        "region_ids": [int(r) for r in common_ids],
        "regression_beta_baseline": coef[0].tolist(),
        "regression_beta_slope_per_year": coef[1].tolist(),
        "per_session_beta": [
            [float(d[r]) for r in common_ids] for d in per_session_betas
        ],
        "r2": r2.tolist(),
        "note": (
            "Per-region OLS of cvr_regression_beta on (age - age_center) "
            "across S sessions.  cvr_regression_beta itself comes from each "
            "session's CVR stage: %BOLD per unit of HRF-convolved data-driven "
            "CO2 stimulus.  Slope is interpretable as 'change in vascular "
            "reactivity per year', though only 3 sessions makes the per-region "
            "estimate noisy — useful as a group-level / pattern proxy."
        ),
    }
    json_path = out_dir / "regression_longitudinal.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[long-reg] saved {json_path} ({len(common_ids)} regions, "
          f"mean |slope|={np.mean(np.abs(coef[1])):.5f}/yr, "
          f"|slope|>1e-4 in {int((np.abs(coef[1]) > 1e-4).sum())} regions)")

    rid_to_idx = {int(r): i for i, r in enumerate(common_ids)}
    slope_map = np.zeros(template_aparc.shape, dtype=np.float32)
    for rid, idx in rid_to_idx.items():
        slope_map[template_aparc == rid] = float(coef[1, idx])
    out_nii = out_dir / "slope_regression_beta_per_year.nii.gz"
    nib.save(nib.Nifti1Image(slope_map, affine), str(out_nii))
    print(f"[long-reg] saved {out_nii}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", help="e.g. sub-1003")
    parser.add_argument("--age-center", type=float, default=60.0)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2.0)
    args = parser.parse_args()
    sys.exit(main(args.subject, args.age_center, args.n_steps, args.learning_rate))
