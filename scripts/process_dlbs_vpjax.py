#!/usr/bin/env python3
"""Process a single DLBS subject/session through the vpjax pipeline.

Stages (each skipped if outputs already exist):

  A. CVR — fit per-region Balloon-Windkessel parameters from the
     hypercapnia BOLD scan, using a data-driven CO₂ stimulus proxy
     (no etCO₂ trace was recorded for DLBS).  Region atlas is the
     FreeSurfer aparc+aseg, registered into the CVR EPI space by
     ``register_freesurfer_to_func.sh``.
  B. Perfusion — pass-through of the oxford_asl perfusion + arrival
     fits (we don't refit; we just copy them under the vpjax tree
     and report region-mean CBF aligned to the same atlas via the
     anat-in-cvr aparc as a near-equivalent space, since DLBS lacks a
     joint ASL/BOLD coregistration).
  C. Metabolism — Fick CMRO₂ from CBF · OEF_prior · CaO₂ using the
     vpjax.metabolism.fick model with a baseline OEF of 0.40.

PET (amyloid / tau) and the longitudinal age-slope fit are *not* run
here — those ride on top of all 3 sessions' aparc-in-CVR being ready
and a separate PET-space registration step.

Usage::

    python process_dlbs_vpjax.py --subject sub-1003 --session ses-wave1
    python process_dlbs_vpjax.py --subject sub-1003 --session ses-wave1 --stage cvr
    python process_dlbs_vpjax.py --subject sub-1003 --all-sessions

Outputs land at::

    /data/datasets/smri-fm-cmp/vpjax/ds004856/<sub>/<ses>/
        cvr/cvr_map.nii.gz, cvr/balloon_params.json, cvr/predicted_bold.nii.gz
        perfusion/cbf_map.nii.gz, perfusion/att_map.nii.gz, perfusion/region_summary.json
        metabolism/cmro2_estimate.nii.gz, metabolism/region_summary.json
        manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dlbs-vpjax")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DS = "ds004856"
FSL_ROOT = Path(f"/data/datasets/smri-fm-cmp/fsl/{DS}")
VPJAX_ROOT = Path(f"/data/datasets/smri-fm-cmp/vpjax/{DS}")
RAW_ROOT = Path(f"/data/raw/openneuro/{DS}")


def cvr_dir(subject: str, session: str) -> Path:
    return FSL_ROOT / "cvr" / f"{subject}_{session}_run-1"


def asl_dir(subject: str, session: str) -> Path:
    return FSL_ROOT / "asl" / f"{subject}_{session}_run-1" / "native_space"


def anat_in_cvr_dir(subject: str, session: str) -> Path:
    return VPJAX_ROOT / subject / session / "anat-in-cvr"


def anat_work_dir(subject: str, session: str) -> Path:
    """Directory holding FreeSurfer mgz → nii.gz conversions in FS/T1 space."""
    return anat_in_cvr_dir(subject, session) / "_work"


PET_TRACER_SUFFIX = {"amyloid": "amyloid_18FAV45", "tau": "tau_18FAV1451"}


def pet_tracer_dir(subject: str, session: str, tracer: str) -> Path:
    return VPJAX_ROOT / subject / session / "pet" / PET_TRACER_SUFFIX[tracer]


def petsurfer_dir(subject: str, session: str, tracer: str) -> Path:
    """PETSurfer outputs from Legion (NFS-shared)."""
    return (
        Path("/data/datasets/smri-fm-cmp/integrated/ds004856")
        / subject / session / "pet" / PET_TRACER_SUFFIX[tracer] / "petsurfer"
    )


def out_session_dir(subject: str, session: str) -> Path:
    return VPJAX_ROOT / subject / session


# FreeSurfer aparc+aseg cerebellum labels.
#   - CEREBELLUM_LABELS         = whole cerebellum (cortex + WM) — used by the
#     vpjax volumetric SUVR helper as a generous reference mask.
#   - CEREBELLUM_CORTEX_LABELS  = cerebellar cortex only — the modern AV45
#     reference and what PETSurfer's mri_gtmpvc was rescaled against.
CEREBELLUM_LABELS = (7, 8, 46, 47)
CEREBELLUM_CORTEX_LABELS = (8, 47)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_nii(path: Path) -> tuple[np.ndarray, np.ndarray, object]:
    import nibabel as nib
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine, img.header


def _save_nii(data: np.ndarray, affine: np.ndarray, path: Path,
              dtype: np.dtype = np.float32) -> None:
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.asarray(data, dtype=dtype), affine),
        str(path),
    )
    log.info("saved %s", path)


def _read_tr(json_path: Path) -> float:
    with open(json_path) as f:
        meta = json.load(f)
    return float(meta["RepetitionTime"])


def _hypercapnia_bold_json(subject: str, session: str) -> Path:
    return (
        RAW_ROOT / subject / session / "func"
        / f"{subject}_{session}_task-Hypercapnia_run-1_bold.json"
    )


# ---------------------------------------------------------------------------
# Stage A: CVR — Balloon biophysics + canonical-HRF regression
# ---------------------------------------------------------------------------

def _canonical_hrf(t: np.ndarray) -> np.ndarray:
    """Single-gamma HRF (Glover 1999, simplified: peak ~6 s, no undershoot)."""
    h = (t / 6.0) ** 5.0 * np.exp(-(t / 6.0))
    h = h / h.max()
    return h


def _ols_per_region_cvr(
    bold_region_ts: np.ndarray,
    stimulus_at_tr: np.ndarray,
    tr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS regression of region-mean BOLD on HRF-convolved stimulus.

    Detrends BOLD with a linear+constant nuisance set, then computes
    the regression coefficient β (%BOLD per unit Δstim) and the
    correlation r.

    Parameters
    ----------
    bold_region_ts : (R, N) per-region BOLD time series, fractional change
    stimulus_at_tr : (N,) stimulus at the TR grid (unitless, peak ~0.3)
    tr : repetition time, s

    Returns
    -------
    beta : (R,) regression slope
    r    : (R,) Pearson correlation
    fitted : (R, N) fitted BOLD response (β · stim_conv + intercept + drift)
    """
    R, N = bold_region_ts.shape
    t_grid = np.arange(N) * tr
    hrf_t = np.arange(0, 30.0 + tr, tr)            # 30 s HRF kernel
    hrf = _canonical_hrf(hrf_t)
    stim_conv_full = np.convolve(stimulus_at_tr, hrf, mode="full")[:N]
    stim_conv = stim_conv_full - stim_conv_full.mean()

    # Design: [convolved_stim, linear_drift, intercept]
    drift = (np.arange(N) - (N - 1) / 2) / N
    drift = drift - drift.mean()
    X = np.stack([stim_conv, drift, np.ones(N)], axis=1)         # (N, 3)
    XtX_inv = np.linalg.inv(X.T @ X)
    Y = bold_region_ts.T                                          # (N, R)
    beta_full = XtX_inv @ X.T @ Y                                 # (3, R)

    fitted_T = X @ beta_full                                       # (N, R)
    resid = Y - fitted_T
    var_y = Y.var(axis=0)
    var_resid = resid.var(axis=0)
    var_y_safe = np.where(var_y > 1e-12, var_y, 1e-12)
    r2 = 1.0 - var_resid / var_y_safe
    r2 = np.clip(r2, 0.0, 1.0)
    # Sign of correlation = sign of beta on the convolved stimulus
    r = np.sign(beta_full[0]) * np.sqrt(r2)

    return beta_full[0], r, fitted_T.T


def run_cvr(subject: str, session: str, *, force: bool = False,
            n_steps: int = 200, learning_rate: float = 2.0,
            min_voxels: int = 20) -> dict:
    out_dir = out_session_dir(subject, session) / "cvr"
    cvr_balloon_path = out_dir / "cvr_balloon_map.nii.gz"
    cvr_regression_path = out_dir / "cvr_regression_map.nii.gz"
    cvr_correlation_path = out_dir / "cvr_correlation_map.nii.gz"
    params_path = out_dir / "balloon_params.json"
    pred_path = out_dir / "predicted_bold.nii.gz"

    if not force and cvr_balloon_path.exists() and params_path.exists() and cvr_regression_path.exists():
        log.info("CVR outputs already exist, skipping (use --force to rerun)")
        return {"status": "skipped", "outputs": [str(cvr_balloon_path), str(params_path)]}

    bold_path = cvr_dir(subject, session) / "mc.nii.gz"
    mask_path = cvr_dir(subject, session) / "brain_mask.nii.gz"
    aparc_path = anat_in_cvr_dir(subject, session) / "aparc+aseg.nii.gz"
    bold_json = _hypercapnia_bold_json(subject, session)

    for p in (bold_path, mask_path, aparc_path, bold_json):
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    tr = _read_tr(bold_json)
    log.info("CVR: TR=%.3fs, BOLD=%s", tr, bold_path)

    bold_4d, affine, _ = _load_nii(bold_path)
    mask, _, _ = _load_nii(mask_path)
    aparc, _, _ = _load_nii(aparc_path)
    aparc = aparc.astype(np.int32)

    log.info("CVR: shape=%s, %d nonzero aparc labels",
             bold_4d.shape, int((aparc > 0).sum()))

    # Lazy import — JAX init is slow.
    from vpjax.hemodynamics.cvr import fit_cvr_per_region, extract_global_stimulus

    # Build stimulus once so we can also use it for the regression.
    stim_dt, _ = extract_global_stimulus(bold_4d, mask, tr=tr, dt=0.1)
    sub = int(round(tr / 0.1))
    stim_tr = np.asarray(stim_dt)[::sub][:bold_4d.shape[-1]]

    t0 = time.time()
    result = fit_cvr_per_region(
        bold_4d=bold_4d,
        brain_mask=mask,
        region_volume=aparc,
        tr=tr,
        dt=0.1,
        stimulus=stim_dt,
        n_steps=n_steps,
        learning_rate=learning_rate,
        min_voxels=min_voxels,
    )
    log.info("CVR: Balloon fit %d regions in %.1fs",
             result.region_ids.shape[0], time.time() - t0)

    # OLS-regression CVR per region on the same observed BOLD time series.
    t0 = time.time()
    beta_reg, r_reg, fitted_reg = _ols_per_region_cvr(
        np.asarray(result.bold_observed), stim_tr, tr=tr,
    )
    log.info("CVR: regression done in %.2fs (mean β=%.4f, mean |r|=%.3f)",
             time.time() - t0, float(np.mean(beta_reg)), float(np.mean(np.abs(r_reg))))

    # Voxel-wise maps: broadcast per-region scalars across voxels.
    cvr_balloon = np.zeros(bold_4d.shape[:3], dtype=np.float32)
    cvr_reg = np.zeros(bold_4d.shape[:3], dtype=np.float32)
    cvr_corr = np.zeros(bold_4d.shape[:3], dtype=np.float32)
    pred_4d = np.zeros(bold_4d.shape, dtype=np.float32)
    rid_to_idx = {int(r): i for i, r in enumerate(result.region_ids)}
    for rid, idx in rid_to_idx.items():
        rmask = (aparc == rid)
        cvr_balloon[rmask] = float(result.cvr_scalar[idx])
        cvr_reg[rmask] = float(beta_reg[idx])
        cvr_corr[rmask] = float(r_reg[idx])
        pred_4d[rmask, :] = result.bold_predicted[idx, :]

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_nii(cvr_balloon, affine, cvr_balloon_path)
    _save_nii(cvr_reg, affine, cvr_regression_path)
    _save_nii(cvr_corr, affine, cvr_correlation_path)
    _save_nii(pred_4d, affine, pred_path)

    # Backwards-compat alias.
    cvr_map_path = out_dir / "cvr_map.nii.gz"
    if cvr_map_path.exists() and not cvr_map_path.is_symlink():
        cvr_map_path.unlink()
    if not cvr_map_path.exists():
        cvr_map_path.symlink_to(cvr_balloon_path.name)

    payload = {
        "region_ids": result.region_ids.tolist(),
        "kappa": result.kappa.tolist(),
        "gamma": result.gamma.tolist(),
        "tau": result.tau.tolist(),
        "alpha": result.alpha.tolist(),
        "E0": result.E0.tolist(),
        "cvr_balloon_scalar": result.cvr_scalar.tolist(),
        "cvr_regression_beta": beta_reg.tolist(),
        "cvr_regression_r": r_reg.tolist(),
        "loss_balloon": result.loss.tolist(),
        "tr": tr,
        "n_steps_balloon": n_steps,
        "learning_rate_balloon": learning_rate,
        "fit_names_balloon": ["kappa", "tau"],
        "regression_method": (
            "OLS of region-mean BOLD ~ HRF-convolved stimulus + linear drift + intercept; "
            "HRF = single-gamma t^5 exp(-t/6) (Glover 1999 simplified, peak ≈ 6 s); "
            "stimulus = whole-brain mean BOLD low-passed at 0.1 Hz, scaled to ±0.3."
        ),
    }
    params_path.write_text(json.dumps(payload, indent=2))
    log.info("saved %s", params_path)

    return {
        "status": "ok",
        "n_regions": int(result.region_ids.shape[0]),
        "mean_loss_balloon": float(np.mean(result.loss)),
        "mean_regression_beta": float(np.mean(beta_reg)),
        "mean_regression_r": float(np.mean(np.abs(r_reg))),
        "outputs": [
            str(cvr_balloon_path), str(cvr_regression_path),
            str(cvr_correlation_path), str(params_path), str(pred_path),
        ],
    }


# ---------------------------------------------------------------------------
# Stage B: Perfusion — quantify CBF from raw pCASL via vpjax.perfusion.kinetic
# ---------------------------------------------------------------------------

def _raw_asl_paths(subject: str, session: str) -> tuple[Path, Path, Path]:
    base = (
        RAW_ROOT / subject / session / "perf"
        / f"{subject}_{session}_acq-ASL_run-1"
    )
    return (
        base.with_name(base.name + "_asl.nii.gz"),
        base.with_name(base.name + "_asl.json"),
        RAW_ROOT / subject / session / "perf"
            / f"{subject}_{session}_aslcontext.tsv",
    )


def _read_aslcontext(tsv_path: Path) -> np.ndarray:
    """Return a 1D array of 'label' / 'control' / 'm0scan' / 'deltam' strings."""
    with open(tsv_path) as f:
        # First line is header 'volume_type'
        types = [line.strip() for line in f.readlines()[1:]]
    return np.asarray(types)


def run_perfusion(subject: str, session: str, *,
                  force: bool = False,
                  motion_correct: bool = True) -> dict:
    """Quantify CBF in ASL native space from raw pCASL pairs.

    Uses ``vpjax.perfusion.kinetic.quantify_cbf`` with the Alsop 2015
    single-PLD formula::

        CBF = 6000·λ·ΔM·exp(PLD/T1b) / (2·α·T1b·M0·(1 − exp(−τ/T1b)))

    With BIDS ``M0Type: Absent`` we use the per-voxel mean of control
    volumes as M0 (per Alsop Table 1).  When ``motion_correct=True``
    (default) FSL ``mcflirt`` aligns the 60 raw volumes to volume 0
    before label/control averaging — a 3D acquisition this short
    typically has sub-mm drift, but mcflirt is cheap (a few seconds)
    and avoids systematic bias from any residual motion.

    Outputs land in ASL native space:
      perfusion/cbf_map.nii.gz       — CBF (mL/100g/min)
      perfusion/att_map.nii.gz       — ATT (s) from oxford_asl passthrough
      perfusion/m0_estimate.nii.gz   — control-mean M0
      perfusion/deltam.nii.gz        — mean(control − label)
      perfusion/asl_mc.nii.gz        — motion-corrected raw 4D (if MC on)
      perfusion/asl_mc.par           — mcflirt 6-DOF parameters per volume
      perfusion/region_summary.json  — provenance + global stats
    """
    out_dir = out_session_dir(subject, session) / "perfusion"
    cbf_path = out_dir / "cbf_map.nii.gz"
    att_path = out_dir / "att_map.nii.gz"
    m0_path = out_dir / "m0_estimate.nii.gz"
    deltam_path = out_dir / "deltam.nii.gz"
    summary_path = out_dir / "region_summary.json"

    if not force and cbf_path.exists() and m0_path.exists():
        log.info("perfusion outputs already exist, skipping")
        return {"status": "skipped", "outputs": [str(cbf_path), str(m0_path)]}

    raw_nii, raw_json, aslcontext = _raw_asl_paths(subject, session)
    for p in (raw_nii, raw_json, aslcontext):
        if not p.exists():
            return {"status": "missing", "reason": f"raw ASL input missing: {p}"}

    out_dir.mkdir(parents=True, exist_ok=True)

    if motion_correct:
        import subprocess
        mc_base = out_dir / "asl_mc"  # mcflirt appends .nii.gz
        mc_nii = out_dir / "asl_mc.nii.gz"
        log.info("ASL: running mcflirt -refvol 0 -cost normmi ...")
        subprocess.run([
            "mcflirt",
            "-in", str(raw_nii),
            "-out", str(mc_base),
            "-refvol", "0",
            "-cost", "normmi",
            "-plots",
        ], check=True)
        asl, affine_asl, _ = _load_nii(mc_nii)
        # mcflirt -plots writes .par next to the output (asl_mc.par).
    else:
        asl, affine_asl, _ = _load_nii(raw_nii)

    if asl.ndim != 4:
        return {"status": "error", "reason": f"ASL not 4D: shape={asl.shape}"}

    types = _read_aslcontext(aslcontext)
    if types.shape[0] != asl.shape[3]:
        return {"status": "error",
                "reason": f"aslcontext rows ({types.shape[0]}) != volumes ({asl.shape[3]})"}

    label_idx = np.where(types == "label")[0]
    ctrl_idx = np.where(types == "control")[0]
    if label_idx.size == 0 or ctrl_idx.size == 0:
        return {"status": "error", "reason": f"unexpected aslcontext types: {set(types)}"}
    log.info("ASL: %d label + %d control volumes", label_idx.size, ctrl_idx.size)

    meta = json.loads(raw_json.read_text())
    pld = float(meta["PostLabelingDelay"])
    tau = float(meta["LabelingDuration"])
    asl_type = meta.get("ArterialSpinLabelingType", "PCASL").upper()
    m0_type = meta.get("M0Type", "Absent")
    log.info("ASL: type=%s, PLD=%.3fs, τ=%.3fs, M0Type=%s",
             asl_type, pld, tau, m0_type)

    label_mean = asl[..., label_idx].mean(axis=-1)
    ctrl_mean = asl[..., ctrl_idx].mean(axis=-1)
    delta_m = ctrl_mean - label_mean

    if m0_type == "Absent":
        m0 = ctrl_mean
        m0_source = "mean(control)"
    else:
        # Could add support for separate M0Scan here; not needed for DLBS.
        return {"status": "error", "reason": f"M0Type={m0_type} not yet supported"}

    # vpjax single-PLD quantification.  alpha=0.85 for pCASL (Alsop 2015).
    import jax.numpy as jnp
    from vpjax.perfusion.kinetic import ASLKineticParams, quantify_cbf

    params = ASLKineticParams(
        M0b=jnp.asarray(m0),
        T1b=jnp.array(1.65),
        alpha=jnp.array(0.85 if asl_type == "PCASL" else 0.95),
        tau=jnp.array(tau),
        delta=jnp.array(1.20),  # nominal; ATT comes from oxford_asl
        lambda_p=jnp.array(0.90),
    )
    cbf_q = np.asarray(quantify_cbf(jnp.asarray(delta_m), pld=pld, params=params))

    # Mask out non-brain (use oxford_asl mask if available).
    mask_path = asl_dir(subject, session) / "mask.nii.gz"
    if mask_path.exists():
        mask = _load_nii(mask_path)[0] > 0
        cbf_q = np.where(mask, cbf_q, 0.0)
    cbf_q = np.where(np.isfinite(cbf_q), cbf_q, 0.0)
    cbf_q = np.clip(cbf_q, 0.0, 200.0)  # physiological upper bound

    _save_nii(cbf_q.astype(np.float32), affine_asl, cbf_path)
    _save_nii(m0.astype(np.float32), affine_asl, m0_path)
    _save_nii(delta_m.astype(np.float32), affine_asl, deltam_path)

    # Compute framewise displacement summary from mcflirt .par if present.
    fd_summary = None
    par_path = out_dir / "asl_mc.par"
    if par_path.exists():
        par = np.loadtxt(str(par_path))  # cols: rx ry rz tx ty tz (rad, mm)
        # Power-style FD: |Δtx|+|Δty|+|Δtz| + 50·(|Δrx|+|Δry|+|Δrz|)
        d = np.diff(par, axis=0)
        fd = np.abs(d[:, :3]).sum(axis=1) * 50.0 + np.abs(d[:, 3:]).sum(axis=1)
        fd_summary = {
            "n_volumes": int(par.shape[0]),
            "fd_mean_mm": float(fd.mean()),
            "fd_max_mm": float(fd.max()),
            "fd_p95_mm": float(np.percentile(fd, 95)),
        }
        log.info("ASL motion: FD mean=%.3f mm, max=%.3f mm, p95=%.3f mm",
                 fd_summary["fd_mean_mm"], fd_summary["fd_max_mm"], fd_summary["fd_p95_mm"])

    # Optional ATT passthrough from oxford_asl if present.
    arr_src = asl_dir(subject, session) / "arrival.nii.gz"
    att_status = None
    if arr_src.exists():
        att, _, _ = _load_nii(arr_src)
        _save_nii(att, affine_asl, att_path)
        att_status = "oxford_asl arrival.nii.gz"

    cbf_brain = cbf_q[cbf_q > 0]

    # Quality flag from coherent ΔM/M0 in the brain mask.  We take the
    # SIGNED brain-mean ΔM (perfusion signal must be positive — controls
    # are brighter than labels) divided by mean M0.  pCASL ΔM/M0 in a
    # healthy adult is ~0.5–2 % (Alsop 2015); below 0.05 % or NEGATIVE
    # means labeling essentially failed and the per-pair signal is
    # dominated by noise — the per-voxel |ΔM| stays large in that case
    # but the coherent contrast vanishes, which is the right diagnostic.
    if mask_path.exists():
        brain_mask = _load_nii(mask_path)[0] > 0
    else:
        brain_mask = m0 > 0
    dm_brain_mean = float(delta_m[brain_mask].mean())
    m0_brain_mean = float(m0[brain_mask].mean())
    delta_ratio = dm_brain_mean / m0_brain_mean if m0_brain_mean > 0 else 0.0
    if delta_ratio < 5e-4:
        quality_flag = "low"
        log.warning(
            "ASL quality LOW: coherent ΔM/M0 = %.5f (labeling likely failed)",
            delta_ratio,
        )
    elif delta_ratio < 2e-3:
        quality_flag = "marginal"
    else:
        quality_flag = "ok"
    log.info("ASL quality flag = %s (ΔM/M0 = %+.5f, ΔM=%.1f, M0=%.1f)",
             quality_flag, delta_ratio, dm_brain_mean, m0_brain_mean)

    summary_path.write_text(json.dumps({
        "asl_type": asl_type,
        "pld_s": pld,
        "tau_s": tau,
        "m0_type": m0_type,
        "m0_source": m0_source,
        "label_volumes": int(label_idx.size),
        "control_volumes": int(ctrl_idx.size),
        "alpha_assumed": 0.85 if asl_type == "PCASL" else 0.95,
        "T1b_s": 1.65,
        "lambda_p": 0.90,
        "cbf_unit": "mL/100g/min",
        "cbf_global_mean": float(cbf_brain.mean()) if cbf_brain.size else None,
        "cbf_global_p10_p90": [
            float(np.percentile(cbf_brain, 10)),
            float(np.percentile(cbf_brain, 90)),
        ] if cbf_brain.size else None,
        "delta_m_over_m0_brain_coherent": delta_ratio,
        "quality_flag": quality_flag,
        "att_source": att_status,
        "note": (
            "vpjax-quantified CBF via Alsop 2015 single-PLD pCASL formula. "
            "Per-voxel M0 from mean(control) since BIDS M0Type=Absent. "
            "ATT not refit (uses oxford_asl arrival.nii.gz when available)."
        ),
    }, indent=2))
    log.info("saved %s", summary_path)

    return {
        "status": "ok" if quality_flag != "low" else "ok_low_quality",
        "n_label": int(label_idx.size),
        "n_control": int(ctrl_idx.size),
        "cbf_global_mean": float(cbf_brain.mean()) if cbf_brain.size else None,
        "quality_flag": quality_flag,
        "delta_m_over_m0_brain_coherent": delta_ratio,
        "outputs": [str(cbf_path), str(m0_path), str(deltam_path), str(summary_path)],
    }


# ---------------------------------------------------------------------------
# Stage C: Metabolism (Fick CMRO₂ with OEF prior)
# ---------------------------------------------------------------------------

def run_metabolism(subject: str, session: str, *,
                   oef_prior: float = 0.40,
                   force: bool = False) -> dict:
    out_dir = out_session_dir(subject, session) / "metabolism"
    cmro2_path = out_dir / "cmro2_estimate.nii.gz"
    summary_path = out_dir / "region_summary.json"

    if not force and cmro2_path.exists():
        log.info("metabolism outputs already exist, skipping")
        return {"status": "skipped", "outputs": [str(cmro2_path)]}

    cbf_path = out_session_dir(subject, session) / "perfusion" / "cbf_map.nii.gz"
    if not cbf_path.exists():
        log.warning("CBF map missing — run perfusion stage first")
        return {"status": "missing", "reason": "cbf_map.nii.gz not found"}

    from vpjax.metabolism.fick import compute_cao2, fick_cmro2
    import jax.numpy as jnp

    cbf, affine, _ = _load_nii(cbf_path)
    cao2 = float(compute_cao2())  # default Fick params (Hb=15, SaO2=0.98)
    cmro2 = np.asarray(
        fick_cmro2(jnp.asarray(cbf), oef=jnp.asarray(oef_prior), cao2=jnp.asarray(cao2))
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_nii(cmro2, affine, cmro2_path)

    summary_path.write_text(json.dumps({
        "oef_prior": oef_prior,
        "cao2_umol_per_ml": cao2,
        "cmro2_unit": "µmol O2 / 100g / min",
        "cbf_source": str(cbf_path),
        "cmro2_mean": float(cmro2[cbf > 0].mean()) if (cbf > 0).any() else None,
        "note": (
            "CMRO2 = CBF * OEF_prior * CaO2 (Fick). OEF is a fixed prior, "
            "not voxelwise — the joint amyloid-CMRO2 model in vpjax.pet.joint "
            "is needed to relax this assumption against PET data."
        ),
    }, indent=2))
    log.info("saved %s", summary_path)

    return {
        "status": "ok",
        "outputs": [str(cmro2_path), str(summary_path)],
        "oef_prior": oef_prior,
    }


# ---------------------------------------------------------------------------
# Stage D: PET (static SUVR with optional flow-corrected BPnd)
# ---------------------------------------------------------------------------

def run_pet(subject: str, session: str, *,
            tracer: str = "amyloid",
            min_voxels: int = 20,
            force: bool = False) -> dict:
    out_dir = pet_tracer_dir(subject, session, tracer)
    pet_in_t1 = out_dir / "pet_in_t1.nii.gz"
    suvr_path = out_dir / "suvr_map.nii.gz"
    bpnd_path = out_dir / "bpnd_map.nii.gz"
    regional_path = out_dir / "regional_suvr.json"

    if not force and suvr_path.exists() and regional_path.exists():
        log.info("pet outputs already exist, skipping")
        return {"status": "skipped", "outputs": [str(suvr_path), str(regional_path)]}

    if not pet_in_t1.exists():
        log.warning(
            "pet_in_t1 missing — run register_pet_to_t1.sh %s %s %s first",
            subject, session, tracer,
        )
        return {"status": "missing", "reason": str(pet_in_t1)}

    aparc_path = anat_work_dir(subject, session) / "aparc+aseg.nii.gz"
    if not aparc_path.exists():
        return {"status": "missing", "reason": str(aparc_path)}

    pet, affine, _ = _load_nii(pet_in_t1)
    aparc, _, _ = _load_nii(aparc_path)
    aparc = aparc.astype(np.int32)

    cb_mask = np.isin(aparc, CEREBELLUM_LABELS).astype(np.float32)
    if cb_mask.sum() < 100:
        return {"status": "error", "reason": "cerebellum mask too small"}
    log.info("pet: cerebellum reference %d voxels", int(cb_mask.sum()))

    # Lazy import — JAX init slow.
    from vpjax.pet.suvr import compute_suvr, regional_suvr
    from vpjax.pet.binding import fit_bpnd_per_region

    suvr_3d, ref_mean = compute_suvr(pet, cb_mask)
    log.info("pet: cerebellum mean uptake = %.1f, SUVR computed", ref_mean)

    # Per-region SUVR (every nonzero aparc label that meets min_voxels).
    suvr_per_region, region_ids = regional_suvr(
        pet, cb_mask, aparc, min_voxels=min_voxels,
    )

    # BPnd ≈ SUVR - 1 (no flow correction at this stage).
    fit_out = fit_bpnd_per_region(
        suvr_per_region, fit_flow=False,
        n_steps=200, learning_rate=0.05,
    )
    bpnd_per_region = fit_out["bpnd"]

    # Voxelwise BPnd map: SUVR - 1 (clip negatives to 0).
    bpnd_map = np.maximum(suvr_3d - 1.0, 0.0).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_nii(suvr_3d, affine, suvr_path)
    _save_nii(bpnd_map, affine, bpnd_path)

    payload = {
        "tracer": tracer,
        "reference_labels": list(CEREBELLUM_LABELS),
        "reference_mean_pet": float(ref_mean),
        "n_regions": int(region_ids.shape[0]),
        "region_ids": region_ids.tolist(),
        "suvr": suvr_per_region.tolist(),
        "bpnd": bpnd_per_region.tolist(),
        "bpnd_global_loss": float(fit_out["loss"]),
        "note": (
            "Static-frame SUVR with whole-cerebellum reference. "
            "BPnd ≈ SUVR - 1 (Logan equilibrium limit, no flow correction). "
            "Use vpjax.pet.joint for flow-aware BPnd + OEF estimation."
        ),
    }
    regional_path.write_text(json.dumps(payload, indent=2))
    log.info("saved %s", regional_path)

    return {
        "status": "ok",
        "n_regions": int(region_ids.shape[0]),
        "outputs": [str(suvr_path), str(bpnd_path), str(regional_path)],
    }


# ---------------------------------------------------------------------------
# Stage D′: PETSurfer pickup — GTM-PVC per-region SUVR from Legion
# ---------------------------------------------------------------------------

def _parse_gtm_stats(stats_path: Path) -> tuple[list[int], list[str], list[str]]:
    """Parse a PETSurfer gtm.stats.dat fixed-width table.

    Returns (label_ids, region_names, tissue_types) — order matches the
    rows of gtm.nii.gz / gtm.tsv exactly.
    """
    label_ids: list[int] = []
    names: list[str] = []
    tissue: list[str] = []
    with open(stats_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                _idx = int(parts[0])
                label_id = int(parts[1])
            except ValueError:
                continue
            label_ids.append(label_id)
            names.append(parts[2])
            tissue.append(parts[3])
    return label_ids, names, tissue


def run_pet_petsurfer(subject: str, session: str, *,
                     tracer: str = "amyloid",
                     force: bool = False) -> dict:
    """Ingest PETSurfer GTM-PVC outputs into the vpjax tree.

    PETSurfer (FreeSurfer's mri_gtmpvc) ran on Legion and lives under
    /data/datasets/smri-fm-cmp/integrated/.../petsurfer/.  Each session
    has a per-region SUVR vector with iterative GTM partial-volume
    correction — strictly better than our volumetric Müller-Gärtner
    helper for cortical regions.

    Output: vpjax/.../pet/<tracer>/regional_suvr_petsurfer.json with
    the same schema as the existing regional_suvr.json
    (region_ids, suvr, bpnd) so the joint stage can swap it in
    transparently.  Adds suvr_nopvc and tissue_type columns for
    diagnostic use.
    """
    out_dir = pet_tracer_dir(subject, session, tracer)
    out_path = out_dir / "regional_suvr_petsurfer.json"

    if not force and out_path.exists():
        log.info("petsurfer SUVR already ingested, skipping")
        return {"status": "skipped", "outputs": [str(out_path)]}

    src = petsurfer_dir(subject, session, tracer)
    if not src.exists():
        log.warning("PETSurfer dir missing: %s", src)
        return {"status": "missing", "reason": str(src)}

    stats_path = src / "pvc" / "gtm.stats.dat"
    gtm_path = src / "pvc" / "gtm.nii.gz"
    nopvc_path = src / "pvc" / "nopvc.nii.gz"
    for p in (stats_path, gtm_path, nopvc_path):
        if not p.exists():
            return {"status": "missing", "reason": f"PETSurfer file missing: {p}"}

    label_ids, names, tissues = _parse_gtm_stats(stats_path)
    gtm_raw = _load_nii(gtm_path)[0].astype(np.float32)
    nopvc_raw = _load_nii(nopvc_path)[0].astype(np.float32)

    # PETSurfer normally writes (N_regions, 1, 1).  Some sessions land
    # with an extra frame axis (N_regions, 1, 1, 2) when mri_gtmpvc
    # processes multiple input frames — we take the first frame for the
    # canonical static SUVR.
    def _to_1d(arr: np.ndarray, n_expected: int) -> np.ndarray:
        if arr.ndim == 4 and arr.shape[-1] > 1:
            arr = arr[..., 0]  # first frame only
        a = arr.squeeze()
        if a.ndim == 2 and a.shape[1] > 1:
            a = a[:, 0]
        elif a.ndim != 1:
            a = a.reshape(-1)
        return a[:n_expected] if a.size > n_expected else a

    gtm = _to_1d(gtm_raw, len(label_ids))
    nopvc = _to_1d(nopvc_raw, len(label_ids))
    n_frames_amy = int(gtm_raw.shape[-1]) if gtm_raw.ndim == 4 else 1

    if gtm.shape != (len(label_ids),):
        return {"status": "error",
                "reason": f"gtm.nii.gz length {gtm.shape} ≠ stats rows {len(label_ids)} "
                          f"(raw shape {gtm_raw.shape})"}
    if nopvc.shape != gtm.shape:
        return {"status": "error",
                "reason": f"nopvc.nii.gz shape {nopvc.shape} ≠ gtm shape {gtm.shape}"}

    # PETSurfer's GTM is already cerebellum-cortex-normalised (--rescale 8 47);
    # the diagnostic mean over (8, 47) should be ≈ 1.0 by construction.
    cb_idx = [i for i, l in enumerate(label_ids) if l in CEREBELLUM_CORTEX_LABELS]
    cb_gtm_mean = float(gtm[cb_idx].mean()) if cb_idx else None
    cb_nopvc_mean = float(nopvc[cb_idx].mean()) if cb_idx else None

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tracer": tracer,
        "source": "petsurfer (mri_gtmpvc)",
        "petsurfer_dir": str(src),
        "n_regions": len(label_ids),
        "region_ids": label_ids,
        "region_names": names,
        "tissue_types": tissues,
        "suvr": gtm.tolist(),                 # GTM-PVC SUVR
        "suvr_nopvc": nopvc.tolist(),         # un-PVCed reference
        "bpnd": (gtm - 1.0).tolist(),         # Logan-equilibrium BPnd
        "cerebellum_gtm_mean": cb_gtm_mean,
        "cerebellum_nopvc_mean": cb_nopvc_mean,
        "n_frames_in_source": n_frames_amy,
        "note": (
            "PETSurfer mri_gtmpvc geometric-transfer-matrix PVC. "
            "Already SUVR-normalised against cerebellar GM (FS labels 8,47); "
            "BPnd = SUVR - 1 (Logan equilibrium limit). Some regions can have "
            "negative GTM-PVC values due to noise amplification — keep them "
            "for downstream regression-style analyses but treat as low-confidence."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("saved %s (%d regions, cerebellum-GTM=%.3f)",
             out_path, len(label_ids), cb_gtm_mean or 0.0)

    return {
        "status": "ok",
        "n_regions": len(label_ids),
        "cerebellum_gtm_mean": cb_gtm_mean,
        "outputs": [str(out_path)],
    }


# ---------------------------------------------------------------------------
# Stage E: ASL → T1 + regional CBF (prep for the joint amyloid–CMRO2 fit)
# ---------------------------------------------------------------------------

def run_asl_in_t1(subject: str, session: str, *, force: bool = False) -> dict:
    """Register the vpjax-quantified CBF map to T1 and compute per-region CBF.

    Consumes ``perfusion/cbf_map.nii.gz`` produced by ``run_perfusion``,
    which is already in mL/100g/min via the Alsop 2015 quantification.
    No post-hoc rescaling is applied.
    """
    out_dir = out_session_dir(subject, session) / "perfusion"
    cbf_t1_path = out_dir / "cbf_in_t1.nii.gz"
    asl_to_t1_mat = out_dir / "asl_to_t1.mat"
    regional_cbf_path = out_dir / "regional_cbf.json"

    if not force and cbf_t1_path.exists() and regional_cbf_path.exists():
        log.info("ASL-in-T1 outputs already exist, skipping")
        return {"status": "skipped", "outputs": [str(cbf_t1_path), str(regional_cbf_path)]}

    cbf_native_path = out_dir / "cbf_map.nii.gz"
    m0_native_path = out_dir / "m0_estimate.nii.gz"
    if not cbf_native_path.exists():
        return {"status": "missing", "reason": "run perfusion stage first"}
    if not m0_native_path.exists():
        return {"status": "missing", "reason": "m0_estimate.nii.gz missing"}

    aparc_path = anat_work_dir(subject, session) / "aparc+aseg.nii.gz"
    brain_path = anat_work_dir(subject, session) / "brain.nii.gz"
    if not aparc_path.exists() or not brain_path.exists():
        return {"status": "missing", "reason": "FS conversions missing — run register_freesurfer_to_func.sh first"}

    import subprocess

    # Register the M0 image (control-mean, T1-like contrast) rather than
    # the CBF map (CSF-zero / GM-high, harder to register cross-modally).
    # Then apply the same transform to the CBF map with trilinear
    # interpolation.  Without this, wave-3 wave registers ~3.5x off in
    # the cerebellum because the CBF/T1 contrast inversion confuses MI.
    log.info("running flirt: ASL M0 → T1 brain (then apply to CBF)...")
    subprocess.run([
        "flirt",
        "-in", str(m0_native_path),
        "-ref", str(brain_path),
        "-omat", str(asl_to_t1_mat),
        "-out", str(out_dir / "m0_in_t1.nii.gz"),
        "-dof", "6",
        "-cost", "normmi",
        "-interp", "trilinear",
    ], check=True)
    subprocess.run([
        "flirt",
        "-in", str(cbf_native_path),
        "-ref", str(brain_path),
        "-applyxfm", "-init", str(asl_to_t1_mat),
        "-out", str(cbf_t1_path),
        "-interp", "trilinear",
    ], check=True)

    # Compute per-region CBF means using aparc.
    cbf_t1, t1_affine, _ = _load_nii(cbf_t1_path)
    aparc, _, _ = _load_nii(aparc_path)
    aparc = aparc.astype(np.int32)

    cb_mask = np.isin(aparc, CEREBELLUM_LABELS)

    region_ids = np.unique(aparc[aparc > 0]).astype(np.int32)
    means = []
    kept = []
    for rid in region_ids:
        m = aparc == rid
        n = int(m.sum())
        if n < 20:
            continue
        cbf_vals = cbf_t1[m]
        cbf_vals = cbf_vals[cbf_vals > 0]
        if cbf_vals.size < 5:
            continue
        means.append(float(cbf_vals.mean()))
        kept.append(int(rid))

    cbf_ref = float(cbf_t1[cb_mask][cbf_t1[cb_mask] > 0].mean()) if cb_mask.any() else None

    regional_cbf_path.write_text(json.dumps({
        "n_regions": len(kept),
        "region_ids": kept,
        "cbf_mean_per_region": means,
        "cbf_ref_cerebellum": cbf_ref,
        "unit": "mL/100g/min",
        "note": (
            "Per-region CBF means in T1 (FreeSurfer) space. "
            "Source CBF is the vpjax-quantified Alsop 2015 single-PLD map "
            "(see perfusion/region_summary.json). ASL→T1 transform via "
            "flirt 6-DOF normmi. cbf_ref_cerebellum is suitable as the "
            "cbf_ref for vpjax.pet.binding / vpjax.pet.joint flow correction."
        ),
    }, indent=2))
    log.info("saved %s (%d regions, cbf_ref=%s)",
             regional_cbf_path, len(kept), f"{cbf_ref:.2f}" if cbf_ref else "n/a")

    return {
        "status": "ok",
        "n_regions": len(kept),
        "cbf_ref_cerebellum": cbf_ref,
        "outputs": [str(cbf_t1_path), str(regional_cbf_path)],
    }


# ---------------------------------------------------------------------------
# Stage F: Joint amyloid–CMRO2 fit
# ---------------------------------------------------------------------------

def run_joint_amyloid_cmro2(subject: str, session: str, *,
                             tracer: str = "amyloid",
                             pet_source: str = "auto",
                             force: bool = False) -> dict:
    """Joint per-region (BPnd, OEF) fit with Vaishnavi-style coupling.

    The PET source can be ``"petsurfer"`` (GTM-PVC, preferred when
    available), ``"vpjax"`` (our volumetric SUVR), or ``"auto"`` —
    which picks PETSurfer if its file is present, otherwise falls back
    to the vpjax pipeline.  The selected source is recorded in the
    output JSON.
    """
    out_path = (out_session_dir(subject, session)
                / "metabolism" / "amyloid_cmro2_joint.json")
    if not force and out_path.exists():
        log.info("joint amyloid-CMRO2 output already exists, skipping")
        return {"status": "skipped", "outputs": [str(out_path)]}

    cbf_summary = out_session_dir(subject, session) / "perfusion" / "regional_cbf.json"
    if not cbf_summary.exists():
        return {"status": "missing",
                "reason": "needs asl_in_t1 stage (regional_cbf.json)"}

    # If the perfusion stage flagged the ASL acquisition as "low" quality
    # (ΔM/M0 below the labeling-noise threshold) the joint fit becomes
    # meaningless — CBF drives both the SUVR flow correction and the Fick
    # CMRO2 term.  Skip with an explicit status so downstream tooling
    # knows not to use this session for joint analyses.
    perf_summary_path = out_session_dir(subject, session) / "perfusion" / "region_summary.json"
    if perf_summary_path.exists():
        perf_summary = json.loads(perf_summary_path.read_text())
        if perf_summary.get("quality_flag") == "low":
            log.warning(
                "joint stage skipped: perfusion quality_flag=low (raw pCASL "
                "labeling failed for this session — see region_summary.json)",
            )
            return {
                "status": "skipped_low_quality_cbf",
                "reason": "perfusion quality_flag=low",
                "delta_m_over_m0_coherent": perf_summary.get(
                    "delta_m_over_m0_brain_coherent"
                ),
            }

    pet_dir = pet_tracer_dir(subject, session, tracer)
    petsurfer_path = pet_dir / "regional_suvr_petsurfer.json"
    vpjax_path = pet_dir / "regional_suvr.json"

    if pet_source == "petsurfer":
        pet_summary = petsurfer_path
    elif pet_source == "vpjax":
        pet_summary = vpjax_path
    elif pet_source == "auto":
        pet_summary = petsurfer_path if petsurfer_path.exists() else vpjax_path
    else:
        return {"status": "error", "reason": f"unknown pet_source: {pet_source}"}

    if not pet_summary.exists():
        return {"status": "missing",
                "reason": f"PET regional summary missing: {pet_summary}"}

    log.info("joint: PET source = %s (%s)",
             "petsurfer" if pet_summary == petsurfer_path else "vpjax",
             pet_summary.name)

    pet_data = json.loads(pet_summary.read_text())
    cbf_data = json.loads(cbf_summary.read_text())

    pet_ids = np.asarray(pet_data["region_ids"], dtype=np.int32)
    pet_suvr = np.asarray(pet_data["suvr"], dtype=np.float32)
    cbf_ids = np.asarray(cbf_data["region_ids"], dtype=np.int32)
    cbf_means = np.asarray(cbf_data["cbf_mean_per_region"], dtype=np.float32)

    # Intersect on region IDs that have both SUVR and CBF.
    common = np.intersect1d(pet_ids, cbf_ids)
    if common.size == 0:
        return {"status": "error", "reason": "no overlapping region IDs"}
    pet_idx = np.searchsorted(pet_ids, common)
    cbf_idx = np.searchsorted(cbf_ids, common)
    suvr = pet_suvr[pet_idx]
    cbf = cbf_means[cbf_idx]
    cbf_ref = (
        float(cbf_data.get("cbf_ref_cerebellum") or cbf.mean())
    )
    log.info("joint: %d regions, cbf_ref=%.2f", common.size, cbf_ref)

    from vpjax.pet.joint import fit_joint_amyloid_cmro2

    out = fit_joint_amyloid_cmro2(
        suvr_obs=suvr,
        cbf_obs=cbf,
        cbf_ref=cbf_ref,
        lambda_couple=0.5,
        lambda_oef=1.0,
        n_steps=500,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pet_source_used = "petsurfer" if pet_summary == petsurfer_path else "vpjax"
    payload = {
        "tracer": tracer,
        "pet_source": pet_source_used,
        "pet_summary_path": str(pet_summary),
        "n_regions": int(common.size),
        "region_ids": common.tolist(),
        "suvr": suvr.tolist(),
        "cbf": cbf.tolist(),
        "cbf_ref_cerebellum": cbf_ref,
        "bpnd": out["bpnd"].tolist(),
        "oef": out["oef"].tolist(),
        "alpha_amy": float(out["alpha_amy"]),
        "beta_flow": float(out["beta_flow"]),
        "cmro2_fick": out["cmro2_fick"].tolist(),
        "cmro2_coupling": out["cmro2_coupling"].tolist(),
        "loss": float(out["loss"]),
        "note": (
            "Joint per-region (BPnd, OEF) recovered from SUVR + CBF with "
            "Vaishnavi 2010 amyloid-CMRO2 coupling (CMRO2 = baseline·(1+α·BPnd)). "
            "Loss = SUVR fit + λ_couple · normalised CMRO2 residual + λ_oef · (OEF-prior)². "
            "When pet_source='petsurfer', SUVR comes from FreeSurfer's mri_gtmpvc "
            "(GTM partial-volume correction) and is more accurate in cortical "
            "regions than the vpjax volumetric Müller-Gärtner output."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("saved %s", out_path)

    return {
        "status": "ok",
        "n_regions": int(common.size),
        "outputs": [str(out_path)],
    }


# ---------------------------------------------------------------------------
# Stage G: Identifiability — local Jacobian-rank + symbolic Gröbner basis
# ---------------------------------------------------------------------------

def run_identifiability(subject: str, session: str, *,
                        force: bool = False) -> dict:
    """Per-session identifiability report for the protocols actually used.

    Mirrors the standalone ``scripts/analyze_identifiability.py`` sweep
    but parameterises every helper with the session's *actual* inputs:

      * the data-driven CO2 stimulus extracted from this session's
        hypercapnia BOLD (so the Balloon conditioning numbers are
        protocol-faithful);
      * the per-region CBF means from this session's ASL fit (so the
        PET flow-correction and joint amyloid-CMRO2 helpers see the
        empirical CBF spread);
      * the actual single PLD (1.525 s) used by DLBS.

    Output: ``identifiability_report.json`` with rank, condition number,
    singular values, and explicit null-space directions for every
    (forward model × fit_names) combination examined.
    """
    out_dir = out_session_dir(subject, session)
    report_path = out_dir / "identifiability_report.json"

    if not force and report_path.exists():
        log.info("identifiability report already exists, skipping")
        return {"status": "skipped", "outputs": [str(report_path)]}

    payload: dict = {"subject": subject, "session": session}

    # ------- Balloon under the actual data-driven CO2 stimulus -------
    bold_path = cvr_dir(subject, session) / "mc.nii.gz"
    mask_path = cvr_dir(subject, session) / "brain_mask.nii.gz"
    bold_json = _hypercapnia_bold_json(subject, session)
    if bold_path.exists() and mask_path.exists() and bold_json.exists():
        from vpjax.hemodynamics.cvr import extract_global_stimulus
        from vpjax.identifiability import balloon_identifiability
        bold_4d, _, _ = _load_nii(bold_path)
        mask, _, _ = _load_nii(mask_path)
        tr = _read_tr(bold_json)
        stim, _ = extract_global_stimulus(bold_4d, mask, tr=tr, dt=0.1)

        balloon_rows = []
        for fit_names in (
            ("kappa", "gamma", "tau", "alpha", "E0"),
            ("kappa", "tau"),
            ("kappa",),
        ):
            for observers in (("bold",), ("bold", "asl"), ("bold", "asl", "vaso")):
                out = balloon_identifiability(
                    fit_names, stim, dt=0.1, tr=tr, observers=observers,
                )
                balloon_rows.append({
                    "fit_names": list(fit_names),
                    "observers": list(observers),
                    "rank": int(out["rank"]),
                    "n_params": int(out["n_params"]),
                    "is_identifiable": bool(out["is_identifiable"]),
                    "condition_number": float(out["condition_number"]),
                    "singular_values": [float(v) for v in np.asarray(out["singular_values"])],
                    "collinear_sets": out["collinear_sets"],
                })
        payload["balloon_data_driven"] = {
            "tr": tr, "dt": 0.1,
            "stimulus_peak_abs": float(np.max(np.abs(np.asarray(stim)))),
            "rows": balloon_rows,
        }
        log.info("identifiability: Balloon sweep done (%d rows)", len(balloon_rows))
    else:
        log.warning("identifiability: hypercapnia BOLD missing — Balloon sweep skipped")
        payload["balloon_data_driven"] = None

    # ------- PET / joint with empirical CBF spread -------
    cbf_summary_path = out_dir / "perfusion" / "regional_cbf.json"
    pet_summary_path = (
        out_dir / "pet" / "amyloid_18FAV45" / "regional_suvr.json"
    )
    if cbf_summary_path.exists():
        cbf_data = json.loads(cbf_summary_path.read_text())
        cbf_per_region = np.asarray(
            cbf_data["cbf_mean_per_region"], dtype=np.float32,
        )
        cbf_ref = float(cbf_data.get("cbf_ref_cerebellum") or cbf_per_region.mean())
        # Use a small representative subset (5 regions sampled across
        # the empirical CBF range) — keeps the Jacobian small and the
        # finding interpretable.
        if cbf_per_region.size > 5:
            idx = np.linspace(0, cbf_per_region.size - 1, 5).astype(int)
            cbf_subset = cbf_per_region[idx]
        else:
            cbf_subset = cbf_per_region

        from vpjax.identifiability import (
            pet_joint_identifiability,
            pet_static_suvr_identifiability,
        )
        import jax.numpy as jnp
        cbf_jnp = jnp.asarray(cbf_subset)
        pet_rows = []
        for fit_flow in (False, True):
            out = pet_static_suvr_identifiability(
                n_regions=int(cbf_subset.size),
                cbf_per_region=cbf_jnp,
                cbf_ref=cbf_ref,
                fit_flow=fit_flow,
            )
            pet_rows.append({
                "fit_flow": fit_flow,
                "rank": int(out["rank"]),
                "n_params": int(out["n_params"]),
                "is_identifiable": bool(out["is_identifiable"]),
                "condition_number": float(out["condition_number"]),
                "collinear_sets": out["collinear_sets"],
            })
        joint_rows = []
        for fit_alpha in (False, True):
            out = pet_joint_identifiability(
                n_regions=int(cbf_subset.size),
                cbf_per_region=cbf_jnp,
                cbf_ref=cbf_ref,
                fit_alpha_amy=fit_alpha,
                fit_beta_flow=True,
            )
            joint_rows.append({
                "fit_alpha_amy": fit_alpha,
                "rank": int(out["rank"]),
                "n_params": int(out["n_params"]),
                "is_identifiable": bool(out["is_identifiable"]),
                "condition_number": float(out["condition_number"]),
                "collinear_sets": out["collinear_sets"],
            })
        payload["pet_static_suvr"] = pet_rows
        payload["pet_joint"] = joint_rows
        payload["pet_cbf_subset"] = {
            "cbf_per_region": [float(v) for v in cbf_subset],
            "cbf_ref": cbf_ref,
            "n_regions_used": int(cbf_subset.size),
            "n_regions_total": int(cbf_per_region.size),
        }
        log.info("identifiability: PET sweeps done")
    else:
        log.warning("identifiability: regional_cbf.json missing — PET sweeps skipped")
        payload["pet_static_suvr"] = None
        payload["pet_joint"] = None

    # ------- ASL Buxton kinetic at the actual single PLD -------
    raw_json = _raw_asl_paths(subject, session)[1]
    if raw_json.exists():
        from vpjax.identifiability import asl_kinetic_identifiability
        meta = json.loads(raw_json.read_text())
        pld = float(meta["PostLabelingDelay"])
        asl_rows = [
            ("single PLD, fit (CBF)", asl_kinetic_identifiability(
                plds=[pld], fit_names=("CBF",))),
            ("single PLD, fit (CBF, delta)", asl_kinetic_identifiability(
                plds=[pld], fit_names=("CBF", "delta"))),
        ]
        payload["asl_kinetic"] = [
            {
                "label": label,
                "rank": int(out["rank"]),
                "n_params": int(out["n_params"]),
                "is_identifiable": bool(out["is_identifiable"]),
                "condition_number": float(out["condition_number"]),
                "collinear_sets": out["collinear_sets"],
            }
            for label, out in asl_rows
        ]
        log.info("identifiability: ASL sweep done at PLD=%.3fs", pld)
    else:
        payload["asl_kinetic"] = None

    # ------- Symbolic Gröbner-basis (informational only — no DLBS multi-echo) -------
    # DLBS is single-echo throughout, so the symbolic helper is included
    # only for cohorts that have multi-echo data (e.g. WAND).  We still
    # record a placeholder so downstream tooling can detect that the
    # symbolic block was considered but not applicable.
    payload["symbolic_multi_echo"] = {
        "note": "DLBS is single-echo BOLD/ASL/PET; symbolic invariants apply only "
                "to multi-echo protocols (e.g. WAND MEGRE). See "
                "vpjax.identifiability_symbolic.multi_echo_identifiability.",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2))
    log.info("saved %s", report_path)

    # Headline summary for the manifest.
    headline = {}
    if payload.get("balloon_data_driven"):
        rows = payload["balloon_data_driven"]["rows"]
        kt_bold = next(
            (r for r in rows
             if r["fit_names"] == ["kappa", "tau"] and r["observers"] == ["bold"]),
            None,
        )
        if kt_bold:
            headline["balloon_kappa_tau_bold_cond"] = kt_bold["condition_number"]
    if payload.get("pet_joint"):
        for r in payload["pet_joint"]:
            if r["fit_alpha_amy"]:
                headline["joint_alpha_amy_free_identifiable"] = r["is_identifiable"]
    if payload.get("asl_kinetic"):
        first = payload["asl_kinetic"][0]
        headline["asl_single_pld_cbf_identifiable"] = first["is_identifiable"]

    return {
        "status": "ok",
        "report": str(report_path),
        "headline": headline,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

STAGES = ("cvr", "perfusion", "metabolism", "pet", "pet_petsurfer",
          "asl_in_t1", "joint", "identifiability")


def process_session(subject: str, session: str, *,
                    stages: tuple[str, ...] = STAGES,
                    tracer: str = "amyloid",
                    force: bool = False) -> dict:
    log.info("=== %s %s ===", subject, session)
    out_dir = out_session_dir(subject, session)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    manifest: dict = {
        "subject": subject,
        "session": session,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": {},
    }

    if "cvr" in stages:
        manifest["stages"]["cvr"] = run_cvr(subject, session, force=force)
    if "perfusion" in stages:
        manifest["stages"]["perfusion"] = run_perfusion(subject, session, force=force)
    if "metabolism" in stages:
        manifest["stages"]["metabolism"] = run_metabolism(subject, session, force=force)
    if "pet" in stages:
        manifest["stages"]["pet"] = run_pet(subject, session, tracer=tracer, force=force)
    if "pet_petsurfer" in stages:
        manifest["stages"]["pet_petsurfer"] = run_pet_petsurfer(
            subject, session, tracer=tracer, force=force,
        )
    if "asl_in_t1" in stages:
        manifest["stages"]["asl_in_t1"] = run_asl_in_t1(subject, session, force=force)
    if "joint" in stages:
        manifest["stages"]["joint"] = run_joint_amyloid_cmro2(
            subject, session, tracer=tracer, force=force,
        )
    if "identifiability" in stages:
        manifest["stages"]["identifiability"] = run_identifiability(subject, session, force=force)

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("manifest -> %s", manifest_path)
    return manifest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, help="e.g. sub-1003")
    parser.add_argument(
        "--session",
        help="single session, e.g. ses-wave1; mutually exclusive with --all-sessions",
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="process all 3 DLBS sessions (wave1/wave2/wave3)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=list(STAGES),
        choices=STAGES,
        help="which stages to run (default: all)",
    )
    parser.add_argument(
        "--tracer",
        choices=("amyloid", "tau"),
        default="amyloid",
        help="PET tracer for the pet/pet_petsurfer/joint stages",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun even if outputs already exist",
    )
    args = parser.parse_args(argv)

    if args.all_sessions and args.session:
        parser.error("--session and --all-sessions are mutually exclusive")
    if not args.all_sessions and not args.session:
        parser.error("either --session or --all-sessions is required")

    sessions = ("ses-wave1", "ses-wave2", "ses-wave3") if args.all_sessions else (args.session,)

    for ses in sessions:
        try:
            process_session(
                args.subject, ses,
                stages=tuple(args.stages),
                tracer=args.tracer,
                force=args.force,
            )
        except FileNotFoundError as e:
            log.error("session %s skipped: %s", ses, e)
        except Exception:
            log.exception("session %s failed", ses)
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
