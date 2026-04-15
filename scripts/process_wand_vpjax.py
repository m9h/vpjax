#!/usr/bin/env python3
"""Process a single WAND subject through the vpjax multi-modal pipeline.

Stages (each skipped if output already exists):
  A. Perfusion: TRUST T₂ → global OEF, pCASL → CBF, Fick → CMRO₂
  B. qBOLD: multi-echo GRE (MEGRE) → per-voxel OEF, DBV, R₂'
  C. Iron-myelin: R₂* + QSM → iron/myelin decomposition
  D. Hemodynamic inversion: task BOLD + events → Balloon params

Usage:
    python process_wand_vpjax.py --subject sub-08033
    python process_wand_vpjax.py --subject sub-08033 --stage perfusion
    python process_wand_vpjax.py --all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wand-vpjax")

WAND_DIR = Path("/data/raw/wand")
DERIV_DIR = WAND_DIR / "derivatives"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_nii(path: Path) -> tuple[np.ndarray, np.ndarray, object]:
    """Load NIfTI, return (data, affine, header)."""
    import nibabel as nib
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine, img.header


def _save_nii(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Save array as NIfTI."""
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), str(path))
    log.info("Saved %s", path.name)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _find_file(pattern: str, root: Path) -> Path | None:
    """Find first file matching glob pattern under root."""
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Stage A: Perfusion (TRUST + ASL → CMRO₂)
# ---------------------------------------------------------------------------

def stage_perfusion(sub: str) -> None:
    """TRUST OEF + pCASL CBF → global CMRO₂."""
    from vpjax.perfusion.trust import TRUSTParams, trust_oef, t2_to_svo2

    out_dir = DERIV_DIR / "vpjax" / sub / "perfusion"
    summary_path = out_dir / "perfusion_summary.json"
    if summary_path.exists():
        log.info("Stage A (perfusion): already done, skipping")
        return

    perf_dir = WAND_DIR / sub / "ses-03" / "perf"
    if not perf_dir.exists():
        log.warning("Stage A: no ses-03/perf/ for %s, skipping", sub)
        return

    # --- TRUST: get global OEF ---
    # Check for existing processed TRUST results first
    existing_trust = DERIV_DIR / "perfusion" / sub / "trust_results.json"
    t2_venous = None

    if existing_trust.exists():
        trust_res = _read_json(existing_trust)
        # Use pre-computed SvO₂ or T₂ if available
        if "T2_blood_s" in trust_res:
            t2_venous = trust_res["T2_blood_s"]
        elif "T2_blood_ms" in trust_res:
            t2_venous = trust_res["T2_blood_ms"] / 1000.0
        elif "SvO2" in trust_res:
            # Back-compute T₂ from SvO₂
            from vpjax.perfusion.trust import svo2_to_t2
            t2_venous = float(svo2_to_t2(jnp.array(trust_res["SvO2"])))
        log.info("Using existing TRUST results: T₂=%.1f ms", (t2_venous or 0) * 1000)

    if t2_venous is None:
        # No pre-processed TRUST: use population average at 3T
        # Healthy adult venous T₂ ≈ 60-80ms (Lu & Ge 2008)
        t2_venous = 0.065
        log.info("No processed TRUST data — using population default T₂=65 ms")

    t2_venous = float(np.clip(t2_venous, 0.020, 0.200))

    params = TRUSTParams()
    svo2 = float(t2_to_svo2(jnp.array(t2_venous), params))
    oef_global = float(trust_oef(jnp.array(t2_venous), params))
    log.info("TRUST: T₂=%.1f ms, SvO₂=%.3f, OEF=%.3f", t2_venous * 1000, svo2, oef_global)

    # --- ASL: load quantified CBF ---
    # Check for existing oxford_asl output first (pre-processed CBF)
    cbf_median = None
    cbf_data = None
    cbf_affine = None

    oxford_cbf = DERIV_DIR / "perfusion" / sub / "oxford_asl" / "native_space" / "perfusion.nii.gz"
    if oxford_cbf.exists():
        cbf_data, cbf_affine, _ = _load_nii(oxford_cbf)
        brain_mask = cbf_data > 5.0
        if np.any(brain_mask):
            cbf_median = float(np.median(cbf_data[brain_mask]))
        log.info("ASL (oxford_asl): median CBF = %.1f mL/100g/min", cbf_median)
    else:
        # Raw pCASL data exists but needs FSL oxford_asl processing first
        log.info("No quantified CBF (oxford_asl not run) — using population default")
        cbf_median = 50.0  # healthy adult default (mL/100g/min)

    if cbf_data is not None and cbf_affine is not None:
        _save_nii(cbf_data, cbf_affine, out_dir / "CBF_map.nii.gz")

    # --- CMRO₂ ---
    cao2 = 8.3  # µmol O₂/mL blood (standard)
    cmro2_global = cbf_median * oef_global * cao2
    log.info("CMRO₂ = %.1f µmol/100g/min", cmro2_global)

    # Save summary
    summary = {
        "subject": sub,
        "T2_venous_s": t2_venous,
        "SvO2": svo2,
        "OEF_global": oef_global,
        "CBF_median_mL_100g_min": cbf_median,
        "CaO2_umol_mL": cao2,
        "CMRO2_global_umol_100g_min": cmro2_global,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Stage A complete")


# ---------------------------------------------------------------------------
# Stage B: qBOLD (multi-echo GRE → OEF/DBV per voxel)
# ---------------------------------------------------------------------------

def stage_qbold(sub: str) -> None:
    """Fit qBOLD model to multi-echo MEGRE → OEF, DBV, R₂' maps."""
    from vpjax.qbold.oef_mapping import fit_oef_volume
    from vpjax.qbold.signal_model import QBOLDParams, compute_r2prime
    from vpjax.qsm.r2star_fitting import fit_r2star_loglinear

    # Find MEGRE data (typically in ses-06 for WAND)
    megre_echo1 = None
    megre_ses = None
    for ses in ["ses-06", "ses-05", "ses-04", "ses-02"]:
        candidate = _find_file(f"{ses}/anat/*echo-01*mag*MEGRE*", WAND_DIR / sub)
        if candidate is not None:
            megre_echo1 = candidate
            megre_ses = ses
            break

    if megre_echo1 is None:
        log.info("Stage B (qBOLD): no MEGRE data for %s, skipping", sub)
        return

    out_dir = DERIV_DIR / "vpjax" / sub / megre_ses
    if (out_dir / "OEF_map.nii.gz").exists():
        log.info("Stage B (qBOLD): already done, skipping")
        return

    # Load all echoes
    anat_dir = megre_echo1.parent
    echo_files = sorted(anat_dir.glob(f"*echo-*_part-mag_MEGRE.nii.gz"))
    if len(echo_files) < 3:
        log.warning("Stage B: only %d MEGRE echoes, need ≥3", len(echo_files))
        return

    # Get echo times from JSON sidecars
    echo_times = []
    for ef in echo_files:
        jf = ef.with_suffix("").with_suffix(".json")
        if jf.exists():
            meta = _read_json(jf)
            echo_times.append(meta.get("EchoTime", 0.0))
        else:
            echo_times.append(0.0)

    te = np.array(echo_times, dtype=np.float32)
    if np.all(te == 0):
        # Fallback: assume standard 6-echo MEGRE
        te = np.array([0.0045, 0.009, 0.0135, 0.018, 0.0225, 0.027], dtype=np.float32)
        te = te[:len(echo_files)]

    log.info("Loading %d MEGRE echoes (TE: %s ms)", len(echo_files),
             ", ".join(f"{t*1000:.1f}" for t in te))

    data_list = []
    affine = None
    for ef in echo_files:
        d, aff, _ = _load_nii(ef)
        data_list.append(d)
        if affine is None:
            affine = aff

    vol_shape = data_list[0].shape
    n_echoes = len(data_list)
    data_4d = np.stack(data_list, axis=-1)  # (*spatial, n_echoes)

    # Create brain mask from first echo
    mask = data_list[0] > np.percentile(data_list[0], 15)
    n_voxels = int(np.sum(mask))
    log.info("Brain mask: %d voxels of %d total", n_voxels, np.prod(vol_shape))

    # Flatten masked voxels: (N, T)
    data_masked = data_4d[mask]  # (N, n_echoes)

    # Step 1: R₂* from log-linear fit (fast)
    log.info("Fitting R₂* (log-linear)...")
    te_jax = jnp.array(te)
    r2star_flat, s0_flat = fit_r2star_loglinear(jnp.array(data_masked), te_jax)

    # Step 2: qBOLD OEF/DBV fit (gradient-based, slower)
    log.info("Fitting qBOLD (OEF/DBV) for %d voxels...", n_voxels)
    qbold_params = QBOLDParams()
    result = fit_oef_volume(jnp.array(data_masked), te_jax, qbold_params)

    # Reconstruct volumes
    def _to_vol(flat_arr: jnp.ndarray) -> np.ndarray:
        vol = np.zeros(vol_shape, dtype=np.float32)
        vol[mask] = np.asarray(flat_arr)
        return vol

    r2star_vol = _to_vol(r2star_flat)
    s0_vol = _to_vol(s0_flat)
    oef_vol = _to_vol(result["oef"])
    dbv_vol = _to_vol(result["dbv"])
    r2prime_vol = _to_vol(result["r2prime"])

    # Save maps
    for name, data in [
        ("R2star_map", r2star_vol),
        ("S0_map", s0_vol),
        ("OEF_map", oef_vol),
        ("DBV_map", dbv_vol),
        ("R2prime_map", r2prime_vol),
    ]:
        _save_nii(data, affine, out_dir / f"{name}.nii.gz")

    log.info("Stage B complete")


# ---------------------------------------------------------------------------
# Stage C: Iron-myelin decomposition
# ---------------------------------------------------------------------------

def stage_iron_myelin(sub: str) -> None:
    """Decompose R₂* + QSM → iron and myelin maps."""
    from vpjax.layers.iron_myelin import decompose_r2star_qsm

    # Find R₂* map from Stage B
    r2star_path = None
    megre_ses = None
    for ses in ["ses-06", "ses-05", "ses-04"]:
        candidate = DERIV_DIR / "vpjax" / sub / ses / "R2star_map.nii.gz"
        if candidate.exists():
            r2star_path = candidate
            megre_ses = ses
            break

    if r2star_path is None:
        log.info("Stage C (iron-myelin): no R₂* map for %s, skipping", sub)
        return

    out_dir = r2star_path.parent
    if (out_dir / "iron_map.nii.gz").exists():
        log.info("Stage C (iron-myelin): already done, skipping")
        return

    # Find QSM chi map
    chi_path = None
    for qsm_dir in sorted(DERIV_DIR.glob("qsmxt*")):
        # Search for Chi maps under this QSM derivatives directory
        for chi_candidate in qsm_dir.rglob("Chi*.nii*"):
            if sub in str(chi_candidate):
                chi_path = chi_candidate
                break
        if chi_path is not None:
            break

    if chi_path is None:
        log.info("Stage C: no QSM chi map for %s, skipping", sub)
        return

    log.info("Loading R₂* from %s", r2star_path.name)
    log.info("Loading QSM from %s", chi_path.name)

    r2star_data, affine, _ = _load_nii(r2star_path)
    chi_data, _, _ = _load_nii(chi_path)

    # Resample chi to R₂* space if shapes differ
    if chi_data.shape != r2star_data.shape:
        log.warning(
            "Shape mismatch: R₂* %s vs QSM %s — skipping iron-myelin",
            r2star_data.shape, chi_data.shape,
        )
        return

    iron, myelin = decompose_r2star_qsm(
        jnp.array(r2star_data), jnp.array(chi_data),
    )

    _save_nii(np.asarray(iron), affine, out_dir / "iron_map.nii.gz")
    _save_nii(np.asarray(myelin), affine, out_dir / "myelin_map.nii.gz")
    log.info("Stage C complete")


# ---------------------------------------------------------------------------
# Stage D: Hemodynamic inversion (BOLD → Balloon params)
# ---------------------------------------------------------------------------

def stage_hemodynamic_inversion(sub: str) -> None:
    """Fit Balloon-Windkessel params to task BOLD data."""
    from vpjax.hemodynamics.inversion import fit_balloon_bold

    out_dir = DERIV_DIR / "vpjax" / sub / "hemodynamics"
    summary_path = out_dir / "balloon_params.json"
    if summary_path.exists():
        log.info("Stage D (hemodynamic inversion): already done, skipping")
        return

    func_dir = WAND_DIR / sub / "ses-03" / "func"
    if not func_dir.exists():
        log.info("Stage D: no ses-03/func/ for %s, skipping", sub)
        return

    # Find task BOLD with events (category localiser preferred)
    bold_nii = None
    events_tsv = None
    for task in ["categorylocaliser", "reversallearning"]:
        candidate = _find_file(f"*task-{task}*run-1*bold.nii.gz", func_dir)
        if candidate is None:
            candidate = _find_file(f"*task-{task}*bold.nii.gz", func_dir)
        if candidate is not None:
            evt = _find_file(f"*task-{task}*events.tsv", func_dir)
            if evt is not None:
                bold_nii = candidate
                events_tsv = evt
                break

    if bold_nii is None or events_tsv is None:
        log.info("Stage D: no task BOLD with events for %s, skipping", sub)
        return

    # Load BOLD metadata
    bold_json = bold_nii.with_suffix("").with_suffix(".json")
    meta = _read_json(bold_json) if bold_json.exists() else {}
    tr = meta.get("RepetitionTime", 2.0)

    # Load BOLD data → global mean time series
    bold_data, _, _ = _load_nii(bold_nii)
    if bold_data.ndim == 4:
        brain_mask = np.mean(bold_data, axis=-1) > np.percentile(
            np.mean(bold_data, axis=-1), 20
        )
        bold_ts = np.mean(bold_data[brain_mask], axis=0)
    else:
        log.warning("Stage D: unexpected BOLD shape %s", bold_data.shape)
        return

    # Convert to fractional change
    bold_mean = np.mean(bold_ts)
    bold_frac = (bold_ts - bold_mean) / bold_mean

    n_vols = len(bold_frac)
    log.info("BOLD: %d volumes, TR=%.2fs, task=%s", n_vols, tr, bold_nii.stem)

    # Load events → construct stimulus at ODE resolution
    dt = 0.5  # 500ms ODE timestep (matches TR/4, fast enough for HRF)
    duration = n_vols * tr
    n_samples = int(duration / dt)
    stimulus = np.zeros(n_samples, dtype=np.float32)

    import csv
    with open(events_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            trial_type = row.get("trial_type", "")
            if trial_type in ("baseline", "n/a", ""):
                continue
            onset = float(row["onset"])
            dur = float(row.get("duration", 1.0))
            i_start = int(onset / dt)
            i_end = int((onset + dur) / dt)
            i_start = max(0, min(i_start, n_samples - 1))
            i_end = max(0, min(i_end, n_samples))
            stimulus[i_start:i_end] = 1.0

    n_active = int(np.sum(stimulus > 0))
    log.info("Stimulus: %d active samples (%.1fs) of %d total",
             n_active, n_active * dt, n_samples)

    if n_active == 0:
        log.warning("Stage D: no active stimulus events, skipping")
        return

    # Fit Balloon model
    log.info("Fitting Balloon-Windkessel to global BOLD...")
    result = fit_balloon_bold(
        jnp.array(bold_frac),
        jnp.array(stimulus),
        tr=tr,
        dt=dt,
        n_steps=200,
    )

    params_out = {
        "subject": sub,
        "task": bold_nii.stem,
        "TR_s": tr,
        "n_volumes": n_vols,
        "kappa": float(result["kappa"]),
        "gamma": float(result["gamma"]),
        "tau": float(result["tau"]),
        "alpha": float(result["alpha"]),
        "E0": float(result["E0"]),
        "loss": float(result["loss"]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(params_out, f, indent=2)
    log.info("Stage D complete: loss=%.2e, E0=%.3f, kappa=%.3f",
             params_out["loss"], params_out["E0"], params_out["kappa"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGES = {
    "perfusion": stage_perfusion,
    "qbold": stage_qbold,
    "iron_myelin": stage_iron_myelin,
    "hemodynamics": stage_hemodynamic_inversion,
}


def process_subject(sub: str, stages: list[str] | None = None) -> None:
    """Run all (or selected) stages for one subject."""
    if stages is None:
        stages = list(STAGES.keys())

    log.info("=" * 60)
    log.info("Processing %s  [stages: %s]", sub, ", ".join(stages))
    log.info("=" * 60)

    for stage_name in stages:
        fn = STAGES[stage_name]
        try:
            fn(sub)
        except Exception:
            log.exception("Stage %s FAILED for %s", stage_name, sub)


def get_all_subjects() -> list[str]:
    """Return sorted list of all WAND subject IDs."""
    return sorted(
        p.name for p in WAND_DIR.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="vpjax WAND processing")
    parser.add_argument("--subject", type=str, help="Single subject ID (e.g. sub-08033)")
    parser.add_argument("--all", action="store_true", help="Process all subjects")
    parser.add_argument("--stage", type=str, choices=list(STAGES.keys()),
                        help="Run only this stage")
    parser.add_argument("--list", action="store_true", help="List subjects and exit")
    args = parser.parse_args()

    if args.list:
        for s in get_all_subjects():
            print(s)
        return

    stages = [args.stage] if args.stage else None

    if args.all:
        for sub in get_all_subjects():
            process_subject(sub, stages)
    elif args.subject:
        sub = args.subject
        if not sub.startswith("sub-"):
            sub = f"sub-{sub}"
        process_subject(sub, stages)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
