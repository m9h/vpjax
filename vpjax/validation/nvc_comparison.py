"""NVC model comparison on WAND task data.

Compares four neurovascular coupling forward models:
1. Linear convolution (canonical double-gamma HRF)
2. Balloon-Windkessel (Friston 2000 / Stephan 2007)
3. Riera multi-compartment (Riera 2006/2007)
4. Riera with pCASL/TRUST calibration constraints

Evaluated against measured BOLD from WAND sub-08033 category
localiser task (block design, 6s blocks, TR=2s).

Usage:
    uv run python -m vpjax.validation.nvc_comparison
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# 1. Linear HRF convolution
# ---------------------------------------------------------------------------

def _canonical_hrf(tr: float, duration: float = 32.0) -> np.ndarray:
    """Double-gamma canonical HRF (Glover 1999)."""
    from scipy.stats import gamma as gamma_dist
    t = np.arange(0, duration, tr)
    # Parameters from SPM/FSL
    a1, b1 = 6.0, 1.0   # peak
    a2, b2 = 16.0, 1.0  # undershoot
    c = 1.0 / 6.0
    hrf = gamma_dist.pdf(t, a1, scale=b1) - c * gamma_dist.pdf(t, a2, scale=b2)
    return hrf / np.max(hrf)


def predict_bold_linear(stimulus: np.ndarray, tr: float = 2.0) -> np.ndarray:
    """Predict BOLD via linear convolution with canonical HRF.

    Parameters
    ----------
    stimulus : neural stimulus time course (one value per TR)
    tr : repetition time (s)

    Returns
    -------
    Predicted BOLD (same length as stimulus)
    """
    hrf = _canonical_hrf(tr)
    bold = np.convolve(stimulus, hrf, mode="full")[:len(stimulus)]
    # Normalize
    if np.max(np.abs(bold)) > 0:
        bold = bold / np.max(np.abs(bold)) * 0.03  # ~3% signal change
    return bold


# ---------------------------------------------------------------------------
# 2. Balloon-Windkessel
# ---------------------------------------------------------------------------

def predict_bold_balloon(stimulus: np.ndarray, tr: float = 2.0) -> np.ndarray:
    """Predict BOLD via Balloon-Windkessel model.

    Upsamples stimulus to fine time grid, integrates ODE, downsamples.
    """
    from vpjax._types import BalloonParams
    from vpjax.hemodynamics.balloon import solve_balloon
    from vpjax.hemodynamics.bold import observe_bold

    # Upsample stimulus to fine grid
    dt = 0.05
    samples_per_tr = int(tr / dt)
    stim_fine = np.repeat(stimulus, samples_per_tr)
    stim_jax = jnp.array(stim_fine, dtype=jnp.float32)

    params = BalloonParams()
    _, traj = solve_balloon(params, stim_jax, dt=dt)
    bold_fine = np.array(observe_bold(traj))

    # Downsample back to TR
    bold = bold_fine[::samples_per_tr][:len(stimulus)]
    return bold


# ---------------------------------------------------------------------------
# 3. Riera multi-compartment
# ---------------------------------------------------------------------------

def predict_bold_riera(stimulus: np.ndarray, tr: float = 2.0) -> np.ndarray:
    """Predict BOLD via Riera multi-compartment NVC model.

    Uses the full arteriolar/capillary/venous pathway with NO + adenosine
    vasodilatory signals and metabolic coupling.
    """
    from vpjax.hemodynamics.riera import RieraNVC, RieraParams, RieraState, riera_to_balloon
    from vpjax.hemodynamics.bold import observe_bold
    from vpjax._types import BalloonState

    dt = 0.05
    samples_per_tr = int(tr / dt)
    stim_fine = np.repeat(stimulus, samples_per_tr)

    model = RieraNVC(params=RieraParams())
    y = RieraState.steady_state()

    bold_trace = []
    for i in range(len(stim_fine)):
        u = jnp.array(stim_fine[i])
        dy = model(jnp.array(0.0), y, u)
        y = jax.tree.map(lambda yi, dyi: yi + dt * dyi, y, dy)

        if i % samples_per_tr == 0:
            v, q = riera_to_balloon(y)
            state = BalloonState(s=jnp.array(0.0), f=y.f_a, v=v, q=q)
            bold_trace.append(float(observe_bold(state)))

    bold = np.array(bold_trace[:len(stimulus)])
    return bold


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wand_task(
    bold_path: str,
    events_path: str,
) -> tuple[np.ndarray, list[dict], float]:
    """Load WAND task BOLD and events.

    Returns
    -------
    bold_ts : global mean BOLD, shape (T,)
    events : list of dicts with onset, duration, trial_type
    tr : repetition time (s)
    """
    import nibabel as nib

    img = nib.load(bold_path)
    data = img.get_fdata()

    # Brain mask
    brain_mask = np.mean(data, axis=-1) > np.percentile(np.mean(data, axis=-1), 10)

    # Top 5% most variable voxels (task-responsive)
    temporal_std = np.std(data, axis=-1)
    temporal_std[~brain_mask] = 0
    threshold = np.percentile(temporal_std[brain_mask], 95)
    active_mask = temporal_std >= threshold
    bold_ts = np.mean(data[active_mask], axis=0).astype(np.float32)

    # Minimal preprocessing: detrend + high-pass filter
    from scipy.signal import detrend
    bold_ts = detrend(bold_ts).astype(np.float32)
    # High-pass: remove frequencies below 1/128 Hz (standard FSL cutoff)
    n_vols = len(bold_ts)
    cutoff_vols = int(128.0 / tr) if tr > 0 else 64
    if n_vols > cutoff_vols:
        # DCT-based high-pass (remove slow drift)
        from scipy.fft import dct, idct
        coeffs = dct(bold_ts, type=2)
        n_remove = max(1, n_vols // cutoff_vols)
        coeffs[:n_remove] = 0
        bold_ts = idct(coeffs, type=2).astype(np.float32) / (2 * n_vols)

    # Get TR
    json_path = bold_path.replace(".nii.gz", ".json")
    tr = 2.0
    try:
        with open(json_path) as f:
            meta = json.load(f)
            tr = meta.get("RepetitionTime", 2.0)
    except FileNotFoundError:
        pass

    # Parse events
    events = []
    with open(events_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            events.append({
                "onset": float(row["onset"]),
                "duration": float(row["duration"]),
                "trial_type": row["trial_type"].strip(),
            })

    return bold_ts, events, float(tr)


def build_stimulus(
    events: list[dict],
    n_vols: int,
    tr: float,
    condition: str,
) -> np.ndarray:
    """Build a binary stimulus regressor for one condition.

    Parameters
    ----------
    events : parsed events list
    n_vols : number of BOLD volumes
    tr : repetition time
    condition : trial type to extract

    Returns
    -------
    stimulus : binary regressor, shape (n_vols,)
    """
    stim = np.zeros(n_vols)
    for ev in events:
        if ev["trial_type"] == condition:
            start_vol = int(ev["onset"] / tr)
            end_vol = int((ev["onset"] + ev["duration"]) / tr)
            start_vol = max(0, min(start_vol, n_vols - 1))
            end_vol = max(0, min(end_vol, n_vols))
            stim[start_vol:end_vol] = 1.0
    return stim


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_nvc_comparison(
    bold_path: str,
    events_path: str,
) -> list[dict]:
    """Run NVC model comparison on WAND task data.

    For each model, builds predicted BOLD from ALL conditions combined
    (full design matrix approach), then correlates with measured BOLD.
    Also reports per-condition variance explained via partial correlation.

    Returns list of {model, condition, r_value, p_value, mse}.
    """
    bold_ts, events, tr = load_wand_task(bold_path, events_path)

    # Demean and normalize BOLD
    bold_ts = bold_ts - np.mean(bold_ts)
    bold_norm = bold_ts / (np.std(bold_ts) + 1e-10)

    # Get unique conditions (excluding baseline)
    conditions = sorted(set(ev["trial_type"] for ev in events if ev["trial_type"] != "baseline"))

    models = {
        "linear": predict_bold_linear,
        "balloon": predict_bold_balloon,
        "riera": predict_bold_riera,
    }

    results = []

    for model_name, predict_fn in models.items():
        # Build combined predicted BOLD from all conditions
        combined_pred = np.zeros(len(bold_ts))
        per_condition_preds = {}

        for condition in conditions:
            stim = build_stimulus(events, len(bold_ts), tr, condition)
            if np.sum(stim) < 1:
                continue
            pred = predict_fn(stim, tr=tr)
            combined_pred += pred
            per_condition_preds[condition] = pred

        # Overall model fit (all conditions combined)
        combined_pred -= np.mean(combined_pred)
        std_comb = np.std(combined_pred)
        if std_comb > 1e-10:
            combined_norm = combined_pred / std_comb
        else:
            combined_norm = combined_pred

        r_overall, p_overall = pearsonr(bold_norm, combined_norm)
        mse_overall = float(np.mean((bold_norm - combined_norm) ** 2))

        results.append({
            "model": model_name,
            "condition": "ALL_COMBINED",
            "r_value": float(r_overall),
            "p_value": float(p_overall),
            "mse": mse_overall,
        })

        # Per-condition contribution
        for condition, pred in per_condition_preds.items():
            pred_norm = pred - np.mean(pred)
            sp = np.std(pred_norm)
            if sp > 1e-10:
                pred_norm = pred_norm / sp
            r, p = pearsonr(bold_norm, pred_norm)
            results.append({
                "model": model_name,
                "condition": condition,
                "r_value": float(r),
                "p_value": float(p),
                "mse": float(np.mean((bold_norm - pred_norm) ** 2)),
            })

    return results


def print_comparison(results: list[dict]):
    """Print formatted comparison table."""
    print(f"\n{'Model':<12} {'Condition':<15} {'r':>8} {'p':>12} {'MSE':>8}")
    print("-" * 58)
    for r in sorted(results, key=lambda x: (x["condition"], x["model"])):
        print(f"{r['model']:<12} {r['condition']:<15} {r['r_value']:>+8.4f} "
              f"{r['p_value']:>12.2e} {r['mse']:>8.4f}")

    # Summary per model
    print(f"\n{'Model':<12} {'Mean r':>8} {'Median r':>10}")
    print("-" * 32)
    for model in ["linear", "balloon", "riera"]:
        rs = [r["r_value"] for r in results if r["model"] == model]
        if rs:
            print(f"{model:<12} {np.mean(rs):>+8.4f} {np.median(rs):>+10.4f}")


def main():
    import os
    bold = os.path.expanduser("~/dev/wand/sub-08033/ses-03/func/sub-08033_ses-03_task-categorylocaliser_run-1_bold.nii.gz")
    events = os.path.expanduser("~/dev/wand/sub-08033/ses-03/func/sub-08033_ses-03_task-categorylocaliser_run-1_events.tsv")

    print("Running NVC model comparison on WAND sub-08033 category localiser...")
    results = run_nvc_comparison(bold, events)
    print_comparison(results)

    # Save
    out_path = os.path.expanduser("~/dev/wand/derivatives/vpjax_nvc_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
