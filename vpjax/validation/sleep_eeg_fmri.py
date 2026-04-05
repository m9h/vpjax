"""Validation pipeline: vpjax sleep models vs OpenNeuro ds003768.

Loads simultaneous EEG-fMRI sleep data, extracts spectral features
per sleep stage, and compares measured BOLD spectra against vpjax
forward model predictions.

Pipeline:
    1. Load sleep staging (TSV) → per-epoch stage labels
    2. Load BOLD (NIfTI) → global mean time series
    3. Load EEG (BrainVision) → band power time courses
    4. Compute BOLD power spectrum per sleep stage
    5. Run vpjax forward model → predicted BOLD spectrum per stage
    6. Compare predicted vs measured

Requires: mne, nibabel, scipy (install with `pip install vpjax[validation]`)
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# 1. Sleep staging
# ---------------------------------------------------------------------------

def load_sleep_stages(
    tsv_path: str | Path,
    subject: int | None = None,
) -> dict[str, list[tuple[float, str]]]:
    """Load sleep staging from ds003768 TSV file.

    Parameters
    ----------
    tsv_path : path to sub-XX-sleep-stage.tsv
    subject : subject number (for filtering, optional)

    Returns
    -------
    Dict mapping session name → list of (epoch_start_sec, stage_label).
    Stage labels normalized to: 'W', '1', '2', '3', 'R'.
    """
    stages: dict[str, list[tuple[float, str]]] = {}

    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            session = row.get("session", "").strip()
            epoch_time = float(row.get("epoch_start_time_sec", 0))
            raw_stage = row.get("30-sec_epoch_sleep_stage", "").strip()

            # Normalize stage label
            stage = _normalize_stage(raw_stage)
            if stage is None:
                continue

            if session not in stages:
                stages[session] = []
            stages[session].append((epoch_time, stage))

    return stages


def _normalize_stage(raw: str) -> str | None:
    """Normalize sleep stage label to standard set."""
    raw = raw.strip()
    if raw in ("W", "W (uncertain)"):
        return "W"
    elif raw in ("1", "1 (uncertain)", "1 (unscorable)"):
        return "1"
    elif raw in ("2", "2 (uncertain)", "2 (unscorable)"):
        return "2"
    elif raw in ("3", "3 (uncertain)", "3 (unscorable)"):
        return "3"
    elif raw in ("R", "R (uncertain)"):
        return "R"
    elif raw in ("unscorable", "2 or 3 (unscorable)"):
        return None
    return None


# ---------------------------------------------------------------------------
# 2. BOLD loading
# ---------------------------------------------------------------------------

def load_bold_global(
    nifti_path: str | Path,
) -> tuple[np.ndarray, float]:
    """Load BOLD fMRI and return global mean time series.

    Parameters
    ----------
    nifti_path : path to *_bold.nii.gz

    Returns
    -------
    ts : global mean BOLD signal, shape (T,)
    tr : repetition time (s)
    """
    import nibabel as nib
    import json

    img = nib.load(str(nifti_path))
    data = img.get_fdata()  # (x, y, z, t)

    # Global mean (exclude zero voxels)
    mask = np.mean(data, axis=-1) > np.percentile(np.mean(data, axis=-1), 10)
    ts = np.mean(data[mask], axis=0)

    # Get TR from sidecar JSON
    json_path = str(nifti_path).replace(".nii.gz", ".json").replace(".nii", ".json")
    tr = 2.0  # default
    try:
        with open(json_path) as f:
            meta = json.load(f)
            tr = meta.get("RepetitionTime", 2.0)
    except (FileNotFoundError, json.JSONDecodeError):
        # Try to get from NIfTI header
        tr = float(img.header.get_zooms()[-1])
        if tr > 100:  # probably in ms
            tr /= 1000.0

    return ts.astype(np.float32), float(tr)


# ---------------------------------------------------------------------------
# 3. EEG loading
# ---------------------------------------------------------------------------

def load_eeg_bandpower(
    vhdr_path: str | Path,
    epoch_duration: float = 30.0,
) -> dict[str, np.ndarray]:
    """Load EEG and compute band power in standard sleep bands.

    Parameters
    ----------
    vhdr_path : path to BrainVision .vhdr file
    epoch_duration : epoch length for power computation (s)

    Returns
    -------
    Dict with keys 'delta', 'theta', 'alpha', 'sigma', 'beta'.
    Each value is a 1-D array of power per epoch.
    """
    import mne

    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

    # Pick EEG channels only (exclude ECG, EOG)
    raw.pick("eeg")

    # Bandpass filter
    raw.filter(0.5, 45.0, verbose=False)

    # Compute power in epochs
    sfreq = raw.info["sfreq"]
    data = raw.get_data()  # (n_channels, n_samples)
    n_samples = data.shape[1]
    epoch_samples = int(epoch_duration * sfreq)
    n_epochs = n_samples // epoch_samples

    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "sigma": (11.0, 16.0),
        "beta": (16.0, 30.0),
    }

    from scipy.signal import welch

    result = {band: [] for band in bands}

    for i in range(n_epochs):
        start = i * epoch_samples
        end = start + epoch_samples
        epoch_data = data[:, start:end]

        # Mean across channels, then Welch PSD
        mean_signal = np.mean(epoch_data, axis=0)
        freqs, psd = welch(mean_signal, fs=sfreq, nperseg=min(int(4 * sfreq), epoch_samples))

        for band, (flo, fhi) in bands.items():
            mask = (freqs >= flo) & (freqs <= fhi)
            result[band].append(float(np.mean(psd[mask])) if np.any(mask) else 0.0)

    return {band: np.array(vals) for band, vals in result.items()}


# ---------------------------------------------------------------------------
# 4. BOLD spectrum per sleep stage
# ---------------------------------------------------------------------------

def bold_spectrum_by_stage(
    bold_ts: np.ndarray,
    tr: float,
    stage_epochs: list[tuple[float, str]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute BOLD power spectrum for each sleep stage.

    Segments the BOLD time series by sleep stage and computes
    the power spectral density for each.

    Parameters
    ----------
    bold_ts : global mean BOLD, shape (T,)
    tr : repetition time (s)
    stage_epochs : list of (epoch_start_sec, stage_label)

    Returns
    -------
    Dict mapping stage → (frequencies, power_spectral_density)
    """
    from scipy.signal import welch

    n_vols = len(bold_ts)

    # Group BOLD volumes by stage
    stage_data: dict[str, list[float]] = {}
    for epoch_time, stage in stage_epochs:
        # Which BOLD volumes fall in this 30s epoch?
        vol_start = int(epoch_time / tr)
        vol_end = int((epoch_time + 30.0) / tr)
        vol_start = max(0, min(vol_start, n_vols - 1))
        vol_end = max(0, min(vol_end, n_vols))
        if vol_end <= vol_start:
            continue
        if stage not in stage_data:
            stage_data[stage] = []
        stage_data[stage].extend(bold_ts[vol_start:vol_end].tolist())

    # Compute PSD per stage
    result = {}
    for stage, data in stage_data.items():
        arr = np.array(data)
        if len(arr) < 10:
            continue
        nperseg = min(len(arr), max(16, len(arr) // 4))
        freqs, psd = welch(arr, fs=1.0 / tr, nperseg=nperseg)
        result[stage] = (freqs, psd)

    return result


# ---------------------------------------------------------------------------
# 5. vpjax forward model prediction
# ---------------------------------------------------------------------------

def predict_bold_spectrum_for_stage(
    stage_label: str,
    freqs: np.ndarray,
    tr: float,
) -> Float[Array, "F"]:
    """Predict BOLD power spectrum for a given sleep stage using vpjax.

    Uses the Balloon model with stage-dependent NVC parameters to
    predict the hemodynamic transfer function, then applies it to
    a white-noise neural input to get the expected BOLD spectrum.

    Parameters
    ----------
    stage_label : 'W', '1', '2', '3', or 'R'
    freqs : frequency axis (Hz)
    tr : repetition time (s)

    Returns
    -------
    Predicted BOLD power spectrum at the given frequencies
    """
    from vpjax.sleep.nvc_state import (
        WAKE, N1, N2, N3, REM,
        balloon_params_for_stage, nvc_gain_for_stage,
    )
    from vpjax.hemodynamics.balloon import BalloonWindkessel
    from vpjax._types import BalloonParams, BalloonState

    # Map label to stage constant
    stage_map = {"W": WAKE, "1": N1, "2": N2, "3": N3, "R": REM}
    stage = stage_map.get(stage_label, WAKE)

    params = balloon_params_for_stage(stage)
    gain = nvc_gain_for_stage(stage)

    # Hemodynamic transfer function (Balloon model frequency response)
    # Linearized around steady state: H(f) = BOLD_response / neural_input
    # Approximate via impulse response → FFT
    model = BalloonWindkessel(params=params)
    dt = 0.1  # 100ms resolution
    n_pts = 600  # 60s impulse response
    t = np.arange(n_pts) * dt

    # Impulse (scaled by NVC gain)
    impulse = np.zeros(n_pts)
    impulse[1:4] = gain  # brief pulse

    # Simulate impulse response
    from vpjax.hemodynamics.balloon import solve_balloon
    from vpjax.hemodynamics.bold import observe_bold

    stim = jnp.array(impulse, dtype=jnp.float32)
    _, traj = solve_balloon(params, stim, dt=dt)
    bold_ir = np.array(observe_bold(traj))

    # FFT of impulse response → transfer function magnitude
    n_fft = max(len(bold_ir), 2048)
    H = np.abs(np.fft.rfft(bold_ir, n=n_fft))
    f_axis = np.fft.rfftfreq(n_fft, d=dt)

    # Interpolate to requested frequencies
    H_interp = np.interp(freqs, f_axis, H)

    # Predicted spectrum = |H(f)|² (power transfer from white noise input)
    predicted = H_interp ** 2

    # Normalize to sum to 1 (shape comparison, not absolute power)
    total = np.sum(predicted)
    if total > 0:
        predicted = predicted / total

    return jnp.array(predicted)


def predict_bold_spectrum_with_vasomotion(
    stage_label: str,
    freqs: np.ndarray,
    tr: float,
) -> Float[Array, "F"]:
    """Predict BOLD spectrum including vasomotion (~0.02 Hz) component.

    Combines the Balloon transfer function with the NE-driven slow
    vasomotion that appears during NREM sleep.

    Parameters
    ----------
    stage_label : 'W', '1', '2', '3', or 'R'
    freqs : frequency axis (Hz)
    tr : repetition time (s)

    Returns
    -------
    Predicted BOLD power spectrum (normalized)
    """
    from vpjax.sleep.nvc_state import WAKE, N1, N2, N3, REM
    from vpjax.sleep.vasomotion import VasomotionParams, bold_vasomotion

    # Balloon component (existing)
    balloon_spec = np.array(predict_bold_spectrum_for_stage(stage_label, freqs, tr))

    # Vasomotion amplitude per stage (minimal wake, strong N3)
    stage_map = {"W": WAKE, "1": N1, "2": N2, "3": N3, "R": REM}
    stage = stage_map.get(stage_label, WAKE)

    vasomotion_amplitudes = {
        WAKE: 0.002,
        N1: 0.010,
        N2: 0.020,
        N3: 0.035,
        REM: 0.005,
    }
    amp = vasomotion_amplitudes.get(stage, 0.002)

    # Generate vasomotion BOLD signal and compute its PSD
    params = VasomotionParams(cbv_amplitude=jnp.array(amp))
    dt_sim = 0.1
    t_sim = jnp.arange(0, 600, dt_sim)  # 10 min simulation
    bold_vaso = np.array(bold_vasomotion(t_sim, params))

    n_fft = max(len(bold_vaso), 4096)
    vaso_spec = np.abs(np.fft.rfft(bold_vaso, n=n_fft)) ** 2
    f_vaso = np.fft.rfftfreq(n_fft, d=dt_sim)
    vaso_interp = np.interp(freqs, f_vaso, vaso_spec)

    # Normalize vasomotion component
    vaso_total = np.sum(vaso_interp)
    if vaso_total > 0:
        vaso_interp = vaso_interp / vaso_total

    # Combine: balloon + vasomotion (weighted)
    combined = balloon_spec + amp * 10.0 * vaso_interp

    # Renormalize
    total = np.sum(combined)
    if total > 0:
        combined = combined / total

    return jnp.array(combined)
