"""Extract cardiac features from EEG-embedded ECG channel.

The ds003768 EEG data includes an ECG channel alongside the 30 EEG
channels.  This module extracts heart rate per epoch and generates
the cardiac pulsatility BOLD confound spectrum.

Pipeline:
    BrainVision .vhdr → pick ECG → R-peak detection → HR per epoch
    HR → PulsatilityParams → cardiac BOLD confound spectrum
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jaxtyping import Array, Float
import jax.numpy as jnp


def extract_heart_rate(
    vhdr_path: str | Path,
    epoch_duration: float = 30.0,
) -> np.ndarray:
    """Extract heart rate (bpm) per epoch from ECG channel.

    Parameters
    ----------
    vhdr_path : path to BrainVision .vhdr file
    epoch_duration : epoch length (s), matching sleep staging

    Returns
    -------
    hr_bpm : heart rate per epoch (bpm), shape (n_epochs,)
    """
    import mne

    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

    # Find the ECG channel
    ecg_channels = [ch for ch in raw.ch_names if "ECG" in ch.upper() or "EKG" in ch.upper()]
    if not ecg_channels:
        # Try to find it by type
        ecg_idx = mne.pick_types(raw.info, ecg=True)
        if len(ecg_idx) == 0:
            raise ValueError("No ECG channel found in EEG data")
        ecg_channels = [raw.ch_names[i] for i in ecg_idx]

    ecg_raw = raw.copy().pick(ecg_channels)
    ecg_data = ecg_raw.get_data()[0]  # single channel
    sfreq = ecg_raw.info["sfreq"]

    # R-peak detection using scipy
    from scipy.signal import butter, filtfilt, find_peaks

    # Bandpass filter ECG (5-45 Hz)
    b, a = butter(3, [5.0, 45.0], btype="bandpass", fs=sfreq)
    ecg_filt = filtfilt(b, a, ecg_data)

    # Detect R-peaks
    # Use absolute value (handles inverted ECG) and find peaks
    ecg_abs = np.abs(ecg_filt)
    min_distance = int(0.4 * sfreq)  # minimum 0.4s between beats (150 bpm max)
    height_threshold = np.percentile(ecg_abs, 90)
    peaks, _ = find_peaks(ecg_abs, distance=min_distance, height=height_threshold * 0.5)

    # Compute HR per epoch
    epoch_samples = int(epoch_duration * sfreq)
    n_epochs = len(ecg_data) // epoch_samples
    hr_bpm = np.zeros(n_epochs)

    for i in range(n_epochs):
        start = i * epoch_samples
        end = start + epoch_samples
        epoch_peaks = peaks[(peaks >= start) & (peaks < end)]
        if len(epoch_peaks) >= 2:
            rr_intervals = np.diff(epoch_peaks) / sfreq  # seconds
            mean_rr = np.mean(rr_intervals)
            if mean_rr > 0:
                hr_bpm[i] = 60.0 / mean_rr
            else:
                hr_bpm[i] = np.nan
        else:
            hr_bpm[i] = np.nan

    return hr_bpm


def cardiac_bold_spectrum(
    hr_bpm: float | np.ndarray,
    freqs: np.ndarray,
    duration: float = 300.0,
) -> Float[Array, "F"]:
    """Generate cardiac BOLD confound spectrum for given heart rate.

    Parameters
    ----------
    hr_bpm : heart rate (bpm), scalar or mean value
    freqs : frequency axis (Hz)
    duration : simulation duration (s)

    Returns
    -------
    Cardiac BOLD power spectrum at requested frequencies
    """
    from vpjax.cardiac.pulsatility import PulsatilityParams, bold_cardiac_confound

    hr = float(np.nanmean(hr_bpm)) if isinstance(hr_bpm, np.ndarray) else float(hr_bpm)
    if np.isnan(hr) or hr < 30 or hr > 200:
        hr = 70.0  # fallback

    params = PulsatilityParams(hr_bpm=jnp.array(hr))

    dt = 0.1  # 100ms resolution
    t = jnp.arange(0, duration, dt)
    confound = np.array(bold_cardiac_confound(t, params))

    # PSD
    n_fft = max(len(confound), 4096)
    power = np.abs(np.fft.rfft(confound, n=n_fft)) ** 2
    f_axis = np.fft.rfftfreq(n_fft, d=dt)

    # Interpolate to requested frequencies
    power_interp = np.interp(freqs, f_axis, power)

    # Normalize
    total = np.sum(power_interp)
    if total > 0:
        power_interp = power_interp / total

    return jnp.array(power_interp)
