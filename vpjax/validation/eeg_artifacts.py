"""In-scanner EEG cleaning for the state-space validation: AAS and envelopes.

Two artifacts dominate EEG recorded inside the bore, and both are
handled here by average artifact subtraction (AAS), which assumes the
artifact repeats with a known trigger and estimates its waveform as a
sliding average over neighbouring repetitions:

* the gradient artifact, locked to the volume trigger (Allen, Josephs &
  Turner 2000, NeuroImage 12:230);
* the ballistocardiogram (BCG), locked to the ECG R peak (Allen,
  Polizzi, Krakow, Fish & Lemieux 1998, NeuroImage 8:229).

The sliding template follows each artifact's slow drift (scanner
heating, subject settling, heart-rate change) at the cost of a small
bias when neighbouring repetitions overlap.  No ICA, OBS or reference-
layer method is attempted: those are the correct next step for a
production pipeline, and the carbon-wire-loop recordings elsewhere in
this protocol exist precisely to replace this stage with a measured
reference. What is wanted *here* is a transparent baseline whose
residuals can be read in the filter's innovation diagnostics.

Everything is NumPy/SciPy on ``(channels, samples)`` arrays. MNE is used
only to read BrainVision files in :func:`load_brainvision`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_brainvision(vhdr_path: str | Path, trigger: str = "R128") -> dict:
    """Read a BrainVision recording and its volume triggers.

    Returns ``data`` (channels, samples) in volts, ``fs``, ``ch_names``
    and ``triggers`` (sample indices of the volume marker).

    Some BIDS exports (NATVIEW, for one) ship the BrainVision header and
    marker files but hold the samples in an EEGLAB ``.set`` next to them;
    when the ``.eeg`` binary is missing, the ``.set`` is read instead and
    the triggers are taken from the ``.vmrk``.
    """
    import mne

    vhdr_path = Path(vhdr_path)
    set_path = vhdr_path.with_suffix(".set")
    try:
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
        events, ids = mne.events_from_annotations(raw, verbose=False)
        key = next((k for k in ids if trigger in k), None)
        if key is None:
            raise ValueError(f"no {trigger!r} marker in {vhdr_path}; found {sorted(ids)}")
        triggers = events[events[:, 2] == ids[key], 0] - raw.first_samp
    except FileNotFoundError:
        if not set_path.exists():
            raise
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
        triggers = _vmrk_triggers(vhdr_path.with_suffix(".vmrk"), trigger)
    return {
        "data": raw.get_data(),
        "fs": float(raw.info["sfreq"]),
        "ch_names": list(raw.ch_names),
        "triggers": np.asarray(triggers, dtype=int),
    }


def _vmrk_triggers(vmrk_path: Path, trigger: str) -> np.ndarray:
    """Sample positions (0-based) of a marker in a BrainVision ``.vmrk``."""
    out = []
    for line in open(vmrk_path, encoding="utf-8", errors="replace"):
        if line.startswith("Mk") and "=" in line:
            fields = line.strip().split("=", 1)[1].split(",")
            if len(fields) >= 3 and fields[1].strip() == trigger:
                out.append(int(fields[2]) - 1)
    if not out:
        raise ValueError(f"no {trigger!r} marker in {vmrk_path}")
    return np.asarray(out, dtype=int)


# ---------------------------------------------------------------------------
# Average artifact subtraction
# ---------------------------------------------------------------------------

def _sliding_mean(epochs: np.ndarray, half_width: int) -> np.ndarray:
    """Mean over a window of ``±half_width`` epochs, truncated at the ends."""
    n = epochs.shape[0]
    csum = np.concatenate([np.zeros((1,) + epochs.shape[1:]), np.cumsum(epochs, axis=0)])
    lo = np.clip(np.arange(n) - half_width, 0, n)
    hi = np.clip(np.arange(n) + half_width + 1, 0, n)
    return (csum[hi] - csum[lo]) / (hi - lo)[:, None]


def subtract_locked_artifact(
    data: np.ndarray,
    onsets: np.ndarray,
    length: int,
    half_width: int,
    exclude_self: bool = False,
) -> np.ndarray:
    """Subtract a sliding-average template at each onset, channel by channel.

    Parameters
    ----------
    data      : (channels, samples)
    onsets    : sample index at which each repetition starts
    length    : template length in samples
    half_width: template is the mean over ``±half_width`` repetitions
    exclude_self : leave the current repetition out of its own template.
                   Unbiased for the artifact but, for the gradient case,
                   makes no practical difference at the usual window and
                   costs a second pass; off by default.

    Repetitions that would run past the end of the recording are
    dropped.  Overlapping windows (BCG at short RR intervals) are
    subtracted sequentially, which double-counts the overlap slightly;
    keep ``length`` at or below the median interval to limit this.
    """
    onsets = np.asarray(onsets, dtype=int)
    onsets = onsets[(onsets >= 0) & (onsets + length <= data.shape[1])]
    if onsets.size < 2 * half_width + 1:
        raise ValueError("too few repetitions for the requested template width")
    out = np.array(data, dtype=np.float64, copy=True)
    idx = onsets[:, None] + np.arange(length)[None, :]
    for c in range(data.shape[0]):
        epochs = data[c][idx]                       # (n_rep, length)
        template = _sliding_mean(epochs, half_width)
        if exclude_self:
            n = epochs.shape[0]
            counts = (
                np.clip(np.arange(n) + half_width + 1, 0, n)
                - np.clip(np.arange(n) - half_width, 0, n)
            )
            template = (template * counts[:, None] - epochs) / np.maximum(counts - 1, 1)[:, None]
        for k, start in enumerate(onsets):
            out[c, start:start + length] -= template[k]
    return out


def subtract_gradient(
    data: np.ndarray, triggers: np.ndarray, half_width: int = 10
) -> tuple[np.ndarray, dict]:
    """Gradient AAS locked to the volume trigger.

    The template length is the median trigger spacing; a recording with
    a trigger spacing that varies by more than one sample is not
    scanner-synchronised and should not be cleaned this way, so that
    raises rather than smearing the template.
    """
    spacing = np.diff(triggers)
    period = int(np.median(spacing))
    jitter = int(np.max(np.abs(spacing - period)))
    if jitter > 1:
        raise ValueError(
            f"volume-trigger spacing varies by {jitter} samples; the EEG "
            "clock is not locked to the scanner and AAS needs resampling first"
        )
    before = _hf_rms(data, triggers[0], triggers[-1])
    cleaned = subtract_locked_artifact(data, triggers, period, half_width)
    after = _hf_rms(cleaned, triggers[0], triggers[-1])
    return cleaned, {
        "period_samples": period,
        "trigger_jitter_samples": jitter,
        "n_volumes": int(triggers.size),
        "rms_before": before,
        "rms_after": after,
        "attenuation_db": 20.0 * np.log10(before / after),
    }


def _hf_rms(data: np.ndarray, start: int, stop: int) -> float:
    """RMS of the in-scan segment; the gradient artifact dominates it."""
    seg = data[:, start:stop]
    return float(np.sqrt(np.mean(seg**2)))


# ---------------------------------------------------------------------------
# ECG and ballistocardiogram
# ---------------------------------------------------------------------------

def detect_r_peaks(ecg: np.ndarray, fs: float, min_rr: float = 0.4) -> np.ndarray:
    """R-peak sample indices from a single ECG channel.

    Band-pass 5–20 Hz to isolate the QRS complex, choose the polarity
    with the larger peaks (the in-bore ECG is often inverted), and keep
    peaks separated by at least *min_rr* seconds with a prominence above
    four times the median absolute deviation.
    """
    sos = butter(2, [5.0, 20.0], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, ecg)
    mad = np.median(np.abs(x - np.median(x))) + 1e-12
    best = None
    for sign in (1.0, -1.0):
        peaks, _ = find_peaks(sign * x, distance=int(min_rr * fs), prominence=4.0 * mad)
        if not peaks.size:
            continue
        # Judge polarity and the second-pass threshold by peak *height*,
        # not prominence: the band-pass ringing on either side of an
        # inverted QRS is as prominent as the QRS itself but half as high.
        # The MAD of a mostly-flat ECG is tiny, so the first pass also
        # admits noise peaks wherever no QRS falls within ``min_rr``.
        typical = np.median(sign * x[peaks])
        peaks, _ = find_peaks(
            sign * x, distance=int(min_rr * fs), prominence=4.0 * mad, height=0.4 * typical
        )
        if best is None or typical > best[0]:
            best = (typical, peaks)
    if best is None:
        raise ValueError("no QRS-like peaks found in the ECG channel")
    return best[1]


def subtract_bcg(
    data: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    half_width: int = 15,
    lead: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """BCG AAS locked to the R peak.

    Each window starts *lead* seconds before the R peak so the pulse
    artifact, which lags the R peak by roughly 0.2 s, is wholly inside
    it, and runs for the median RR interval.
    """
    rr = np.diff(r_peaks) / fs
    length = int(np.median(rr) * fs)
    onsets = r_peaks - int(lead * fs)
    cleaned = subtract_locked_artifact(data, onsets, length, half_width)
    return cleaned, {
        "n_beats": int(r_peaks.size),
        "heart_rate_bpm": float(60.0 / np.median(rr)),
        "rr_cv": float(np.std(rr) / np.mean(rr)),
        "window_s": length / fs,
    }


# ---------------------------------------------------------------------------
# Resampling and envelopes
# ---------------------------------------------------------------------------

def downsample(data: np.ndarray, fs: float, new_fs: float) -> tuple[np.ndarray, float]:
    """Anti-alias (Butterworth at 0.4·new_fs) and take every ``fs/new_fs``-th sample."""
    step = int(round(fs / new_fs))
    if abs(fs / new_fs - step) > 1e-6:
        raise ValueError("new_fs must divide fs")
    sos = butter(4, 0.4 * new_fs, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)[..., ::step], fs / step


def band_envelope(
    signal: np.ndarray,
    fs: float,
    band: tuple[float, float],
    out_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert amplitude envelope of a band, averaged into ``out_dt`` bins.

    Returns bin centres (s, on the recording's clock) and the envelope.
    Bin averaging rather than decimation, because the envelope's
    fluctuations faster than ``out_dt`` are what the filter will treat
    as observation noise and they should enter as such, not aliased.
    """
    sos = butter(4, list(band), btype="band", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, signal)))
    width = int(round(out_dt * fs))
    n = env.size // width
    binned = env[: n * width].reshape(n, width).mean(axis=1)
    centres = (np.arange(n) + 0.5) * width / fs
    return centres, binned


def clean_rest_run(
    vhdr_path: str | Path,
    picks: tuple[str, ...] = ("O1", "O2", "Oz", "P3", "P4", "Pz"),
    band: tuple[float, float] = (8.0, 13.0),
    out_dt: float = 0.1,
    work_fs: float = 250.0,
    ecg: str = "ECG",
) -> dict:
    """Gradient AAS → downsample → BCG AAS → band envelope on chosen channels.

    Returns the envelope on the recording clock, the volume-trigger times
    on the same clock (so BOLD volumes can be stamped without any offset
    guess), and the per-stage diagnostics.
    """
    rec = load_brainvision(vhdr_path)
    data, fs, names, trig = rec["data"], rec["fs"], rec["ch_names"], rec["triggers"]
    data, grad = subtract_gradient(data, trig)
    data, fs_ds = downsample(data, fs, work_fs)
    scale = fs_ds / fs
    r = detect_r_peaks(data[names.index(ecg)], fs_ds)
    data, bcg = subtract_bcg(data, r, fs_ds)
    missing = [p for p in picks if p not in names]
    if missing:
        raise ValueError(f"channels not in recording: {missing}")
    sig = data[[names.index(p) for p in picks]].mean(axis=0)
    t_env, env = band_envelope(sig, fs_ds, band, out_dt)
    return {
        "t_env": t_env,
        "envelope": env,
        "t_volumes": trig / fs,
        "n_volumes": int(trig.size),
        "fs_work": fs_ds,
        "picks": tuple(picks),
        "band": tuple(band),
        "gradient": grad,
        "bcg": bcg,
        "r_peaks_s": r / fs_ds,
        "scale": scale,
    }
