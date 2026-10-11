"""Load a NATVIEW (NKI naturalistic-viewing EEG-fMRI) run onto one clock.

Telesford et al. 2023, Sci Data 10:554; data on FCP-INDI
(``s3://fcp-indi/data/Projects/NATVIEW_EEGFMRI/raw_data``, CC BY 4.0).
Per run: 64-channel BrainCap MR at 5 kHz (61 EEG, EOGL/EOGU, ECG),
Siemens Trio BOLD at TR 2.1 s, EyeLink 1000 at 250 Hz (gaze, pupil area,
blink/fixation/saccade flags, and a column marking the fMRI volume
triggers), and a BIOPAC respiration belt at 62.5 Hz with the scanner
trigger on the same clock.

Everything is put on the EEG clock, with the volume triggers as the
common reference: the ``R128`` markers in the EEG, the
``fMRI_Volume_Trigger`` column in the eye-tracking file, and the
trigger channel in the BIOPAC file are three views of the same pulses,
so each stream is aligned by matching its first trigger to the EEG's,
not by any offset guess.

Pupil area is an arousal observation: it covaries with locus coeruleus
activity and with the slow BOLD fluctuations of the arousal network
(Yellin et al. 2015; Schneider et al. 2016; Joshi & Gold 2020), which is
exactly the kind of slow shared drive the state-space model is after.
It is also sparse -- blinks and eye closure remove it, and in a resting
run a drowsy subject may supply it a quarter of the time.  That is
handled by the presence mask, not by interpolation.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from vpjax.validation.eeg_artifacts import clean_rest_run

OCCIPITAL = ("O1", "O2", "Oz", "POz", "PO3", "PO4", "PO7", "PO8")


def run_paths(root: str | Path, subject: str, task: str = "rest", session: str = "01") -> dict:
    """Paths of one NATVIEW run in the raw BIDS layout."""
    root = Path(root)
    base = root / f"sub-{subject}" / f"ses-{session}"
    stem = f"sub-{subject}_ses-{session}_task-{task}"
    return {
        "stem": stem,
        "vhdr": base / "eeg" / f"{stem}_eeg.vhdr",
        "bold": base / "func" / f"{stem}_bold.nii.gz",
        "bold_json": base / "func" / f"{stem}_bold.json",
        "t1": base / "anat" / f"sub-{subject}_ses-{session}_T1w.nii.gz",
        "eye": base / "eeg" / f"{stem}_recording-eyetracking_physio.tsv.gz",
        "eye_json": base / "eeg" / f"{stem}_recording-eyetracking_physio.json",
        "resp": base / "eeg" / f"{stem}_recording-respiratory_physio.tsv.gz",
        "resp_json": base / "eeg" / f"{stem}_recording-respiratory_physio.json",
        "trigger": base / "eeg" / f"{stem}_recording-trigger_physio.tsv.gz",
    }


def _align(
    stream_trigger_times: np.ndarray,
    eeg_trigger_times: np.ndarray,
    fs: float,
    max_jitter_s: float = 0.025,
) -> tuple[float, float, float]:
    """Linear map from a stream's clock onto the EEG clock, via shared triggers.

    Matches the trigger trains by their *end*: the eye-tracker and BIOPAC
    may miss the first few pulses (they start recording on the task
    start), but all of them stop with the scanner.  A straight line
    ``t_eeg = a + b * t_stream`` is fitted through the paired triggers so
    that a slow clock drift is absorbed.  The residual spread must be
    within the larger of two sampling intervals (timestamp quantisation
    on that stream) and *max_jitter_s*: the auxiliary streams are binned
    at 0.1 s or coarser, so trigger-logging jitter below 25 ms cannot
    matter, while a spread of a tenth of a TR means the clocks are not
    locked.  Returns ``(a, b, residual_spread_s)``.
    """
    n = min(stream_trigger_times.size, eeg_trigger_times.size)
    if n < 10:
        raise ValueError("too few shared triggers to align")
    x, y = stream_trigger_times[-n:], eeg_trigger_times[-n:]
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    spread = float(np.max(resid) - np.min(resid))
    if spread > max(2.0 / fs, max_jitter_s):
        raise ValueError(
            f"trigger trains disagree by {spread*1e3:.0f} ms after a linear fit; the "
            "streams are not on locked clocks and cannot be aligned"
        )
    if abs(b - 1.0) > 1e-3:
        raise ValueError(f"clock rate ratio {b:.5f} is not plausible for locked clocks")
    return float(a), float(b), spread


def load_eye(paths: dict, eeg_trigger_times: np.ndarray, out_dt: float = 0.25) -> dict:
    """Pupil area binned to ``out_dt`` on the EEG clock, with a validity mask.

    A bin is valid when at least half its samples carry a positive pupil
    area and no blink flag.  The value is the mean area of those samples;
    standardisation (log, unit SD) is left to the caller, as for the EEG.
    """
    meta = json.load(open(paths["eye_json"]))
    cols = {c: i for i, c in enumerate(meta["Columns"])}
    eye = np.genfromtxt(gzip.open(paths["eye"]), delimiter="\t")
    t = eye[:, cols["Time"]]
    pupil = eye[:, cols["Pupil_Area"]]
    blink = eye[:, cols["Blinks"]]
    trig = t[eye[:, cols["fMRI_Volume_Trigger"]] > 0]
    a, b, spread = _align(trig, eeg_trigger_times, float(meta["SamplingFrequency"]))
    t = a + b * t

    good = np.isfinite(pupil) & (pupil > 0) & ~(np.nan_to_num(blink) > 0)
    edges = np.arange(np.floor(t[0] / out_dt) * out_dt, t[-1] + out_dt, out_dt)
    which = np.digitize(t, edges) - 1
    n_bins = edges.size - 1
    count = np.bincount(which, minlength=n_bins)[:n_bins]
    n_good = np.bincount(which[good], minlength=n_bins)[:n_bins]
    total = np.bincount(which[good], weights=pupil[good], minlength=n_bins)[:n_bins]
    valid = (count > 0) & (n_good >= 0.5 * np.maximum(count, 1))
    values = np.where(valid, total / np.maximum(n_good, 1), np.nan)
    centres = edges[:-1] + out_dt / 2
    return {
        "time_s": centres[count > 0],
        "values": values[count > 0],
        "valid": valid[count > 0],
        "offset_s": a, "rate": b, "trigger_residual_s": spread,
        "fraction_valid": float(valid[count > 0].mean()),
        "n_triggers": int(trig.size),
    }


def load_respiration(paths: dict, eeg_trigger_times: np.ndarray, out_dt: float = 1.0) -> dict:
    """Respiratory amplitude envelope on the EEG clock.

    The belt voltage is band-passed to breathing frequencies (0.1–0.6 Hz)
    and its Hilbert amplitude averaged into ``out_dt`` bins -- the
    breathing *depth*, which is what drives arterial CO2 and hence CBF
    (Birn et al. 2006), rather than the raw belt trace.
    """
    from scipy.signal import butter, hilbert, sosfiltfilt

    meta = json.load(open(paths["resp_json"]))
    fs = float(meta["SamplingFrequency"])
    resp = np.genfromtxt(gzip.open(paths["resp"]), delimiter="\t")
    trig = np.genfromtxt(gzip.open(paths["trigger"]), delimiter="\t")
    t = resp[:, 0]
    # The trigger file is a sparse list of (time, value) rows; successive
    # rows closer than half a TR are the same pulse sampled twice.
    tt = trig[:, 0]
    pulses = tt[np.r_[True, np.diff(tt) > 0.5]]
    a, b, spread = _align(pulses, eeg_trigger_times, fs)
    t = a + b * t
    sos = butter(2, [0.1, 0.6], btype="band", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, resp[:, 1])))
    width = int(round(out_dt * fs))
    n = env.size // width
    binned = env[: n * width].reshape(n, width).mean(axis=1)
    centres = t[0] + (np.arange(n) + 0.5) * width / fs
    return {"time_s": centres, "values": binned, "offset_s": a, "rate": b,
            "trigger_residual_s": spread, "n_triggers": int(pulses.size)}


def load_run(paths: dict, picks: tuple[str, ...] = OCCIPITAL, band=(8.0, 13.0),
             eeg_dt: float = 0.1, eye_dt: float = 0.25, resp_dt: float = 1.0) -> dict:
    """Clean the EEG and align every auxiliary stream to its clock.

    Returns ``t_volumes`` (EEG clock), ``tr``, and per-stream dicts for
    ``eeg`` (alpha envelope), ``pupil`` and ``respiration``.  BOLD series
    are extracted separately (global mean or ROI) and stamped at
    ``t_volumes + tr / 2``.
    """
    eeg = clean_rest_run(paths["vhdr"], picks=picks, band=band, out_dt=eeg_dt)
    tr = float(json.load(open(paths["bold_json"]))["RepetitionTime"])
    t_vol = eeg["t_volumes"]
    out = {
        "stem": paths["stem"], "tr": tr, "t_volumes": t_vol,
        "eeg": {"time_s": eeg["t_env"], "values": eeg["envelope"],
                "gradient": eeg["gradient"], "bcg": eeg["bcg"]},
    }
    if Path(paths["eye"]).exists():
        out["pupil"] = load_eye(paths, t_vol, out_dt=eye_dt)
    if Path(paths["resp"]).exists() and Path(paths["trigger"]).exists():
        out["respiration"] = load_respiration(paths, t_vol, out_dt=resp_dt)
    return out
