"""Cerebrovascular reactivity (CVR) fitting from hypercapnia BOLD.

Wraps :func:`vpjax.hemodynamics.inversion.fit_balloon_bold` for the
specific case of block-design CO2 challenges, with a fallback when no
end-tidal CO2 (etCO2) trace is available.  Most large cohorts (UK
Biobank, ABCD, ADNI, DLBS) lack a synchronised capnograph, so the
data-driven stimulus path is the common one.

Two stimulus modes:

* **Known etCO2** — pass the etCO2 trace directly; it is normalised and
  resampled onto the ODE grid.
* **Data-driven** — the whole-brain mean BOLD signal is used as a CO2
  proxy after band-pass filtering (Liu et al. 2017, NeuroImage 187:104).
  The shape of the global signal is dominated by the CO2 challenge in
  hypercapnia experiments; this is *not* a substitute for true etCO2,
  but it recovers the relative timing and amplitude needed to fit
  per-region Balloon parameters.

Per-region fitting loops the BOLD time series of each label in a region
volume (FastSurfer aparc, FreeSurfer aparc+aseg, Glasser, etc.) against
the shared stimulus, using the JIT-compiled batch fitter.

References
----------
Liu P et al. (2017) NeuroImage 187:104-115
    "Cerebrovascular reactivity (CVR) MRI with CO2 challenge: a
    technical review"
Murphy K et al. (2011) NeuroImage 54:369-379
    "Resting-state fMRI confounds and cleanup"
Stephan KJ et al. (2007) NeuroImage 38:387-401
    "Comparing hemodynamic models with DCM"
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from vpjax._types import BalloonParams
from vpjax.hemodynamics.bold import BOLDParams
from vpjax.hemodynamics.inversion import fit_balloon_bold_batch


# ---------------------------------------------------------------------------
# Stimulus extraction
# ---------------------------------------------------------------------------

def _butter_lowpass(
    x: Float[Array, "N"],
    tr: float,
    cutoff_hz: float,
) -> Float[Array, "N"]:
    """Low-pass filter via SciPy butter+filtfilt (numpy, not JAX).

    CVR signal is a slow ~0.05 Hz fluctuation; cutoff_hz=0.1 removes
    cardiac/respiratory aliasing while preserving the CO2 envelope.
    """
    from scipy.signal import butter, filtfilt
    fs = 1.0 / tr
    nyq = fs / 2.0
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = butter(4, wn, btype="low")
    return filtfilt(b, a, np.asarray(x))


def extract_global_stimulus(
    bold_4d: Float[np.ndarray, "X Y Z N"],
    brain_mask: Float[np.ndarray, "X Y Z"],
    tr: float,
    dt: float,
    cutoff_hz: float = 0.05,
    amplitude: float = 0.3,
) -> tuple[Float[Array, "T"], Float[Array, "N"]]:
    """Build a CO2 stimulus proxy from the whole-brain mean BOLD signal.

    The global mean BOLD timecourse during a block-design hypercapnia
    paradigm is dominated by the CO2-induced flow change (Liu 2017).
    We low-pass filter it, normalise to ``±amplitude``, and upsample to
    the ODE grid.

    The amplitude default of 0.3 keeps the Balloon ODE in its stable
    regime: a neural input of magnitude ~1 drives flow ``f`` below zero
    via the vasodilatory signal ``s``, after which ``(1-E0)^(1/f)`` in
    the deoxyhemoglobin equation hits a singularity and the trajectory
    diverges to NaN.  The scale of the data-driven CO2 proxy is in any
    case arbitrary (it's a unit-free regressor); the per-region CVR
    scalar from the Balloon fit absorbs the global amplitude.

    Parameters
    ----------
    bold_4d : 4D BOLD volume (X, Y, Z, N)
    brain_mask : binary brain mask (X, Y, Z); nonzero voxels averaged
    tr : repetition time, seconds
    dt : ODE integration timestep, seconds (T = N * round(tr/dt))
    cutoff_hz : low-pass cutoff (default 0.05 Hz).  CO2 block-design
        paradigms have fundamental frequencies under 0.02 Hz; 0.05 Hz
        retains harmonics up to ~4× the fundamental while suppressing
        residual respiratory (~0.2-0.3 Hz) and cardiac (~1 Hz) signals
        that survive the TR=2 s sampling.
    amplitude : peak absolute value of the returned stimulus (default
        0.3, see note above)

    Returns
    -------
    stimulus : array of shape (T,) at the ODE timestep, zero-mean,
        scaled so ``max(|stim|) ≈ amplitude``
    global_mean : array of shape (N,) the raw whole-brain mean (TR grid)
    """
    mask = np.asarray(brain_mask) > 0
    voxels = np.asarray(bold_4d)[mask]                  # (V, N)
    g = voxels.mean(axis=0)                             # (N,)

    g_lp = _butter_lowpass(g, tr=tr, cutoff_hz=cutoff_hz)
    g_centered = g_lp - g_lp.mean()
    peak = float(np.max(np.abs(g_centered)))
    g_scaled = g_centered * (amplitude / peak) if peak > 1e-8 else g_centered

    # Upsample TR grid → dt grid by linear interpolation.
    n = g_scaled.shape[0]
    subsample = int(round(tr / dt))
    t_target = n * subsample
    src_t = np.arange(n) * tr
    tgt_t = np.arange(t_target) * dt
    stim = np.interp(tgt_t, src_t, g_scaled)

    return jnp.asarray(stim), jnp.asarray(g)


def resample_etco2_stimulus(
    etco2: Float[np.ndarray, "M"],
    etco2_tr: float,
    n_volumes: int,
    bold_tr: float,
    dt: float,
) -> Float[Array, "T"]:
    """Resample a known etCO2 trace onto the ODE grid.

    Parameters
    ----------
    etco2 : etCO2 samples, mmHg or arbitrary
    etco2_tr : etCO2 sampling interval, seconds
    n_volumes : number of BOLD volumes (used to size output)
    bold_tr : BOLD repetition time, seconds
    dt : ODE timestep, seconds

    Returns
    -------
    stimulus : array of shape (n_volumes * round(bold_tr/dt),) z-scored
    """
    etco2 = np.asarray(etco2, dtype=np.float64)

    subsample = int(round(bold_tr / dt))
    t_target = n_volumes * subsample
    src_t = np.arange(etco2.shape[0]) * etco2_tr
    tgt_t = np.arange(t_target) * dt
    interp = np.interp(tgt_t, src_t, etco2, left=etco2[0], right=etco2[-1])
    stim = (interp - interp.mean()) / (interp.std() + 1e-8)

    return jnp.asarray(stim)


# ---------------------------------------------------------------------------
# Per-region fitting
# ---------------------------------------------------------------------------

class CVRFitResult(NamedTuple):
    """Per-region CVR fit output.

    All arrays have leading axis R (number of regions).

    Attributes
    ----------
    region_ids : integer label values, shape (R,)
    kappa, gamma, tau, alpha, E0 : Balloon parameters per region (R,)
    cvr_scalar : %BOLD per unit z-scored stimulus, shape (R,)
        Computed as the peak fitted BOLD response across the time
        series.  Sign follows the stimulus sign convention.
    loss : MSE per region (R,)
    bold_predicted : fitted BOLD time series (R, N)
    bold_observed : observed (region-mean) BOLD time series (R, N)
    """
    region_ids: np.ndarray
    kappa: np.ndarray
    gamma: np.ndarray
    tau: np.ndarray
    alpha: np.ndarray
    E0: np.ndarray
    cvr_scalar: np.ndarray
    loss: np.ndarray
    bold_predicted: np.ndarray
    bold_observed: np.ndarray


def _region_mean_timeseries(
    bold_4d: Float[np.ndarray, "X Y Z N"],
    region_volume: Float[np.ndarray, "X Y Z"],
    region_ids: np.ndarray,
    min_voxels: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-region BOLD means; drop regions below voxel threshold."""
    bold = np.asarray(bold_4d)
    region_volume = np.asarray(region_volume)
    n_t = bold.shape[-1]

    means = []
    kept = []
    for rid in region_ids:
        mask = region_volume == rid
        n_vox = int(mask.sum())
        if n_vox < min_voxels:
            continue
        ts = bold[mask].mean(axis=0)
        # Convert to fractional change relative to per-region baseline.
        baseline = ts.mean()
        if baseline <= 0:
            continue
        means.append((ts - baseline) / baseline)
        kept.append(int(rid))

    if not kept:
        return np.zeros((0, n_t), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return (
        np.stack(means, axis=0).astype(np.float32),
        np.asarray(kept, dtype=np.int32),
    )


def fit_cvr_per_region(
    bold_4d: Float[np.ndarray, "X Y Z N"],
    brain_mask: Float[np.ndarray, "X Y Z"],
    region_volume: Float[np.ndarray, "X Y Z"],
    tr: float,
    dt: float = 0.1,
    stimulus: Float[Array, "T"] | None = None,
    region_ids: np.ndarray | None = None,
    balloon_params: BalloonParams | None = None,
    bold_params: BOLDParams | None = None,
    fit_names: tuple[str, ...] = ("kappa", "tau"),
    n_steps: int = 200,
    learning_rate: float = 2.0,
    min_voxels: int = 5,
) -> CVRFitResult:
    """Fit Balloon-Windkessel parameters per region from hypercapnia BOLD.

    If *stimulus* is omitted, the data-driven global signal is used as
    the CO2 proxy — appropriate for cohorts without a capnograph.

    Parameters
    ----------
    bold_4d : preprocessed BOLD volume (motion-corrected, in native
        space matching ``brain_mask`` and ``region_volume``)
    brain_mask : binary mask used for the global stimulus
    region_volume : integer label volume; each unique nonzero label
        becomes a region (e.g. FastSurfer ``aparc+aseg.nii.gz``)
    tr : BOLD repetition time, seconds
    dt : ODE integration timestep (default 0.1 s, matches the stable
        regime used in ``fit_balloon_bold``)
    stimulus : optional pre-computed stimulus on the ODE grid; if None
        we derive it from the global mean
    region_ids : optional restriction to a subset of labels; if None we
        fit every nonzero label that has at least *min_voxels* voxels
    balloon_params, bold_params, fit_names, n_steps, learning_rate :
        forwarded to :func:`fit_balloon_bold_batch`
    min_voxels : drop regions with fewer voxels than this

    Returns
    -------
    :class:`CVRFitResult` with per-region parameters and CVR scalars
    """
    bold_4d = np.asarray(bold_4d)
    brain_mask = np.asarray(brain_mask)
    region_volume = np.asarray(region_volume)

    if stimulus is None:
        stimulus, _ = extract_global_stimulus(
            bold_4d, brain_mask, tr=tr, dt=dt,
        )

    if region_ids is None:
        region_ids = np.unique(region_volume[region_volume > 0]).astype(np.int32)

    region_ts, kept = _region_mean_timeseries(
        bold_4d, region_volume, region_ids, min_voxels=min_voxels,
    )
    if kept.shape[0] == 0:
        raise ValueError(
            "No regions met the min_voxels threshold "
            f"(min_voxels={min_voxels}, candidate labels={len(region_ids)})"
        )

    fit_out = fit_balloon_bold_batch(
        jnp.asarray(region_ts),
        stimulus,
        tr=tr,
        dt=dt,
        balloon_params=balloon_params,
        bold_params=bold_params,
        fit_names=fit_names,
        n_steps=n_steps,
        learning_rate=learning_rate,
    )

    bold_pred = np.asarray(fit_out["bold_predicted"])     # (R, N)
    cvr_scalar = bold_pred.max(axis=1) - bold_pred.min(axis=1)

    return CVRFitResult(
        region_ids=kept,
        kappa=np.asarray(fit_out["kappa"]),
        gamma=np.asarray(fit_out["gamma"]),
        tau=np.asarray(fit_out["tau"]),
        alpha=np.asarray(fit_out["alpha"]),
        E0=np.asarray(fit_out["E0"]),
        cvr_scalar=cvr_scalar,
        loss=np.asarray(fit_out["loss"]),
        bold_predicted=bold_pred,
        bold_observed=region_ts,
    )


__all__ = [
    "CVRFitResult",
    "extract_global_stimulus",
    "resample_etco2_stimulus",
    "fit_cvr_per_region",
]
