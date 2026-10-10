"""Fit the drive-plus-Balloon model to a recorded EEG-fMRI run.

The simulation study in :mod:`vpjax.validation.statespace_recovery`
knows the truth; here nothing is known, so three things change:

* the observation noise variances are estimated, not supplied;
* the EEG envelope enters through a free loading ``gain`` on the drive
  (the envelope is standardised, so ``gain`` is in drive units per SD of
  log-envelope), with its sign chosen by the data -- alpha power is
  usually inversely related to cortical activation in resting
  EEG-fMRI (Goldman et al. 2002; Laufs et al. 2003), but "usually" is
  not a modelling assumption worth hard-coding;
* the EEG envelope gets its own nuisance state: a fast OU process that
  the BOLD never sees.  On a recorded run the alpha envelope has
  structure on the 0.1--0.3 s scale that no drive passing through the
  Balloon can share with a 2 s BOLD series; with a single shared state
  the fit pins that state to the envelope, drives the EEG noise
  variance to zero and leaves the BOLD innovations autocorrelated at
  0.9.  The nuisance state lets the shared drive be the slow component,
  which is the only part BOLD can corroborate.  ``nuisance=False``
  keeps the one-state model for comparison;
* the verdict comes from innovation whiteness *and* identifiability,
  on exactly the footing the simulation established: whiteness says
  the model can describe the run, identifiability says whether the
  parameters it did so with mean anything.

Everything is a single global time series per modality.  That is the
coarsest possible test and deliberately so: if the identifiability
contrast between BOLD-only and BOLD+EEG does not appear at the global
level, no regional analysis will rescue it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from vpjax.statespace import (
    ObservationGrid,
    build_observation_grid,
    fit_statespace,
    parameter_uncertainty,
    residual_diagnostics,
)
from vpjax._types import BalloonParams
from vpjax.validation.statespace_recovery import (
    Z,
    _flatten_diagnostics,
    augmented_drift,
    augmented_observation,
    baseline_state,
)

PARAM_NAMES = ("kappa", "tau", "tau_z", "q_z", "gain", "tau_n", "q_n", "r_eeg", "r_bold")
ONE_STATE_NAMES = ("kappa", "tau", "tau_z", "q_z", "gain", "r_eeg", "r_bold")
BOLD_ONLY_NAMES = ("kappa", "tau", "tau_z", "q_z", "r_bold")
N = 5   # index of the EEG nuisance state, after the five drive+Balloon states


# ---------------------------------------------------------------------------
# Preparing the two series
# ---------------------------------------------------------------------------

def bold_fractional(ts: np.ndarray, drop: int = 0) -> np.ndarray:
    """Fractional signal change after dropping *drop* volumes and a linear detrend.

    The Balloon observation is a fractional change about baseline, so the
    series is divided by its mean, not z-scored: its amplitude carries
    information about the drive.
    """
    y = np.asarray(ts, dtype=float)[drop:]
    k = np.arange(y.size)
    slope, intercept = np.polyfit(k, y, 1)
    baseline = intercept + slope * k
    return (y - baseline) / np.mean(y)


def standardise_envelope(
    env: np.ndarray, t_env: np.ndarray, t0: float, t1: float, log: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict an envelope to ``[t0, t1]`` and standardise it to unit SD.

    The log is taken by default: an amplitude envelope is positive and
    right-skewed, and its log is the quantity whose fluctuations are
    closest to Gaussian, which is what a Kalman observation assumes.
    """
    keep = (t_env >= t0) & (t_env <= t1)
    x = np.asarray(env, dtype=float)[keep]
    if log:
        x = np.log(np.maximum(x, 1e-12 * np.max(x)))
    x = (x - x.mean()) / x.std()
    return np.asarray(t_env)[keep], x


def run_grid(
    t_bold: np.ndarray,
    bold: np.ndarray,
    t_env: np.ndarray | None = None,
    env: np.ndarray | None = None,
    max_dt: float = 0.25,
) -> ObservationGrid:
    """Observation grid for one run; noise SDs are placeholders.

    The grid carries unit noise variances, which the model overrides:
    :func:`run_model` puts ``r_diag`` in its spec, so the values stored
    here are never read by the fit.
    """
    streams = {"bold": {"time_s": t_bold, "values": bold, "noise_sd": 1.0}}
    if env is not None:
        streams["eeg"] = {"time_s": t_env, "values": env, "noise_sd": 1.0}
    return build_observation_grid(streams, max_dt=max_dt)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def run_model(
    grid: ObservationGrid,
    fit_names: tuple[str, ...],
    fixed: dict[str, float],
    sign: float = -1.0,
    nuisance: bool = True,
    balloon: BalloonParams | None = None,
    jitter: float = 1e-8,
):
    """``build(theta)`` for a recorded run.

    *fit_names* are free (log scale); *fixed* supplies the rest.  ``gamma``,
    ``alpha`` and ``E0`` come from *balloon* and are never free, for the
    reason given in :mod:`vpjax.validation.statespace_recovery`.  With
    *nuisance* the state is six-dimensional: the EEG observation is
    ``gain * z + n`` where ``n`` is an OU process with time constant
    ``tau_n`` and diffusion ``q_n`` that enters no other equation.
    """
    bp = BalloonParams() if balloon is None else balloon
    has_eeg = "eeg" in grid.channels
    nuisance = nuisance and has_eeg
    if has_eeg and "gain" not in fit_names and "gain" not in fixed:
        raise ValueError("an EEG channel needs a gain, free or fixed")
    needed = PARAM_NAMES if nuisance else (ONE_STATE_NAMES if has_eeg else BOLD_ONLY_NAMES)
    missing = [n for n in needed if n not in fit_names and n not in fixed]
    if missing:
        raise ValueError(f"neither free nor fixed: {missing}")
    d = 6 if nuisance else 5
    free = {n: i for i, n in enumerate(fit_names)}
    cols = {name: i for i, name in enumerate(grid.channels)}
    full_h = augmented_observation()

    def build(theta):
        def get(name):
            return theta[free[name]] if name in free else fixed[name]

        kappa, tau, tau_z, q_z = (get(n) for n in ("kappa", "tau", "tau_z", "q_z"))
        r = jnp.zeros(len(grid.channels)).at[cols["bold"]].set(get("r_bold"))
        if has_eeg:
            gain = sign * get("gain")
            r = r.at[cols["eeg"]].set(get("r_eeg"))
        if nuisance:
            tau_n, q_n = get("tau_n"), get("q_n")

        core = augmented_drift(kappa, tau, tau_z, bp.gamma, bp.alpha, bp.E0)

        def f(t, x, args):
            dx = core(t, x[:5], args)
            if nuisance:
                dx = jnp.concatenate([dx, jnp.array([-x[N] / tau_n])])
            return dx

        def h(t, x, args):
            both = full_h(t, x[:5], args)
            out = jnp.zeros(len(grid.channels)).at[cols["bold"]].set(both[1])
            if has_eeg:
                eeg = gain * x[Z] + (x[N] if nuisance else 0.0)
                out = out.at[cols["eeg"]].set(eeg)
            return out

        Qm = jnp.zeros((d, d)).at[Z, Z].set(q_z) + jitter * jnp.eye(d)
        # Stationary variances (q tau / 2) as the priors on the OU states,
        # so the filter does not spend the first minute converging from a
        # wrong guess about their scale.
        diag = [q_z * tau_z / 2.0, 0.1, 0.1, 0.01, 0.01]
        m0 = baseline_state()
        if nuisance:
            Qm = Qm.at[N, N].set(q_n)
            diag.append(q_n * tau_n / 2.0)
            m0 = jnp.concatenate([m0, jnp.zeros(1)])
        return {
            "f": f,
            "h": h,
            "Q": Qm,
            "m0": m0,
            "P0": jnp.diag(jnp.array(diag)),
            "r_diag": r,
        }

    return build


def default_init(
    bold: np.ndarray, has_eeg: bool, nuisance: bool = True,
    balloon: BalloonParams | None = None,
) -> dict:
    """Starting values that are scaled to the run, not to any dataset.

    ``q_z`` is set so the drive's stationary SD roughly reproduces the
    observed BOLD SD through the Balloon's unit-gain steady state; the
    noise variances start at half the observed variance of each series.
    """
    bp = BalloonParams() if balloon is None else balloon
    bold_var = float(np.var(bold))
    tau_z = 2.0
    init = {
        "kappa": float(bp.kappa), "tau": float(bp.tau), "tau_z": tau_z,
        # BOLD fractional SD ~ 0.03 * drive SD at these Balloon gains
        # (measured in the simulation), so drive var ~ bold_var / 1e-3.
        "q_z": 2.0 * bold_var / 1e-3 / tau_z,
        "r_bold": 0.5 * bold_var,
    }
    if has_eeg:
        init["gain"] = 1.0 / np.sqrt(init["q_z"] * tau_z / 2.0)
        init["r_eeg"] = 0.5
    if has_eeg and nuisance:
        # The envelope is standardised, so split its unit variance evenly
        # between the fast nuisance state and observation noise; the shared
        # drive's share is what the BOLD will argue for.
        tau_n = 0.3
        init.update({"tau_n": tau_n, "q_n": 2.0 * 0.4 / tau_n, "r_eeg": 0.3})
    return init


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_run(
    t_bold: np.ndarray,
    bold: np.ndarray,
    t_env: np.ndarray | None = None,
    env: np.ndarray | None = None,
    sign: float | None = None,
    nuisance: bool = True,
    init: dict | None = None,
    fixed: dict | None = None,
    max_dt: float = 0.25,
    restarts: int = 8,
    max_steps: int = 300,
    seed: int = 1,
) -> dict:
    """Fit one run; with an envelope, try both loading signs unless given.

    Returns the same keys as ``statespace_recovery.fit_simulation`` less
    the truth-dependent ones, plus ``sign`` and, when both signs were
    tried, ``log_likelihood_by_sign``.
    """
    has_eeg = env is not None
    nuisance = nuisance and has_eeg
    grid = run_grid(t_bold, bold, t_env, env, max_dt=max_dt)
    fixed = {} if fixed is None else dict(fixed)
    all_names = PARAM_NAMES if nuisance else (ONE_STATE_NAMES if has_eeg else BOLD_ONLY_NAMES)
    names = tuple(n for n in all_names if n not in fixed)
    init = default_init(bold, has_eeg, nuisance) if init is None else dict(init)
    init_vec = jnp.array([init[n] for n in names])

    signs = (sign,) if sign is not None else ((-1.0, 1.0) if has_eeg else (1.0,))
    fits = {}
    for s in signs:
        build = run_model(grid, names, fixed, sign=s, nuisance=nuisance)
        fit = fit_statespace(
            build, grid, init=init_vec, max_steps=max_steps,
            restarts=restarts, seed=seed,
        )
        fits[s] = (build, fit)
    best = max(fits, key=lambda s: float(fits[s][1]["log_likelihood"]))
    build, fit = fits[best]
    unc = parameter_uncertainty(build, grid, fit["log_theta"])

    return {
        "fit_names": names,
        "nuisance": nuisance,
        "sign": best,
        "log_likelihood_by_sign": {
            str(s): float(f["log_likelihood"]) for s, (_, f) in fits.items()
        },
        "estimate": {n: float(v) for n, v in zip(names, fit["theta"])},
        "initial": {n: init[n] for n in names},
        "fixed": fixed,
        "log_likelihood": float(fit["log_likelihood"]),
        "n_obs": float(fit["n_obs"]),
        "success": bool(fit["success"]),
        "diagnostics": _flatten_diagnostics(
            residual_diagnostics(fit["filter"], channels=grid.channels)
        ),
        "channels": grid.channels,
        "log_likelihood_by_start": [float(v) for v in fit["all_log_likelihoods"]],
        "identifiable": bool(unc["identifiable"]),
        "positive_definite": bool(unc["positive_definite"]),
        "curvature_ratio": float(unc["curvature_ratio"]),
        "condition_number": float(unc["condition_number"]),
        "standard_error": {n: float(v) for n, v in zip(names, unc["standard_error"])},
        "correlation": np.asarray(unc["correlation"]).tolist(),
    }


def format_run(label: str, r: dict) -> str:
    """One arm of a run, as text."""
    lines = [f"{label}: log-lik {r['log_likelihood']:.1f} over {r['n_obs']:.0f} obs, "
             f"converged={r['success']}, sign={r['sign']:+.0f}"]
    if len(r["log_likelihood_by_sign"]) > 1:
        lines.append("  by sign: " + ", ".join(
            f"{k}: {v:.1f}" for k, v in r["log_likelihood_by_sign"].items()))
    for n in r["fit_names"]:
        se = r["standard_error"][n]
        lines.append(f"  {n:<7} {r['estimate'][n]:>11.4g}   rel SE "
                     + (f"{se:.3f}" if np.isfinite(se) else "inf"))
    for name, d in r["diagnostics"]["per_channel"].items():
        lines.append(f"  innovations {name:<5} n={d['n']:>6.0f} variance {d['variance']:.3f} "
                     f"lag-1 {d['lag1']:+.3f}")
    lines.append(f"  identifiable={r['identifiable']} curvature ratio={r['curvature_ratio']:+.3g} "
                 f"positive definite={r['positive_definite']}")
    return "\n".join(lines)
