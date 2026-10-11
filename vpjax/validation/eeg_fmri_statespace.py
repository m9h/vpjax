"""Fit the drive-plus-Balloon model to a recorded run with auxiliary channels.

The simulation study in :mod:`vpjax.validation.statespace_recovery`
knows the truth; here nothing is known, so three things change:

* the observation noise variances are estimated, not supplied;
* every fast channel -- an EEG band envelope, pupil area, a respiratory
  or cardiac measure -- enters through its own free loading ``gain_<c>``
  on the shared drive (the channel is standardised, so the gain is in
  drive units per SD), with its sign chosen by the data.  Alpha power is
  usually inversely related to cortical activation in resting EEG-fMRI
  (Goldman et al. 2002; Laufs et al. 2003), but "usually" is not a
  modelling assumption worth hard-coding, and for pupil or respiration
  there is no such convention at all;
* each fast channel gets its own nuisance state: an OU process the BOLD
  never sees.  On a recorded run the alpha envelope has structure on the
  0.1--0.3 s scale that no drive passing through the Balloon can share
  with a 2 s BOLD series; with a single shared state the fit pins that
  state to the envelope, drives the EEG noise variance to zero and
  leaves the BOLD innovations autocorrelated at 0.9.  The nuisance state
  lets the shared drive be the slow component, which is the only part
  BOLD can corroborate.  ``nuisance=False`` keeps the one-state model
  for comparison;
* the verdict comes from innovation whiteness *and* identifiability, on
  exactly the footing the simulation established: whiteness says the
  model can describe the run, identifiability says whether the
  parameters it did so with mean anything.

State layout is ``[z, s, f, v, q, n_1, ..., n_k]`` for *k* auxiliary
channels with nuisance states.  Channels may be sparse -- a pupil trace
that is valid a quarter of the time is a legitimate observation of the
drive a quarter of the time, and the presence mask handles it without
interpolation.

Everything is a single time series per channel.  That is the coarsest
possible test and deliberately so: if the identifiability contrast
between BOLD-only and BOLD-plus-fast-channel does not appear here, no
regional analysis will rescue it.
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

CORE_NAMES = ("kappa", "tau", "tau_z", "q_z")
N_CORE = 5   # drive + four Balloon states


def param_names(channels: tuple[str, ...] = (), nuisance: bool = True) -> tuple[str, ...]:
    """Free-parameter names for a model with these auxiliary *channels*."""
    names = list(CORE_NAMES)
    for c in channels:
        names.append(f"gain_{c}")
        if nuisance:
            names += [f"tau_n_{c}", f"q_n_{c}"]
        names.append(f"r_{c}")
    names.append("r_bold")
    return tuple(names)


PARAM_NAMES = param_names(("eeg",), nuisance=True)
ONE_STATE_NAMES = param_names(("eeg",), nuisance=False)
BOLD_ONLY_NAMES = param_names(())


# ---------------------------------------------------------------------------
# Preparing the series
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
    env: np.ndarray,
    t_env: np.ndarray,
    t0: float,
    t1: float,
    log: bool = True,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a positive series to ``[t0, t1]`` and standardise it to unit SD.

    The log is taken by default: an amplitude envelope (or a pupil area)
    is positive and right-skewed, and its log is the quantity whose
    fluctuations are closest to Gaussian, which is what a Kalman
    observation assumes.  Samples flagged invalid, or non-positive, are
    dropped rather than filled.
    """
    t_env = np.asarray(t_env, dtype=float)
    x = np.asarray(env, dtype=float)
    keep = (t_env >= t0) & (t_env <= t1) & np.isfinite(x)
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool)
    if log:
        keep &= x > 0
    x = x[keep]
    if log:
        x = np.log(x)
    x = (x - x.mean()) / x.std()
    return t_env[keep], x


def rebin(t: np.ndarray, v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Average a regularly sampled series into bins of width *dt*.

    Used to take a 10 Hz envelope down to 2 Hz before fitting.  At 10 Hz
    the envelope's own fast structure (0.1–0.3 s) makes the EEG
    observation-noise variance redundant with the nuisance state and
    the fit runs it to zero; at 2 Hz that structure is sub-sample and
    enters as white noise, where it belongs, while the slow drive (2–5 s)
    is still sampled four times per time constant.
    """
    t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float)
    step = float(np.median(np.diff(t)))
    k = int(round(dt / step))
    if k <= 1:
        return t, v
    n = v.size // k
    return (t[: n * k].reshape(n, k).mean(axis=1), v[: n * k].reshape(n, k).mean(axis=1))


def run_grid(
    t_bold: np.ndarray,
    bold: np.ndarray,
    aux: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    max_dt: float = 0.25,
) -> ObservationGrid:
    """Observation grid for one run; noise SDs are placeholders.

    *aux* maps a channel name to ``(time_s, values)``.  The grid carries
    unit noise variances, which the model overrides: :func:`run_model`
    puts ``r_diag`` in its spec, so the values stored here are never
    read by the fit.
    """
    streams = {"bold": {"time_s": t_bold, "values": bold, "noise_sd": 1.0}}
    for name, (t, v) in (aux or {}).items():
        streams[name] = {"time_s": t, "values": v, "noise_sd": 1.0}
    return build_observation_grid(streams, max_dt=max_dt)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def run_model(
    grid: ObservationGrid,
    fit_names: tuple[str, ...],
    fixed: dict[str, float],
    sign: float | dict[str, float] = 1.0,
    nuisance: bool = True,
    balloon: BalloonParams | None = None,
    jitter: float = 1e-8,
):
    """``build(theta)`` for a recorded run.

    *fit_names* are free (log scale); *fixed* supplies the rest.  *sign*
    is the loading sign per auxiliary channel (a scalar applies to all).
    ``gamma``, ``alpha`` and ``E0`` come from *balloon* and are never
    free, for the reason given in
    :mod:`vpjax.validation.statespace_recovery`.
    """
    bp = BalloonParams() if balloon is None else balloon
    channels = tuple(c for c in grid.channels if c != "bold")
    nuisance = nuisance and bool(channels)
    needed = param_names(channels, nuisance)
    missing = [n for n in needed if n not in fit_names and n not in fixed]
    if missing:
        raise ValueError(f"neither free nor fixed: {missing}")
    signs = {c: float(sign) for c in channels} if np.isscalar(sign) else dict(sign)
    free = {n: i for i, n in enumerate(fit_names)}
    cols = {name: i for i, name in enumerate(grid.channels)}
    d = N_CORE + (len(channels) if nuisance else 0)
    nidx = {c: N_CORE + i for i, c in enumerate(channels)}
    full_h = augmented_observation()

    def build(theta):
        def get(name):
            return theta[free[name]] if name in free else fixed[name]

        kappa, tau, tau_z, q_z = (get(n) for n in CORE_NAMES)
        r = jnp.zeros(len(grid.channels)).at[cols["bold"]].set(get("r_bold"))
        gains = {c: signs[c] * get(f"gain_{c}") for c in channels}
        for c in channels:
            r = r.at[cols[c]].set(get(f"r_{c}"))
        core = augmented_drift(kappa, tau, tau_z, bp.gamma, bp.alpha, bp.E0)

        def f(t, x, args):
            dx = core(t, x[:N_CORE], args)
            if nuisance:
                dn = jnp.stack([-x[nidx[c]] / get(f"tau_n_{c}") for c in channels])
                dx = jnp.concatenate([dx, dn])
            return dx

        def h(t, x, args):
            both = full_h(t, x[:N_CORE], args)
            out = jnp.zeros(len(grid.channels)).at[cols["bold"]].set(both[1])
            for c in channels:
                y = gains[c] * x[Z] + (x[nidx[c]] if nuisance else 0.0)
                out = out.at[cols[c]].set(y)
            return out

        Qm = jnp.zeros((d, d)).at[Z, Z].set(q_z) + jitter * jnp.eye(d)
        # Stationary variances (q tau / 2) as the priors on the OU states,
        # so the filter does not spend the first minute converging from a
        # wrong guess about their scale.
        diag = [q_z * tau_z / 2.0, 0.1, 0.1, 0.01, 0.01]
        m0 = baseline_state()
        if nuisance:
            for c in channels:
                tau_n, q_n = get(f"tau_n_{c}"), get(f"q_n_{c}")
                Qm = Qm.at[nidx[c], nidx[c]].set(q_n)
                diag.append(q_n * tau_n / 2.0)
            m0 = jnp.concatenate([m0, jnp.zeros(len(channels))])
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
    bold: np.ndarray,
    channels: tuple[str, ...] | bool = (),
    nuisance: bool = True,
    balloon: BalloonParams | None = None,
) -> dict:
    """Starting values that are scaled to the run, not to any dataset.

    ``q_z`` is set so the drive's stationary SD roughly reproduces the
    observed BOLD SD through the Balloon's unit-gain steady state; the
    noise variances start at half the observed variance of each series.
    *channels* may be ``True`` as shorthand for a single ``"eeg"`` channel.
    """
    if channels is True:
        channels = ("eeg",)
    elif channels is False:
        channels = ()
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
    for c in channels:
        init[f"gain_{c}"] = 1.0 / np.sqrt(init["q_z"] * tau_z / 2.0)
        if nuisance:
            # The channel is standardised, so split its unit variance
            # between a fast nuisance state and observation noise; the
            # shared drive's share is what the BOLD will argue for.
            tau_n = 0.3
            init.update({f"tau_n_{c}": tau_n, f"q_n_{c}": 2.0 * 0.4 / tau_n, f"r_{c}": 0.3})
        else:
            init[f"r_{c}"] = 0.5
    return init


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _sign_key(signs: dict[str, float]) -> str:
    return ",".join(f"{c}:{s:+.0f}" for c, s in signs.items()) or "none"


def fit_run(
    t_bold: np.ndarray,
    bold: np.ndarray,
    t_env: np.ndarray | None = None,
    env: np.ndarray | None = None,
    aux: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    sign: float | dict[str, float] | None = None,
    nuisance: bool = True,
    init: dict | None = None,
    fixed: dict | None = None,
    max_dt: float = 0.25,
    restarts: int = 8,
    max_steps: int = 300,
    seed: int = 1,
) -> dict:
    """Fit one run with any set of auxiliary channels.

    ``t_env, env`` is shorthand for ``aux={"eeg": (t_env, env)}``.  With
    *sign* unspecified the loading signs are chosen greedily: all start
    positive, then each channel's sign is flipped in turn and the flip
    kept if the likelihood improves — ``k + 1`` fits for *k* channels
    rather than ``2**k``.  For one channel this is exactly both signs.

    Returns the same keys as ``statespace_recovery.fit_simulation`` less
    the truth-dependent ones, plus ``sign`` (per channel) and
    ``log_likelihood_by_sign``.
    """
    aux = dict(aux or {})
    if env is not None:
        aux["eeg"] = (t_env, env)
    channels = tuple(aux)
    nuisance = nuisance and bool(channels)
    grid = run_grid(t_bold, bold, aux, max_dt=max_dt)
    fixed = {} if fixed is None else dict(fixed)
    names = tuple(n for n in param_names(channels, nuisance) if n not in fixed)
    init = default_init(bold, channels, nuisance) if init is None else dict(init)
    init_vec = jnp.array([init[n] for n in names])

    def one(signs):
        build = run_model(grid, names, fixed, sign=signs, nuisance=nuisance)
        fit = fit_statespace(
            build, grid, init=init_vec, max_steps=max_steps,
            restarts=restarts, seed=seed,
        )
        return build, fit

    tried = {}
    if sign is None:
        signs = {c: 1.0 for c in channels}
    else:
        signs = {c: float(sign) for c in channels} if np.isscalar(sign) else dict(sign)
    best = (signs, *one(signs))
    tried[_sign_key(signs)] = float(best[2]["log_likelihood"])
    if sign is None:
        for c in channels:
            trial = dict(best[0]); trial[c] = -trial[c]
            cand = (trial, *one(trial))
            tried[_sign_key(trial)] = float(cand[2]["log_likelihood"])
            if float(cand[2]["log_likelihood"]) > float(best[2]["log_likelihood"]):
                best = cand
    signs, build, fit = best
    unc = parameter_uncertainty(build, grid, fit["log_theta"])

    return {
        "fit_names": names,
        "channels": grid.channels,
        "nuisance": nuisance,
        "sign": dict(signs),
        "log_likelihood_by_sign": tried,
        "estimate": {n: float(v) for n, v in zip(names, fit["theta"])},
        "initial": {n: init[n] for n in names},
        "fixed": fixed,
        "log_likelihood": float(fit["log_likelihood"]),
        "n_obs": float(fit["n_obs"]),
        "n_present": {
            c: int(np.asarray(grid.mask)[:, i].sum()) for i, c in enumerate(grid.channels)
        },
        "success": bool(fit["success"]),
        "diagnostics": _flatten_diagnostics(
            residual_diagnostics(fit["filter"], channels=grid.channels)
        ),
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
    signs = " ".join(f"{c}{s:+.0f}" for c, s in r["sign"].items()) or "none"
    lines = [f"{label}: log-lik {r['log_likelihood']:.1f} over {r['n_obs']:.0f} obs, "
             f"converged={r['success']}, signs {signs}"]
    if len(r["log_likelihood_by_sign"]) > 1:
        lines.append("  by sign: " + ", ".join(
            f"{k}: {v:.1f}" for k, v in r["log_likelihood_by_sign"].items()))
    for n in r["fit_names"]:
        se = r["standard_error"][n]
        lines.append(f"  {n:<10} {r['estimate'][n]:>11.4g}   rel SE "
                     + (f"{se:.3f}" if np.isfinite(se) else "inf"))
    for name, d in r["diagnostics"]["per_channel"].items():
        lines.append(f"  innovations {name:<6} n={d['n']:>6.0f} variance {d['variance']:.3f} "
                     f"lag-1 {d['lag1']:+.3f}")
    lines.append(f"  identifiable={r['identifiable']} curvature ratio={r['curvature_ratio']:+.3g} "
                 f"positive definite={r['positive_definite']}")
    return "\n".join(lines)
