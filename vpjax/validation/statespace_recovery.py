"""Does the LL filter recover neurovascular parameters, and does EEG help?

Two questions, in order.

**First, a gate.** Before any estimator is pointed at a recording it has
to recover known parameters from data simulated by the same forward
model. If it cannot, nothing it reports from real data means anything.
This module simulates the vpjax Balloon-Windkessel model driven by a
stochastic neural drive, observes it the way a simultaneous EEG-fMRI
experiment does, and checks the estimates against the truth.

**Second, a design question.** Simultaneous EEG costs a cap, an artifact
problem and a limit on the sequences that can be run. It is worth that
only if the fast channel constrains something the slow one cannot. The
latent drive varies on the timescale of the EEG and is observed by BOLD
only after convolution with a sluggish, nonlinear transfer -- so adding
EEG should sharpen the parameters governing that transfer. This module
quantifies how much, by fitting the same simulated data twice: once from
BOLD alone, once from both. The ratio of the errors is the measurable
benefit of the second modality, under this model and at this SNR.

What the comparison is not: a claim about any real recording. It
measures what the fast channel buys *given that this model is correct*.
A model misspecified in a way EEG cannot see would show a benefit here
and none in practice. Treat the number as an upper bound and read
:func:`vpjax.statespace.residual_diagnostics` on real data.

State vector, in the order the filter sees it::

    x = [z, s, f, v, q]

``z`` is the neural drive, an Ornstein-Uhlenbeck process with time
constant ``tau_z`` and diffusion ``q_z``; the remaining four are the
Balloon-Windkessel states from :class:`vpjax.BalloonState`. Making the
drive part of the state, rather than a known input, is the point: in a
resting or drug study the drive is exactly what is unknown.

The EEG channel is a band-limited power envelope at ~10 Hz standing in
for ``z``, not raw EEG. That is the quantity a fusion model observes in
practice and it keeps the grid tractable; a 250 Hz raw channel would add
compute without adding information about a drive that varies far more
slowly.

Amplitude limit, measured
-------------------------
The local linearization is first order, and the BOLD observation is
nonlinear in ``v`` and ``q``, so the approximation degrades as the
fluctuation grows. Running the filter at the *true* parameters and
reading the BOLD channel's innovation variance (1.0 if the filter is
calibrated) gives, at 180 s and TR 2 s:

===========  ==========  ==================
BOLD SD      drive SD    BOLD innov. var.
===========  ==========  ==================
0.9%         0.055       0.94
1.6%         0.10        0.95
2.6%         0.17        0.97
4.0%         0.32        1.15
4.1%         0.71        1.84
===========  ==========  ==================

The EEG channel stays at 1.00 throughout, because its observation map
is exactly linear -- which is what identifies the cause as the
observation nonlinearity rather than the dynamics or the grid. The
figures are flat in ``max_dt`` from 0.5 s down to 0.02 s, so this is
not a step-size effect either.

So: trustworthy for resting-state and drug-infusion fluctuations of a
couple of percent, biased for large evoked responses. For those, the
innovation variance is the warning and a second-order or
sigma-point filter is the fix.

References
----------
Riera JJ et al. (2004) NeuroImage 21:547-567
Valdes-Sosa PA et al. (2009) Hum Brain Mapp 30:2701-2721
Friston KJ et al. (2000) NeuroImage 12:466-477
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from vpjax._types import BalloonParams
from vpjax.hemodynamics.bold import BOLDParams
from vpjax.statespace import (
    ObservationGrid,
    build_observation_grid,
    fit_statespace,
    parameter_uncertainty,
    residual_diagnostics,
)

#: Indices into the augmented state vector.
Z, S, F, V, Q = 0, 1, 2, 3, 4

#: Floor applied to v and f inside the drift.  The filter's mean is not
#: constrained to be physiological, and v <= 0 would make v^(1/alpha)
#: non-finite and destroy the whole trajectory rather than just that
#: step.  The floor is far below any plausible state.
_FLOOR = 1e-3


def augmented_drift(
    kappa: Float[Array, ""],
    tau: Float[Array, ""],
    tau_z: Float[Array, ""],
    gamma: Float[Array, ""],
    alpha: Float[Array, ""],
    E0: Float[Array, ""],
    beta: Float[Array, ""] = 0.0,
):
    """Build ``f(t, x, args)`` for the drive-plus-Balloon system.

    ``beta`` is the gain on an exogenous input ``u(t)`` -- a drug
    effect-site or occupancy time course, for instance.  The input
    arrives through ``args`` as ``(static_args, u_k)``, which is the
    convention :func:`vpjax.statespace.ll_filter` uses for its
    ``inputs`` argument; with plain ``args`` there is no input.
    """

    def drift(t, x, args):
        u = args[1] if isinstance(args, tuple) else 0.0
        z, s = x[Z], x[S]
        f = jnp.maximum(x[F], _FLOOR)
        v = jnp.maximum(x[V], _FLOOR)
        q = x[Q]

        fout = jnp.power(v, 1.0 / alpha)
        extraction = 1.0 - jnp.power(1.0 - E0, 1.0 / f)

        return jnp.array([
            -z / tau_z + beta * u,
            z - kappa * s - gamma * (f - 1.0),
            s,
            (f - fout) / tau,
            (f * extraction / E0 - fout * q / v) / tau,
        ])

    return drift


def augmented_observation(bold_params: BOLDParams | None = None):
    """Build ``h(t, x, args) -> [eeg_proxy, bold]``."""
    bp = BOLDParams() if bold_params is None else bold_params

    def observe(t, x, args):
        v = jnp.maximum(x[V], _FLOOR)
        q = x[Q]
        bold = bp.V0 * (
            bp.k1 * (1.0 - q) + bp.k2 * (1.0 - q / v) + bp.k3 * (1.0 - v)
        )
        return jnp.array([x[Z], bold])

    return observe


def baseline_state() -> Float[Array, "5"]:
    """Resting state: no drive, flow/volume/dHb at baseline."""
    return jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])


def drug_shaped_input(
    onset: float, rise: float, decay: float
):
    """A unit-peak two-exponential (Bateman-shaped) input ``u(t)``.

    This is a *shape*, not a pharmacokinetic model of any compound: it
    has the qualitative form of an effect-site curve after a single
    administration -- zero before *onset*, rising on the time scale
    *rise*, decaying on the time scale *decay* -- and is normalised to
    peak at one so that ``beta`` in :func:`augmented_drift` is the
    peak drive in the drive's own units.  Every number that would make
    it a model of a specific drug (the two time scales and the gain)
    is supplied by the caller.  For real PK/PD input use
    :mod:`vpjax.pharmacokinetics`.
    """
    if not (rise > 0 and decay > rise):
        raise ValueError("need 0 < rise < decay for a single-peaked input")

    def u(t):
        t = np.asarray(t, dtype=float)
        s = np.clip(t - onset, 0.0, None)
        raw = np.exp(-s / decay) - np.exp(-s / rise)
        t_peak = (rise * decay / (decay - rise)) * np.log(decay / rise)
        peak = np.exp(-t_peak / decay) - np.exp(-t_peak / rise)
        return np.where(t >= onset, raw / peak, 0.0)

    return u


def simulate(
    duration: float = 360.0,
    dt: float = 0.01,
    eeg_dt: float = 0.1,
    tr: float = 2.0,
    tau_z: float = 2.0,
    q_z: float = 0.01,
    eeg_snr: float = 3.0,
    eeg_sd: float | None = None,
    bold_sd: float = 0.002,
    balloon: BalloonParams | None = None,
    bold_params: BOLDParams | None = None,
    seed: int = 0,
    drive_input=None,
    beta: float = 0.0,
) -> dict:
    """Simulate a resting EEG-fMRI run from the augmented model.

    Integrated by Euler-Maruyama on a fine grid (*dt*), then sampled at
    *eeg_dt* and *tr*.  The fine grid is the ground truth; the filter
    never sees it, which is what makes this a real test rather than a
    round trip through the same discretisation.

    Parameters
    ----------
    duration : run length (s)
    dt       : simulation step (s)
    eeg_dt   : EEG envelope sample interval (s)
    tr       : fMRI repetition time (s)
    tau_z    : drive time constant (s)
    q_z      : drive diffusion (per s).  The default gives a BOLD
               fluctuation of roughly 1.5% SD, which is resting-state
               scale.  See the note on amplitude below before raising it.
    eeg_snr  : ratio of the drive's stationary SD to the EEG channel's
               noise SD.  Specified as a ratio rather than an absolute
               level because the drive's amplitude is set by ``q_z`` and
               ``tau_z``: a fixed noise SD silently becomes a different
               SNR whenever the amplitude changes, which makes runs at
               different amplitudes incomparable.
    eeg_sd   : absolute override for the EEG noise SD, in units of the
               drive.  Overrides *eeg_snr* when given.
    bold_sd  : BOLD observation noise SD, as a fractional signal change
    balloon  : BalloonParams; defaults to the Friston/Stephan values
    bold_params : BOLDParams; defaults to 3T
    seed     : RNG seed
    drive_input : optional callable ``u(t)`` evaluated on the fine grid,
               e.g. :func:`drug_shaped_input`.  Enters the drive as
               ``beta * u(t)``.  With no input the run is stationary
               resting state.
    beta     : gain on ``drive_input``, in the drive's units.  Only
               meaningful with an input; recorded in ``truth`` and
               estimated by :func:`fit_simulation` when nonzero.

    Returns
    -------
    dict with ``truth`` (the parameters used), ``t_fine``, ``x``,
    ``t_eeg``, ``eeg``, ``t_bold``, ``bold``.
    """
    bp = BalloonParams() if balloon is None else balloon
    kappa = float(bp.kappa)
    gamma = float(bp.gamma)
    tau = float(bp.tau)
    alpha = float(bp.alpha)
    E0 = float(bp.E0)

    drift = augmented_drift(kappa, tau, tau_z, gamma, alpha, E0, beta)
    observe = augmented_observation(bold_params)

    # Stationary SD of the OU drive: Var[z] = q_z tau_z / 2.
    drive_sd = float(np.sqrt(q_z * tau_z / 2.0))
    if eeg_sd is None:
        if eeg_snr <= 0:
            raise ValueError("eeg_snr must be positive")
        eeg_sd = drive_sd / eeg_snr

    n = int(round(duration / dt)) + 1
    t_fine = np.round(np.arange(n) * dt, 9)
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=np.sqrt(q_z * dt), size=n)

    u_fine = (
        np.zeros(n) if drive_input is None
        else np.asarray(drive_input(t_fine), dtype=float)
    )
    if u_fine.shape != (n,):
        raise ValueError("drive_input must return one value per fine-grid time")

    x = np.zeros((n, 5))
    x[0] = np.asarray(baseline_state())
    step = jax.jit(drift)
    for k in range(1, n):
        dx = np.asarray(step(0.0, jnp.asarray(x[k - 1]), (None, u_fine[k - 1])))
        x[k] = x[k - 1] + dx * dt
        x[k, Z] += noise[k - 1]

    y_clean = np.asarray(jax.vmap(lambda xi: observe(0.0, xi, None))(jnp.asarray(x)))

    eeg_every = int(round(eeg_dt / dt))
    bold_every = int(round(tr / dt))
    i_eeg = np.arange(0, n, eeg_every)
    i_bold = np.arange(0, n, bold_every)

    return {
        "truth": {
            "kappa": kappa, "tau": tau, "tau_z": tau_z, "q_z": q_z,
            "gamma": gamma, "alpha": alpha, "E0": E0, "beta": beta,
        },
        "t_fine": t_fine,
        "x": x,
        "u_fine": u_fine,
        "t_eeg": t_fine[i_eeg],
        "eeg": y_clean[i_eeg, 0] + rng.normal(scale=eeg_sd, size=i_eeg.size),
        "t_bold": t_fine[i_bold],
        "bold": y_clean[i_bold, 1] + rng.normal(scale=bold_sd, size=i_bold.size),
        "eeg_sd": eeg_sd,
        "bold_sd": bold_sd,
        "drive_sd": drive_sd,
        "eeg_snr": drive_sd / eeg_sd,
    }


def make_grid(
    sim: dict,
    use_eeg: bool = True,
    max_dt: float = 0.25,
) -> ObservationGrid:
    """Assemble a simulation into a filter-ready grid.

    With ``use_eeg=False`` only the BOLD channel is included, which is
    the comparison condition: the same data, one modality.

    ``max_dt`` subdivides the grid so no propagation step is longer than
    a quarter second.  A 2 s step is far too long for the Balloon
    nonlinearity, and the resulting stale linearization, not the filter
    itself, is what would show up as bias.
    """
    streams = {
        "bold": {
            "time_s": sim["t_bold"],
            "values": sim["bold"],
            "noise_sd": sim["bold_sd"],
        }
    }
    if use_eeg:
        streams["eeg"] = {
            "time_s": sim["t_eeg"],
            "values": sim["eeg"],
            "noise_sd": sim["eeg_sd"],
        }
    return build_observation_grid(streams, max_dt=max_dt)


_FIT_NAMES = ("kappa", "tau", "tau_z", "q_z")
_INPUT_NAMES = _FIT_NAMES + ("beta",)


def default_fit_names(sim: dict) -> tuple[str, ...]:
    """The parameters a simulation exposes: ``beta`` only with an input."""
    return _INPUT_NAMES if sim["truth"].get("beta", 0.0) != 0.0 else _FIT_NAMES


def _build_model(
    grid: ObservationGrid,
    sim: dict,
    fit_names: tuple[str, ...] = _FIT_NAMES,
    jitter: float = 1e-8,
):
    """Return a ``build(theta)`` closure for :func:`fit_statespace`.

    ``theta`` holds the parameters named in *fit_names*, in that order;
    every other parameter is held at its simulated value.  Fixing a
    parameter at truth is how a "known from elsewhere" counterfactual is
    expressed -- a drive time constant supplied by a separate EEG
    session, say -- so the subset is a modelling choice, not a
    convenience. ``gamma``, ``alpha`` and ``E0`` are never free: they are
    not identifiable from BOLD alone (Stephan et al. 2007), and leaving
    them free would confound this comparison with a separate question.

    If the simulation carries an input, it is sampled onto the grid and
    passed to the filter as ``inputs``.  The input is a known smooth
    deterministic function, so sampling it is not interpolation of data.
    """
    truth = sim["truth"]
    unknown = set(fit_names) - set(_INPUT_NAMES)
    if unknown:
        raise ValueError(f"cannot fit {sorted(unknown)}; choose from {_INPUT_NAMES}")
    if "beta" in fit_names and truth.get("beta", 0.0) == 0.0:
        raise ValueError("beta is only estimable from a simulation with an input")
    free = {n: i for i, n in enumerate(fit_names)}
    u_fine = sim.get("u_fine")
    inputs = (
        None if u_fine is None or not np.any(u_fine)
        else jnp.asarray(np.interp(np.asarray(grid.t), sim["t_fine"], u_fine))
    )
    has_eeg = "eeg" in grid.channels
    # Channel order follows grid.channels, which follows insertion order.
    eeg_col = grid.channels.index("eeg") if has_eeg else None
    bold_col = grid.channels.index("bold")

    full_h = augmented_observation()

    def h(t, x, args):
        both = full_h(t, x, args)
        if not has_eeg:
            return both[1:2]
        out = jnp.zeros(2)
        return out.at[eeg_col].set(both[0]).at[bold_col].set(both[1])

    def build(theta):
        def get(name):
            return theta[free[name]] if name in free else truth[name]

        kappa, tau, tau_z, q_z = (get(n) for n in _FIT_NAMES)
        beta = get("beta") if "beta" in truth else 0.0
        Qm = jnp.zeros((5, 5)).at[Z, Z].set(q_z)
        # A floor on the hemodynamic states' process noise keeps the
        # predicted covariance invertible for the smoother; it is six
        # orders below the drive noise and does not shape the fit.
        Qm = Qm + jitter * jnp.eye(5)
        spec = {
            "f": augmented_drift(
                kappa, tau, tau_z,
                truth["gamma"], truth["alpha"], truth["E0"], beta,
            ),
            "h": h,
            "Q": Qm,
            "m0": baseline_state(),
            "P0": jnp.diag(jnp.array([1.0, 0.1, 0.1, 0.01, 0.01])),
        }
        if inputs is not None:
            spec["inputs"] = inputs
        return spec

    return build


def _flatten_diagnostics(d: dict) -> dict:
    """Convert nested residual diagnostics to plain floats for reporting."""
    out = {k: float(v) for k, v in d.items() if k != "per_channel"}
    out["per_channel"] = {
        name: {k: float(v) for k, v in stats.items()}
        for name, stats in d["per_channel"].items()
    }
    return out


def fit_simulation(
    sim: dict,
    use_eeg: bool = True,
    init: dict[str, float] | None = None,
    max_dt: float = 0.25,
    max_steps: int = 200,
    restarts: int = 8,
    seed: int = 1,
    fit_names: tuple[str, ...] | None = None,
) -> dict:
    """Fit the augmented model to a simulation and score the recovery.

    *fit_names* selects the free parameters (default: all the simulation
    exposes, see :func:`default_fit_names`); the rest are fixed at truth.

    Restarts are on by default and necessary, not cautious. ``kappa``
    and ``tau`` both set the width of the hemodynamic impulse response,
    so the likelihood has a ridge along their product; a single
    gradient-based fit started off that ridge slides along it and
    terminates, reporting convergence, several hundred log-likelihood
    units below the truth. The basins are far apart in likelihood, so
    they are trivial to rank once found -- which makes multi-start
    sufficient here, and a single start misleading.

    Returns
    -------
    dict with ``estimate`` and ``truth`` per parameter, ``relative_error``,
    ``log_likelihood``, ``diagnostics`` (innovation whiteness), ``success``,
    ``n_obs``, and ``log_likelihood_by_start``.
    """
    fit_names = default_fit_names(sim) if fit_names is None else tuple(fit_names)
    grid = make_grid(sim, use_eeg=use_eeg, max_dt=max_dt)
    build = _build_model(grid, sim, fit_names)

    if init is None:
        # Deliberately wrong by roughly a factor of two in each direction,
        # so recovery is not an artifact of starting at the answer.
        init = {
            "kappa": 1.5 * sim["truth"]["kappa"],
            "tau": 0.6 * sim["truth"]["tau"],
            "tau_z": 1.7 * sim["truth"]["tau_z"],
            "q_z": 0.5 * sim["truth"]["q_z"],
            "beta": 0.5 * sim["truth"].get("beta", 1.0),
        }
    init = {n: init[n] for n in fit_names}
    init_vec = jnp.array([init[n] for n in fit_names])

    fit = fit_statespace(
        build, grid, init=init_vec, max_steps=max_steps,
        restarts=restarts, seed=seed,
    )
    unc = parameter_uncertainty(build, grid, fit["log_theta"])
    est = {n: float(v) for n, v in zip(fit_names, fit["theta"])}
    truth = {n: float(sim["truth"][n]) for n in fit_names}

    return {
        "fit_names": fit_names,
        "estimate": est,
        "truth": truth,
        "initial": dict(init),
        "relative_error": {
            n: abs(est[n] - truth[n]) / truth[n] for n in fit_names
        },
        "log_likelihood": float(fit["log_likelihood"]),
        "n_obs": float(fit["n_obs"]),
        "success": bool(fit["success"]),
        "diagnostics": _flatten_diagnostics(
            residual_diagnostics(fit["filter"], channels=grid.channels)
        ),
        "channels": grid.channels,
        "log_likelihood_by_start": [
            float(v) for v in fit["all_log_likelihoods"]
        ],
        "identifiable": bool(unc["identifiable"]),
        "positive_definite": bool(unc["positive_definite"]),
        "curvature_ratio": float(unc["curvature_ratio"]),
        "condition_number": float(unc["condition_number"]),
        "standard_error": {
            n: float(v) for n, v in zip(fit_names, unc["standard_error"])
        },
    }


def is_calibrated(fit: dict, tolerance: float = 0.5) -> bool:
    """Whether every channel's innovations are close to standard normal.

    Necessary but **not sufficient**. A model whose innovation variance
    is far from 1 has not described the data. But the converse does not
    hold: this simulation's BOLD-only arm whitens its residuals
    perfectly while estimating the drive diffusion three orders of
    magnitude wrong, because a fast large drive through a slow balloon
    is observationally identical, at a 2 s TR, to a slow small drive
    through a fast balloon. Calibration says the model *can* describe
    the data; it says nothing about whether the parameters that did so
    are the right ones. For that, read ``identifiable``, which comes
    from the curvature of the likelihood
    (:func:`vpjax.statespace.parameter_uncertainty`).
    """
    return all(
        abs(d["variance"] - 1.0) <= tolerance and abs(d["lag1"]) <= 0.3
        for d in fit["diagnostics"]["per_channel"].values()
    )


def compare_modalities(sim: dict, **kwargs) -> dict:
    """Fit the same simulation from BOLD alone and from BOLD plus EEG.

    Returns
    -------
    dict with ``bold_only``, ``both``, ``error_ratio`` — the BOLD-only
    relative error over the joint relative error, per parameter — and
    ``comparable``.

    **Read ``comparable`` first.** When one arm fails to fit at all,
    which is the usual outcome for BOLD alone at a 2 s TR, the error
    ratio is a number describing that failure rather than a measurement
    of what the second modality contributes. The distinction is not
    pedantic: a ratio of 50 from a diverged BOLD-only fit says nothing
    about how much EEG sharpens a fit that works, and quoting it as if
    it did would overstate the case for the EEG cap by an unknown
    factor. When ``comparable`` is False the defensible claim is the
    qualitative one -- that BOLD alone does not identify this model --
    and that claim is checkable from the per-channel diagnostics.
    """
    bold_only = fit_simulation(sim, use_eeg=False, **kwargs)
    both = fit_simulation(sim, use_eeg=True, **kwargs)
    ratio = {
        n: bold_only["relative_error"][n] / max(both["relative_error"][n], 1e-12)
        for n in bold_only["fit_names"]
    }
    return {
        "bold_only": bold_only,
        "both": both,
        "error_ratio": ratio,
        "calibrated": {
            "bold_only": is_calibrated(bold_only),
            "both": is_calibrated(both),
        },
        "identifiable": {
            "bold_only": bold_only["identifiable"],
            "both": both["identifiable"],
        },
        "comparable": bold_only["identifiable"] and both["identifiable"],
    }


def format_comparison(comparison: dict) -> str:
    """Render :func:`compare_modalities` as a table."""
    lines = [
        f"{'parameter':<10} {'truth':>10} {'BOLD only':>12} {'BOLD+EEG':>12}"
        f" {'err ratio':>10}"
    ]
    b, j = comparison["bold_only"], comparison["both"]
    for n in b["fit_names"]:
        lines.append(
            f"{n:<10} {b['truth'][n]:>10.4g} {b['estimate'][n]:>12.4g}"
            f" {j['estimate'][n]:>12.4g} {comparison['error_ratio'][n]:>10.2f}"
        )
    lines.append("")
    lines.append(
        f"observations: BOLD only {b['n_obs']:.0f}, BOLD+EEG {j['n_obs']:.0f}"
    )
    lines.append("")
    lines.append("innovation whiteness (variance should be ~1, lag-1 ~0):")
    for label, r in (("BOLD only", b), ("BOLD+EEG", j)):
        for name, d in r["diagnostics"]["per_channel"].items():
            lines.append(
                f"  {label:<10} {name:<6} n={d['n']:>6.0f}  "
                f"variance {d['variance']:>8.3f}  lag-1 {d['lag1']:>+7.3f}"
            )
    lines.append("")
    lines.append("identifiability, from the curvature of the likelihood")
    lines.append(
        "(relative standard errors, so 0.03 is 3%; a curvature ratio near "
        "zero or negative is a flat or non-concave direction)"
    )
    for label, r in (("BOLD only", b), ("BOLD+EEG", j)):
        ses = " ".join(
            f"{n}={r['standard_error'][n]:.3f}"
            if np.isfinite(r["standard_error"][n]) else f"{n}=inf"
            for n in r["fit_names"]
        )
        lines.append(
            f"  {label:<10} identifiable={str(r['identifiable']):<5} "
            f"curvature ratio={r['curvature_ratio']:+.3g} "
            f"(positive definite={r['positive_definite']})"
        )
        lines.append(f"  {'':<10} {ses}")
    lines.append("")
    if comparison["comparable"]:
        lines.append(
            "Both arms are identifiable, so the error ratios above compare "
            "precision like for like."
        )
    else:
        failed = [k for k, ok in comparison["identifiable"].items() if not ok]
        lines.append(
            f"NOT comparable: {', '.join(failed)} is not identifiable -- the "
            "likelihood has a flat or non-concave direction, so its "
            "estimates are not estimates of anything and the error ratios "
            "above quantify that, not the precision gained from the second "
            "modality. Note that the unidentifiable arm may still whiten "
            "its residuals; calibration does not detect this."
        )
    return "\n".join(lines)
