"""Inverse problems for hemodynamic models.

Fits Balloon-Windkessel or Riera NVC parameters to observed fMRI data
(BOLD, ASL, VASO) using gradient-based optimization through the
differentiable forward models.

The ODE is solved exactly via Diffrax, and JAX autodiff provides exact
gradients through the solver for parameter estimation.  This is the
direct-integration alternative to physics-informed neural networks
(PINNs): no neural network approximation, no physics-vs-data loss
balancing, and exact ODE solutions at every optimization step.

Usage::

    from vpjax.hemodynamics.inversion import fit_balloon_bold

    result = fit_balloon_bold(bold_data, stimulus, tr=2.0, dt=0.01)
    print(result['E0'], result['loss'])

References
----------
Friston KJ et al. (2000) NeuroImage 12:466-477
    "Nonlinear responses in fMRI: the Balloon model, Volterra kernels,
    and other hemodynamics"
Stephan KJ et al. (2007) NeuroImage 38:387-401
    "Comparing hemodynamic models with DCM"
Riera JJ et al. (2007) NeuroImage 36:1179-1196
    "Nonlinear local electrovascular coupling. II"
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax._types import BalloonParams, BalloonState
from vpjax.hemodynamics.balloon import solve_balloon
from vpjax.hemodynamics.bold import BOLDParams, observe_bold
from vpjax.hemodynamics.riera import (
    RieraParams,
    RieraState,
    riera_to_balloon,
    solve_riera,
)
from vpjax.perfusion.asl import observe_asl
from vpjax.perfusion.vaso import observe_vaso


# ---------------------------------------------------------------------------
# Physiological parameter bounds
# ---------------------------------------------------------------------------

_BALLOON_BOUNDS: dict[str, tuple[float, float]] = {
    "kappa": (0.3, 2.0),
    "gamma": (0.1, 1.0),
    "tau": (0.5, 3.0),
    "alpha": (0.15, 0.50),
    "E0": (0.15, 0.65),
}

_BALLOON_ALL_NAMES: tuple[str, ...] = ("kappa", "gamma", "tau", "alpha", "E0")

# Default: fit only HRF-shaping params.  alpha and E0 are poorly
# identifiable from BOLD alone (Stephan et al. 2007, DCM convention).
_BALLOON_DEFAULT_FIT: tuple[str, ...] = ("kappa", "gamma", "tau")

_RIERA_BOUNDS: dict[str, tuple[float, float]] = {
    "kappa_no": (0.2, 2.0),
    "kappa_ade": (0.1, 1.5),
    "gamma_no": (0.1, 1.0),
    "gamma_ade": (0.05, 0.8),
    "c_no": (0.1, 3.0),
    "c_ade": (0.05, 2.0),
    "tau_a": (0.1, 2.0),
    "tau_c": (0.3, 3.0),
    "tau_v": (0.5, 5.0),
    "alpha_a": (0.05, 0.50),
    "alpha_c": (0.02, 0.30),
    "alpha_v": (0.10, 0.50),
    "E0": (0.15, 0.65),
    "phi": (0.1, 2.0),
    "tau_m": (0.5, 10.0),
}

_RIERA_ALL_NAMES: tuple[str, ...] = (
    "kappa_no", "kappa_ade", "gamma_no", "gamma_ade",
    "c_no", "c_ade", "tau_a", "tau_c", "tau_v",
    "alpha_a", "alpha_c", "alpha_v", "E0", "phi", "tau_m",
)

_RIERA_DEFAULT_FIT: tuple[str, ...] = (
    "c_no", "kappa_no", "gamma_no", "tau_v", "alpha_v", "E0",
)


# ---------------------------------------------------------------------------
# Balloon-Windkessel inverse problems
# ---------------------------------------------------------------------------

def _make_balloon_params(
    theta: Float[Array, "P"],
    base_params: BalloonParams,
    fit_names: tuple[str, ...],
) -> BalloonParams:
    """Reconstruct BalloonParams, replacing fitted fields with clipped theta."""
    vals = {name: getattr(base_params, name) for name in _BALLOON_ALL_NAMES}
    for i, name in enumerate(fit_names):
        lo, hi = _BALLOON_BOUNDS[name]
        vals[name] = jnp.clip(theta[i], lo, hi)
    return BalloonParams(**vals)


def fit_balloon_bold(
    bold_data: Float[Array, "N"],
    stimulus: Float[Array, "T"],
    tr: float,
    dt: float = 0.01,
    balloon_params: BalloonParams | None = None,
    bold_params: BOLDParams | None = None,
    fit_names: tuple[str, ...] = _BALLOON_DEFAULT_FIT,
    n_steps: int = 500,
    learning_rate: float = 5.0,
) -> dict[str, Float[Array, ""]]:
    """Fit Balloon-Windkessel parameters to a BOLD time series.

    Integrates the Balloon ODE at resolution *dt*, observes the BOLD
    signal, subsamples to *tr*, and minimizes MSE against *bold_data*.
    Gradients flow through the ODE solver via JAX autodiff.

    Only the parameters named in *fit_names* are optimised; the rest
    are held fixed at their values in *balloon_params*.  The default
    fits (kappa, gamma, tau) — the HRF-shaping parameters — and holds
    alpha and E0 fixed, following DCM convention (Stephan et al. 2007).

    Parameters
    ----------
    bold_data : observed BOLD signal (fractional change), shape (N,)
    stimulus : neural stimulus sampled at *dt*, shape (T,).
        T must satisfy ``T >= N * round(tr / dt)``.
    tr : BOLD repetition time (seconds)
    dt : ODE integration timestep (seconds)
    balloon_params : initial BalloonParams (default: DCM standard)
    bold_params : BOLDParams held fixed (default: 3T)
    fit_names : which BalloonParams fields to optimise
        (default: kappa, gamma, tau)
    n_steps : gradient descent iterations
    learning_rate : step size (default 5.0 — gradients through ODE
        solvers are O(1e-3), requiring a larger step than typical)

    Returns
    -------
    Dict with all 5 Balloon parameter values (fitted and fixed),
    plus 'bold_predicted' and 'loss'.
    """
    if balloon_params is None:
        balloon_params = BalloonParams()
    if bold_params is None:
        bold_params = BOLDParams()

    subsample = int(round(tr / dt))
    n = bold_data.shape[0]

    # Pack only the fitted parameters into theta
    theta = jnp.array([float(getattr(balloon_params, name)) for name in fit_names])

    def loss_fn(theta):
        bp = _make_balloon_params(theta, balloon_params, fit_names)
        _, traj = solve_balloon(bp, stimulus, dt=dt)
        bold_pred = observe_bold(traj, bold_params)[::subsample][:n]
        return jnp.mean((bold_pred - bold_data) ** 2)

    # JIT a single gradient step — avoids compiling the full unrolled
    # scan which can take hours with diffeqsolve inside jax.grad.
    # NaN-safe: if the ODE diverges for a parameter combination, the
    # gradient is NaN and we skip the update (keep previous theta).
    @jax.jit
    def step(th, vel):
        g = jax.grad(loss_fn)(th)
        safe = jnp.all(jnp.isfinite(g))
        vel = jnp.where(safe, 0.9 * vel + 0.1 * g, vel)
        th = jnp.where(safe, th - learning_rate * vel, th)
        return th, vel

    velocity = jnp.zeros_like(theta)
    for _ in range(n_steps):
        theta, velocity = step(theta, velocity)

    # Final forward pass
    bp_final = _make_balloon_params(theta, balloon_params, fit_names)
    _, traj_final = solve_balloon(bp_final, stimulus, dt=dt)
    bold_pred = observe_bold(traj_final, bold_params)[::subsample][:n]
    loss = jnp.mean((bold_pred - bold_data) ** 2)

    result: dict[str, Float[Array, "..."]] = {}
    for name in _BALLOON_ALL_NAMES:
        result[name] = getattr(bp_final, name)
    result["bold_predicted"] = bold_pred
    result["loss"] = loss

    return result


def fit_balloon_multimodal(
    bold_data: Float[Array, "N"],
    stimulus: Float[Array, "T"],
    tr: float,
    dt: float = 0.01,
    asl_data: Float[Array, "N"] | None = None,
    vaso_data: Float[Array, "N"] | None = None,
    weights: dict[str, float] | None = None,
    balloon_params: BalloonParams | None = None,
    bold_params: BOLDParams | None = None,
    fit_names: tuple[str, ...] = _BALLOON_ALL_NAMES,
    n_steps: int = 500,
    learning_rate: float = 5.0,
) -> dict[str, Float[Array, ""]]:
    """Fit Balloon-Windkessel parameters to multi-modal fMRI data.

    Jointly fits BOLD, ASL (fractional CBF change), and/or VASO
    (fractional CBV change) time series.  Multiple modalities improve
    parameter identifiability by constraining complementary aspects of
    the hemodynamic response.

    The default fits all 5 parameters, since multi-modal data provides
    enough constraints.  For BOLD-only, use :func:`fit_balloon_bold`
    which defaults to the identifiable subset (kappa, gamma, tau).

    Parameters
    ----------
    bold_data : observed BOLD signal (fractional change), shape (N,)
    stimulus : neural stimulus sampled at *dt*, shape (T,)
    tr : repetition time shared by all modalities (seconds)
    dt : ODE integration timestep (seconds)
    asl_data : observed ASL signal (f - 1), shape (N,), optional
    vaso_data : observed VASO signal (1 - v), shape (N,), optional
    weights : per-modality loss weights, keys 'bold'/'asl'/'vaso'
        (default: 1.0 for each)
    balloon_params : initial BalloonParams
    bold_params : BOLDParams held fixed
    fit_names : which BalloonParams fields to optimise
        (default: all 5 — appropriate for multi-modal data)
    n_steps : gradient descent iterations
    learning_rate : step size

    Returns
    -------
    Dict with all 5 Balloon parameter values (fitted and fixed),
    'bold_predicted', optionally 'asl_predicted'/'vaso_predicted',
    and 'loss'.
    """
    if balloon_params is None:
        balloon_params = BalloonParams()
    if bold_params is None:
        bold_params = BOLDParams()
    if weights is None:
        weights = {}

    w_bold = weights.get("bold", 1.0)
    w_asl = weights.get("asl", 1.0)
    w_vaso = weights.get("vaso", 1.0)

    use_asl = asl_data is not None
    use_vaso = vaso_data is not None
    subsample = int(round(tr / dt))
    n = bold_data.shape[0]

    theta = jnp.array([float(getattr(balloon_params, name)) for name in fit_names])

    def loss_fn(theta):
        bp = _make_balloon_params(theta, balloon_params, fit_names)
        _, traj = solve_balloon(bp, stimulus, dt=dt)

        bold_pred = observe_bold(traj, bold_params)[::subsample][:n]
        loss = w_bold * jnp.mean((bold_pred - bold_data) ** 2)

        if use_asl:
            asl_pred = observe_asl(traj)[::subsample][:n]
            loss = loss + w_asl * jnp.mean((asl_pred - asl_data) ** 2)

        if use_vaso:
            vaso_pred = observe_vaso(traj)[::subsample][:n]
            loss = loss + w_vaso * jnp.mean((vaso_pred - vaso_data) ** 2)

        return loss

    @jax.jit
    def step(th, vel):
        g = jax.grad(loss_fn)(th)
        safe = jnp.all(jnp.isfinite(g))
        vel = jnp.where(safe, 0.9 * vel + 0.1 * g, vel)
        th = jnp.where(safe, th - learning_rate * vel, th)
        return th, vel

    velocity = jnp.zeros_like(theta)
    for _ in range(n_steps):
        theta, velocity = step(theta, velocity)

    bp_final = _make_balloon_params(theta, balloon_params, fit_names)
    _, traj_final = solve_balloon(bp_final, stimulus, dt=dt)

    bold_pred = observe_bold(traj_final, bold_params)[::subsample][:n]
    loss = jnp.mean((bold_pred - bold_data) ** 2)

    result: dict[str, Float[Array, "..."]] = {}
    for name in _BALLOON_ALL_NAMES:
        result[name] = getattr(bp_final, name)
    result["bold_predicted"] = bold_pred
    result["loss"] = loss

    if use_asl:
        result["asl_predicted"] = observe_asl(traj_final)[::subsample][:n]
    if use_vaso:
        result["vaso_predicted"] = observe_vaso(traj_final)[::subsample][:n]

    return result


def fit_balloon_bold_batch(
    bold_data: Float[Array, "R N"],
    stimulus: Float[Array, "T"],
    tr: float,
    dt: float = 0.01,
    balloon_params: BalloonParams | None = None,
    bold_params: BOLDParams | None = None,
    fit_names: tuple[str, ...] = _BALLOON_DEFAULT_FIT,
    n_steps: int = 500,
    learning_rate: float = 5.0,
) -> dict[str, Float[Array, "R"]]:
    """Fit Balloon params independently for R ROIs sharing one stimulus.

    Parameters
    ----------
    bold_data : BOLD signals, shape (R, N) — R ROIs, N timepoints
    stimulus : shared stimulus, shape (T,)
    tr, dt, balloon_params, bold_params, fit_names, n_steps,
    learning_rate : forwarded to :func:`fit_balloon_bold`

    Returns
    -------
    Dict with each value shape (R,) or (R, N).
    """
    fit_fn = jax.vmap(
        lambda d: fit_balloon_bold(
            d, stimulus, tr, dt, balloon_params, bold_params,
            fit_names, n_steps, learning_rate,
        )
    )
    return fit_fn(bold_data)


# ---------------------------------------------------------------------------
# Riera NVC inverse problem
# ---------------------------------------------------------------------------

def fit_riera_bold(
    bold_data: Float[Array, "N"],
    stimulus: Float[Array, "T"],
    tr: float,
    dt: float = 0.01,
    riera_params: RieraParams | None = None,
    bold_params: BOLDParams | None = None,
    fit_names: tuple[str, ...] = _RIERA_DEFAULT_FIT,
    n_steps: int = 800,
    learning_rate: float = 2.0,
) -> dict[str, Float[Array, ""]]:
    """Fit Riera NVC parameters to a BOLD time series.

    Integrates the 8-state Riera ODE at resolution *dt*, maps the
    multi-compartment state to a BOLD signal via :func:`riera_to_balloon`
    + :func:`observe_bold`, and minimizes MSE.

    Only the parameters named in *fit_names* are optimised; the rest are
    held fixed at their values in *riera_params*.

    Parameters
    ----------
    bold_data : observed BOLD signal (fractional change), shape (N,)
    stimulus : neural stimulus sampled at *dt*, shape (T,)
    tr : BOLD repetition time (seconds)
    dt : ODE integration timestep (seconds)
    riera_params : initial RieraParams (default: literature values)
    bold_params : BOLDParams held fixed (default: 3T)
    fit_names : which RieraParams fields to optimise
        (default: c_no, kappa_no, gamma_no, tau_v, alpha_v, E0)
    n_steps : gradient descent iterations
    learning_rate : step size

    Returns
    -------
    Dict with all 15 Riera parameter values (fitted and fixed),
    plus 'bold_predicted' and 'loss'.
    """
    if riera_params is None:
        riera_params = RieraParams()
    if bold_params is None:
        bold_params = BOLDParams()

    subsample = int(round(tr / dt))
    n = bold_data.shape[0]

    # Pack fitted parameters into theta
    theta = jnp.array([float(getattr(riera_params, name)) for name in fit_names])

    # Cache the base (fixed) parameter values
    base_vals = {name: getattr(riera_params, name) for name in _RIERA_ALL_NAMES}

    def _make_params(theta):
        vals = dict(base_vals)
        for i, name in enumerate(fit_names):
            lo, hi = _RIERA_BOUNDS[name]
            vals[name] = jnp.clip(theta[i], lo, hi)
        return RieraParams(**vals)

    def loss_fn(theta):
        rp = _make_params(theta)
        _, traj = solve_riera(rp, stimulus, dt=dt)

        # Map Riera multi-compartment state to BOLD observation
        v, q = riera_to_balloon(traj)
        pseudo = BalloonState(
            s=jnp.zeros_like(v), f=traj.f_a, v=v, q=q,
        )
        bold_pred = observe_bold(pseudo, bold_params)[::subsample][:n]
        return jnp.mean((bold_pred - bold_data) ** 2)

    @jax.jit
    def step(th, vel):
        g = jax.grad(loss_fn)(th)
        safe = jnp.all(jnp.isfinite(g))
        vel = jnp.where(safe, 0.9 * vel + 0.1 * g, vel)
        th = jnp.where(safe, th - learning_rate * vel, th)
        return th, vel

    velocity = jnp.zeros_like(theta)
    for _ in range(n_steps):
        theta, velocity = step(theta, velocity)

    # Final forward pass
    rp_final = _make_params(theta)
    _, traj_final = solve_riera(rp_final, stimulus, dt=dt)
    v, q = riera_to_balloon(traj_final)
    pseudo = BalloonState(
        s=jnp.zeros_like(v), f=traj_final.f_a, v=v, q=q,
    )
    bold_pred = observe_bold(pseudo, bold_params)[::subsample][:n]
    loss = jnp.mean((bold_pred - bold_data) ** 2)

    # Return all parameter values (fitted ones updated, fixed ones as-is)
    result: dict[str, Float[Array, "..."]] = {}
    for name in _RIERA_ALL_NAMES:
        result[name] = getattr(rp_final, name)
    result["bold_predicted"] = bold_pred
    result["loss"] = loss

    return result
