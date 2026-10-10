"""Innovation-based maximum likelihood for state-space models.

The filter's log-likelihood is already the exact one-step-ahead
predictive likelihood of the data under the linearized model — the
"innovation likelihood" of the Ozaki/Riera line of work — so parameter
estimation is just maximising the output of
:func:`vpjax.statespace.ll_filter` over its parameters.  JAX
differentiates straight through the filter recursion, which is what
makes this practical: no finite differences, no EM, and the same code
path for the gradient as for the likelihood.

Why this rather than fitting a forward model to the data by least
squares, as :mod:`vpjax.hemodynamics.inversion` does: a least-squares
fit assumes the model explains the data up to observation noise, with no
allowance for the dynamics themselves being driven by something
unmodelled.  Physiological recordings are not like that — spontaneous
fluctuations, drift, and arousal all enter the state, not the sensor.
Allowing process noise (``Q``) absorbs that into the state estimate
instead of biasing the parameters, and the innovation likelihood is the
correct objective once it does.

What this cannot do: ``Q`` and ``R`` trade off against each other, and
estimating both freely alongside the dynamic parameters is usually not
identifiable from a single recording.  Fix ``R`` from a measurement the
noise can be read off directly, estimate ``Q``, and check the result
against :func:`vpjax.identifiability.check_local_identifiability`.

References
----------
Ozaki T (1994) in "Handbook of Statistics" 10:63-87
    "The local linearization filter with application to nonlinear system
    identification"
Riera JJ et al. (2004) NeuroImage 21:547-567
Schweppe FC (1965) IEEE Trans Inform Theory 11:61-70
    Innovation-form likelihood for Gaussian state-space models
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax.statespace.filters import FilterResult, ll_filter
from vpjax.statespace.multirate import ObservationGrid


def innovation_log_likelihood(
    build: Callable[[Float[Array, "P"]], dict],
    log_theta: Float[Array, "P"],
    grid: ObservationGrid,
) -> Float[Array, ""]:
    """Log-likelihood of *grid* under the model that *build* returns.

    Parameters
    ----------
    build     : maps a parameter vector to a dict of keyword arguments
                for :func:`ll_filter` — at minimum ``f``, ``h``, ``Q``,
                ``m0``, ``P0``, and optionally ``r_diag``, ``args`` and
                ``inputs``.  Anything it does not supply is taken from
                *grid*.
    log_theta : parameters on the log scale.  Variances, rate constants
                and time constants are positive and span orders of
                magnitude; estimating their logarithms enforces
                positivity without a bound whose gradient vanishes.
    grid      : the data

    Returns
    -------
    Scalar log-likelihood.
    """
    spec = dict(build(jnp.exp(log_theta)))
    spec.setdefault("r_diag", grid.r_diag)
    return ll_filter(t=grid.t, y=grid.y, mask=grid.mask, **spec).log_likelihood


def fit_statespace(
    build: Callable[[Float[Array, "P"]], dict],
    grid: ObservationGrid,
    init: Float[Array, "P"],
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_steps: int = 256,
    throw: bool = False,
    restarts: int = 0,
    restart_scale: float = 0.5,
    seed: int = 0,
) -> dict[str, Float[Array, "..."]]:
    """Maximise the innovation likelihood with respect to the parameters.

    Uses Optimistix BFGS on the negative mean log-likelihood per
    observation.  Dividing by the observation count keeps the
    convergence tolerances interpretable across recordings of different
    length, and keeps the objective in a range where the default BFGS
    line search behaves.

    **Use ``restarts``.** Neurovascular likelihoods have ridges — in the
    Balloon model the signal decay ``kappa`` and the transit time
    ``tau`` both set the width of the impulse response, so they trade
    off against each other and a gradient method started off the ridge
    slides along it to a local optimum. The basins are usually far apart
    in likelihood, so they are easy to tell apart once found; the
    difficulty is finding them. A fit reported from a single start
    should not be trusted, and ``success`` being ``True`` means the
    optimizer converged, not that it converged to the right place.

    Parameters
    ----------
    build : as in :func:`innovation_log_likelihood`
    grid  : the data
    init  : starting parameters on the *natural* scale, positive
    rtol, atol, max_steps, throw : passed to Optimistix.  ``throw=False``
            returns the best iterate rather than raising, so a failure
            to converge shows up in ``success`` instead of aborting.
    restarts : number of extra starts, drawn log-normally around *init*.
            The returned fit is the one with the highest likelihood.
    restart_scale : SD of the log-scale perturbation for the restarts.
            ``0.5`` spans roughly a factor of e either side of *init*.
    seed  : RNG seed for the restart perturbations

    Returns
    -------
    dict with ``theta`` (natural scale), ``log_theta``,
    ``log_likelihood``, ``n_obs``, ``success``, ``filter`` — the
    :class:`FilterResult` at the estimate, so the residuals can be
    checked without re-running — and ``all_log_likelihoods``, one per
    start, whose spread is the honest measure of how multi-modal the
    surface turned out to be.
    """
    import optimistix as optx

    init = jnp.asarray(init, dtype=float)
    if jnp.any(init <= 0):
        raise ValueError("init must be strictly positive (it is log-transformed)")
    if restarts < 0:
        raise ValueError("restarts must be non-negative")
    n_obs = grid.mask.sum()

    def objective(log_theta, args):
        return -innovation_log_likelihood(build, log_theta, grid) / n_obs

    # BestSoFarMinimiser is not a refinement here, it is a correctness
    # requirement: with throw=False Optimistix returns the *last* iterate,
    # which after a failed line search can be far worse than the starting
    # point. Wrapping the solver makes a non-converged fit return the best
    # point it actually saw.
    solver = optx.BestSoFarMinimiser(optx.BFGS(rtol=rtol, atol=atol))

    log_init = jnp.log(init)
    starts = [log_init]
    if restarts:
        key = jax.random.PRNGKey(seed)
        starts.extend(
            log_init
            + restart_scale * jax.random.normal(k, log_init.shape)
            for k in jax.random.split(key, restarts)
        )

    solutions = [
        optx.minimise(objective, solver, s0, max_steps=max_steps, throw=throw)
        for s0 in starts
    ]
    lls = jnp.stack([
        innovation_log_likelihood(build, sol.value, grid) for sol in solutions
    ])
    best = int(jnp.argmax(lls))
    sol = solutions[best]

    spec = dict(build(jnp.exp(sol.value)))
    spec.setdefault("r_diag", grid.r_diag)
    result = ll_filter(t=grid.t, y=grid.y, mask=grid.mask, **spec)

    return {
        "theta": jnp.exp(sol.value),
        "log_theta": sol.value,
        "log_likelihood": result.log_likelihood,
        "n_obs": n_obs,
        "success": sol.result == optx.RESULTS.successful,
        "filter": result,
        "all_log_likelihoods": lls,
        "best_start": best,
    }


def residual_diagnostics(
    result: FilterResult,
    channels: tuple[str, ...] | None = None,
) -> dict[str, Float[Array, "..."]]:
    """Whiteness checks on the standardized innovations.

    Under a correct model the standardized innovations are independent
    standard normals, so these numbers say whether the model is
    adequate -- which the likelihood alone does not. Two diagnoses:

    * ``variance`` far from 1 means the noise covariances are
      mis-scaled: above 1, ``Q`` or ``R`` is too small for the data.
    * ``lag1`` far from 0 means the *dynamics* are wrong. The filter
      cannot whiten what the state equation does not describe, so
      structured residuals point at the model, not the noise.

    Reported **per channel** as well as pooled, because pooling across
    channels with very different sample counts hides exactly the
    failure worth catching: a fast channel with thousands of samples
    will dominate the pooled variance and leave a badly fitted slow
    channel invisible.

    The lag-1 correlation pairs each sample with the next *present*
    sample of the same channel, not with the next grid row. On a grid
    subdivided by ``max_dt``, or wherever channels are sampled at
    different rates, adjacent rows of one channel are generally not
    both present, and a row-adjacent lag-1 would be computed over an
    empty set and silently report zero.

    Parameters
    ----------
    result   : a :class:`FilterResult`
    channels : optional channel names, as carried by
               :class:`~vpjax.statespace.ObservationGrid`. When given,
               the per-channel entries are keyed by name instead of index.

    Returns
    -------
    dict with pooled ``mean``, ``variance``, ``lag1``, ``n``, and a
    ``per_channel`` mapping to the same quantities for each channel.
    """
    resid = result.residual
    present = (resid != 0.0).astype(float)

    def one_channel(r, w):
        n = jnp.maximum(jnp.sum(w), 1.0)
        mean = jnp.sum(r * w) / n
        var = jnp.sum(w * r**2) / n - mean**2

        # Pair each present sample with the next present sample of this
        # channel, carrying the last seen value across absent rows.  The
        # products are of mean-removed residuals, so a channel with an
        # offset reports its autocorrelation rather than offset²/var.
        def body(carry, inputs):
            last, has_last = carry
            r_i, w_i = inputs
            r_i = r_i - mean
            paired = w_i * has_last
            contrib = paired * last * r_i
            last = jnp.where(w_i > 0, r_i, last)
            has_last = jnp.maximum(has_last, w_i)
            return (last, has_last), (contrib, paired)

        _, (prod, paired) = jax.lax.scan(
            body, (jnp.array(0.0), jnp.array(0.0)), (r, w)
        )
        npair = jnp.sum(paired)
        lag1 = jnp.sum(prod) / jnp.maximum(npair, 1.0) / jnp.maximum(var, 1e-12)
        return mean, var, lag1, jnp.sum(w), npair

    means, vars_, lag1s, ns, npairs = jax.vmap(
        one_channel, in_axes=(1, 1)
    )(resid, present)

    n_total = jnp.maximum(jnp.sum(present), 1.0)
    pooled_mean = jnp.sum(resid * present) / n_total
    pooled_var = jnp.sum(present * resid**2) / n_total - pooled_mean**2

    m = resid.shape[1]
    keys = channels if channels is not None else tuple(range(m))
    if len(keys) != m:
        raise ValueError(
            f"got {len(keys)} channel names for {m} residual columns"
        )

    return {
        "mean": pooled_mean,
        "variance": pooled_var,
        "lag1": lag1s[jnp.argmax(ns)],
        "n": jnp.sum(present),
        "per_channel": {
            key: {
                "mean": means[i],
                "variance": vars_[i],
                "lag1": lag1s[i],
                "n": ns[i],
                "n_pairs": npairs[i],
            }
            for i, key in enumerate(keys)
        },
    }


def parameter_uncertainty(
    build: Callable[[Float[Array, "P"]], dict],
    grid: ObservationGrid,
    log_theta: Float[Array, "P"],
) -> dict[str, Float[Array, "..."]]:
    """Asymptotic uncertainty and an identifiability verdict, from curvature.

    Computes the observed Fisher information -- the negative Hessian of
    the log-likelihood at *log_theta* -- and inverts it for standard
    errors. Because the parameters are on the log scale, the standard
    errors are directly interpretable as *relative* uncertainties: 0.1
    means roughly 10%.

    **This is the check that innovation whiteness cannot do.** A
    state-space model can whiten its residuals with badly wrong
    parameters whenever two of them trade off: a fast large-amplitude
    drive through a slow hemodynamic filter looks, to a slow modality,
    exactly like a slow small drive through a fast filter. The
    likelihood is then flat along that trade-off, the fit is perfectly
    calibrated, and the estimates are meaningless. A flat direction
    shows up here as a near-zero Hessian eigenvalue and a large
    standard error, and unlike parameter recovery it needs no ground
    truth, so it works on real recordings.

    Returns
    -------
    dict with

    ``standard_error``   relative SE per parameter (log scale).  Always
                         finite, from a pseudo-inverse; read it together
                         with ``positive_definite``, since at a
                         non-maximum it is not a standard error at all
    ``eigenvalues``      eigenvalues of the information matrix, ascending
    ``min_eigenvalue``   the smallest, signed; at or below zero means
                         this point is not a maximum
    ``curvature_ratio``  smallest signed eigenvalue over largest
                         absolute one.  The headline number: order one
                         is well conditioned, near zero is a flat
                         direction, negative is not a maximum
    ``condition_number`` largest over smallest *absolute* eigenvalue.
                         Reported for familiarity, but it cannot see a
                         non-concave direction -- use
                         ``curvature_ratio``
    ``correlation``      parameter correlation matrix implied by the
                         inverse information, which names *which*
                         parameters are trading off.  An entry outside
                         [-1, 1] is itself diagnostic: it can only
                         happen when the information is not positive
                         definite
    ``positive_definite`` whether every eigenvalue is above zero
    ``identifiable``     True when the Hessian is positive definite,
                         ``curvature_ratio`` exceeds ``1e-6``, and every
                         SE is below ``0.5`` (50% relative). A coarse
                         screen, not a proof -- structural
                         identifiability is what
                         :mod:`vpjax.identifiability_symbolic` is for.

    Notes
    -----
    The Hessian is taken of the *total* log-likelihood, not the mean,
    so the standard errors shrink with the amount of data as they
    should. It is a dense ``P x P`` Hessian through the whole filter
    recursion, so cost grows with the square of the parameter count.
    """
    def total_ll(lt):
        return innovation_log_likelihood(build, lt, grid)

    log_theta = jnp.asarray(log_theta, dtype=float)
    information = -jax.hessian(total_ll)(log_theta)
    information = 0.5 * (information + information.T)

    eigs = jnp.linalg.eigvalsh(information)
    cond = jnp.max(jnp.abs(eigs)) / jnp.maximum(jnp.min(jnp.abs(eigs)), 1e-300)
    # The signed ratio, not the condition number, is what detects a flat
    # direction. Two parameters entering only through their sum give
    # eigenvalues of order +b and -a, whose *absolute* ratio is near one
    # -- so the condition number looks healthy while the Hessian is not
    # even concave. The smallest signed eigenvalue relative to the
    # largest catches both that case and a genuinely flat one.
    curvature_ratio = jnp.min(eigs) / jnp.max(jnp.abs(eigs))

    # Always pseudo-invert, rather than returning infinities when the
    # information is singular: the magnitudes and the correlation
    # structure are what name the offending parameters, and they are
    # most wanted in exactly the degenerate case. A non-positive-definite
    # Hessian means this point is not a maximum, so the standard errors
    # do not have their usual meaning there -- hence the separate
    # ``positive_definite`` flag rather than silently finite numbers.
    pd = jnp.all(eigs > 0)
    cov = jnp.linalg.pinv(information)
    se = jnp.sqrt(jnp.abs(jnp.diag(cov)))
    outer = se[:, None] * se[None, :]
    corr = jnp.where(outer > 0, cov / outer, jnp.nan)

    return {
        "standard_error": se,
        "eigenvalues": eigs,
        "min_eigenvalue": jnp.min(eigs),
        "curvature_ratio": curvature_ratio,
        "condition_number": cond,
        "correlation": corr,
        "positive_definite": pd,
        "identifiable": pd & jnp.all(se < 0.5) & (curvature_ratio > 1e-6),
    }


def profile_likelihood(
    build: Callable[[Float[Array, "P"]], dict],
    grid: ObservationGrid,
    theta: Float[Array, "P"],
    index: int,
    values: Float[Array, "G"],
) -> Float[Array, "G"]:
    """Log-likelihood along one parameter, the others held fixed.

    A conditional slice, not a true profile likelihood — the other
    parameters are *not* re-optimised at each point, so the curve is
    narrower than a real profile and overstates precision. It is still
    the quickest way to see a flat direction, which is the failure mode
    that matters: a parameter whose slice is flat is not estimable from
    the data, whatever the optimizer reports for it.
    """
    log_theta = jnp.log(jnp.asarray(theta, dtype=float))

    def one(v):
        lt = log_theta.at[index].set(jnp.log(v))
        return innovation_log_likelihood(build, lt, grid)

    return jax.lax.map(one, jnp.asarray(values, dtype=float))
