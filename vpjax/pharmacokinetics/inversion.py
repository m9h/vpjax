"""Single-subject PK parameter estimation.

Fits a compartment model to one concentration time series by weighted
nonlinear least squares, using Optimistix's Levenberg-Marquardt solver.
LM is the right tool here rather than the hand-rolled gradient descent
used elsewhere in vpjax: PK residuals are a genuine least-squares
problem with a well-conditioned Gauss-Newton structure once the
parameters are on the log scale, and LM converges in tens of
evaluations where plain descent needs hundreds.

Parameters are estimated as logarithms.  Clearances and volumes are
strictly positive and vary over orders of magnitude between subjects,
so the log scale both enforces positivity without clipping — which
would zero the gradient at the bound, as the Balloon fits in
:mod:`vpjax.hemodynamics.inversion` have to work around — and makes the
curvature roughly isotropic.

A single-subject fit is the right first step but not the end: with
sparse sampling the individual parameters are weakly identified, and the
population model in :mod:`vpjax.pharmacokinetics.nlme` is what makes
them estimable by sharing strength across subjects.  Run
:func:`vpjax.identifiability.check_local_identifiability` on the chosen
structure before trusting any individual estimate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax.pharmacokinetics.compartments import (
    ONE_COMPARTMENT,
    PKParams,
    PKStructure,
    concentration,
    secondary_parameters,
)
from vpjax.pharmacokinetics.dosing import DosingRegimen, build_schedule
from vpjax.pharmacokinetics.error import (
    ResidualErrorParams,
    log_likelihood,
    weighted_residuals,
)

_PK_ALL_NAMES: tuple[str, ...] = ("CL", "V1", "Q2", "V2", "Q3", "V3", "ka", "F")

#: Parameters to fit for each structure, in the absence of an explicit choice.
#: ``F`` is omitted throughout: bioavailability is not identifiable from
#: extravascular data alone, only the ratios ``CL/F`` and ``V/F`` are.
_PK_DEFAULT_FIT: dict[tuple[int, bool], tuple[str, ...]] = {
    (0, False): ("CL", "V1"),
    (1, False): ("CL", "V1", "Q2", "V2"),
    (2, False): ("CL", "V1", "Q2", "V2", "Q3", "V3"),
    (0, True): ("CL", "V1", "ka"),
    (1, True): ("CL", "V1", "Q2", "V2", "ka"),
}


def default_fit_names(structure: PKStructure) -> tuple[str, ...]:
    """Identifiable parameter set for a structure, given dense sampling."""
    return _PK_DEFAULT_FIT[(structure.n_peripheral, structure.absorption)]


def _make_pk_params(
    log_theta: Float[Array, "P"],
    base: PKParams,
    fit_names: tuple[str, ...],
) -> PKParams:
    """Rebuild PKParams, replacing fitted fields with ``exp(log_theta)``."""
    vals = {name: getattr(base, name) for name in _PK_ALL_NAMES}
    for i, name in enumerate(fit_names):
        vals[name] = jnp.exp(log_theta[i])
    return PKParams(**vals)


def fit_pk_subject(
    conc_obs: Float[Array, "N"],
    t_obs: Float[Array, "N"],
    regimen: DosingRegimen,
    structure: PKStructure = ONE_COMPARTMENT,
    init_params: PKParams | None = None,
    error_params: ResidualErrorParams | None = None,
    fit_names: tuple[str, ...] | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 256,
    throw: bool = False,
) -> dict[str, Float[Array, "..."]]:
    """Fit compartment parameters to one subject's concentration series.

    Parameters
    ----------
    conc_obs     : measured concentrations, shape (N,)
    t_obs        : measurement times, shape (N,)
    regimen      : the dosing regimen that produced the data
    structure    : PKStructure — number of compartments and absorption
    init_params  : starting values; defaults are generic and usually
                   adequate after :func:`initial_guess` has been applied
    error_params : residual error model used to weight the residuals
    fit_names    : parameters to estimate; defaults to
                   :func:`default_fit_names` for the structure
    rtol, atol, max_steps, throw : passed to Optimistix.  ``throw=False``
                   returns the best iterate instead of raising when the
                   solver hits *max_steps*, so a failed fit is visible
                   in the returned ``success`` flag rather than fatal.

    Returns
    -------
    dict with one entry per parameter in ``_PK_ALL_NAMES``, plus
    ``conc_predicted``, ``residuals``, ``loss`` (sum of squared weighted
    residuals), ``log_likelihood``, ``success``, and the derived
    quantities from :func:`secondary_parameters`.
    """
    import optimistix as optx

    if init_params is None:
        init_params = PKParams()
    if error_params is None:
        error_params = ResidualErrorParams()
    if fit_names is None:
        fit_names = default_fit_names(structure)

    schedule = build_schedule(regimen, t_obs)
    log_theta0 = jnp.log(
        jnp.stack([jnp.asarray(getattr(init_params, n), dtype=float) for n in fit_names])
    )

    def residual_fn(log_theta, args):
        params = _make_pk_params(log_theta, init_params, fit_names)
        pred = concentration(params, regimen, t_obs, structure, schedule)
        return weighted_residuals(pred, conc_obs, error_params)

    # Wrapped so that a fit stopped by max_steps returns the best iterate
    # rather than the last one, which throw=False would otherwise hand back.
    solver = optx.BestSoFarLeastSquares(
        optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    )
    sol = optx.least_squares(
        residual_fn,
        solver,
        log_theta0,
        max_steps=max_steps,
        throw=throw,
    )

    params = _make_pk_params(sol.value, init_params, fit_names)
    pred = concentration(params, regimen, t_obs, structure, schedule)
    resid = weighted_residuals(pred, conc_obs, error_params)

    result: dict[str, Float[Array, "..."]] = {
        name: getattr(params, name) for name in _PK_ALL_NAMES
    }
    result["conc_predicted"] = pred
    result["residuals"] = resid
    result["loss"] = jnp.sum(resid**2)
    result["log_likelihood"] = log_likelihood(pred, conc_obs, error_params)
    result["success"] = sol.result == optx.RESULTS.successful
    result.update(secondary_parameters(params, structure))
    return result


def initial_guess(
    conc_obs: Float[Array, "N"],
    t_obs: Float[Array, "N"],
    dose: float,
) -> PKParams:
    """Crude non-compartmental starting values for an IV bolus fit.

    ``V1`` comes from back-extrapolating to the first observed
    concentration and ``CL`` from dose over the trapezoidal AUC, with
    the terminal phase extrapolated by the log-linear slope of the last
    three points.  These are rough — the point is only to start LM in
    the right order of magnitude, which for PK is most of the battle.

    Peripheral and absorption parameters are left at the class defaults,
    since no non-compartmental analogue of them exists.
    """
    c = jnp.asarray(conc_obs)
    t = jnp.asarray(t_obs)

    v1 = dose / jnp.maximum(c[0], 1e-12)

    auc = jnp.trapezoid(c, t)
    # Extrapolate the terminal tail using the slope of the last 3 points.
    tail_t, tail_c = t[-3:], jnp.maximum(c[-3:], 1e-12)
    slope = jnp.polyfit(tail_t, jnp.log(tail_c), 1)[0]
    lam_z = jnp.maximum(-slope, 1e-6)
    auc_total = auc + c[-1] / lam_z

    cl = dose / jnp.maximum(auc_total, 1e-12)
    return PKParams(CL=cl, V1=v1)


def fit_pk_batch(
    conc_obs: Float[Array, "S N"],
    t_obs: Float[Array, "N"],
    regimen: DosingRegimen,
    structure: PKStructure = ONE_COMPARTMENT,
    **kwargs,
) -> dict[str, Float[Array, "S"]]:
    """Fit S subjects independently on a shared time grid and regimen.

    Independent per-subject fits ("naive pooled" individual estimates)
    are a diagnostic, not a population analysis: they ignore the
    hierarchy and so overstate between-subject variability. Use them to
    check that the structural model can describe each subject at all
    before moving to :mod:`vpjax.pharmacokinetics.nlme`.
    """
    results = [
        fit_pk_subject(c, t_obs, regimen, structure, **kwargs) for c in conc_obs
    ]
    keys = results[0].keys()
    return {k: jnp.stack([r[k] for r in results]) for k in keys}
