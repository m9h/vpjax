"""Nonlinear mixed-effects (population) PK model.

A population model treats each subject's parameters as a draw from a
distribution rather than as free quantities:

    theta_i = theta_pop * exp(eta_i),    eta_i ~ N(0, Omega)

This log-normal form is the pharmacometrics convention and is not
arbitrary: clearances and volumes are positive and right-skewed across
subjects, and on the log scale a covariate effect becomes an additive
shift, so allometric scaling is linear in the parameters.

Two things in the implementation are worth stating explicitly because
they decide whether sampling works at all:

* **Non-centred parameterisation.** The random effects are sampled as
  standard normals and scaled, not sampled from ``N(0, omega)``
  directly.  Centred hierarchies have a funnel geometry that gradient
  samplers negotiate badly whenever the data are sparse — which, for
  population PK, is always.
* **An LKJ prior on the correlation matrix** rather than independent
  random effects.  Clearance and volume are correlated across subjects;
  forcing ``Omega`` diagonal pushes that correlation into the residual
  error and biases the variance estimates.

This module is optional: it imports NumPyro lazily so that the rest of
the package works without it.  Install with ``pip install numpyro``.

Cost ordering, cheapest first — all three are worth running, in order:

1. :func:`vpjax.pharmacokinetics.inversion.fit_pk_batch` for per-subject
   estimates, as a structural-model check.
2. :func:`fit_population_map` — maximum a posteriori, a few seconds.
   This is the closest analogue to the FOCE estimation that NONMEM and
   Pumas use by default, and it is the one to report if sampling is out
   of reach.
3. :func:`fit_population_nuts` — full posterior, minutes to hours.

References
----------
Sheiner LB, Beal SL (1980) J Pharmacokinet Biopharm 8:553-571
    "Evaluation of methods for estimating population pharmacokinetic
    parameters"
Betancourt M, Girolami M (2015) in "Current Trends in Bayesian
    Methodology with Applications" — non-centred hierarchical geometry
Lewandowski D et al. (2009) J Multivar Anal 100:1989-2001
    "Generating random correlation matrices based on vines" (LKJ)
Margossian CC et al. (2022) CPT Pharmacometrics Syst Pharmacol 11:1452-1466
    "Flexible and efficient Bayesian pharmacometrics modeling using Stan
    and Torsten"
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax.pharmacokinetics.compartments import (
    ONE_COMPARTMENT,
    PKParams,
    PKStructure,
    solve_linear_pk,
)
from vpjax.pharmacokinetics.dosing import DosingRegimen, Schedule, build_schedule


#: Default prior medians, as plain Python floats.  They must not be JAX
#: arrays: the model is traced by NumPyro's inference machinery, and an
#: array built during tracing is a tracer, which cannot be converted to a
#: concrete number.  These are generic adult-human orders of magnitude in
#: litres and litres/hour — replace them with values from the compound
#: being studied before fitting anything.
DEFAULT_PRIOR_MEDIAN: dict[str, float] = {
    "CL": 10.0,
    "V1": 10.0,
    "Q2": 5.0,
    "V2": 30.0,
    "Q3": 1.0,
    "V3": 100.0,
    "ka": 1.0,
    "F": 1.0,
}


def _require_numpyro():
    try:
        import numpyro
        import numpyro.distributions as dist

        return numpyro, dist
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The population PK model needs NumPyro. "
            "Install it with `pip install numpyro`."
        ) from exc


def population_model(
    conc_obs: Float[Array, "S N"] | None,
    regimen: DosingRegimen,
    schedule: Schedule,
    structure: PKStructure = ONE_COMPARTMENT,
    fit_names: tuple[str, ...] = ("CL", "V1"),
    prior_median: dict[str, float] | None = None,
    prior_log_sd: float = 1.0,
    covariates: Float[Array, "S C"] | None = None,
) -> None:
    """NumPyro model for a population PK fit.

    Mirrors the block structure a Pumas or NONMEM model would use:
    population parameters and variance components, then per-subject
    random effects, then individual parameters, then the dynamics, then
    the observation model.

    Parameters
    ----------
    conc_obs     : (S, N) concentrations; ``None`` draws from the prior
                   predictive, which is how to check that the priors
                   imply plausible curves before seeing any data
    regimen      : shared dosing regimen
    schedule     : shared integration grid, from :func:`build_schedule`
    structure    : PKStructure
    fit_names    : which parameters get population and random effects.
                   Everything else stays at its :class:`PKParams` default.
    prior_median : prior median per fitted parameter; defaults to the
                   :class:`PKParams` defaults
    prior_log_sd : prior standard deviation on the log population
                   parameters — ``1.0`` is weakly informative, spanning
                   roughly a factor of 7 either side of the median
    covariates   : (S, C) subject covariates, entered as additive
                   effects on the log parameters.  Centre and scale them
                   before passing them in; the prior on the covariate
                   coefficients assumes unit-scale predictors.
    """
    numpyro, dist = _require_numpyro()

    n_subj = (
        conc_obs.shape[0]
        if conc_obs is not None
        else (covariates.shape[0] if covariates is not None else 1)
    )
    p = len(fit_names)
    if prior_median is None:
        prior_median = {n: DEFAULT_PRIOR_MEDIAN[n] for n in fit_names}

    # --- population parameters (@param in Pumas) -------------------------
    log_median = jnp.log(
        jnp.stack([jnp.asarray(prior_median[n], dtype=float) for n in fit_names])
    )
    log_theta_pop = numpyro.sample(
        "log_theta_pop",
        dist.Normal(log_median, prior_log_sd).to_event(1),
    )

    # --- variance components (@random) -----------------------------------
    omega = numpyro.sample(
        "omega", dist.HalfNormal(jnp.full((p,), 0.5)).to_event(1)
    )
    if p > 1:
        chol_corr = numpyro.sample("chol_corr", dist.LKJCholesky(p, concentration=2.0))
        chol_cov = omega[:, None] * chol_corr
    else:
        chol_cov = jnp.diag(omega)

    sigma_prop = numpyro.sample("sigma_prop", dist.HalfNormal(0.3))
    sigma_add = numpyro.sample("sigma_add", dist.HalfNormal(0.1))

    # --- covariate coefficients ------------------------------------------
    if covariates is not None:
        beta = numpyro.sample(
            "beta",
            dist.Normal(jnp.zeros((p, covariates.shape[1])), 0.5).to_event(2),
        )
        cov_shift = covariates @ beta.T  # (S, P)
    else:
        cov_shift = jnp.zeros((n_subj, p))

    with numpyro.plate("subject", n_subj, dim=-1):
        # Non-centred: sample standard normals, then scale by the Cholesky
        # factor.  Sampling eta ~ N(0, Omega) directly creates the funnel.
        z = numpyro.sample(
            "z", dist.Normal(jnp.zeros((p,)), 1.0).to_event(1)
        )
    eta = z @ chol_cov.T  # (S, P)

    # --- individual parameters (@pre) ------------------------------------
    log_theta_i = log_theta_pop[None, :] + eta + cov_shift
    theta_i = jnp.exp(log_theta_i)
    numpyro.deterministic("theta_i", theta_i)

    # --- dynamics (@dynamics) --------------------------------------------
    def one_subject(theta_row):
        kwargs = {n: theta_row[i] for i, n in enumerate(fit_names)}
        params = PKParams(**kwargs)
        amounts = solve_linear_pk(params, regimen, schedule, structure)
        return amounts[schedule.obs_idx, structure.central] / params.V1

    pred = jax.vmap(one_subject)(theta_i)  # (S, N)
    numpyro.deterministic("pred", pred)

    # --- observation model (@derived) ------------------------------------
    sd = jnp.sqrt(sigma_add**2 + (sigma_prop * pred) ** 2)
    numpyro.sample("obs", dist.Normal(pred, sd), obs=conc_obs)


def fit_population_nuts(
    conc_obs: Float[Array, "S N"],
    t_obs: Float[Array, "N"],
    regimen: DosingRegimen,
    structure: PKStructure = ONE_COMPARTMENT,
    fit_names: tuple[str, ...] = ("CL", "V1"),
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 4,
    seed: int = 0,
    target_accept_prob: float = 0.9,
    **model_kwargs: Any,
):
    """Sample the population posterior with NUTS.

    Returns the ``MCMC`` object, not a summary: inspect
    ``mcmc.print_summary()`` and check that ``r_hat`` is below 1.01 and
    the effective sample size is adequate before using the estimates.
    A high divergence count usually means the structural model is
    over-parameterised for the data rather than that the sampler needs
    more steps.
    """
    numpyro, _ = _require_numpyro()
    from numpyro.infer import MCMC, NUTS

    schedule = build_schedule(regimen, t_obs)
    kernel = NUTS(population_model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        conc_obs=conc_obs,
        regimen=regimen,
        schedule=schedule,
        structure=structure,
        fit_names=fit_names,
        **model_kwargs,
    )
    return mcmc


def fit_population_map(
    conc_obs: Float[Array, "S N"],
    t_obs: Float[Array, "N"],
    regimen: DosingRegimen,
    structure: PKStructure = ONE_COMPARTMENT,
    fit_names: tuple[str, ...] = ("CL", "V1"),
    n_steps: int = 2000,
    learning_rate: float = 0.05,
    seed: int = 0,
    **model_kwargs: Any,
) -> dict[str, Array]:
    """Maximum a posteriori population fit by stochastic optimisation.

    The practical substitute for NUTS: same model, point estimate, and
    fast enough to run inside a simulation study. It gives no
    uncertainty, so use it to choose a structural model and then sample
    the model you settled on.
    """
    numpyro, _ = _require_numpyro()
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta
    import optax

    schedule = build_schedule(regimen, t_obs)
    guide = AutoDelta(population_model)
    svi = SVI(population_model, guide, optax.adam(learning_rate), Trace_ELBO())
    result = svi.run(
        jax.random.PRNGKey(seed),
        n_steps,
        conc_obs=conc_obs,
        regimen=regimen,
        schedule=schedule,
        structure=structure,
        fit_names=fit_names,
        progress_bar=False,
        **model_kwargs,
    )
    return {"params": result.params, "losses": result.losses}
