"""Pharmacokinetic and pharmacodynamic models.

Supplies the top of the causal chain the rest of vpjax models: an
administered dose becomes a plasma concentration, a plasma concentration
becomes an effect-site concentration and a receptor occupancy, and that
occupancy becomes a drive entering both the neural models in vbjax and
the vascular models here.  Fitting that chain end to end is what lets a
drug study separate a change in neural activity from a change in
vascular tone, rather than reporting the sum of the two as a BOLD
effect.

Layout
------
``dosing``       dose events and the integration grid they imply
``compartments`` linear models solved in closed form; nonlinear via Diffrax
``occupancy``    effect-site lag, Hill occupancy, regional weighting
``error``        residual error models for concentration data
``inversion``    single-subject estimation (Optimistix Levenberg-Marquardt)
``nlme``         population model (NumPyro; optional dependency)

Enable double precision before fitting real data::

    import jax
    jax.config.update("jax_enable_x64", True)

Concentration curves span several orders of magnitude between the peak
and the terminal phase, and in float32 the closed-form solutions are
only accurate to about 1e-7 relative — enough to limit how tightly a
terminal slope can be estimated.

The structural conventions — clearance-and-volume parameterisation,
log-normal between-subject variability, combined residual error, and
closed-form solutions for linear disposition — follow NONMEM and Pumas
deliberately, so that a fit here can be cross-checked against an
established estimator.  That check is not optional: a PK model with a
misplaced rate constant or a mis-scaled volume still produces a
plausible-looking curve, so agreement with an independent
implementation is the only cheap way to know the model is the one
intended.
"""

from vpjax.pharmacokinetics.compartments import (
    ONE_COMPARTMENT,
    ORAL_ONE_COMPARTMENT,
    ORAL_TWO_COMPARTMENT,
    THREE_COMPARTMENT,
    TWO_COMPARTMENT,
    MichaelisMentenParams,
    PKParams,
    PKStructure,
    concentration,
    disposition_rate_constants,
    rate_matrix,
    secondary_parameters,
    solve_linear_pk,
    solve_nonlinear_pk,
)
from vpjax.pharmacokinetics.dosing import (
    DosingRegimen,
    Schedule,
    bolus,
    build_schedule,
    combine,
    dose_terms,
    infusion,
    repeated,
)
from vpjax.pharmacokinetics.error import (
    ResidualErrorParams,
    log_likelihood,
    log_scale_residuals,
    residual_sd,
    weighted_residuals,
)
from vpjax.pharmacokinetics.inversion import (
    default_fit_names,
    fit_pk_batch,
    fit_pk_subject,
    initial_guess,
)
from vpjax.pharmacokinetics.occupancy import (
    EffectSiteParams,
    HillParams,
    competitive_drive,
    effect_site_concentration,
    hill_effect,
    occupancy,
    regional_drive,
)

__all__ = [
    # Dosing
    "DosingRegimen",
    "Schedule",
    "bolus",
    "infusion",
    "repeated",
    "combine",
    "build_schedule",
    "dose_terms",
    # Compartment models
    "PKParams",
    "PKStructure",
    "MichaelisMentenParams",
    "ONE_COMPARTMENT",
    "TWO_COMPARTMENT",
    "THREE_COMPARTMENT",
    "ORAL_ONE_COMPARTMENT",
    "ORAL_TWO_COMPARTMENT",
    "rate_matrix",
    "disposition_rate_constants",
    "solve_linear_pk",
    "solve_nonlinear_pk",
    "concentration",
    "secondary_parameters",
    # Pharmacodynamics
    "EffectSiteParams",
    "HillParams",
    "effect_site_concentration",
    "hill_effect",
    "occupancy",
    "regional_drive",
    "competitive_drive",
    # Error models
    "ResidualErrorParams",
    "residual_sd",
    "weighted_residuals",
    "log_likelihood",
    "log_scale_residuals",
    # Estimation
    "fit_pk_subject",
    "fit_pk_batch",
    "initial_guess",
    "default_fit_names",
]
