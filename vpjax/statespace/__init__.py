"""Continuous-discrete state-space estimation.

Supplies the estimator that multimodal physiological recordings need:
one latent state, several sensors observing it at different rates, and
a nonlinear continuous-time model connecting them.

Layout
------
``propagators`` exact mean, transition and process-noise propagation
``filters``     local-linearization filter and RTS smoother
``multirate``   asynchronous streams to a masked observation grid
``estimation``  innovation-based maximum likelihood and diagnostics

The pieces compose with the rest of vpjax rather than replacing
anything. :mod:`vpjax.hemodynamics.inversion` fits a deterministic
forward model by least squares and is the right tool when the stimulus
is known and the response is driven by it. This module is for the case
where the drive is itself unknown or noisy — resting state, a drug
infusion, sleep — and so has to be estimated as part of the state.

Typical use for the EEG-fMRI problem::

    from vpjax.statespace import build_observation_grid, fit_statespace

    grid = build_observation_grid(
        {
            "eeg_delta": {"time_s": t_eeg, "values": delta_power,
                          "noise_sd": eeg_sd},
            "bold":      {"time_s": t_bold, "values": bold_rois,
                          "noise_sd": bold_sd},
        },
        max_dt=0.25,            # keep the linearization fresh between TRs
    )

    def build(theta):
        return {"f": drift, "h": observe, "Q": theta[0] * Q_shape,
                "m0": m0, "P0": P0, "args": theta}

    fit = fit_statespace(build, grid, init=jnp.array([...]))

Then check :func:`residual_diagnostics` on ``fit["filter"]`` before
reading anything off the parameters: a likelihood can always be
maximised, and only the innovations say whether the model fits.

The local linearization is a first-order approximation to a nonlinear
filter. Validate any new model on data simulated from that same model
first -- if parameters are not recovered from simulation, nothing
recovered from a recording means anything.
"""

from vpjax.statespace.estimation import (
    fit_statespace,
    innovation_log_likelihood,
    parameter_uncertainty,
    profile_likelihood,
    residual_diagnostics,
)
from vpjax.statespace.filters import (
    FilterResult,
    SmootherResult,
    ll_filter,
    rts_smoother,
    sequential_update,
)
from vpjax.statespace.multirate import (
    ObservationGrid,
    build_observation_grid,
    regular_grid,
)
from vpjax.statespace.propagators import (
    ll_mean_step,
    ll_propagate,
    symmetrize,
    van_loan_noise,
)

__all__ = [
    # Propagators
    "ll_mean_step",
    "ll_propagate",
    "van_loan_noise",
    "symmetrize",
    # Filtering
    "FilterResult",
    "SmootherResult",
    "ll_filter",
    "rts_smoother",
    "sequential_update",
    # Observation grids
    "ObservationGrid",
    "build_observation_grid",
    "regular_grid",
    # Estimation
    "fit_statespace",
    "innovation_log_likelihood",
    "parameter_uncertainty",
    "profile_likelihood",
    "residual_diagnostics",
]
