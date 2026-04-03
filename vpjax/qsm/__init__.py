"""QSM (Quantitative Susceptibility Mapping) utilities."""

from vpjax.qsm.susceptibility import (
    SusceptibilityParams,
    oef_from_susceptibility,
    susceptibility_from_sources,
)
from vpjax.qsm.r2star_fitting import (
    fit_r2star_loglinear,
    fit_r2star_nonlinear,
    fit_r2star_volume,
    multi_echo_combine,
)
from vpjax.qsm.phase import (
    echo_combination_weights,
    frequency_to_susceptibility,
    phase_to_frequency,
    temporal_unwrap,
)

__all__ = [
    "SusceptibilityParams",
    "susceptibility_from_sources",
    "oef_from_susceptibility",
    "fit_r2star_loglinear",
    "fit_r2star_nonlinear",
    "fit_r2star_volume",
    "multi_echo_combine",
    "temporal_unwrap",
    "phase_to_frequency",
    "frequency_to_susceptibility",
    "echo_combination_weights",
]
