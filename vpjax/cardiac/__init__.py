"""Cardiac physiology: heart-brain coupling models."""

from vpjax.cardiac.vagal import (
    VagalODE,
    VagalParams,
    VagalState,
    hr_to_rr_interval,
    vagal_hr_response,
)
from vpjax.cardiac.baroreceptor import (
    BaroreceptorParams,
    arterial_pressure,
    cortical_excitability,
    modulate_neural_drive,
)
from vpjax.cardiac.pulsatility import (
    PulsatilityParams,
    asl_cardiac_confound,
    bold_cardiac_confound,
    cbv_pulsation,
)

__all__ = [
    "VagalParams",
    "VagalState",
    "VagalODE",
    "vagal_hr_response",
    "hr_to_rr_interval",
    "BaroreceptorParams",
    "cortical_excitability",
    "modulate_neural_drive",
    "arterial_pressure",
    "PulsatilityParams",
    "cbv_pulsation",
    "bold_cardiac_confound",
    "asl_cardiac_confound",
]
