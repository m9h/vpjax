"""Sleep physiology: state-dependent NVC, vasomotion, and glymphatic coupling."""

from vpjax.sleep.nvc_state import (
    WAKE, N1, N2, N3, REM,
    balloon_params_for_stage,
    hrf_peak_time,
    nvc_gain_continuous,
    nvc_gain_for_stage,
    riera_params_for_stage,
)
from vpjax.sleep.vasomotion import (
    VasomotionParams,
    bold_vasomotion,
    cbv_from_ne,
    cbv_vasomotion,
    ne_sawtooth_oscillation,
    norepinephrine_oscillation,
)
from vpjax.sleep.csf_coupling import (
    CSFParams,
    csf_bold_contribution,
    csf_flow_from_cbv,
    csf_flow_from_cbv_delayed,
    glymphatic_clearance,
)
from vpjax.sleep.global_waves import (
    global_bold_amplitude,
    global_bold_wave,
)
from vpjax.sleep.locus_coeruleus import (
    LCParams,
    lc_firing_rate,
    lc_firing_timecourse,
    lc_to_norepinephrine,
)

__all__ = [
    "WAKE", "N1", "N2", "N3", "REM",
    "balloon_params_for_stage",
    "riera_params_for_stage",
    "nvc_gain_for_stage",
    "nvc_gain_continuous",
    "hrf_peak_time",
    "VasomotionParams",
    "norepinephrine_oscillation",
    "ne_sawtooth_oscillation",
    "cbv_vasomotion",
    "cbv_from_ne",
    "bold_vasomotion",
    "CSFParams",
    "csf_flow_from_cbv",
    "csf_flow_from_cbv_delayed",
    "csf_bold_contribution",
    "glymphatic_clearance",
    "global_bold_amplitude",
    "global_bold_wave",
    "LCParams",
    "lc_firing_rate",
    "lc_firing_timecourse",
    "lc_to_norepinephrine",
]
