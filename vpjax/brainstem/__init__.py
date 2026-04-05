"""Brainstem nuclei identification from MELODIC ICA."""

from vpjax.brainstem.atlas import (
    NUCLEI,
    create_synthetic_atlas,
    load_brainstem_navigator,
)
from vpjax.brainstem.ica_identify import (
    identify_brainstem_components,
    load_melodic_ics,
    spatial_overlap,
)
from vpjax.brainstem.extract import (
    brainstem_to_vpjax_inputs,
    extract_timecourses,
    lc_timecourse_to_ne,
)

__all__ = [
    "NUCLEI",
    "create_synthetic_atlas",
    "load_brainstem_navigator",
    "spatial_overlap",
    "identify_brainstem_components",
    "load_melodic_ics",
    "extract_timecourses",
    "lc_timecourse_to_ne",
    "brainstem_to_vpjax_inputs",
]
