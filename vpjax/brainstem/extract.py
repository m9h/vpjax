"""Extract brainstem nuclei time courses from ICA decomposition.

Once brainstem components are identified via spatial overlap with
atlas ROIs, this module extracts their time courses from the MELODIC
mixing matrix and converts them to physiological signals for vpjax.

Key conversion: LC ICA time course → NE concentration → vasomotion

References
----------
Turker HB et al. (2021) Current Biology 31:5019-5025
    "Estimates of locus coeruleus function with fMRI"
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float


def extract_timecourses(
    mixing: np.ndarray,
    matches: dict[str, dict],
) -> dict[str, np.ndarray]:
    """Extract time courses for identified brainstem components.

    Parameters
    ----------
    mixing : MELODIC mixing matrix, shape (n_timepoints, n_components)
    matches : output from identify_brainstem_components

    Returns
    -------
    Dict mapping nucleus name → time course array (n_timepoints,)
    """
    timecourses = {}
    for nucleus_name, match_info in matches.items():
        idx = match_info["component_idx"]
        timecourses[nucleus_name] = mixing[:, idx]
    return timecourses


def lc_timecourse_to_ne(
    lc_tc: np.ndarray,
    tr: float = 1.0,
    tau_ne: float = 5.0,
) -> np.ndarray:
    """Convert LC ICA time course to estimated NE concentration.

    The LC BOLD signal is a delayed, smoothed proxy for LC firing.
    We model NE as a smoothed, sign-preserved version of the LC signal.

    NE release tracks LC firing, with exponential clearance:
        d[NE]/dt = gain × LC_signal - NE / τ_NE

    Parameters
    ----------
    lc_tc : LC component time course (from ICA mixing matrix)
    tr : repetition time (s)
    tau_ne : NE clearance time constant (s)

    Returns
    -------
    Estimated NE concentration (a.u., same length as lc_tc)
    """
    alpha = 1.0 - np.exp(-tr / tau_ne)

    ne = np.zeros_like(lc_tc)
    ne[0] = lc_tc[0]
    for i in range(1, len(lc_tc)):
        ne[i] = ne[i - 1] + alpha * (lc_tc[i] - ne[i - 1])

    # Center
    ne = ne - np.mean(ne)
    return ne


def brainstem_to_vpjax_inputs(
    timecourses: dict[str, np.ndarray],
    tr: float = 1.0,
) -> dict[str, Float[Array, "T"]]:
    """Convert brainstem time courses to vpjax model inputs.

    Maps:
    - LC → NE concentration → vpjax.sleep.vasomotion input
    - NTS → vagal afferent signal → vpjax.cardiac.baroreceptor input
    - DR → serotonin proxy → sleep state modulation

    Parameters
    ----------
    timecourses : dict from extract_timecourses
    tr : repetition time (s)

    Returns
    -------
    Dict with JAX arrays ready for vpjax models:
        'ne' : norepinephrine (from LC)
        'vagal_afferent' : vagal signal (from NTS)
        'serotonin' : serotonin proxy (from DR)
    """
    result = {}

    if "LC" in timecourses:
        ne = lc_timecourse_to_ne(timecourses["LC"], tr=tr)
        result["ne"] = jnp.array(ne)

    if "NTS" in timecourses:
        # NTS signal is a proxy for vagal afferent activity
        nts = timecourses["NTS"]
        result["vagal_afferent"] = jnp.array(nts - np.mean(nts))

    if "DR" in timecourses:
        # DR signal is a proxy for serotonin
        dr = timecourses["DR"]
        result["serotonin"] = jnp.array(dr - np.mean(dr))

    return result
