"""Neural-population layer: the excitation/inhibition block the EEG observable is derived from.

The only member so far is :mod:`vpjax.neural.ei_block`, which wraps the
linearised graph Wilson-Cowan model of gnf-jax (Aqil et al. 2021) as a
linear-Gaussian latent for :mod:`vpjax.statespace`.  gnf-jax is an
optional dependency (``vpjax[ei]``); importing this subpackage without it
raises a clear error rather than failing at the first call.
"""

from vpjax.neural import connectome  # noqa: F401
from vpjax.neural.ei_block import (
    ENTRY_POINTS,
    EIBlock,
    band_power,
    linear_ei_block,
    statespace_spec,
)

__all__ = ["ENTRY_POINTS", "EIBlock", "band_power", "linear_ei_block", "statespace_spec"]
