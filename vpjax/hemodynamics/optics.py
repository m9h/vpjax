"""Optical properties for fNIRS / DOT (dot-jax interface).

Maps hemoglobin concentrations (HbO, HbR) to tissue optical properties
(absorption coefficient mua, reduced scattering coefficient musp) at
given wavelengths.  These are consumed by dot-jax's forward model.

vpjax does NOT import dot-jax.  The user composes them:

    mua, musp = vpjax.to_optical_properties(state, wavelengths)
    measurements = dot_jax.forward_model(mua, musp)  # user composes

References
----------
Prahl S. Tabulated molar extinction coefficients for hemoglobin.
    https://omlc.org/spectra/hemoglobin/
Jacques SL (2013) Phys Med Biol 58:R37 (optical properties of tissue)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


# Extinction coefficients (cm^-1 / mM) at common fNIRS wavelengths.
# From Prahl / compiled tables.  Indexed: [wavelength, {HbO, HbR}]
# Format: {wavelength_nm: (eps_HbO, eps_HbR)}
_EXTINCTION_COEFFICIENTS = {
    690: (0.0096, 0.1174),
    750: (0.0283, 0.0682),
    760: (0.0317, 0.0620),
    780: (0.0402, 0.0492),
    800: (0.0507, 0.0420),  # isosbestic point
    830: (0.0704, 0.0352),
    850: (0.0878, 0.0316),
}

# Baseline reduced scattering: musp(lambda) = a * (lambda/500nm)^(-b)
# Typical brain tissue values from Jacques (2013).
_SCATTER_A = 24.2   # cm^-1 at 500nm
_SCATTER_B = 1.611  # scattering power


def _baseline_musp(wavelengths: Float[Array, "W"]) -> Float[Array, "W"]:
    """Baseline reduced scattering coefficient for brain tissue."""
    return _SCATTER_A * jnp.power(wavelengths / 500.0, -_SCATTER_B)


def _get_extinction(
    wavelengths: Float[Array, "W"],
) -> tuple[Float[Array, "W"], Float[Array, "W"]]:
    """Linearly interpolate extinction coefficients at given wavelengths.

    Returns (eps_HbO, eps_HbR) arrays of shape (W,).
    """
    # Build sorted reference arrays
    wl_ref = jnp.array(sorted(_EXTINCTION_COEFFICIENTS.keys()), dtype=jnp.float32)
    eps_hbo_ref = jnp.array(
        [_EXTINCTION_COEFFICIENTS[int(w)][0] for w in sorted(_EXTINCTION_COEFFICIENTS.keys())],
        dtype=jnp.float32,
    )
    eps_hbr_ref = jnp.array(
        [_EXTINCTION_COEFFICIENTS[int(w)][1] for w in sorted(_EXTINCTION_COEFFICIENTS.keys())],
        dtype=jnp.float32,
    )

    eps_hbo = jnp.interp(wavelengths, wl_ref, eps_hbo_ref)
    eps_hbr = jnp.interp(wavelengths, wl_ref, eps_hbr_ref)
    return eps_hbo, eps_hbr


def to_optical_properties(
    hbo: Float[Array, "..."],
    hbr: Float[Array, "..."],
    wavelengths: Float[Array, "W"],
    dpf: float = 6.0,
) -> tuple[Float[Array, "... W"], Float[Array, "... W"]]:
    """Convert hemoglobin concentrations to optical properties.

    Parameters
    ----------
    hbo : oxyhemoglobin concentration change (mM).
    hbr : deoxyhemoglobin concentration change (mM).
    wavelengths : wavelengths in nm, shape (W,).
    dpf : differential pathlength factor (default 6.0 for adult brain).

    Returns
    -------
    mua : absorption coefficient (cm^-1), shape (..., W).
    musp : reduced scattering coefficient (cm^-1), shape (..., W).
         Scattering is assumed constant (baseline tissue value).
    """
    eps_hbo, eps_hbr = _get_extinction(wavelengths)

    # Beer-Lambert: mua(lambda) = eps_HbO(lambda)*[HbO] + eps_HbR(lambda)*[HbR]
    # Broadcast: hbo is (...), eps is (W,) -> (..., W)
    mua = (
        hbo[..., None] * eps_hbo
        + hbr[..., None] * eps_hbr
    )

    musp = _baseline_musp(wavelengths)
    # Broadcast musp to match leading dims of mua
    musp = jnp.broadcast_to(musp, mua.shape)

    return mua, musp
