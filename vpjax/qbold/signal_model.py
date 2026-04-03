"""Quantitative BOLD (qBOLD) signal model.

Multi-echo GRE signal model from He & Yablonskiy (2007):

    S(TE) = S₀ · F(OEF, DBV, TE) · exp(-R₂·TE)

where F models the intravoxel frequency distribution from randomly
oriented deoxygenated blood vessels.

At short TE: signal dominated by R₂ (tissue T2)
At long TE:  signal dominated by R₂' (reversible, from deoxy-Hb)
The difference: R₂' = R₂* - R₂ ∝ OEF × DBV × B₀

References
----------
He X, Yablonskiy DA (2007) MRM 57:115-126
    "Quantitative BOLD: mapping of human cerebral deoxygenated blood
    volume and oxygen extraction fraction"
Yablonskiy DA, Haacke EM (1994) MRM 32:749-763
    "Theory of NMR signal behavior in magnetically inhomogeneous tissues"
Christen T et al. (2012) NeuroImage 60:582-591
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


# Gyromagnetic ratio for ¹H (rad/s/T)
GAMMA = 2.675e8

# Susceptibility difference between fully deoxygenated and oxygenated blood (SI)
DELTA_CHI0 = 0.264e-6  # 0.264 ppm in SI (4π × 0.18 ppm CGS)

# Hematocrit
HCT_DEFAULT = 0.40


class QBOLDParams(eqx.Module):
    """Parameters for the qBOLD signal model.

    Attributes
    ----------
    B0   : main magnetic field strength (T)
    Hct  : hematocrit fraction
    R2t  : tissue transverse relaxation rate (s⁻¹), ~1/T2_tissue
    S0   : baseline signal (arbitrary units)
    """
    B0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(3.0))
    Hct: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(HCT_DEFAULT))
    R2t: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(11.5))
    S0: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))


def characteristic_frequency(oef: Float[Array, "..."],
                              params: QBOLDParams | None = None,
                              ) -> Float[Array, "..."]:
    """Compute the characteristic frequency shift δω from OEF.

    δω = (4/3) π γ Δχ₀ Hct OEF B₀

    Parameters
    ----------
    oef : oxygen extraction fraction
    params : QBOLDParams (default: 3T)

    Returns
    -------
    δω in rad/s
    """
    if params is None:
        params = QBOLDParams()

    delta_omega = (4.0 / 3.0) * jnp.pi * GAMMA * DELTA_CHI0 * params.Hct * oef * params.B0
    return delta_omega


def qbold_signal(te: Float[Array, "T"],
                 oef: Float[Array, "..."],
                 dbv: Float[Array, "..."],
                 params: QBOLDParams | None = None,
                 ) -> Float[Array, "... T"]:
    """Compute the qBOLD signal at given echo times.

    Implements the static dephasing regime model (Yablonskiy & Haacke 1994,
    He & Yablonskiy 2007):

        S(TE) = S₀ · exp(-R₂·TE) · F(OEF, DBV, TE)

    where F is the extravascular signal function accounting for
    deoxygenated blood vessels modeled as randomly oriented infinite
    cylinders.

    In the static dephasing regime (TE >> 1/δω):
        F ≈ exp(-DBV · f(δω·TE))

    where f(x) = 1 - (2/3) for the long-TE limit, and for
    intermediate TE we use the full expression.

    Parameters
    ----------
    te : echo times in seconds, shape (T,)
    oef : oxygen extraction fraction, shape (...)
    dbv : deoxygenated blood volume fraction, shape (...)
    params : QBOLDParams

    Returns
    -------
    Signal array of shape (..., T).
    """
    if params is None:
        params = QBOLDParams()

    dw = characteristic_frequency(oef, params)

    # Reshape for broadcasting: oef/dbv are (...), te is (T,)
    te_ = te  # (T,)
    dw_ = dw[..., None]  # (..., 1)
    dbv_ = dbv[..., None]  # (..., 1)

    # τ = δω · TE  (dimensionless dephasing parameter)
    tau = dw_ * te_

    # Static dephasing regime signal function (He & Yablonskiy 2007)
    # For |τ| > 0:
    #   F_ext = exp(-dbv · g(τ))
    # where g(τ) uses the analytical expression for randomly oriented
    # cylinders in the static dephasing regime:
    #   For short τ: g(τ) ≈ τ²/3  (quadratic regime, reversible)
    #   For long τ:  g(τ) ≈ τ - 1  (linear regime, irreversible R2')
    #
    # Smooth interpolation (He & Yablonskiy 2007, Eq. 6):
    tau_c = 1.5  # crossover point

    # Short-TE regime (quadratic dephasing)
    g_short = tau ** 2 / 3.0

    # Long-TE regime (linear dephasing, the R2' regime)
    g_long = jnp.abs(tau) - 1.0

    # Smooth transition using the analytical form
    g = jnp.where(jnp.abs(tau) < tau_c, g_short, g_long)

    # Extravascular signal attenuation
    F_ext = jnp.exp(-dbv_ * g)

    # Full signal: S0 * exp(-R2t * TE) * F_ext
    R2_decay = jnp.exp(-params.R2t[..., None] * te_)
    signal = params.S0[..., None] * R2_decay * F_ext

    return signal


def compute_r2prime(oef: Float[Array, "..."],
                    dbv: Float[Array, "..."],
                    params: QBOLDParams | None = None,
                    ) -> Float[Array, "..."]:
    """Compute R₂' (reversible transverse relaxation rate) from OEF and DBV.

    R₂' = DBV · δω  (in the linear/long-TE regime)

    This is the additional relaxation rate due to microscopic field
    inhomogeneities from deoxygenated blood.

    Parameters
    ----------
    oef : oxygen extraction fraction
    dbv : deoxygenated blood volume fraction
    params : QBOLDParams

    Returns
    -------
    R₂' in s⁻¹
    """
    if params is None:
        params = QBOLDParams()

    dw = characteristic_frequency(oef, params)
    return dbv * dw


def compute_r2star(oef: Float[Array, "..."],
                   dbv: Float[Array, "..."],
                   params: QBOLDParams | None = None,
                   ) -> Float[Array, "..."]:
    """Compute R₂* = R₂ + R₂' (effective transverse relaxation rate).

    Parameters
    ----------
    oef : oxygen extraction fraction
    dbv : deoxygenated blood volume fraction
    params : QBOLDParams (R2t field provides R₂)

    Returns
    -------
    R₂* in s⁻¹
    """
    if params is None:
        params = QBOLDParams()

    r2p = compute_r2prime(oef, dbv, params)
    return params.R2t + r2p
