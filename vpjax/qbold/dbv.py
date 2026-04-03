"""Deoxygenated blood volume (DBV) estimation.

DBV is the fraction of tissue volume occupied by deoxygenated blood
(primarily venous and capillary compartments).  Together with OEF,
DBV determines the R₂' contribution to the MRI signal.

Methods:
1. From the qBOLD fit: DBV is fitted jointly with OEF (see oef_mapping.py)
2. From R₂' and OEF: DBV = R₂' / δω (known OEF → known δω)
3. From CBV partitioning: DBV ≈ venous_fraction × CBV

References
----------
He X, Yablonskiy DA (2007) MRM 57:115-126
An H, Lin W (2003) MRM 50:708-716
Blockley NP et al. (2015) NeuroImage 112:225-234
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax.qbold.signal_model import QBOLDParams, characteristic_frequency


class DBVParams(eqx.Module):
    """Parameters for DBV estimation.

    Attributes
    ----------
    venous_fraction : fraction of total CBV that is venous (default ~0.77)
        From Grubb/Ito: venous blood dominates total CBV.
    capillary_fraction : fraction of total CBV that is capillary (~0.21)
    """
    venous_fraction: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.77)
    )
    capillary_fraction: Float[Array, "..."] = eqx.field(
        default_factory=lambda: jnp.array(0.21)
    )


def dbv_from_r2prime(
    r2prime: Float[Array, "..."],
    oef: Float[Array, "..."],
    qbold_params: QBOLDParams | None = None,
) -> Float[Array, "..."]:
    """Estimate DBV from R₂' and OEF.

    Since R₂' = DBV × δω(OEF), we can invert:
        DBV = R₂' / δω

    Parameters
    ----------
    r2prime : reversible relaxation rate R₂' (s⁻¹)
    oef : oxygen extraction fraction
    qbold_params : QBOLDParams (B0, Hct)

    Returns
    -------
    DBV (deoxygenated blood volume fraction)
    """
    if qbold_params is None:
        qbold_params = QBOLDParams()

    dw = characteristic_frequency(oef, qbold_params)
    # Avoid division by zero
    dw_safe = jnp.where(dw > 1e-6, dw, 1e-6)
    return r2prime / dw_safe


def dbv_from_cbv(
    cbv: Float[Array, "..."],
    params: DBVParams | None = None,
) -> Float[Array, "..."]:
    """Estimate DBV from total CBV using compartmental fractions.

    DBV ≈ (venous_fraction + capillary_fraction) × CBV

    Assumes all venous blood and capillary blood contributes to
    the deoxygenated blood signal.

    Parameters
    ----------
    cbv : total cerebral blood volume fraction
    params : DBVParams

    Returns
    -------
    Estimated DBV (deoxygenated blood volume fraction)
    """
    if params is None:
        params = DBVParams()

    # Venous + capillary blood is deoxygenated (partially)
    return (params.venous_fraction + params.capillary_fraction) * cbv


def dbv_change_from_balloon(
    v: Float[Array, "..."],
    dbv0: float = 0.038,
) -> Float[Array, "..."]:
    """Compute DBV change from Balloon model volume state.

    In the Balloon-Windkessel model, v = CBV/CBV₀. The absolute
    DBV is approximately:
        DBV(t) = DBV₀ × v(t)

    Parameters
    ----------
    v : CBV/CBV₀ from Balloon model (1.0 at baseline)
    dbv0 : baseline DBV fraction (default ~3.8% from literature)

    Returns
    -------
    Time-varying DBV fraction.
    """
    return dbv0 * v
