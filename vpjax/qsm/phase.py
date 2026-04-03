"""Multi-echo phase processing utilities for QSM.

QSM reconstruction requires phase data from multi-echo GRE.
The processing pipeline is:
    1. Phase combination across echoes (temporal unwrapping)
    2. Spatial phase unwrapping (external: ROMEO, Laplacian)
    3. Background field removal (external: V-SHARP, PDF, LBV)
    4. Dipole inversion (external: TKD, MEDI, TGV-QSM)

vpjax provides the phase combination step (step 1) and interface
utilities for connecting to external QSM tools (via Neurodesk).

References
----------
Robinson SD et al. (2017) MRM 77:1678-1689
    "ROMEO: phase unwrapping"
Li W et al. (2015) NeuroImage 108:111-122
Eckstein K et al. (2018) MRM 79:2996-3008
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def temporal_unwrap(
    phase: Float[Array, "... T"],
    te: Float[Array, "T"],
) -> Float[Array, "... T"]:
    """Temporal phase unwrapping across echoes.

    Removes 2π wraps between consecutive echoes by detecting and
    correcting phase jumps that exceed π.

    Parameters
    ----------
    phase : wrapped phase data (radians), shape (..., T)
    te : echo times (s), shape (T,)

    Returns
    -------
    Temporally unwrapped phase, shape (..., T)
    """
    # Phase difference between consecutive echoes
    dphase = jnp.diff(phase, axis=-1)

    # Wrap to [-π, π]
    dphase_wrapped = jnp.angle(jnp.exp(1j * dphase))

    # Reconstruct unwrapped phase from first echo
    unwrapped = jnp.concatenate([
        phase[..., :1],
        phase[..., :1] + jnp.cumsum(dphase_wrapped, axis=-1),
    ], axis=-1)

    return unwrapped


def phase_to_frequency(
    phase: Float[Array, "... T"],
    te: Float[Array, "T"],
) -> Float[Array, "..."]:
    """Convert multi-echo phase to frequency map via linear regression.

    φ(TE) = φ₀ + 2π·Δf·TE

    The slope gives the frequency offset Δf (Hz).

    Parameters
    ----------
    phase : unwrapped phase (radians), shape (..., T)
    te : echo times (s), shape (T,)

    Returns
    -------
    Frequency offset Δf (Hz), shape (...)
    """
    # Linear regression: phase = intercept + slope * TE
    n = te.shape[0]
    te_mean = jnp.mean(te)
    phase_mean = jnp.mean(phase, axis=-1)

    te_centered = te - te_mean
    phase_centered = phase - phase_mean[..., None]

    cov = jnp.sum(phase_centered * te_centered, axis=-1)
    var_te = jnp.sum(te_centered ** 2)
    var_safe = jnp.where(var_te > 1e-12, var_te, 1e-12)

    slope = cov / var_safe  # rad/s

    # Convert to Hz
    freq = slope / (2.0 * jnp.pi)
    return freq


def frequency_to_susceptibility(
    freq: Float[Array, "..."],
    B0: float = 3.0,
    gyro: float = 42.576e6,
) -> Float[Array, "..."]:
    """Approximate local susceptibility from frequency offset.

    For a sphere (rough approximation):
        Δf = γ · B₀ · Δχ / 3

    This is NOT a proper dipole inversion — use QSMxT/MEDI for that.
    This provides only a rough estimate useful for quick inspection.

    Parameters
    ----------
    freq : frequency offset (Hz)
    B0 : main field strength (T)
    gyro : gyromagnetic ratio (Hz/T)

    Returns
    -------
    Approximate susceptibility (ppm)
    """
    # Δχ = 3·Δf / (γ·B₀)  in ppm
    chi_ppm = 3.0 * freq / (gyro * B0) * 1e6
    return chi_ppm


def echo_combination_weights(
    te: Float[Array, "T"],
    snr: Float[Array, "... T"] | None = None,
) -> Float[Array, "... T"]:
    """Compute optimal weights for multi-echo phase combination.

    Optimal weighting for phase estimation (Robinson et al. 2017):
        w(TE) ∝ TE × SNR(TE)

    If SNR is not provided, assumes monoexponential decay:
        SNR(TE) ∝ exp(-TE/T₂*)  with T₂* ≈ 25ms at 3T.

    Parameters
    ----------
    te : echo times (s), shape (T,)
    snr : per-echo SNR map, shape (..., T). Optional.

    Returns
    -------
    Normalized weights, shape (..., T). Sum to 1 along last axis.
    """
    if snr is None:
        # Assume T2* ~ 25ms for cortical gray matter at 3T
        t2star = 0.025
        snr = jnp.exp(-te / t2star)

    weights = te * snr
    w_sum = jnp.sum(weights, axis=-1, keepdims=True)
    w_sum_safe = jnp.where(w_sum > 1e-10, w_sum, 1e-10)
    return weights / w_sum_safe
