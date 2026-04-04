"""Field-strength presets and pipeline helpers.

Provides pre-configured parameter bundles for common scanner setups
and convenience functions that chain multiple vpjax modules into
complete analysis pipelines.

Usage::

    from vpjax.presets import params_3T, params_7T

    # Get all 3T-appropriate parameters in one call
    p = params_3T()
    bold = observe_bold(state, p.bold)
    oef = trust_oef(t2_venous, p.trust)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax._types import BalloonParams, BalloonState
from vpjax.hemodynamics.bold import BOLDParams, observe_bold
from vpjax.hemodynamics.balloon import solve_balloon
from vpjax.perfusion.asl import observe_asl
from vpjax.perfusion.vaso import observe_vaso
from vpjax.perfusion.trust import TRUSTParams
from vpjax.perfusion.kinetic import ASLKineticParams
from vpjax.perfusion.calibration import blood_t1
from vpjax.qbold.signal_model import QBOLDParams
from vpjax.qbold.calibrated import CalibratedBOLDParams
from vpjax.vaso.signal_model import VASOParams
from vpjax.metabolism.fick import FickParams, fick_cmro2, compute_cao2
from vpjax.metabolism.oef import compute_oef


# ---------------------------------------------------------------------------
# Parameter bundles
# ---------------------------------------------------------------------------

@dataclass
class ScannerPreset:
    """Bundle of parameters appropriate for a given field strength."""
    balloon: BalloonParams
    bold: BOLDParams
    trust: TRUSTParams
    asl: ASLKineticParams
    qbold: QBOLDParams
    calibrated: CalibratedBOLDParams
    vaso: VASOParams
    fick: FickParams
    field_strength: float


def params_3T() -> ScannerPreset:
    """Standard parameter bundle for 3T scanners.

    Uses default values from the literature for 3T:
    - Blood T1 ~ 1.65s
    - TE ~ 30ms for BOLD
    - qBOLD R2_tissue ~ 11.5 s⁻¹
    """
    return ScannerPreset(
        balloon=BalloonParams(),
        bold=BOLDParams(),
        trust=TRUSTParams(
            K1=jnp.array(6.56),
            K2=jnp.array(188.2),
            T2_0=jnp.array(0.220),
        ),
        asl=ASLKineticParams(
            T1b=jnp.array(1.65),
            T1t=jnp.array(1.30),
        ),
        qbold=QBOLDParams(B0=jnp.array(3.0), R2t=jnp.array(11.5)),
        calibrated=CalibratedBOLDParams(
            beta=jnp.array(1.3),
            TE=jnp.array(0.030),
        ),
        vaso=VASOParams(
            T1b=jnp.array(1.65),
            T1t=jnp.array(1.30),
        ),
        fick=FickParams(),
        field_strength=3.0,
    )


def params_7T() -> ScannerPreset:
    """Parameter bundle for 7T scanners.

    Key differences from 3T:
    - Longer blood T1 (~2.09s)
    - Shorter optimal TE for BOLD (~22ms)
    - Higher qBOLD sensitivity (B0-dependent δω)
    - Different TRUST calibration constants
    - Higher BOLD signal (k1/k2/k3 scale with B0)
    """
    return ScannerPreset(
        balloon=BalloonParams(),
        bold=BOLDParams(
            k1=jnp.array(7.0),
            k2=jnp.array(1.0),
            k3=jnp.array(0.40),
        ),
        trust=TRUSTParams(
            K1=jnp.array(12.0),
            K2=jnp.array(340.0),
            T2_0=jnp.array(0.085),
        ),
        asl=ASLKineticParams(
            T1b=jnp.array(2.09),
            T1t=jnp.array(1.80),
            alpha=jnp.array(0.80),
        ),
        qbold=QBOLDParams(B0=jnp.array(7.0), R2t=jnp.array(16.0)),
        calibrated=CalibratedBOLDParams(
            beta=jnp.array(1.5),
            TE=jnp.array(0.022),
        ),
        vaso=VASOParams(
            T1b=jnp.array(2.09),
            T1t=jnp.array(1.80),
        ),
        fick=FickParams(),
        field_strength=7.0,
    )


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def balloon_to_signals(
    stimulus: Float[Array, "T"],
    params: BalloonParams | None = None,
    bold_params: BOLDParams | None = None,
    dt: float = 0.01,
) -> dict[str, Float[Array, "T"]]:
    """Run the full Balloon pipeline: stimulus → BOLD + ASL + VASO.

    Parameters
    ----------
    stimulus : neural stimulus time course, shape (T,)
    params : BalloonParams (default: standard DCM)
    bold_params : BOLDParams (default: 3T)
    dt : sampling interval (s)

    Returns
    -------
    Dict with keys: 'ts', 'bold', 'asl', 'vaso', 'cbf', 'cbv', 'dhb'
    """
    if params is None:
        params = BalloonParams()

    ts, traj = solve_balloon(params, stimulus, dt=dt)
    bold = observe_bold(traj, bold_params)
    asl = observe_asl(traj)
    vaso = observe_vaso(traj)

    return {
        "ts": ts,
        "bold": bold,
        "asl": asl,
        "vaso": vaso,
        "cbf": traj.f,
        "cbv": traj.v,
        "dhb": traj.q,
    }


def trust_cmro2_pipeline(
    t2_venous: Float[Array, "..."],
    cbf: Float[Array, "..."],
    trust_params: TRUSTParams | None = None,
    fick_params: FickParams | None = None,
) -> dict[str, Float[Array, "..."]]:
    """Level 1 CMRO₂: TRUST global OEF + CBF → CMRO₂.

    Parameters
    ----------
    t2_venous : venous blood T2 from TRUST (s)
    cbf : cerebral blood flow (mL/100g/min)
    trust_params : TRUSTParams
    fick_params : FickParams

    Returns
    -------
    Dict with: 'svo2', 'oef', 'cao2', 'cmro2'
    """
    from vpjax.perfusion.trust import t2_to_svo2, trust_oef

    if trust_params is None:
        trust_params = TRUSTParams()

    svo2 = t2_to_svo2(t2_venous, trust_params)
    oef = trust_oef(t2_venous, trust_params)
    cao2 = compute_cao2(fick_params)
    cmro2 = fick_cmro2(cbf, oef, cao2)

    return {
        "svo2": svo2,
        "oef": oef,
        "cao2": cao2,
        "cmro2": cmro2,
    }


def qbold_cmro2_pipeline(
    oef_regional: Float[Array, "..."],
    cbf_regional: Float[Array, "..."],
    fick_params: FickParams | None = None,
) -> dict[str, Float[Array, "..."]]:
    """Level 2 CMRO₂: qBOLD regional OEF + regional CBF → CMRO₂.

    Parameters
    ----------
    oef_regional : per-voxel OEF from qBOLD fitting
    cbf_regional : per-voxel CBF from ASL (mL/100g/min)
    fick_params : FickParams

    Returns
    -------
    Dict with: 'cao2', 'cmro2'
    """
    cao2 = compute_cao2(fick_params)
    cmro2 = fick_cmro2(cbf_regional, oef_regional, cao2)

    return {
        "cao2": cao2,
        "cmro2": cmro2,
    }


def calibrated_fmri_pipeline(
    r2prime: Float[Array, "..."],
    delta_bold: Float[Array, "..."],
    cbf_ratio: Float[Array, "..."],
    cal_params: CalibratedBOLDParams | None = None,
) -> dict[str, Float[Array, "..."]]:
    """Calibrated fMRI: R₂' → M → ΔCMRO₂.

    Parameters
    ----------
    r2prime : resting R₂' from qBOLD (s⁻¹)
    delta_bold : task-evoked BOLD signal change (fractional)
    cbf_ratio : task-evoked CBF/CBF₀
    cal_params : CalibratedBOLDParams

    Returns
    -------
    Dict with: 'M', 'cmro2_ratio', 'oef_ratio'
    """
    from vpjax.qbold.calibrated import estimate_M_from_r2prime, estimate_cmro2_change

    M = estimate_M_from_r2prime(r2prime, cal_params)
    cmro2_ratio = estimate_cmro2_change(delta_bold, cbf_ratio, M, cal_params)

    # OEF ratio from flow-metabolism coupling
    oef_ratio = compute_oef(cbf_ratio, cmro2_ratio) / jnp.array(0.34)

    return {
        "M": M,
        "cmro2_ratio": cmro2_ratio,
        "oef_ratio": oef_ratio,
    }
