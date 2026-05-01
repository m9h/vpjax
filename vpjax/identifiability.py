"""Local structural identifiability via the Jacobian-rank test.

Implements the Walter & Pronzato (1997) local identifiability test:
given a parametric forward model ``y = h(θ)``, evaluate the Jacobian
``J = ∂h/∂θ`` at a nominal parameter point and check whether
``rank(J) = dim(θ)``.  When the rank is deficient, the right null
space of ``J`` identifies the parameter combinations that the model
*cannot* distinguish — these are the directions in which the loss is
exactly flat to first order, regardless of how much data is collected.

This lineage traces back to Bellman & Åström (1970) and Joseph
DiStefano III's UCLA Biocybernetics Lab series on input-output
identifiability of compartmental ODE models.  The function here is a
JAX-native port of the implementation in the ``sbi4dwi`` repository
(``dmipy_jax/analysis/local_identifiability.py``), generalised so that
*any* Equinox-compatible forward model — Balloon, Riera, ASL kinetic,
PET binding — can be screened.

Usage
-----

>>> def fwd(theta):
...     # returns observed signal (any shape)
...     ...
>>> result = check_local_identifiability(fwd, theta0, names=["kappa", "tau"])
>>> if not result["is_identifiable"]:
...     for col in result["collinear_sets"]:
...         print(col)

References
----------
Bellman R, Åström KJ (1970) Math Biosci 7:329-339
    "On structural identifiability"
DiStefano JJ (2014) "Dynamic Systems Biology Modeling and Simulation"
    UCLA Biocybernetics Lab; Academic Press
Walter E, Pronzato L (1997) "Identification of Parametric Models from
    Experimental Data" Springer
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def check_local_identifiability(
    forward_fn: Callable[[Array], Array],
    params: Float[Array, "P"],
    names: Sequence[str] | None = None,
    rtol: float = 1e-5,
    null_threshold: float = 0.1,
) -> dict[str, Any]:
    """Test local structural identifiability at a nominal parameter point.

    Computes ``J = ∂(forward_fn)/∂params`` via :func:`jax.jacfwd`, reduces
    the per-observation Jacobian to a 2-D ``(M, P)`` matrix (flattening
    any output structure), and returns the singular-value spectrum
    plus an analysis of the null space.

    Parameters
    ----------
    forward_fn : ``params -> signal`` Equinox / JAX function.  The
        signal can have any shape; it's flattened before the SVD.
    params : flat parameter vector at which to evaluate identifiability.
        For models where the nominal point matters (nonlinear systems),
        choose a physiologically realistic baseline.
    names : optional parameter names for reporting (length P).
    rtol : relative threshold for declaring a singular value zero
        (``threshold = rtol · max_sv``).  Default ``1e-5`` matches the
        DiStefano-style convention for ill-conditioned biological
        Jacobians.
    null_threshold : minimum absolute coefficient in a null-space
        vector for a parameter to be reported as part of the collinear
        set (default 0.1, i.e. >10% loading).

    Returns
    -------
    Dict::

        is_identifiable : bool   – rank == P at the chosen rtol
        rank : int                – numeric rank of the Jacobian
        n_params : int
        condition_number : float – ``S[0] / S[-1]`` (large ⇒ near-singular)
        singular_values : (P,) array, padded with zeros if M < P
        collinear_sets : list of dicts, one per zero singular value::
            { "singular_value": float,
              "params": [name, ...],
              "coefficients": [float, ...] }
            describing the linear combination ``Σ c_i · θ_i`` that the
            data cannot constrain.
        jacobian_shape : (M, P) shape of the flattened Jacobian.
        nominal_params : echo of ``params`` (numpy).
    """
    params = jnp.asarray(params)
    P = int(params.shape[0])

    if names is None:
        names = tuple(f"p{i}" for i in range(P))
    elif len(names) != P:
        raise ValueError(f"len(names)={len(names)} ≠ P={P}")

    # Reverse-mode Jacobian.  We'd prefer jax.jacfwd for tall Jacobians
    # (M >> P, typical for time-series models), but diffrax's solvers
    # are wrapped in custom_vjp and only support reverse-mode AD —
    # jacfwd would raise "can't apply forward-mode autodiff to a
    # custom_vjp function".  jacrev is correct for any Jacobian and is
    # cheap when P is small (≤ ~10), which is the usual case here.
    raw_jac = jax.jacrev(forward_fn)(params)
    J = jnp.reshape(raw_jac, (-1, P))

    if not bool(jnp.all(jnp.isfinite(J))):
        return {
            "error": "Jacobian contains non-finite entries",
            "is_identifiable": False,
            "rank": 0,
            "n_params": P,
            "jacobian_shape": tuple(J.shape),
        }

    # Full SVD so V is square (P, P); when M < P the thin SVD would
    # drop the (P − M) extra null-space directions that come from the
    # geometry alone (more parameters than measurements).
    U, S, Vh = jnp.linalg.svd(J, full_matrices=True)
    V = Vh.T  # (P, P)

    s_short = np.asarray(S)  # length min(M, P)
    max_sv = float(s_short[0]) if s_short.size else 0.0
    threshold = rtol * max_sv if max_sv > 0 else 0.0
    rank = int(np.sum(s_short > threshold))
    is_identifiable = rank == P

    # Pad the singular-value vector to length P with explicit zeros so
    # geometric null directions (M < P case) are surfaced too.
    s_full = np.zeros(P, dtype=float)
    s_full[:s_short.shape[0]] = s_short

    collinear_sets: list[dict[str, Any]] = []
    null_indices = np.where(s_full <= threshold)[0]
    for idx in null_indices:
        null_vec = np.asarray(V[:, idx])
        contributing = np.where(np.abs(null_vec) > null_threshold)[0]
        collinear_sets.append({
            "singular_value": float(s_full[idx]),
            "params": [names[i] for i in contributing],
            "coefficients": [float(null_vec[i]) for i in contributing],
        })

    cond = float(s_full[0] / (s_full[-1] + 1e-30)) if s_full.size else float("inf")

    return {
        "is_identifiable": bool(is_identifiable),
        "rank": rank,
        "n_params": P,
        "condition_number": cond,
        "singular_values": s_full,
        "collinear_sets": collinear_sets,
        "jacobian_shape": tuple(J.shape),
        "nominal_params": np.asarray(params),
        "names": tuple(names),
    }


# ---------------------------------------------------------------------------
# Balloon-specific helpers (the most-used model in vpjax)
# ---------------------------------------------------------------------------

def balloon_identifiability(
    fit_names: Sequence[str],
    stimulus: Float[Array, "T"],
    dt: float = 0.1,
    tr: float | None = None,
    nominal: dict[str, float] | None = None,
    observers: Sequence[str] = ("bold",),
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Local identifiability of selected Balloon-Windkessel parameters.

    Builds a forward function ``θ → stacked observations`` for the
    requested observers (``bold``, ``asl``, ``vaso``), evaluates the
    Jacobian at the nominal Balloon point, and reports rank/null-space.

    Parameters
    ----------
    fit_names : Balloon parameter names to test (subset of
        ``("kappa", "gamma", "tau", "alpha", "E0")``).
    stimulus : neural input on the ODE grid (length T).
    dt : ODE timestep (s).
    tr : if given, the observed signal is sub-sampled at this TR before
        the Jacobian is computed (matches the optimisation setup).
    nominal : overrides for any of the 5 Balloon parameters; defaults
        to the Stephan 2007 DCM standard values for any name not given.
    observers : which observation models to stack into the forward
        signal — combinations of ``"bold"``, ``"asl"``, ``"vaso"``.
    rtol : forwarded to :func:`check_local_identifiability`.

    Returns
    -------
    Dict from :func:`check_local_identifiability` plus an ``observers``
    field for provenance.
    """
    from vpjax._types import BalloonParams
    from vpjax.hemodynamics.balloon import solve_balloon
    from vpjax.hemodynamics.bold import BOLDParams, observe_bold
    from vpjax.perfusion.asl import observe_asl
    from vpjax.perfusion.vaso import observe_vaso

    valid = ("kappa", "gamma", "tau", "alpha", "E0")
    bad = [n for n in fit_names if n not in valid]
    if bad:
        raise ValueError(f"unknown Balloon param(s): {bad}; valid: {valid}")

    nominal = dict(nominal or {})
    base = BalloonParams()
    base_vals = {
        n: float(nominal.get(n, getattr(base, n)))
        for n in valid
    }
    bold_params = BOLDParams()

    sub = int(round(tr / dt)) if tr is not None else 1

    def forward_fn(theta: Float[Array, "P"]) -> Float[Array, "..."]:
        vals = dict(base_vals)
        for i, n in enumerate(fit_names):
            vals[n] = theta[i]
        bp = BalloonParams(**{k: jnp.asarray(v) for k, v in vals.items()})
        _, traj = solve_balloon(bp, stimulus, dt=dt)
        outs = []
        for obs in observers:
            if obs == "bold":
                y = observe_bold(traj, bold_params)
            elif obs == "asl":
                y = observe_asl(traj)
            elif obs == "vaso":
                y = observe_vaso(traj)
            else:
                raise ValueError(f"unknown observer: {obs}")
            outs.append(y[::sub] if sub > 1 else y)
        return jnp.concatenate([jnp.ravel(o) for o in outs])

    theta0 = jnp.array([base_vals[n] for n in fit_names])
    result = check_local_identifiability(
        forward_fn, theta0, names=tuple(fit_names), rtol=rtol,
    )
    result["observers"] = tuple(observers)
    return result


# ---------------------------------------------------------------------------
# Static-SUVR / BPnd binding model
# ---------------------------------------------------------------------------

def pet_static_suvr_identifiability(
    n_regions: int,
    cbf_per_region: Float[Array, "R"] | None = None,
    cbf_ref: float = 50.0,
    fit_flow: bool = True,
    nominal_bpnd: Float[Array, "R"] | None = None,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Identifiability of the static-SUVR binding-potential model.

    Forward (per region):
        SUVR = 1 + BPnd + β_flow · (CBF / CBF_ref − 1)

    Latent: BPnd₁,…,BPnd_R  (and optionally a single global β_flow).
    Observation: SUVR₁,…,SUVR_R (R measurements).

    The Jacobian is exactly linear, so the rank test is closed-form, but
    we run it through :func:`check_local_identifiability` for symmetry.

    Parameters
    ----------
    n_regions : R
    cbf_per_region : CBF values used for the flow-correction term
        (length R).  If None, CBF varies as ``np.linspace(40, 60, R)``.
    cbf_ref : reference-region mean CBF.
    fit_flow : if True, β_flow is added as a free parameter.
    nominal_bpnd : nominal BPnd values for the Jacobian evaluation.
        Defaults to zero (Logan equilibrium baseline).
    """
    if cbf_per_region is None:
        cbf_per_region = jnp.asarray(np.linspace(40.0, 60.0, n_regions))
    if nominal_bpnd is None:
        nominal_bpnd = jnp.zeros(n_regions)

    if fit_flow:
        names = tuple([f"BPnd_{i}" for i in range(n_regions)] + ["beta_flow"])
        params0 = jnp.concatenate([nominal_bpnd, jnp.array([0.0])])

        def fwd(theta):
            bpnd = theta[:n_regions]
            beta = theta[n_regions]
            return 1.0 + bpnd + beta * (cbf_per_region / cbf_ref - 1.0)
    else:
        names = tuple(f"BPnd_{i}" for i in range(n_regions))
        params0 = nominal_bpnd

        def fwd(theta):
            return 1.0 + theta

    out = check_local_identifiability(fwd, params0, names=names, rtol=rtol)
    out["model"] = "pet_static_suvr"
    out["fit_flow"] = bool(fit_flow)
    out["n_regions"] = int(n_regions)
    return out


# ---------------------------------------------------------------------------
# Joint amyloid–CMRO2 model (vpjax.pet.joint)
# ---------------------------------------------------------------------------

def pet_joint_identifiability(
    n_regions: int,
    cbf_per_region: Float[Array, "R"] | None = None,
    cbf_ref: float = 50.0,
    fit_alpha_amy: bool = False,
    fit_beta_flow: bool = True,
    cmro2_baseline: float = 160.0,
    alpha_amy: float = -0.10,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Identifiability of the joint amyloid–CMRO2 model.

    Forward (per region):
        SUVR    = 1 + BPnd + β_flow · (CBF / CBF_ref − 1)
        CMRO2   = CBF · OEF · CaO2                          (Fick)
        CMRO2_c = CMRO2_baseline · (1 + α_amy · BPnd)       (Vaishnavi)
        residual_couple = (CMRO2 − CMRO2_c) / CMRO2_baseline (normalised)
        residual_oef    = OEF − OEF_prior

    Stacked observations (used as the implicit "data" for the Jacobian):
        [SUVR_r, residual_couple_r, residual_oef_r]  for r in 1..R

    Latent: BPnd_r, OEF_r per region; optional global α_amy, β_flow.
    """
    from vpjax.metabolism.fick import compute_cao2

    if cbf_per_region is None:
        cbf_per_region = jnp.asarray(np.linspace(40.0, 60.0, n_regions))
    cao2 = float(compute_cao2())
    oef_prior = jnp.array(0.40)

    bpnd0 = jnp.zeros(n_regions)
    oef0 = jnp.full((n_regions,), 0.40)

    names: list[str] = []
    parts = [bpnd0, oef0]
    names += [f"BPnd_{i}" for i in range(n_regions)]
    names += [f"OEF_{i}" for i in range(n_regions)]
    if fit_beta_flow:
        parts.append(jnp.array([0.0]))
        names.append("beta_flow")
    if fit_alpha_amy:
        parts.append(jnp.array([alpha_amy]))
        names.append("alpha_amy")
    params0 = jnp.concatenate(parts)

    def fwd(theta):
        bpnd = theta[:n_regions]
        oef = theta[n_regions:2 * n_regions]
        idx = 2 * n_regions
        beta = theta[idx] if fit_beta_flow else jnp.array(0.0)
        if fit_beta_flow:
            idx += 1
        alpha = theta[idx] if fit_alpha_amy else jnp.array(alpha_amy)

        suvr = 1.0 + bpnd + beta * (cbf_per_region / cbf_ref - 1.0)
        cmro2_fick = cbf_per_region * oef * cao2
        cmro2_coup = cmro2_baseline * (1.0 + alpha * bpnd)
        couple_resid = (cmro2_fick - cmro2_coup) / cmro2_baseline
        oef_resid = oef - oef_prior
        return jnp.concatenate([suvr, couple_resid, oef_resid])

    out = check_local_identifiability(fwd, params0, names=tuple(names), rtol=rtol)
    out["model"] = "pet_joint"
    out["n_regions"] = int(n_regions)
    out["fit_beta_flow"] = bool(fit_beta_flow)
    out["fit_alpha_amy"] = bool(fit_alpha_amy)
    return out


# ---------------------------------------------------------------------------
# ASL Buxton kinetic model (single- or multi-PLD)
# ---------------------------------------------------------------------------

def asl_kinetic_identifiability(
    plds: Sequence[float],
    fit_names: Sequence[str] = ("CBF",),
    nominal_cbf: float = 60.0,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Identifiability of the Buxton single-PLD pCASL forward model.

    The forward signal at a given PLD is given by
    :func:`vpjax.perfusion.kinetic.asl_kinetic_signal`.  We treat each
    PLD as one observation (in mL/100g/min units after the standard
    quantification), and ask whether the requested ``fit_names`` are
    locally identifiable from those observations.

    ``fit_names`` is a subset of
    ``("CBF", "delta", "tau", "alpha", "T1b", "T1t", "lambda_p")``;
    other parameters stay at their literature defaults.

    Parameters
    ----------
    plds : sequence of post-label delays (s).  Single-PLD acquisitions
        give one measurement; multi-PLD give as many as you supply.
    fit_names : parameter names to test for identifiability.
    nominal_cbf : nominal CBF in mL/100g/min for the Jacobian point.
    """
    from vpjax.perfusion.kinetic import ASLKineticParams, asl_kinetic_signal

    base_params = ASLKineticParams()
    base_vals = {
        "M0b": float(base_params.M0b),
        "T1b": float(base_params.T1b),
        "T1t": float(base_params.T1t),
        "alpha": float(base_params.alpha),
        "tau": float(base_params.tau),
        "delta": float(base_params.delta),
        "lambda_p": float(base_params.lambda_p),
    }
    valid = ("CBF",) + tuple(base_vals)
    bad = [n for n in fit_names if n not in valid]
    if bad:
        raise ValueError(f"unknown ASL params: {bad}; valid: {valid}")

    plds_arr = jnp.asarray([float(p) for p in plds])
    nominal_cbf_jnp = jnp.array(float(nominal_cbf))

    theta0 = []
    for n in fit_names:
        if n == "CBF":
            theta0.append(nominal_cbf)
        else:
            theta0.append(base_vals[n])
    theta0_jnp = jnp.asarray(theta0)

    def fwd(theta):
        vals = dict(base_vals)
        cbf = nominal_cbf_jnp
        for i, n in enumerate(fit_names):
            if n == "CBF":
                cbf = theta[i]
            else:
                vals[n] = theta[i]
        params = ASLKineticParams(
            M0b=jnp.asarray(vals["M0b"]),
            T1b=jnp.asarray(vals["T1b"]),
            T1t=jnp.asarray(vals["T1t"]),
            alpha=jnp.asarray(vals["alpha"]),
            tau=jnp.asarray(vals["tau"]),
            delta=jnp.asarray(vals["delta"]),
            lambda_p=jnp.asarray(vals["lambda_p"]),
        )
        # asl_kinetic_signal returns (T,) for scalar cbf, but expects
        # cbf to be at-least-1D so it can broadcast.
        return asl_kinetic_signal(plds_arr, jnp.atleast_1d(cbf), params).reshape(-1)

    out = check_local_identifiability(fwd, theta0_jnp, names=tuple(fit_names), rtol=rtol)
    out["model"] = "asl_kinetic"
    out["plds"] = list(plds)
    out["nominal_cbf"] = float(nominal_cbf)
    return out


# ---------------------------------------------------------------------------
# Riera 8-state NVC model — for task-evoked BOLD
# ---------------------------------------------------------------------------

def riera_identifiability(
    fit_names: Sequence[str],
    stimulus: Float[Array, "T"],
    dt: float = 0.1,
    tr: float | None = None,
    nominal: dict[str, float] | None = None,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Local identifiability of selected Riera-NVC parameters from BOLD.

    The Riera 2007 model has 15 parameters across NO / adenosine / glutamate
    /  vascular sub-systems.  At any given task-stimulus protocol, only a
    handful are typically rank-recoverable — the rest live near a flat
    direction of the loss surface.  This helper runs the same Walter–Pronzato
    Jacobian-rank test as :func:`balloon_identifiability` against the Riera
    forward model, so we can pre-screen which ``fit_names`` are even worth
    optimising before we spend GPU minutes on a fit.

    Parameters
    ----------
    fit_names : Riera parameter names to test (subset of
        ``("kappa_no", "kappa_ade", "gamma_no", "gamma_ade", "c_no", "c_ade",
        "tau_a", "tau_c", "tau_v", "alpha_a", "alpha_c", "alpha_v", "E0",
        "phi", "tau_m")``).
    stimulus : neural-input (or boxcar event) timecourse on the ODE grid.
    dt : ODE timestep (s).
    tr : sub-sample BOLD to this TR before computing the Jacobian
        (matches the optimisation setup).
    nominal : overrides for any of the 15 Riera parameters; defaults to the
        Riera 2007 literature values for any name not given.
    rtol : relative threshold for declaring a singular value zero.
    """
    from vpjax.hemodynamics.bold import BOLDParams, observe_bold
    from vpjax.hemodynamics.inversion import _RIERA_ALL_NAMES, _RIERA_BOUNDS
    from vpjax.hemodynamics.riera import (
        RieraParams,
        riera_to_balloon,
        solve_riera,
    )
    from vpjax._types import BalloonState

    valid = _RIERA_ALL_NAMES
    bad = [n for n in fit_names if n not in valid]
    if bad:
        raise ValueError(f"unknown Riera param(s): {bad}; valid: {valid}")

    nominal = dict(nominal or {})
    base = RieraParams()
    base_vals = {n: float(nominal.get(n, getattr(base, n))) for n in valid}
    bold_params = BOLDParams()

    sub = int(round(tr / dt)) if tr is not None else 1

    def forward_fn(theta: Float[Array, "P"]) -> Float[Array, "..."]:
        vals = dict(base_vals)
        for i, n in enumerate(fit_names):
            vals[n] = theta[i]
        rp = RieraParams(**{k: jnp.asarray(v) for k, v in vals.items()})
        _, traj = solve_riera(rp, stimulus, dt=dt)
        v, q = riera_to_balloon(traj)
        pseudo = BalloonState(s=jnp.zeros_like(v), f=traj.f_a, v=v, q=q)
        y = observe_bold(pseudo, bold_params)
        return jnp.ravel(y[::sub] if sub > 1 else y)

    theta0 = jnp.array([base_vals[n] for n in fit_names])
    result = check_local_identifiability(
        forward_fn, theta0, names=tuple(fit_names), rtol=rtol,
    )
    result["model"] = "riera"
    return result


__all__ = [
    "check_local_identifiability",
    "balloon_identifiability",
    "pet_static_suvr_identifiability",
    "pet_joint_identifiability",
    "asl_kinetic_identifiability",
    "riera_identifiability",
]
