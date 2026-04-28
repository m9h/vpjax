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


__all__ = [
    "check_local_identifiability",
    "balloon_identifiability",
]
