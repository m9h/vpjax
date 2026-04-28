"""Symbolic / Gröbner-basis identifiability for sums-of-exponentials models.

Companion to :mod:`vpjax.identifiability` (which does the local
Walter & Pronzato Jacobian-rank test).  This module is for *global*
structural identifiability — proving algebraically whether the
parameters of a model can be uniquely determined from the noise-free
observations, no matter what the data values are.

We focus on the sum-of-exponentials family because:

  * dMRI: ``S(b) = S_0 · Σ f_i · exp(-b · D_i)``       (multi-compartment)
  * R2* / R2:   ``S(TE) = S_0 · Σ f_i · exp(-TE · R_i)``
  * qBOLD:      ``S(TE) = S_0 · Σ f_i · exp(-TE · R2*_i)`` (extended)

The Gröbner-basis trick (Allard, Yvinec, Cherruault 1995; reproduced in
the ``sbi4dwi`` repo) is to substitute ``X_i = exp(-t_base · D_i)``
where ``t_base = gcd(t_1, t_2, …)``.  Each measurement at
``t_k = k · t_base`` becomes a polynomial ``Σ w_i · X_i^k`` and the
problem reduces to a polynomial system in
``(w_1, …, w_n, X_1, …, X_n)``.  A Gröbner basis with lex elimination
of parameters then either proves identifiability (basis is in
triangular form with one polynomial per parameter) or surfaces an
algebraic invariant — a polynomial in the *signals only* that vanishes
on the model manifold.

References
----------
Allard L, Yvinec M, Cherruault Y (1995) Math Modelling 9:153-160
Bellman R, Åström KJ (1970) Math Biosci 7:329-339
DiStefano JJ III (2014) Dynamic Systems Biology Modeling and Simulation
Pohjanpalo H (1978) Math Biosci 41:21-33
"""

from __future__ import annotations

from typing import Sequence

import sympy


# ---------------------------------------------------------------------------
# Polynomial-system construction
# ---------------------------------------------------------------------------

def define_exponential_components(n_compartments: int) -> dict[str, list]:
    """Symbolic variables for a sum-of-exponentials model.

    Returns a dict with keys::

        f      : list of fraction symbols f_1, …, f_n     (positive)
        D      : list of decay-rate symbols D_1, …, D_n   (positive)
        S0     : single proton-density-style symbol       (positive)
        w      : list of "weight" symbols w_i = S0·f_i    (free)
        X      : list of substitution symbols X_i         (free)
    """
    f = [sympy.Symbol(f"f_{i}", real=True, positive=True) for i in range(1, n_compartments + 1)]
    D = [sympy.Symbol(f"D_{i}", real=True, positive=True) for i in range(1, n_compartments + 1)]
    S0 = sympy.Symbol("S0", real=True, positive=True)
    w = [sympy.Symbol(f"w_{i}") for i in range(1, n_compartments + 1)]
    X = [sympy.Symbol(f"X_{i}") for i in range(1, n_compartments + 1)]
    return {"f": f, "D": D, "S0": S0, "w": w, "X": X}


def construct_exponential_polynomial_system(
    sample_points: Sequence[float],
    measured_signals: Sequence[sympy.Expr],
    n_compartments: int,
) -> tuple[list[sympy.Expr], list[sympy.Symbol], dict[str, list]]:
    """Build the polynomial system for ``S(t_k) = Σ w_i · X_i^k``.

    Parameters
    ----------
    sample_points : positive scalars — could be b-values (dMRI), echo
        times (multi-echo GRE), or post-label delays (multi-PLD ASL).
        We assume they are integer (or near-integer rational) multiples
        of their smallest non-zero element; the substitution
        ``X_i = exp(-t_base · D_i)`` then gives ``S(k·t_base) = Σ w_i · X_i^k``.
    measured_signals : per-sample symbolic placeholders ``y_k`` (or
        actual numeric values).  Use symbols when computing invariants.
    n_compartments : number of exponential components.

    Returns
    -------
    polynomials : list of ``Σ w_i · X_i^k − y_k`` (one per sample).
    variables   : ``[w_1, …, w_n, X_1, …, X_n]`` — the unknowns.
    components  : dict from :func:`define_exponential_components`.
    """
    nonzero = [t for t in sample_points if t > 1e-9]
    if not nonzero:
        raise ValueError("at least one positive sample point required")
    t_base = min(nonzero)
    ks = [int(round(t / t_base)) for t in sample_points]

    comps = define_exponential_components(n_compartments)
    w = comps["w"]
    X = comps["X"]

    polys: list[sympy.Expr] = []
    for k, y in zip(ks, measured_signals):
        model_term = sum(w[i] * (X[i] ** k) for i in range(n_compartments))
        polys.append(model_term - y)

    variables = w + X
    return polys, variables, comps


# ---------------------------------------------------------------------------
# Gröbner-basis identifiability
# ---------------------------------------------------------------------------

def analyze_groebner_identifiability(
    polynomials: Sequence[sympy.Expr],
    variables: Sequence[sympy.Symbol],
    *,
    order: str = "lex",
) -> dict:
    """Compute a Gröbner basis and read off identifiability.

    Returns a dict with::

        is_trivial      : bool — basis is {1} (system is inconsistent)
        is_zero_dim     : bool — finite solutions in algebraic closure
        basis_length    : int
        basis_degrees   : list of total degrees of basis elements
        univariate_polys: per variable, the lowest-degree univariate
                          polynomial in the basis (or None).  When such
                          a polynomial exists for *every* variable, the
                          ideal is zero-dimensional and the parameters
                          are globally identifiable up to the (finite)
                          ambiguity given by that polynomial's roots.
        basis_str       : list of ``str(p)`` for each basis element
        variables       : echo of the variable list (str)
    """
    gb = sympy.groebner(list(polynomials), list(variables), order=order)
    basis_list = list(gb)

    is_trivial = bool(basis_list) and basis_list == [sympy.Integer(1)]

    basis_degrees: list[int] = []
    for p in basis_list:
        try:
            basis_degrees.append(int(sympy.Poly(p, *variables).total_degree()))
        except sympy.PolynomialError:
            basis_degrees.append(-1)

    # Univariate polynomial per variable: a polynomial whose only free
    # symbols among `variables` is that single variable.
    var_set = set(variables)
    univariate: dict[str, str | None] = {str(v): None for v in variables}
    for p in basis_list:
        free = p.free_symbols & var_set
        if len(free) == 1:
            v = next(iter(free))
            existing = univariate.get(str(v))
            if existing is None:
                univariate[str(v)] = str(p)

    is_zero_dim = all(univariate[str(v)] is not None for v in variables)

    return {
        "is_trivial": is_trivial,
        "is_zero_dim": is_zero_dim,
        "basis_length": len(basis_list),
        "basis_degrees": basis_degrees,
        "univariate_polys": univariate,
        "basis_str": [str(p) for p in basis_list],
        "variables": [str(v) for v in variables],
    }


# ---------------------------------------------------------------------------
# Signal-only invariants (Pohjanpalo elimination)
# ---------------------------------------------------------------------------

def compute_signal_invariants(
    sample_points: Sequence[float],
    n_compartments: int,
) -> dict:
    """Find polynomial invariants ``P(y_1, …, y_M) = 0`` on the model manifold.

    These are relationships that the noise-free observations *must*
    satisfy regardless of the parameter values.  The presence of
    invariants demonstrates that the parameter-to-signal map is *not*
    surjective onto ``R^M``: data that violates the invariant cannot
    have come from this model.

    Parameters
    ----------
    sample_points : positive sample locations (b-values, TEs, PLDs).
    n_compartments : number of exponential components.

    Returns
    -------
    Dict::

        invariants : list of ``str(p)`` — polynomials in y_k only
        n_invariants : int
        signal_symbols : list of str — y_0, y_1, …
        n_unknowns : int
        n_observations : int
    """
    M = len(sample_points)
    y_syms = [sympy.Symbol(f"y_{i}") for i in range(M)]

    polys, params, _ = construct_exponential_polynomial_system(
        sample_points, y_syms, n_compartments,
    )

    # Lex elimination: parameters first, signals last.  The tail of the
    # basis contains polynomials free of parameters — those are the
    # invariants.
    all_vars = list(params) + list(y_syms)
    gb = sympy.groebner(polys, all_vars, order="lex")
    param_set = set(params)

    invariants: list[sympy.Expr] = []
    for p in gb:
        if not (p.free_symbols & param_set):
            invariants.append(p)

    return {
        "invariants": [str(p) for p in invariants],
        "n_invariants": len(invariants),
        "signal_symbols": [str(s) for s in y_syms],
        "n_unknowns": 2 * n_compartments,
        "n_observations": M,
    }


# ---------------------------------------------------------------------------
# Convenience: multi-echo / multi-compartment screening
# ---------------------------------------------------------------------------

def multi_echo_identifiability(
    echo_times: Sequence[float],
    n_compartments: int,
) -> dict:
    """Quick identifiability summary for an MR multi-echo experiment.

    Models the magnitude signal as
    ``S(TE) = S_0 · Σ f_i · exp(-TE · R2*_i)`` and tests whether the
    measurement schedule (``echo_times``) is rich enough to determine
    the (S0, f_i, R2*_i) parameters.

    Parameters
    ----------
    echo_times : multi-echo TEs (s) — must be near-integer multiples of
        the smallest non-zero TE for the polynomial reduction to apply.
        For the WAND 7-echo MEGRE, ``[0.005, 0.010, …, 0.035]``
        satisfies this (multiples of 5 ms).
    n_compartments : 1 (R2* fitting), 2 (intra/extravascular qBOLD),
        or more.

    Returns
    -------
    Dict combining :func:`analyze_groebner_identifiability` (with
    parameter symbols only) and :func:`compute_signal_invariants`.
    """
    # 1. Identifiability of the parameters at concrete y_k symbols.
    y_syms = [sympy.Symbol(f"y_{i}") for i in range(len(echo_times))]
    polys, params, comps = construct_exponential_polynomial_system(
        echo_times, y_syms, n_compartments,
    )
    param_only = analyze_groebner_identifiability(polys, params)

    # 2. Signal-space invariants (parameter-free polynomial relations).
    invariants = compute_signal_invariants(echo_times, n_compartments)

    return {
        "n_compartments": n_compartments,
        "n_echoes": len(echo_times),
        "echo_times": list(echo_times),
        "param_identifiability": param_only,
        "signal_invariants": invariants,
        "summary": (
            f"{n_compartments}-compartment model, {len(echo_times)} echoes: "
            f"{2 * n_compartments} unknowns, {len(echo_times)} equations; "
            f"{invariants['n_invariants']} algebraic invariant(s); "
            f"parameters {'globally identifiable' if param_only['is_zero_dim'] else 'not zero-dimensional'}."
        ),
    }


__all__ = [
    "define_exponential_components",
    "construct_exponential_polynomial_system",
    "analyze_groebner_identifiability",
    "compute_signal_invariants",
    "multi_echo_identifiability",
]
