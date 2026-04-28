"""Tests for vpjax.identifiability_symbolic — Gröbner-basis identifiability."""

import pytest

from vpjax.identifiability_symbolic import (
    compute_signal_invariants,
    construct_exponential_polynomial_system,
    define_exponential_components,
    multi_echo_identifiability,
)


class TestPolynomialSystem:

    def test_components_have_expected_keys(self):
        comps = define_exponential_components(2)
        assert set(comps) >= {"f", "D", "S0", "w", "X"}
        assert len(comps["f"]) == 2
        assert len(comps["D"]) == 2
        assert len(comps["w"]) == 2
        assert len(comps["X"]) == 2

    def test_polynomial_system_one_compartment(self):
        """For 1 compartment at 3 echoes [b0, 2b0, 3b0]:
        S(b_k) = w_1 · X_1^k → polys are [w_1·X_1 − y_0,
        w_1·X_1² − y_1, w_1·X_1³ − y_2]."""
        import sympy
        y_syms = sympy.symbols("y_0 y_1 y_2")
        polys, vars_, comps = construct_exponential_polynomial_system(
            sample_points=[1.0, 2.0, 3.0],
            measured_signals=y_syms,
            n_compartments=1,
        )
        assert len(polys) == 3
        # Variables ordered (w_1, X_1)
        assert [str(v) for v in vars_] == ["w_1", "X_1"]


class TestSymbolicIdentifiability:

    def test_one_compartment_two_samples_exact(self):
        """1 compartment, 2 samples: 2 unknowns (w_1, X_1), 2 equations,
        exactly determined → no signal invariants."""
        out = compute_signal_invariants(
            sample_points=[1.0, 2.0], n_compartments=1,
        )
        assert out["n_invariants"] == 0
        assert out["n_unknowns"] == 2
        assert out["n_observations"] == 2

    def test_one_compartment_three_samples_one_invariant(self):
        """1 compartment, 3 samples: over-determined; the third
        observation must satisfy one polynomial in (y_0, y_1, y_2)."""
        out = compute_signal_invariants(
            sample_points=[1.0, 2.0, 3.0], n_compartments=1,
        )
        # Should produce at least one invariant — typically y_1² − y_0·y_2
        # (the generic mono-exponential consistency relation).
        assert out["n_invariants"] >= 1
        # Confirm at least one invariant uses all 3 signals.
        used_y2 = any("y_2" in inv for inv in out["invariants"])
        assert used_y2

    def test_two_compartment_under_determined(self):
        """2 compartments, 3 samples: 4 unknowns vs 3 equations —
        infinitely many solutions; no invariants in the consistency
        sense (every observation tuple is reachable)."""
        out = compute_signal_invariants(
            sample_points=[1.0, 2.0, 3.0], n_compartments=2,
        )
        assert out["n_invariants"] == 0
        assert out["n_unknowns"] == 4
        assert out["n_observations"] == 3


class TestMultiEchoIdentifiability:

    def test_one_compartment_seven_echoes_returns_summary(self):
        """Sanity: the public helper produces a non-empty summary."""
        out = multi_echo_identifiability(
            echo_times=[0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035],
            n_compartments=1,
        )
        assert out["n_compartments"] == 1
        assert out["n_echoes"] == 7
        assert "summary" in out
        # Over-determined: 1 compartment ⇒ 2 unknowns; 7 echoes ⇒ 5
        # invariants (over-determined consistency).
        assert out["signal_invariants"]["n_invariants"] >= 1

    def test_two_compartment_under_determined(self):
        out = multi_echo_identifiability(
            echo_times=[0.010, 0.020, 0.030],
            n_compartments=2,
        )
        # 2 compartments need 4 measurements minimum; 3 is below that.
        assert out["signal_invariants"]["n_invariants"] == 0
