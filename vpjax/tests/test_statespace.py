"""Tests for the continuous-discrete state-space estimator.

The filter is exact for linear-Gaussian models, so most of these tests
compare it against an independently written reference: a textbook joint
Kalman filter whose discrete transition and process-noise matrices come
from quadrature rather than from Van Loan's identity.  Two
implementations agreeing through different algebra is the only check
worth much here, because a filter with a sign error or a stale
linearization still produces smooth, plausible state estimates.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vpjax.statespace import (
    build_observation_grid,
    parameter_uncertainty,
    fit_statespace,
    innovation_log_likelihood,
    ll_filter,
    ll_mean_step,
    ll_propagate,
    profile_likelihood,
    residual_diagnostics,
    rts_smoother,
    sequential_update,
    van_loan_noise,
)


# ---------------------------------------------------------------------------
# Independent reference implementations
# ---------------------------------------------------------------------------

def quadrature_noise(A, Q, dt, n=20001):
    """int_0^dt e^{As} Q e^{A's} ds by Simpson's rule — no Van Loan."""
    s = np.linspace(0.0, dt, n)
    import scipy.linalg as sla

    terms = np.stack([
        sla.expm(A * si) @ Q @ sla.expm(A.T * si) for si in s
    ])
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return (s[1] - s[0]) / 3.0 * np.einsum("i,ijk->jk", w, terms)


def reference_kalman(A, b, Q, H, r_diag, m0, P0, t, y, mask):
    """Joint-update continuous-discrete Kalman filter, written from scratch.

    Updates at each grid point then propagates, matching the ordering in
    ``ll_filter``.  Missing channels are dropped from the update by
    selecting rows, not by masking arithmetic.
    """
    import scipy.linalg as sla

    m, P = np.asarray(m0, float).copy(), np.asarray(P0, float).copy()
    R = np.diag(np.asarray(r_diag, float))
    means, covs, ll = [], [], 0.0

    for k in range(len(t)):
        rows = np.flatnonzero(np.asarray(mask[k]) > 0)
        if rows.size:
            Hk, yk = np.asarray(H)[rows], np.asarray(y[k])[rows]
            Rk = R[np.ix_(rows, rows)]
            S = Hk @ P @ Hk.T + Rk
            v = yk - Hk @ m
            K = P @ Hk.T @ np.linalg.inv(S)
            m = m + K @ v
            P = P - K @ Hk @ P
            sign, logdet = np.linalg.slogdet(2.0 * np.pi * S)
            ll += -0.5 * (logdet + v @ np.linalg.solve(S, v))
        means.append(m.copy())
        covs.append(P.copy())

        if k + 1 < len(t):
            dt = float(t[k + 1] - t[k])
            Phi = sla.expm(A * dt)
            Qd = quadrature_noise(A, Q, dt, n=2001)
            # Forcing for a constant b: int_0^dt e^{As} b ds
            d = A.shape[0]
            aug = np.zeros((d + 1, d + 1))
            aug[:d, :d], aug[:d, d] = A, b
            m = Phi @ m + sla.expm(aug * dt)[:d, d]
            P = Phi @ P @ Phi.T + Qd

    return np.array(means), np.array(covs), ll


def joint_update(m, P, y, H, r_diag, mask):
    """Single joint Kalman update over the present rows."""
    rows = np.flatnonzero(np.asarray(mask) > 0)
    Hk, yk = np.asarray(H)[rows], np.asarray(y)[rows]
    Rk = np.diag(np.asarray(r_diag)[rows])
    P, m = np.asarray(P, float), np.asarray(m, float)
    S = Hk @ P @ Hk.T + Rk
    v = yk - Hk @ m
    K = P @ Hk.T @ np.linalg.inv(S)
    _, logdet = np.linalg.slogdet(2.0 * np.pi * S)
    ll = -0.5 * (logdet + v @ np.linalg.solve(S, v))
    return m + K @ v, P - K @ Hk @ P, ll


class TestPropagators:
    def test_van_loan_matches_quadrature(self):
        """The process-noise integral is the one Van Loan's identity claims."""
        A = jnp.array([[-0.7, 0.2, 0.0],
                       [0.1, -1.3, 0.4],
                       [0.0, 0.3, -0.5]])
        Q = jnp.array([[0.30, 0.05, 0.0],
                       [0.05, 0.20, 0.02],
                       [0.00, 0.02, 0.10]])
        dt = 0.37
        Phi, Qd = van_loan_noise(A, Q, dt)

        import scipy.linalg as sla
        assert np.allclose(np.asarray(Phi), sla.expm(np.asarray(A) * dt), atol=1e-6)
        assert np.allclose(np.asarray(Qd), quadrature_noise(np.asarray(A), np.asarray(Q), dt), atol=1e-6)

    def test_process_noise_is_positive_semidefinite(self):
        A = jnp.array([[-2.0, 1.0], [0.5, -3.0]])
        Q = jnp.array([[0.4, 0.0], [0.0, 0.1]])
        _, Qd = van_loan_noise(A, Q, 0.5)
        assert jnp.all(jnp.linalg.eigvalsh(Qd) >= -1e-12)
        assert jnp.allclose(Qd, Qd.T)

    def test_zero_step_is_identity(self):
        """A zero-length interval must not move or inflate the state."""
        A = jnp.array([[-1.0, 0.3], [0.0, -2.0]])
        Q = jnp.eye(2) * 0.2
        Phi, Qd = van_loan_noise(A, Q, 0.0)
        assert jnp.allclose(Phi, jnp.eye(2), atol=1e-12)
        assert jnp.allclose(Qd, 0.0, atol=1e-12)

    def test_ll_mean_step_exact_for_linear_system(self):
        """For an affine field the LL step is the exact solution."""
        A = jnp.array([[-0.5, 0.2], [0.1, -0.9]])
        b = jnp.array([0.3, -0.1])

        def f(t, x, args):
            return A @ x + b

        x0 = jnp.array([1.0, -2.0])
        dt = 0.8
        x1, Phi = ll_mean_step(f, jnp.array(0.0), x0, jnp.array(dt))

        import scipy.linalg as sla
        expA = sla.expm(np.asarray(A) * dt)
        d = 2
        aug = np.zeros((d + 1, d + 1))
        aug[:d, :d], aug[:d, d] = np.asarray(A), np.asarray(b)
        want = expA @ np.asarray(x0) + sla.expm(aug * dt)[:d, d]
        assert np.allclose(np.asarray(x1), want, atol=1e-6)
        assert np.allclose(np.asarray(Phi), expA, atol=1e-6)

    def test_ll_step_survives_singular_jacobian(self):
        """A pure random walk has J = 0; the augmented form stays exact."""
        def f(t, x, args):
            return jnp.array([0.0, 0.0])

        x1, Phi = ll_mean_step(f, jnp.array(0.0), jnp.array([1.0, 2.0]), jnp.array(0.5))
        assert jnp.allclose(x1, jnp.array([1.0, 2.0]))
        assert jnp.allclose(Phi, jnp.eye(2))

    def test_ll_step_matches_diffrax_on_nonlinear_field(self):
        """Over a short step LL agrees with an accurate ODE solve."""
        import diffrax as dfx

        def f(t, x, args):
            return jnp.array([x[1], -jnp.sin(x[0]) - 0.1 * x[1]])

        x0 = jnp.array([0.4, -0.2])
        dt = 0.02
        x_ll, _ = ll_mean_step(f, jnp.array(0.0), x0, jnp.array(dt))
        sol = dfx.diffeqsolve(
            dfx.ODETerm(f), dfx.Tsit5(), t0=0.0, t1=dt, dt0=None, y0=x0,
            stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-12),
        )
        assert jnp.allclose(x_ll, sol.ys[-1], atol=1e-6)

    def test_ll_propagate_inflates_covariance(self):
        def f(t, x, args):
            return -0.5 * x

        m0, P0 = jnp.array([1.0]), jnp.array([[0.1]])
        Q = jnp.array([[0.4]])
        m1, P1, Phi = ll_propagate(f, jnp.array(0.0), m0, P0, jnp.array(1.0), Q)
        assert float(m1[0]) < float(m0[0])         # decays toward zero
        assert float(P1[0, 0]) > float(P0[0, 0])   # noise dominates the decay


class TestSequentialUpdate:
    def test_equals_joint_update(self):
        m = jnp.array([0.5, -1.0, 0.2])
        P = jnp.array([[1.0, 0.2, 0.0],
                       [0.2, 0.8, 0.1],
                       [0.0, 0.1, 0.5]])
        H = jnp.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.5],
                       [0.3, 0.0, 1.0]])
        r = jnp.array([0.1, 0.2, 0.05])
        y = jnp.array([0.7, -0.8, 0.1])
        mask = jnp.ones(3)

        m_s, P_s, ll_s, _ = sequential_update(m, P, y, H @ m, H, r, mask)
        m_j, P_j, ll_j = joint_update(m, P, y, H, r, mask)

        assert np.allclose(np.asarray(m_s), m_j, atol=1e-6)
        assert np.allclose(np.asarray(P_s), P_j, atol=1e-6)
        assert float(ll_s) == pytest.approx(ll_j, rel=1e-5)

    def test_masking_equals_dropping_the_channel(self):
        m = jnp.array([0.5, -1.0])
        P = jnp.array([[1.0, 0.3], [0.3, 0.6]])
        H = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        r = jnp.array([0.1, 0.2, 0.3])
        y = jnp.array([0.7, jnp.nan, 0.4])       # middle channel absent
        mask = jnp.array([1.0, 0.0, 1.0])

        m_s, P_s, ll_s, _ = sequential_update(m, P, y, H @ m, H, r, mask)
        m_j, P_j, ll_j = joint_update(m, P, y, H, r, mask)

        assert np.all(np.isfinite(np.asarray(m_s)))
        assert np.allclose(np.asarray(m_s), m_j, atol=1e-6)
        assert np.allclose(np.asarray(P_s), P_j, atol=1e-6)
        assert float(ll_s) == pytest.approx(ll_j, rel=1e-5)

    def test_fully_masked_update_is_a_no_op(self):
        m = jnp.array([0.5, -1.0])
        P = jnp.array([[1.0, 0.3], [0.3, 0.6]])
        H = jnp.eye(2)
        m_s, P_s, ll, _ = sequential_update(
            m, P, jnp.array([jnp.nan, jnp.nan]), H @ m, H,
            jnp.array([0.1, 0.1]), jnp.zeros(2),
        )
        assert jnp.allclose(m_s, m)
        assert jnp.allclose(P_s, P)
        assert float(ll) == 0.0

    def test_gradient_is_finite_with_missing_data(self):
        """NaNs in masked entries must not poison the gradient."""
        H = jnp.array([[1.0], [1.0]])
        y = jnp.array([0.7, jnp.nan])
        mask = jnp.array([1.0, 0.0])

        def loss(log_r):
            r = jnp.exp(log_r) * jnp.ones(2)
            _, _, ll, _ = sequential_update(
                jnp.array([0.0]), jnp.array([[1.0]]), y, jnp.array([0.0, 0.0]),
                H, r, mask,
            )
            return -ll

        g = jax.grad(loss)(jnp.array(-1.0))
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# A linear-Gaussian two-rate test problem
# ---------------------------------------------------------------------------

def two_rate_problem(seed=0, n_fast=120, every=8):
    """Latent 2-D OU state, one fast channel and one slow channel."""
    rng = np.random.default_rng(seed)
    A = np.array([[-0.8, 0.3], [0.0, -0.4]])
    b = np.array([0.1, 0.0])
    Q = np.array([[0.25, 0.0], [0.0, 0.09]])
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    r = np.array([0.04, 0.09])

    dt = 0.1
    t = np.round(np.arange(n_fast) * dt, 9)

    import scipy.linalg as sla
    Phi = sla.expm(A * dt)
    Qd = quadrature_noise(A, Q, dt, n=2001)
    L = np.linalg.cholesky(Qd + 1e-12 * np.eye(2))
    d = 2
    aug = np.zeros((d + 1, d + 1))
    aug[:d, :d], aug[:d, d] = A, b
    forcing = sla.expm(aug * dt)[:d, d]

    x = np.zeros((n_fast, 2))
    x[0] = rng.normal(size=2)
    for k in range(1, n_fast):
        x[k] = Phi @ x[k - 1] + forcing + L @ rng.normal(size=2)

    y = np.full((n_fast, 2), np.nan)
    mask = np.zeros((n_fast, 2))
    y[:, 0] = x[:, 0] + rng.normal(scale=np.sqrt(r[0]), size=n_fast)
    mask[:, 0] = 1.0
    slow = np.arange(0, n_fast, every)
    y[slow, 1] = x[slow, 1] + rng.normal(scale=np.sqrt(r[1]), size=slow.size)
    mask[slow, 1] = 1.0

    return dict(A=A, b=b, Q=Q, H=H, r=r, t=t, y=y, mask=mask, x=x, dt=dt)


class TestFilter:
    def test_matches_reference_kalman(self):
        """Exact agreement with an independently written joint-update filter."""
        p = two_rate_problem()
        A, b, H = jnp.asarray(p["A"]), jnp.asarray(p["b"]), jnp.asarray(p["H"])

        def f(t, x, args):
            return A @ x + b

        def h(t, x, args):
            return H @ x

        m0, P0 = jnp.zeros(2), jnp.eye(2)
        got = ll_filter(
            f, h, jnp.asarray(p["t"]), jnp.asarray(p["y"]), jnp.asarray(p["mask"]),
            jnp.asarray(p["Q"]), jnp.asarray(p["r"]), m0, P0,
        )
        want_m, want_P, want_ll = reference_kalman(
            p["A"], p["b"], p["Q"], p["H"], p["r"], np.zeros(2), np.eye(2),
            p["t"], p["y"], p["mask"],
        )

        assert np.allclose(np.asarray(got.mean), want_m, atol=1e-5)
        assert np.allclose(np.asarray(got.cov), want_P, atol=1e-5)
        assert float(got.log_likelihood) == pytest.approx(want_ll, rel=1e-5)

    def test_counts_only_present_observations(self):
        p = two_rate_problem(n_fast=40, every=8)
        assert float(p["mask"].sum()) == 40 + 5

    def test_slow_channel_still_constrains_its_state(self):
        """Dropping the slow channel entirely must widen its posterior."""
        p = two_rate_problem()
        A, b, H = jnp.asarray(p["A"]), jnp.asarray(p["b"]), jnp.asarray(p["H"])
        f = lambda t, x, args: A @ x + b
        h = lambda t, x, args: H @ x
        kw = dict(Q=jnp.asarray(p["Q"]), r_diag=jnp.asarray(p["r"]),
                  m0=jnp.zeros(2), P0=jnp.eye(2))

        full = ll_filter(f, h, jnp.asarray(p["t"]), jnp.asarray(p["y"]),
                         jnp.asarray(p["mask"]), **kw)
        dropped_mask = p["mask"].copy()
        dropped_mask[:, 1] = 0.0
        dropped = ll_filter(f, h, jnp.asarray(p["t"]), jnp.asarray(p["y"]),
                            jnp.asarray(dropped_mask), **kw)

        assert float(full.cov[-1, 1, 1]) < float(dropped.cov[-1, 1, 1])
        assert float(full.log_likelihood) != float(dropped.log_likelihood)

    def test_recovers_the_latent_state(self):
        """Filtered mean should track the simulated state within its own SD."""
        p = two_rate_problem()
        A, b, H = jnp.asarray(p["A"]), jnp.asarray(p["b"]), jnp.asarray(p["H"])
        res = ll_filter(
            lambda t, x, args: A @ x + b, lambda t, x, args: H @ x,
            jnp.asarray(p["t"]), jnp.asarray(p["y"]), jnp.asarray(p["mask"]),
            jnp.asarray(p["Q"]), jnp.asarray(p["r"]), jnp.zeros(2), jnp.eye(2),
        )
        err = np.asarray(res.mean) - p["x"]
        sd = np.sqrt(np.asarray(res.cov)[:, [0, 1], [0, 1]])
        # Mean absolute z-score should be around 0.8 for a calibrated filter.
        assert np.mean(np.abs(err / sd)) < 1.5

    def test_jit_and_gradient(self):
        p = two_rate_problem(n_fast=40)
        H = jnp.asarray(p["H"])

        @jax.jit
        def nll(log_theta):
            decay = jnp.exp(log_theta)
            A = jnp.array([[-decay, 0.3], [0.0, -0.4]])
            res = ll_filter(
                lambda t, x, args: A @ x, lambda t, x, args: H @ x,
                jnp.asarray(p["t"]), jnp.asarray(p["y"]), jnp.asarray(p["mask"]),
                jnp.asarray(p["Q"]), jnp.asarray(p["r"]), jnp.zeros(2), jnp.eye(2),
            )
            return -res.log_likelihood

        v, g = jax.value_and_grad(nll)(jnp.log(0.8))
        assert jnp.isfinite(v) and jnp.isfinite(g)


class TestSmoother:
    def test_smoothing_never_increases_variance(self):
        p = two_rate_problem()
        A, H = jnp.asarray(p["A"]), jnp.asarray(p["H"])
        res = ll_filter(
            lambda t, x, args: A @ x, lambda t, x, args: H @ x,
            jnp.asarray(p["t"]), jnp.asarray(p["y"]), jnp.asarray(p["mask"]),
            jnp.asarray(p["Q"]), jnp.asarray(p["r"]), jnp.zeros(2), jnp.eye(2),
        )
        sm = rts_smoother(res)
        fv = np.asarray(res.cov)[:, [0, 1], [0, 1]]
        sv = np.asarray(sm.cov)[:, [0, 1], [0, 1]]
        assert np.all(sv <= fv + 1e-8)

    def test_last_point_unchanged(self):
        p = two_rate_problem(n_fast=30)
        A, H = jnp.asarray(p["A"]), jnp.asarray(p["H"])
        res = ll_filter(
            lambda t, x, args: A @ x, lambda t, x, args: H @ x,
            jnp.asarray(p["t"]), jnp.asarray(p["y"]), jnp.asarray(p["mask"]),
            jnp.asarray(p["Q"]), jnp.asarray(p["r"]), jnp.zeros(2), jnp.eye(2),
        )
        sm = rts_smoother(res)
        assert jnp.allclose(sm.mean[-1], res.mean[-1])
        assert jnp.allclose(sm.cov[-1], res.cov[-1])

    def test_smoother_beats_filter_on_the_slow_channel(self):
        """Hindsight should help most where observations are sparsest."""
        p = two_rate_problem()
        A, H = jnp.asarray(p["A"]), jnp.asarray(p["H"])
        res = ll_filter(
            lambda t, x, args: A @ x, lambda t, x, args: H @ x,
            jnp.asarray(p["t"]), jnp.asarray(p["y"]), jnp.asarray(p["mask"]),
            jnp.asarray(p["Q"]), jnp.asarray(p["r"]), jnp.zeros(2), jnp.eye(2),
        )
        sm = rts_smoother(res)
        f_err = np.abs(np.asarray(res.mean)[:, 1] - p["x"][:, 1]).mean()
        s_err = np.abs(np.asarray(sm.mean)[:, 1] - p["x"][:, 1]).mean()
        assert s_err < f_err


class TestMultirate:
    def _streams(self):
        t_fast = np.round(np.arange(0.0, 10.0, 0.05), 9)
        t_slow = np.round(np.arange(0.0, 10.0, 2.0), 9)
        return {
            "eeg": {"time_s": t_fast, "values": np.sin(t_fast), "noise_sd": 0.1},
            "bold": {
                "time_s": t_slow,
                "values": np.stack([np.cos(t_slow), np.cos(t_slow) * 0.5], axis=1),
                "noise_sd": 0.2,
            },
        }

    def test_grid_is_the_union_without_interpolation(self):
        g = build_observation_grid(self._streams())
        assert g.channels == ("eeg", "bold[0]", "bold[1]")
        assert g.t.shape[0] == 200          # slow times are a subset of fast
        assert jnp.all(jnp.diff(g.t) > 0)
        m = np.asarray(g.mask)
        assert m[:, 0].sum() == 200         # fast channel everywhere
        assert m[:, 1].sum() == 5           # slow channel only where sampled
        # Absent entries are NaN, never a filled-in value.
        assert np.all(np.isnan(np.asarray(g.y)[m == 0]))

    def test_noise_variances_are_squared_sds(self):
        g = build_observation_grid(self._streams())
        assert np.allclose(np.asarray(g.r_diag), [0.01, 0.04, 0.04])

    def test_coverage_reports_the_imbalance(self):
        g = build_observation_grid(self._streams())
        cov = g.coverage()
        assert cov["eeg"] == pytest.approx(1.0)
        assert cov["bold[0]"] == pytest.approx(5 / 200)

    def test_max_dt_subdivides_without_adding_observations(self):
        streams = {
            "bold": {"time_s": np.array([0.0, 2.0, 4.0]),
                     "values": np.array([1.0, 2.0, 3.0]), "noise_sd": 0.1},
        }
        coarse = build_observation_grid(streams)
        fine = build_observation_grid(streams, max_dt=0.25)
        assert coarse.t.shape[0] == 3
        assert fine.t.shape[0] == 17
        assert float(coarse.mask.sum()) == float(fine.mask.sum()) == 3.0
        assert float(jnp.max(jnp.diff(fine.t))) <= 0.25 + 1e-9

    def test_valid_flags_are_honoured(self):
        t = np.arange(5.0)
        streams = {
            "x": {"time_s": t, "values": np.arange(5.0),
                  "valid": np.array([1, 1, 0, 1, 0], dtype=bool),
                  "noise_sd": 0.1},
        }
        g = build_observation_grid(streams)
        assert np.allclose(np.asarray(g.mask)[:, 0], [1, 1, 0, 1, 0])

    def test_nonfinite_values_are_dropped(self):
        streams = {
            "x": {"time_s": np.arange(4.0),
                  "values": np.array([1.0, np.nan, 3.0, np.inf]),
                  "noise_sd": 0.1},
        }
        g = build_observation_grid(streams)
        assert np.allclose(np.asarray(g.mask)[:, 0], [1, 0, 1, 0])

    def test_missing_noise_sd_is_an_error(self):
        with pytest.raises(ValueError, match="noise_sd"):
            build_observation_grid(
                {"x": {"time_s": np.arange(3.0), "values": np.arange(3.0)}}
            )

    def test_non_overlapping_streams_are_an_error(self):
        with pytest.raises(ValueError, match="no common instant"):
            build_observation_grid({
                "a": {"time_s": np.array([0.0, 1.0]), "values": np.zeros(2),
                      "noise_sd": 0.1},
                "b": {"time_s": np.array([5.0, 6.0]), "values": np.zeros(2),
                      "noise_sd": 0.1},
            })

    def test_unsorted_times_are_an_error(self):
        with pytest.raises(ValueError, match="increasing"):
            build_observation_grid({
                "a": {"time_s": np.array([1.0, 0.0, 2.0]), "values": np.zeros(3),
                      "noise_sd": 0.1},
            })

    def test_grid_feeds_the_filter_directly(self):
        g = build_observation_grid(self._streams(), max_dt=0.05)
        H = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5]])
        res = ll_filter(
            lambda t, x, args: jnp.array([-x[0], -0.2 * x[1]]),
            lambda t, x, args: H @ x,
            g.t, g.y, g.mask, jnp.eye(2) * 0.1, g.r_diag,
            jnp.zeros(2), jnp.eye(2),
        )
        assert jnp.isfinite(res.log_likelihood)
        assert float(res.n_obs) == float(g.mask.sum())


class TestEstimation:
    def _build(self, p):
        H = jnp.asarray(p["H"])

        def build(theta):
            decay, q0 = theta[0], theta[1]
            A = jnp.array([[-decay, 0.3], [0.0, -0.4]])
            return {
                "f": lambda t, x, args: A @ x,
                "h": lambda t, x, args: H @ x,
                "Q": jnp.diag(jnp.array([q0, 0.09])),
                "m0": jnp.zeros(2),
                "P0": jnp.eye(2),
            }

        return build

    def _grid(self, p):
        from vpjax.statespace import ObservationGrid
        return ObservationGrid(
            t=jnp.asarray(p["t"]), y=jnp.asarray(p["y"]),
            mask=jnp.asarray(p["mask"]), r_diag=jnp.asarray(p["r"]),
            channels=("fast", "slow"),
        )

    def test_recovers_decay_and_process_noise(self):
        p = two_rate_problem(seed=3, n_fast=600, every=8)
        fit = fit_statespace(
            self._build(p), self._grid(p), init=jnp.array([0.3, 0.6]),
            max_steps=200,
        )
        decay, q0 = float(fit["theta"][0]), float(fit["theta"][1])
        assert decay == pytest.approx(0.8, rel=0.35)
        assert q0 == pytest.approx(0.25, rel=0.5)

    def test_likelihood_peaks_near_the_truth(self):
        p = two_rate_problem(seed=4, n_fast=400)
        build, grid = self._build(p), self._grid(p)
        ll_true = innovation_log_likelihood(
            build, jnp.log(jnp.array([0.8, 0.25])), grid
        )
        for wrong in ([0.1, 0.25], [4.0, 0.25], [0.8, 0.02], [0.8, 3.0]):
            ll_wrong = innovation_log_likelihood(
                build, jnp.log(jnp.array(wrong)), grid
            )
            assert float(ll_true) > float(ll_wrong)

    def test_profile_slice_is_curved_at_the_optimum(self):
        p = two_rate_problem(seed=5, n_fast=400)
        vals = jnp.array([0.2, 0.4, 0.8, 1.6, 3.2])
        ll = profile_likelihood(
            self._build(p), self._grid(p), jnp.array([0.8, 0.25]), 0, vals
        )
        assert int(jnp.argmax(ll)) == 2       # peak at the true value
        assert float(ll[2]) - float(ll[0]) > 1.0

    def test_residuals_are_white_under_the_true_model(self):
        p = two_rate_problem(seed=6, n_fast=800)
        spec = self._build(p)(jnp.array([0.8, 0.25]))
        res = ll_filter(t=jnp.asarray(p["t"]), y=jnp.asarray(p["y"]),
                        mask=jnp.asarray(p["mask"]), r_diag=jnp.asarray(p["r"]),
                        **spec)
        d = residual_diagnostics(res)
        assert float(d["variance"]) == pytest.approx(1.0, abs=0.2)
        assert abs(float(d["lag1"])) < 0.15

    def test_residual_variance_flags_understated_noise(self):
        """Shrinking Q and R by 10x should inflate the innovation variance."""
        p = two_rate_problem(seed=7, n_fast=600)
        spec = self._build(p)(jnp.array([0.8, 0.025]))
        res = ll_filter(t=jnp.asarray(p["t"]), y=jnp.asarray(p["y"]),
                        mask=jnp.asarray(p["mask"]),
                        r_diag=jnp.asarray(p["r"]) / 10.0, **spec)
        assert float(residual_diagnostics(res)["variance"]) > 1.5

    def test_rejects_nonpositive_init(self):
        p = two_rate_problem(n_fast=40)
        with pytest.raises(ValueError, match="positive"):
            fit_statespace(self._build(p), self._grid(p), init=jnp.array([0.0, 0.1]))


class TestParameterUncertainty:
    """Curvature-based identifiability, the check whiteness cannot do."""

    def _grid(self, p):
        from vpjax.statespace import ObservationGrid
        return ObservationGrid(
            t=jnp.asarray(p["t"]), y=jnp.asarray(p["y"]),
            mask=jnp.asarray(p["mask"]), r_diag=jnp.asarray(p["r"]),
            channels=("fast", "slow"),
        )

    def test_identifiable_model_has_finite_standard_errors(self):
        p = two_rate_problem(seed=3, n_fast=400)
        H = jnp.asarray(p["H"])

        def build(theta):
            A = jnp.array([[-theta[0], 0.3], [0.0, -0.4]])
            return {
                "f": lambda t, x, args: A @ x,
                "h": lambda t, x, args: H @ x,
                "Q": jnp.diag(jnp.array([theta[1], 0.09])),
                "m0": jnp.zeros(2), "P0": jnp.eye(2),
            }

        u = parameter_uncertainty(
            build, self._grid(p), jnp.log(jnp.array([0.8, 0.25]))
        )
        assert bool(u["positive_definite"])
        assert jnp.all(jnp.isfinite(u["standard_error"]))
        assert jnp.all(u["standard_error"] < 0.5)
        assert bool(u["identifiable"])
        assert float(u["condition_number"]) < 1e4
        assert float(u["curvature_ratio"]) > 1e-3
        corr = np.asarray(u["correlation"])
        assert np.all(np.abs(corr) <= 1.0 + 1e-5)

    @pytest.mark.parametrize("redundancy", ["sum", "product"])
    def test_flat_direction_is_detected(self, redundancy):
        """Parameters entering only through one combination are not estimable.

        Both redundancies must be caught, and they look different: a
        product redundancy makes the information nearly singular (tiny
        smallest eigenvalue), while a sum redundancy makes it
        *non-concave* (negative smallest eigenvalue) with an
        unremarkable condition number. Only the signed curvature ratio
        sees both, which is why the verdict is based on it.
        """
        p = two_rate_problem(seed=3, n_fast=400)
        H = jnp.asarray(p["H"])
        combine = (
            (lambda a, b: a + b) if redundancy == "sum" else (lambda a, b: a * b)
        )
        start = 0.4 if redundancy == "sum" else 0.9

        def build(theta):
            A = jnp.array([[-combine(theta[0], theta[1]), 0.3], [0.0, -0.4]])
            return {
                "f": lambda t, x, args: A @ x,
                "h": lambda t, x, args: H @ x,
                "Q": jnp.diag(jnp.array([0.25, 0.09])),
                "m0": jnp.zeros(2), "P0": jnp.eye(2),
            }

        u = parameter_uncertainty(
            build, self._grid(p), jnp.log(jnp.array([start, start]))
        )
        assert not bool(u["identifiable"])
        assert not bool(u["positive_definite"])
        assert float(u["curvature_ratio"]) < 1e-6

    def test_correlation_names_the_trade_off(self):
        """A redundant pair should come out near-perfectly correlated."""
        p = two_rate_problem(seed=3, n_fast=400)
        H = jnp.asarray(p["H"])

        def build(theta):
            A = jnp.array([[-(theta[0] * theta[1]), 0.3], [0.0, -0.4]])
            return {
                "f": lambda t, x, args: A @ x,
                "h": lambda t, x, args: H @ x,
                "Q": jnp.diag(jnp.array([0.25, 0.09])),
                "m0": jnp.zeros(2), "P0": jnp.eye(2),
            }

        u = parameter_uncertainty(
            build, self._grid(p), jnp.log(jnp.array([0.9, 0.9]))
        )
        corr = np.asarray(u["correlation"])
        assert abs(corr[0, 1]) > 0.9

    def test_standard_errors_shrink_with_more_data(self):
        """Four times the data should roughly halve the standard errors."""
        H = jnp.asarray(two_rate_problem()["H"])

        def build(theta):
            A = jnp.array([[-theta[0], 0.3], [0.0, -0.4]])
            return {
                "f": lambda t, x, args: A @ x,
                "h": lambda t, x, args: H @ x,
                "Q": jnp.diag(jnp.array([theta[1], 0.09])),
                "m0": jnp.zeros(2), "P0": jnp.eye(2),
            }

        lt = jnp.log(jnp.array([0.8, 0.25]))
        se_short = parameter_uncertainty(
            build, self._grid(two_rate_problem(seed=3, n_fast=200)), lt
        )["standard_error"]
        se_long = parameter_uncertainty(
            build, self._grid(two_rate_problem(seed=3, n_fast=800)), lt
        )["standard_error"]
        ratio = np.asarray(se_short / se_long)
        assert np.all(ratio > 1.4) and np.all(ratio < 3.0)


class TestNonlinear:
    def test_recovers_a_nonlinear_rate_constant(self):
        """A saturating drift, observed with noise; estimate its rate."""
        rng = np.random.default_rng(11)
        kappa_true, dt, n = 1.5, 0.05, 600
        t = np.round(np.arange(n) * dt, 9)

        # dx = kappa (tanh(u) - x) dt + sqrt(q) dW, with a known input u.
        u = np.sin(2.0 * np.pi * t / 5.0)
        q, r = 0.02, 0.01
        x = np.zeros(n)
        for k in range(1, n):
            x[k] = x[k - 1] + kappa_true * (np.tanh(u[k - 1]) - x[k - 1]) * dt \
                 + np.sqrt(q * dt) * rng.normal()
        y = x + rng.normal(scale=np.sqrt(r), size=n)

        grid = build_observation_grid(
            {"obs": {"time_s": t, "values": y, "noise_sd": np.sqrt(r)}}
        )
        u_j = jnp.asarray(u)

        def build(theta):
            kappa, qq = theta[0], theta[1]
            return {
                "f": lambda tt, xx, args: kappa * (jnp.tanh(args[1]) - xx),
                "h": lambda tt, xx, args: xx,
                "Q": jnp.array([[qq]]),
                "m0": jnp.zeros(1),
                "P0": jnp.eye(1) * 0.1,
                "args": None,
                "inputs": u_j,
            }

        fit = fit_statespace(build, grid, init=jnp.array([0.5, 0.05]), max_steps=200)
        assert float(fit["theta"][0]) == pytest.approx(kappa_true, rel=0.3)
        d = residual_diagnostics(fit["filter"])
        assert abs(float(d["lag1"])) < 0.2

    def test_inputs_reach_the_vector_field(self):
        """A per-step input must actually change the trajectory."""
        t = jnp.linspace(0.0, 1.0, 11)
        y = jnp.zeros((11, 1))
        mask = jnp.zeros((11, 1))        # no observations: pure propagation

        def run(u):
            return ll_filter(
                lambda tt, xx, args: jnp.array([args[1]]),
                lambda tt, xx, args: xx,
                t, y, mask, jnp.zeros((1, 1)), jnp.array([1.0]),
                jnp.zeros(1), jnp.eye(1), args=None, inputs=u,
            ).mean[-1, 0]

        assert float(run(jnp.ones(11))) == pytest.approx(1.0, abs=1e-5)
        assert float(run(jnp.full((11,), 2.0))) == pytest.approx(2.0, abs=1e-5)
