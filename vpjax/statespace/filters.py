"""Local-linearization filter and smoother for continuous-discrete models.

Estimates the latent physiological state of a nonlinear SDE observed at
discrete, possibly asynchronous times:

    dx = f(t, x) dt + L dW,     y_k = h(t_k, x_k) + e_k,  e_k ~ N(0, R)

This is the estimator the simultaneous EEG-fMRI problem needs, for one
specific reason: EEG and BOLD observe the *same* latent state at rates
three orders of magnitude apart.  Fitting them separately and comparing
the results afterwards throws away the constraint that makes a
neurovascular model identifiable at all, and interpolating the slow
modality to the fast grid invents data.  A filter instead accepts a
channel being absent at a given instant and lets the remaining channels
constrain the state, which is what the ``mask`` argument is for.

Two implementation choices follow from that:

* **Sequential scalar updates.**  Observation channels are assimilated
  one at a time.  For diagonal ``R`` this is algebraically identical to
  the joint update, needs no matrix inverse, and — the reason it is used
  here — makes both masking and the log-likelihood decompose exactly per
  channel.  A joint update with a masked innovation covariance has to
  special-case the likelihood of the absent channels; this does not.

* **No interpolation anywhere.**  The grid is the union of the
  observation times (see :mod:`vpjax.statespace.multirate`), and every
  channel carries a presence mask.

The nonlinearity is handled by linearizing once per interval at the
current mean, as in Riera's LL filter.  That is a first-order
approximation: it is accurate when the posterior is narrow relative to
the curvature of ``f``, and it is not a particle filter.  Check it on
simulated data from the model being fitted before trusting it on real
data.

References
----------
Jimenez JC, Ozaki T (2003) J Time Ser Anal 24:463-482
    "Local linearization filters for nonlinear continuous-discrete state
    space models with multiplicative noise"
Riera JJ et al. (2004) NeuroImage 21:547-567
    "A state-space model of the hemodynamic approach"
Rauch HE, Tung F, Striebel CT (1965) AIAA J 3:1445-1450
    Fixed-interval smoothing
Sarkka S (2013) "Bayesian Filtering and Smoothing", CUP
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from vpjax.statespace.propagators import symmetrize, van_loan_noise

_LOG_2PI = float(jnp.log(2.0 * jnp.pi))


class FilterResult(eqx.Module):
    """Output of :func:`ll_filter`.

    Attributes
    ----------
    t           : grid times, shape (K,)
    mean        : filtered means E[x_k | y_{1:k}], shape (K, D)
    cov         : filtered covariances, shape (K, D, D)
    pred_mean   : one-step predicted means E[x_k | y_{1:k-1}], shape (K, D).
                  ``pred_mean[0]`` is the prior mean.
    pred_cov    : one-step predicted covariances, shape (K, D, D)
    transition  : linearized transition matrices for interval k -> k+1,
                  shape (K, D, D).  The last entry is the identity.
    log_likelihood : summed innovation log-likelihood over present channels
    residual    : standardized innovations, shape (K, M); zero where masked
    n_obs       : number of assimilated (unmasked) observations
    """

    t: Float[Array, "K"]
    mean: Float[Array, "K D"]
    cov: Float[Array, "K D D"]
    pred_mean: Float[Array, "K D"]
    pred_cov: Float[Array, "K D D"]
    transition: Float[Array, "K D D"]
    log_likelihood: Float[Array, ""]
    residual: Float[Array, "K M"]
    n_obs: Float[Array, ""]


def sequential_update(
    m: Float[Array, "D"],
    P: Float[Array, "D D"],
    y: Float[Array, "M"],
    h_pred: Float[Array, "M"],
    H: Float[Array, "M D"],
    r_diag: Float[Array, "M"],
    mask: Float[Array, "M"],
) -> tuple[Float[Array, "D"], Float[Array, "D D"], Float[Array, ""], Float[Array, "M"]]:
    """Assimilate M scalar observations one at a time.

    Exactly equivalent to the joint Kalman update when ``R`` is
    diagonal, because each channel's residual is taken against the mean
    already updated by the preceding channels.

    Parameters
    ----------
    m, P    : prior mean and covariance for this time point
    y       : observations, shape (M,).  Entries where ``mask`` is zero
              are never read, so they may be NaN.
    h_pred  : ``h(t, m)`` evaluated at the prior mean, shape (M,)
    H       : observation Jacobian at the prior mean, shape (M, D)
    r_diag  : observation noise variances, shape (M,); must be positive
    mask    : 1.0 where the channel is present, 0.0 where absent

    Returns
    -------
    m, P, log_likelihood, standardized_residuals
    """
    m_prior = m

    def body(carry, inputs):
        m_c, P_c, ll = carry
        y_i, hp_i, H_i, r_i, w_i = inputs

        present = w_i > 0.0
        # Sanitize before arithmetic: a missing entry may be NaN, and
        # 0 * NaN is NaN, so masking after the fact would not be enough.
        y_i = jnp.where(present, y_i, 0.0)
        hp_i = jnp.where(present, hp_i, 0.0)

        Ph = P_c @ H_i
        s = H_i @ Ph + r_i
        s = jnp.where(present, s, 1.0)

        pred = hp_i + H_i @ (m_c - m_prior)
        resid = jnp.where(present, y_i - pred, 0.0)
        gain = jnp.where(present, Ph / s, 0.0)

        m_next = m_c + gain * resid
        P_next = P_c - jnp.outer(gain, Ph) * w_i
        ll_next = ll + w_i * (-0.5 * (_LOG_2PI + jnp.log(s) + resid**2 / s))
        return (m_next, P_next, ll_next), resid / jnp.sqrt(s)

    (m_post, P_post, ll_total), resid = jax.lax.scan(
        body, (m, P, jnp.array(0.0)), (y, h_pred, H, r_diag, mask)
    )
    return m_post, symmetrize(P_post), ll_total, resid


def ll_filter(
    f: Callable[[Float[Array, ""], Float[Array, "D"], object], Float[Array, "D"]],
    h: Callable[[Float[Array, ""], Float[Array, "D"], object], Float[Array, "M"]],
    t: Float[Array, "K"],
    y: Float[Array, "K M"],
    mask: Bool[Array, "K M"] | Float[Array, "K M"],
    Q: Float[Array, "D D"],
    r_diag: Float[Array, "M"],
    m0: Float[Array, "D"],
    P0: Float[Array, "D D"],
    args: object = None,
    inputs: object = None,
) -> FilterResult:
    """Run the local-linearization filter over a multirate observation grid.

    Parameters
    ----------
    f      : drift ``f(t, x, args) -> dx/dt``, Diffrax argument order
    h      : observation map ``h(t, x, args) -> y``, shape (M,)
    t      : grid times, shape (K,), strictly increasing
    y      : observations on the grid, shape (K, M).  Masked entries are
             never read and may be NaN.
    mask   : presence indicator, shape (K, M)
    Q      : continuous-time process-noise covariance, shape (D, D)
    r_diag : observation noise variances, shape (M,), positive
    m0, P0 : prior mean and covariance at ``t[0]``
    args   : static arguments for *f* and *h*
    inputs : optional exogenous input, a pytree whose leaves have leading
             axis K.  When given, *f* and *h* are called with
             ``(args, inputs_k)`` in place of ``args`` — this is how a
             PK/PD drive time course enters the dynamics without being
             interpolated inside the vector field.

    Returns
    -------
    FilterResult
    """
    t = jnp.asarray(t, dtype=float)
    y = jnp.asarray(y, dtype=float)
    w = jnp.asarray(mask, dtype=float)
    dt = jnp.concatenate([jnp.diff(t), jnp.zeros((1,))])

    def step(carry, step_inputs):
        m_pred, P_pred = carry
        t_k, dt_k, y_k, w_k, u_k = step_inputs
        local = args if inputs is None else (args, u_k)

        h_pred = h(t_k, m_pred, local)
        H = jax.jacobian(h, argnums=1)(t_k, m_pred, local)
        m_post, P_post, ll, resid = sequential_update(
            m_pred, P_pred, y_k, h_pred, H, r_diag, w_k
        )

        # Propagate the posterior to the next grid point.  The final
        # dt is zero, so the last step is the identity and costs only
        # one wasted Jacobian.
        fx = f(t_k, m_post, local)
        J = jax.jacobian(f, argnums=1)(t_k, m_post, local)
        d = m_post.shape[0]
        aug = jnp.zeros((d + 1, d + 1))
        aug = aug.at[:d, :d].set(J)
        aug = aug.at[:d, d].set(fx)
        forcing = jax.scipy.linalg.expm(aug * dt_k)[:d, d]
        Phi, Qd = van_loan_noise(J, Q, dt_k)

        m_next = m_post + forcing
        P_next = symmetrize(Phi @ P_post @ Phi.T + Qd)

        out = (m_post, P_post, m_pred, P_pred, Phi, ll, resid, jnp.sum(w_k))
        return (m_next, P_next), out

    u = inputs if inputs is not None else jnp.zeros((t.shape[0],))
    (_, _), (mean, cov, pm, pc, Phi, lls, resid, counts) = jax.lax.scan(
        step, (m0, symmetrize(P0)), (t, dt, y, w, u)
    )

    return FilterResult(
        t=t,
        mean=mean,
        cov=cov,
        pred_mean=pm,
        pred_cov=pc,
        transition=Phi,
        log_likelihood=jnp.sum(lls),
        residual=resid,
        n_obs=jnp.sum(counts),
    )


class SmootherResult(eqx.Module):
    """Output of :func:`rts_smoother`.

    Attributes
    ----------
    t    : grid times, shape (K,)
    mean : smoothed means E[x_k | y_{1:K}], shape (K, D)
    cov  : smoothed covariances, shape (K, D, D)
    """

    t: Float[Array, "K"]
    mean: Float[Array, "K D"]
    cov: Float[Array, "K D D"]


def rts_smoother(result: FilterResult) -> SmootherResult:
    """Rauch-Tung-Striebel smoothing pass over a :class:`FilterResult`.

    Reuses the transition matrices the filter already linearized, so the
    smoother is consistent with the forward pass rather than
    re-linearizing at different points.

    Use the smoothed state to *report* a latent time course — a
    deconvolved neural drive, say — and the filter's log-likelihood to
    *estimate* parameters.  The smoother conditions on future data,
    which is what makes it the better state estimate and the wrong
    object for a one-step-ahead likelihood.
    """
    mean_f, cov_f = result.mean, result.cov
    pm, pc, Phi = result.pred_mean, result.pred_cov, result.transition

    def step(carry, inputs):
        m_next, P_next = carry
        m_k, P_k, Phi_k, pm_next, pc_next = inputs

        # G = P_k Phi_k' pc_next^{-1}, formed by a solve rather than an
        # explicit inverse.
        G = jnp.linalg.solve(pc_next, Phi_k @ P_k).T
        m_s = m_k + G @ (m_next - pm_next)
        P_s = symmetrize(P_k + G @ (P_next - pc_next) @ G.T)
        return (m_s, P_s), (m_s, P_s)

    _, (ms, Ps) = jax.lax.scan(
        step,
        (mean_f[-1], cov_f[-1]),
        (mean_f[:-1], cov_f[:-1], Phi[:-1], pm[1:], pc[1:]),
        reverse=True,
    )

    return SmootherResult(
        t=result.t,
        mean=jnp.concatenate([ms, mean_f[-1:]]),
        cov=jnp.concatenate([Ps, cov_f[-1:]]),
    )
