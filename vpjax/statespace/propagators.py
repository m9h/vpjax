"""Exact propagators for the linearized continuous-time dynamics.

A continuous-discrete filter needs three things from the interval
between two observations: where the mean goes, how the linearized flow
maps a perturbation, and how much process noise accumulates.  All three
have closed forms once the vector field is linearized, and all three are
obtained here from matrix exponentials rather than from a sub-integrator,
so no step-size tolerance enters the likelihood.

For ``dx = f(x) dt + L dW`` linearized at ``x_n`` with ``J = df/dx``:

* **Mean.**  The local linearization (LL) update

      x_{n+1} = x_n + (e^{J dt} - I) J^{-1} f(x_n)

  is computed from one exponential of the augmented generator
  ``[[J, f], [0, 0]]``, whose top-right column *is* that product.  This
  avoids forming ``J^{-1}``, so the step stays exact as ``J`` approaches
  singularity instead of needing the ridge that
  :func:`vpjax.integrators.ll_step` applies there.

* **Transition matrix.**  ``Phi = e^{J dt}``.

* **Process noise.**  ``Q_d = int_0^dt e^{J s} Q e^{J' s} ds`` with
  ``Q = L L'``, by Van Loan's identity: exponentiate
  ``[[-J, Q], [0, J']] dt`` and read ``Q_d`` off the blocks.  Computing
  this integral by quadrature instead is the usual source of filters
  whose covariance slowly loses positive-definiteness.

References
----------
Van Loan CF (1978) IEEE Trans Automat Contr 23:395-404
    "Computing integrals involving the matrix exponential"
Ozaki T (1992) Statistica Sinica 2:113-135
    "A bridge between nonlinear time series models and nonlinear
    stochastic dynamical systems"
Jimenez JC, Ozaki T (2003) J Time Ser Anal 24:463-482
    "Local linearization filters for nonlinear continuous-discrete
    state space models with multiplicative noise"
Riera JJ et al. (2004) NeuroImage 21:547-567
    "A state-space model of the hemodynamic approach" — the LL filter
    applied to neurovascular coupling
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float


def symmetrize(P: Float[Array, "D D"]) -> Float[Array, "D D"]:
    """Average a covariance with its transpose.

    Round-off makes a propagated covariance drift out of symmetry, and
    an asymmetric covariance makes the innovation variance in a filter
    update eventually go negative.  Applied after every propagation and
    update.
    """
    return 0.5 * (P + P.T)


def ll_mean_step(
    f: Callable[[Float[Array, ""], Float[Array, "D"], object], Float[Array, "D"]],
    t: Float[Array, ""],
    x: Float[Array, "D"],
    dt: Float[Array, ""],
    args: object = None,
) -> tuple[Float[Array, "D"], Float[Array, "D D"]]:
    """One local-linearization step, returning the mean and ``e^{J dt}``.

    Parameters
    ----------
    f    : vector field ``f(t, x, args) -> dx/dt`` (Diffrax convention)
    t    : interval start time
    x    : state at *t*, shape (D,)
    dt   : interval length
    args : passed through to *f*

    Returns
    -------
    x_next : state at ``t + dt``, shape (D,)
    Phi    : linearized transition matrix ``e^{J dt}``, shape (D, D)
    """
    d = x.shape[0]
    fx = f(t, x, args)
    J = jax.jacobian(f, argnums=1)(t, x, args)

    aug = jnp.zeros((d + 1, d + 1))
    aug = aug.at[:d, :d].set(J)
    aug = aug.at[:d, d].set(fx)
    E = jsl.expm(aug * dt)

    return x + E[:d, d], E[:d, :d]


def van_loan_noise(
    J: Float[Array, "D D"],
    Q: Float[Array, "D D"],
    dt: Float[Array, ""],
) -> tuple[Float[Array, "D D"], Float[Array, "D D"]]:
    """Discrete transition matrix and process-noise covariance.

    Returns ``(Phi, Q_d)`` with ``Phi = e^{J dt}`` and
    ``Q_d = int_0^dt e^{J s} Q e^{J' s} ds``.

    Parameters
    ----------
    J  : Jacobian of the drift, shape (D, D)
    Q  : continuous-time process-noise covariance ``L L'``, shape (D, D)
    dt : interval length
    """
    d = J.shape[0]
    M = jnp.zeros((2 * d, 2 * d))
    M = M.at[:d, :d].set(-J)
    M = M.at[:d, d:].set(Q)
    M = M.at[d:, d:].set(J.T)
    E = jsl.expm(M * dt)

    Phi = E[d:, d:].T
    Qd = Phi @ E[:d, d:]
    return Phi, symmetrize(Qd)


def ll_propagate(
    f: Callable[[Float[Array, ""], Float[Array, "D"], object], Float[Array, "D"]],
    t: Float[Array, ""],
    m: Float[Array, "D"],
    P: Float[Array, "D D"],
    dt: Float[Array, ""],
    Q: Float[Array, "D D"],
    args: object = None,
) -> tuple[Float[Array, "D"], Float[Array, "D D"], Float[Array, "D D"]]:
    """Propagate a Gaussian state over one interval.

    The mean follows the LL step; the covariance follows the linearized
    flow with Van Loan process noise.  Both linearizations use the same
    Jacobian, evaluated at the mean at the start of the interval, so the
    mean and covariance stay mutually consistent.

    Returns
    -------
    m_next, P_next, Phi
    """
    d = m.shape[0]
    fx = f(t, m, args)
    J = jax.jacobian(f, argnums=1)(t, m, args)

    aug = jnp.zeros((d + 1, d + 1))
    aug = aug.at[:d, :d].set(J)
    aug = aug.at[:d, d].set(fx)
    forcing = jsl.expm(aug * dt)[:d, d]

    Phi, Qd = van_loan_noise(J, Q, dt)
    return m + forcing, symmetrize(Phi @ P @ Phi.T + Qd), Phi
