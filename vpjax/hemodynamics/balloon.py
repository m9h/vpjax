"""Single-compartment Balloon-Windkessel model.

Implements the hemodynamic model from:
  Buxton RB et al. (1998) MRM — original Balloon model
  Friston KJ et al. (2000) NeuroImage — nonlinear extension
  Stephan KJ et al. (2007) NeuroImage — DCM parameterisation

The ODE maps a neural stimulus u(t) to venous volume v(t) and
deoxyhemoglobin content q(t) via an intermediate vasodilatory signal
s(t) and blood inflow f(t).

Usage with Diffrax::

    from vpjax.hemodynamics.balloon import BalloonWindkessel
    from vpjax._types import BalloonState, BalloonParams

    model = BalloonWindkessel(params=BalloonParams())
    y0 = BalloonState.steady_state()
    # ... integrate with diffrax.diffeqsolve(...)
"""

from __future__ import annotations

import diffrax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from vpjax._types import BalloonParams, BalloonState


class BalloonWindkessel(eqx.Module):
    """Balloon-Windkessel ODE as a Diffrax vector field.

    Parameters are stored as a ``BalloonParams`` module so the whole system
    is differentiable via ``jax.grad``.
    """

    params: BalloonParams

    def __call__(
        self,
        t: Float[Array, ""],
        y: BalloonState,
        args: Float[Array, "..."],
    ) -> BalloonState:
        """Evaluate the RHS of the Balloon-Windkessel ODE.

        Parameters
        ----------
        t : current time (unused — autonomous system)
        y : current BalloonState
        args : neural stimulus u(t) at current time (scalar or batched)

        Returns
        -------
        BalloonState with time-derivatives (ds/dt, df/dt, dv/dt, dq/dt).
        """
        u = args
        p = self.params

        # Vasodilatory signal
        ds = u - p.kappa * y.s - p.gamma * (y.f - 1.0)

        # Blood inflow (CBF)
        df = y.s

        # Blood volume — Windkessel (venous balloon)
        # tau * dv/dt = f - v^(1/alpha)
        fout = jnp.power(y.v, 1.0 / p.alpha)  # venous outflow
        dv = (y.f - fout) / p.tau

        # Deoxyhemoglobin content
        # tau * dq/dt = f * E(f)/E0 - q * v^(1/alpha-1)
        # E(f) = 1 - (1 - E0)^(1/f)  (oxygen extraction as fn of flow)
        extraction = 1.0 - jnp.power(1.0 - p.E0, 1.0 / y.f)
        dq = (y.f * extraction / p.E0 - fout * y.q / y.v) / p.tau

        return BalloonState(s=ds, f=df, v=dv, q=dq)


def solve_balloon(
    params: BalloonParams,
    stimulus: Float[Array, "T"],
    dt: float = 0.01,
    t0: float = 0.0,
    t1: float | None = None,
    solver: diffrax.AbstractSolver | None = None,
) -> tuple[Float[Array, "T"], BalloonState]:
    """Integrate the Balloon-Windkessel model over a stimulus time-course.

    Parameters
    ----------
    params : BalloonParams
    stimulus : 1-D array of neural stimulus, sampled at interval *dt*.
    dt : sampling interval of stimulus (seconds).
    t0 : start time.
    t1 : end time (default: t0 + len(stimulus)*dt).
    solver : Diffrax solver (default: Tsit5).

    Returns
    -------
    ts : time points (shape ``(T,)``)
    trajectory : BalloonState with arrays of shape ``(T,)``
    """
    n_steps = stimulus.shape[0]
    if t1 is None:
        t1 = t0 + n_steps * dt

    ts = jnp.linspace(t0, t1, n_steps)
    model = BalloonWindkessel(params=params)

    # Interpolate stimulus as a piecewise-constant control
    control = diffrax.LinearInterpolation(ts=ts, ys=stimulus)

    if solver is None:
        solver = diffrax.Tsit5()

    term = diffrax.ODETerm(
        lambda t, y, args: model(t, y, control.evaluate(t))
    )

    y0 = BalloonState.steady_state()

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=n_steps * 100,
    )

    return ts, sol.ys
