"""Dosing regimens and the time grid a PK solver integrates over.

A regimen is a list of dose events.  Each event has a *time*, an
*amount*, a *duration* (0 for an intravenous bolus, > 0 for a
zero-order infusion) and a target *compartment*.  This is the same
information NONMEM and Pumas carry in their event records (``TIME``,
``AMT``, ``RATE``, ``CMT``), so a regimen built here can be written out
in that format for cross-checking against an established estimator.

Times are **static** and amounts are **traced**.  That split matters:
dose times are fixed design constants, so they can determine array
shapes and the integration grid, while amounts stay differentiable in
case a fit treats actual-vs-nominal dose as unknown.

Dose events are known-time discontinuities, not state-triggered events.
They are therefore handled by scanning over the intervals between them
and applying each bolus at an interval boundary — *not* by Diffrax's
event system, whose root finding is for conditions on the state.

References
----------
Bauer RJ (2019) CPT Pharmacometrics Syst Pharmacol 8:525-537
    "NONMEM tutorial part I"
Pumas-AI (2023) "Dosage regimens and events"
    https://docs.pumas.ai/stable/basics/doses_subjects_populations/
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


class DosingRegimen(eqx.Module):
    """A set of dose events.

    Attributes
    ----------
    amounts     : dose amounts, shape (D,) — traced, differentiable
    times       : dose start times (same units as the rate constants),
                  length D — static
    durations   : infusion durations, length D; 0 means an IV bolus — static
    compartments: target state index for each dose, length D — static.
                  0 is the first state of the model: the depot for an
                  oral model, the central compartment otherwise.
    """

    amounts: Float[Array, "D"]
    times: tuple[float, ...] = eqx.field(static=True)
    durations: tuple[float, ...] = eqx.field(static=True)
    compartments: tuple[int, ...] = eqx.field(static=True)

    @property
    def n_doses(self) -> int:
        return len(self.times)


def bolus(amount: float, time: float = 0.0, compartment: int = 0) -> DosingRegimen:
    """A single intravenous bolus."""
    return DosingRegimen(
        amounts=jnp.atleast_1d(jnp.asarray(amount, dtype=float)),
        times=(float(time),),
        durations=(0.0,),
        compartments=(int(compartment),),
    )


def infusion(
    amount: float,
    duration: float,
    time: float = 0.0,
    compartment: int = 0,
) -> DosingRegimen:
    """A single zero-order infusion delivering *amount* over *duration*."""
    if duration <= 0:
        raise ValueError("infusion duration must be > 0; use bolus() instead")
    return DosingRegimen(
        amounts=jnp.atleast_1d(jnp.asarray(amount, dtype=float)),
        times=(float(time),),
        durations=(float(duration),),
        compartments=(int(compartment),),
    )


def repeated(
    amount: float,
    interval: float,
    n_doses: int,
    start: float = 0.0,
    duration: float = 0.0,
    compartment: int = 0,
) -> DosingRegimen:
    """*n_doses* identical doses spaced *interval* apart."""
    times = tuple(float(start + i * interval) for i in range(n_doses))
    return DosingRegimen(
        amounts=jnp.full((n_doses,), float(amount)),
        times=times,
        durations=(float(duration),) * n_doses,
        compartments=(int(compartment),) * n_doses,
    )


def combine(*regimens: DosingRegimen) -> DosingRegimen:
    """Concatenate regimens, e.g. a loading bolus plus a maintenance infusion."""
    if not regimens:
        raise ValueError("combine() needs at least one regimen")
    return DosingRegimen(
        amounts=jnp.concatenate([r.amounts for r in regimens]),
        times=sum((r.times for r in regimens), ()),
        durations=sum((r.durations for r in regimens), ()),
        compartments=sum((r.compartments for r in regimens), ()),
    )


class Schedule(eqx.Module):
    """Pre-computed integration grid for a regimen and observation times.

    Built once from static times by :func:`build_schedule`, then reused
    across solver calls and optimizer steps.

    Attributes
    ----------
    t_grid          : merged, sorted grid of dose starts, infusion ends
                      and observation times, shape (K,)
    dt              : interval lengths, shape (K-1,)
    bolus_map       : (K, D) indicator — dose *d* is a bolus landing on
                      grid point *k*
    infusion_map    : (K-1, D) indicator — dose *d* is infusing during
                      interval *k*
    inv_duration    : (D,) 1/duration for infusions, 0 for boluses
    comp_idx        : (D,) target state index per dose
    obs_idx         : (N,) index into ``t_grid`` of each observation time
    """

    t_grid: Float[Array, "K"]
    dt: Float[Array, "K-1"]
    bolus_map: Float[Array, "K D"]
    infusion_map: Float[Array, "K-1 D"]
    inv_duration: Float[Array, "D"]
    comp_idx: Int[Array, "D"]
    obs_idx: Int[Array, "N"]


def build_schedule(
    regimen: DosingRegimen,
    t_obs: Float[Array, "N"],
    t0: float | None = None,
    _tol: int = 9,
) -> Schedule:
    """Merge dose events and observation times into one integration grid.

    Every dose start and every infusion end becomes a grid point, so no
    interval ever contains a discontinuity in the infusion rate.  Within
    an interval the rate is constant, which is what makes the matrix
    exponential propagator in :mod:`vpjax.pharmacokinetics.compartments`
    exact.
    """
    times = np.asarray(regimen.times, dtype=float)
    durations = np.asarray(regimen.durations, dtype=float)
    comps = np.asarray(regimen.compartments, dtype=int)
    obs = np.asarray(t_obs, dtype=float)

    start = float(np.min(times)) if t0 is None else float(t0)
    if obs.size and float(np.min(obs)) < start:
        raise ValueError("observation times precede the start of the grid")

    marks = np.concatenate([[start], times, times + durations, obs])
    marks = marks[marks >= start]
    grid = np.unique(np.round(marks, _tol))

    # A bolus (duration 0) lands on its own grid point; an infusion
    # contributes a rate over [time, time + duration) instead.
    is_bolus = durations <= 0.0
    bolus_map = np.zeros((grid.size, times.size))
    for d, (t, isb) in enumerate(zip(times, is_bolus)):
        if isb:
            k = int(np.searchsorted(grid, np.round(t, _tol)))
            bolus_map[k, d] = 1.0

    mid = 0.5 * (grid[:-1] + grid[1:])
    infusion_map = np.zeros((max(grid.size - 1, 0), times.size))
    for d, (t, dur, isb) in enumerate(zip(times, durations, is_bolus)):
        if not isb:
            infusion_map[:, d] = ((mid >= t) & (mid < t + dur)).astype(float)

    inv_duration = np.where(is_bolus, 0.0, 1.0 / np.where(is_bolus, 1.0, durations))
    obs_idx = np.searchsorted(grid, np.round(obs, _tol))
    if obs.size and not np.allclose(grid[obs_idx], obs, atol=10.0**-_tol):
        raise ValueError(
            "observation times did not land on grid points; this means the "
            "rounding tolerance is too coarse for the requested times"
        )

    return Schedule(
        t_grid=jnp.asarray(grid),
        dt=jnp.asarray(np.diff(grid)),
        bolus_map=jnp.asarray(bolus_map),
        infusion_map=jnp.asarray(infusion_map),
        inv_duration=jnp.asarray(inv_duration),
        comp_idx=jnp.asarray(comps),
        obs_idx=jnp.asarray(obs_idx),
    )


def dose_terms(
    regimen: DosingRegimen,
    schedule: Schedule,
    n_states: int,
) -> tuple[Float[Array, "K S"], Float[Array, "K-1 S"]]:
    """Expand a regimen onto the state space of a compartment model.

    Returns
    -------
    bolus_states : (K, S) amount added to each state at each grid point
    rate_states  : (K-1, S) infusion rate into each state per interval
    """
    import jax

    onehot = jax.nn.one_hot(schedule.comp_idx, n_states)  # (D, S)
    bolus_states = (schedule.bolus_map * regimen.amounts) @ onehot
    rates = schedule.infusion_map * (regimen.amounts * schedule.inv_duration)
    rate_states = rates @ onehot
    return bolus_states, rate_states
