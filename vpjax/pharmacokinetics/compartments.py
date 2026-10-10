"""Linear and nonlinear compartment models for plasma concentration.

The linear mamillary models (one, two or three compartments, with an
optional first-order absorption depot) are solved **exactly**, not
numerically.  Within an interval of constant infusion rate the system is

    dx/dt = A x + b,

whose solution is obtained from one matrix exponential of the augmented
generator

    M = [[A, b], [0, 0]],    exp(M·dt) = [[e^{A·dt}, A⁻¹(e^{A·dt} - I)b],
                                          [0,        1                ]]

so ``x(t+dt) = E x(t) + c`` with ``E, c`` read off the blocks of
``exp(M·dt)``.  The augmented form avoids inverting ``A`` and stays
well behaved as ``b → 0``.  This is the same identity the local
linearization integrator in :mod:`vpjax.integrators.local_linearization`
applies to a Jacobian; here it is exact rather than a linearization,
because the PK generator really is constant.

Using the closed form rather than an ODE solve is the convention in
NONMEM (``ADVAN1``-``ADVAN4``) and Pumas, and it matters: a population
fit evaluates the model once per subject per likelihood evaluation, and
exactness removes solver tolerance as a source of parameter bias.

Nonlinear elimination (Michaelis-Menten) has no closed form and is
integrated with Diffrax instead, interval by interval, with the same
bolus-at-the-boundary treatment.

Parameterisation follows clinical practice — clearances and volumes
(CL, V1, Q2, V2, Q3, V3, ka) rather than micro rate constants — because
that is the scale on which covariate models and between-subject
variability are specified, and it is far better conditioned for
estimation.

References
----------
Beal SL, Sheiner LB (1982) Am Stat 36:118-119
    "Estimating population kinetics"
Jacquez JA (1996) "Compartmental Analysis in Biology and Medicine", 3rd ed
Bauer RJ (2019) CPT Pharmacometrics Syst Pharmacol 8:525-537
    "NONMEM tutorial part I" (ADVAN closed-form solutions)
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float

from vpjax.pharmacokinetics.dosing import (
    DosingRegimen,
    Schedule,
    build_schedule,
    dose_terms,
)


class PKParams(eqx.Module):
    """Clearance-and-volume parameters for a mamillary compartment model.

    Units are left to the caller but must be mutually consistent: if
    volumes are in L and clearances in L/h, then times are in h and
    concentrations are amount/L.

    Attributes
    ----------
    CL : elimination clearance from the central compartment
    V1 : central volume of distribution
    Q2 : intercompartmental clearance to the first peripheral compartment
    V2 : first peripheral volume
    Q3 : intercompartmental clearance to the second peripheral compartment
    V3 : second peripheral volume
    ka : first-order absorption rate constant (depot models only)
    F  : bioavailable fraction of the dose entering the depot

    Defaults describe a one-compartment model; peripheral parameters are
    ignored unless the corresponding compartments are requested.
    """

    CL: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))
    V1: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(10.0))
    Q2: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))
    V2: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(20.0))
    Q3: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(0.5))
    V3: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(50.0))
    ka: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))
    F: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))


class PKStructure(eqx.Module):
    """Static structural choices: how many compartments, and absorption or not.

    Held separately from :class:`PKParams` so that the state dimension is
    a compile-time constant while every number stays differentiable.
    """

    n_peripheral: int = eqx.field(static=True, default=0)
    absorption: bool = eqx.field(static=True, default=False)

    @property
    def n_states(self) -> int:
        return 1 + self.n_peripheral + int(self.absorption)

    @property
    def central(self) -> int:
        """Index of the central compartment in the state vector."""
        return 1 if self.absorption else 0


ONE_COMPARTMENT = PKStructure(n_peripheral=0, absorption=False)
TWO_COMPARTMENT = PKStructure(n_peripheral=1, absorption=False)
THREE_COMPARTMENT = PKStructure(n_peripheral=2, absorption=False)
ORAL_ONE_COMPARTMENT = PKStructure(n_peripheral=0, absorption=True)
ORAL_TWO_COMPARTMENT = PKStructure(n_peripheral=1, absorption=True)


def rate_matrix(
    params: PKParams,
    structure: PKStructure = ONE_COMPARTMENT,
) -> Float[Array, "S S"]:
    """Build the linear generator ``A`` with state ordered as amounts.

    State order is ``[depot?, central, peripheral1?, peripheral2?]``.
    """
    s = structure.n_states
    c = structure.central
    A = jnp.zeros((s, s))

    k10 = params.CL / params.V1
    loss = k10

    if structure.n_peripheral >= 1:
        k12 = params.Q2 / params.V1
        k21 = params.Q2 / params.V2
        loss = loss + k12
        A = A.at[c, c + 1].set(k21)
        A = A.at[c + 1, c].set(k12)
        A = A.at[c + 1, c + 1].set(-k21)

    if structure.n_peripheral >= 2:
        k13 = params.Q3 / params.V1
        k31 = params.Q3 / params.V3
        loss = loss + k13
        A = A.at[c, c + 2].set(k31)
        A = A.at[c + 2, c].set(k13)
        A = A.at[c + 2, c + 2].set(-k31)

    A = A.at[c, c].set(-loss)

    if structure.absorption:
        A = A.at[0, 0].set(-params.ka)
        A = A.at[c, 0].set(params.ka)

    return A


def _propagator(
    A: Float[Array, "S S"],
    rate: Float[Array, "S"],
    dt: Float[Array, ""],
) -> tuple[Float[Array, "S S"], Float[Array, "S"]]:
    """Exact propagator for ``dx/dt = A x + rate`` over an interval ``dt``."""
    s = A.shape[0]
    M = jnp.zeros((s + 1, s + 1))
    M = M.at[:s, :s].set(A)
    M = M.at[:s, s].set(rate)
    P = jsl.expm(M * dt)
    return P[:s, :s], P[:s, s]


def solve_linear_pk(
    params: PKParams,
    regimen: DosingRegimen,
    schedule: Schedule,
    structure: PKStructure = ONE_COMPARTMENT,
) -> Float[Array, "K S"]:
    """Propagate compartment amounts over the whole schedule.

    Returns amounts at every grid point, *after* any bolus landing on
    that point has been added.
    """
    s = structure.n_states
    A = rate_matrix(params, structure)
    bolus_states, rate_states = dose_terms(regimen, schedule, s)

    if structure.absorption:
        bolus_states = bolus_states.at[:, 0].multiply(params.F)
        rate_states = rate_states.at[:, 0].multiply(params.F)

    def step(x, inputs):
        dt, rate, next_bolus = inputs
        E, c = _propagator(A, rate, dt)
        x_next = E @ x + c + next_bolus
        return x_next, x_next

    x0 = bolus_states[0]
    _, xs = jax.lax.scan(
        step, x0, (schedule.dt, rate_states, bolus_states[1:])
    )
    return jnp.concatenate([x0[None, :], xs], axis=0)


def concentration(
    params: PKParams,
    regimen: DosingRegimen,
    t_obs: Float[Array, "N"],
    structure: PKStructure = ONE_COMPARTMENT,
    schedule: Schedule | None = None,
) -> Float[Array, "N"]:
    """Central-compartment concentration at the requested observation times.

    Parameters
    ----------
    params    : PKParams
    regimen   : DosingRegimen
    t_obs     : observation times, shape (N,)
    structure : PKStructure — fixes the number of compartments
    schedule  : pre-built Schedule; built from *t_obs* if omitted.  Pass
                one explicitly inside an optimizer loop so the grid is
                constructed once rather than at every step.

    Returns
    -------
    Concentration in the central compartment, shape (N,)
    """
    if schedule is None:
        schedule = build_schedule(regimen, t_obs)
    amounts = solve_linear_pk(params, regimen, schedule, structure)
    return amounts[schedule.obs_idx, structure.central] / params.V1


class MichaelisMentenParams(eqx.Module):
    """Saturable elimination replacing (or adding to) linear clearance.

    Attributes
    ----------
    Vmax : maximum elimination rate (amount / time)
    Km   : concentration at half-maximal elimination rate
    """

    Vmax: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))
    Km: Float[Array, "..."] = eqx.field(default_factory=lambda: jnp.array(1.0))


def solve_nonlinear_pk(
    params: PKParams,
    regimen: DosingRegimen,
    schedule: Schedule,
    structure: PKStructure = ONE_COMPARTMENT,
    mm: MichaelisMentenParams | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 4096,
) -> Float[Array, "K S"]:
    """As :func:`solve_linear_pk`, but with saturable elimination.

    Integrated with Diffrax using an implicit solver (``Kvaerno5``),
    because saturable elimination makes the system stiff once
    concentrations fall well below ``Km``.  Doses are still applied at
    interval boundaries, so each ``diffeqsolve`` call sees a smooth
    right-hand side.

    Set *mm* to ``None`` to recover the linear model — useful for
    checking this path against :func:`solve_linear_pk`.
    """
    import diffrax as dfx

    s = structure.n_states
    c = structure.central
    A = rate_matrix(params, structure)
    bolus_states, rate_states = dose_terms(regimen, schedule, s)

    if structure.absorption:
        bolus_states = bolus_states.at[:, 0].multiply(params.F)
        rate_states = rate_states.at[:, 0].multiply(params.F)

    def vector_field(t, x, args):
        rate = args
        dx = A @ x + rate
        if mm is not None:
            conc = x[c] / params.V1
            dx = dx.at[c].add(-mm.Vmax * conc / (mm.Km + conc))
        return dx

    term = dfx.ODETerm(vector_field)
    solver = dfx.Kvaerno5()
    controller = dfx.PIDController(rtol=rtol, atol=atol)

    def step(x, inputs):
        t0, dt, rate, next_bolus = inputs
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t0 + dt,
            dt0=None,
            y0=x,
            args=rate,
            stepsize_controller=controller,
            max_steps=max_steps,
        )
        x_next = sol.ys[-1] + next_bolus
        return x_next, x_next

    x0 = bolus_states[0]
    _, xs = jax.lax.scan(
        step,
        x0,
        (schedule.t_grid[:-1], schedule.dt, rate_states, bolus_states[1:]),
    )
    return jnp.concatenate([x0[None, :], xs], axis=0)


def disposition_rate_constants(
    params: PKParams,
    structure: PKStructure = ONE_COMPARTMENT,
) -> Float[Array, "P"]:
    """Hybrid rate constants of disposition, fastest first.

    These are the negated eigenvalues of the disposition generator; the
    depot row is excluded, since absorption is not part of disposition.

    One and two compartments use the closed-form expressions, both
    because they are exact and because ``jnp.linalg.eigvals`` is only
    implemented on JAX's CPU backend and has no derivative rule — so
    the general path would make this function fail on GPU and inside
    ``grad``.  The three-compartment case falls back to it and inherits
    those restrictions; move the computation to CPU, or report
    ``secondary_parameters`` outside the fit, if that matters.
    """
    k10 = params.CL / params.V1

    if structure.n_peripheral == 0:
        return jnp.atleast_1d(k10)

    if structure.n_peripheral == 1:
        k12 = params.Q2 / params.V1
        k21 = params.Q2 / params.V2
        s = k10 + k12 + k21
        # Discriminant is non-negative for any positive parameters; the
        # clip only guards float round-off near the repeated-root case.
        disc = jnp.sqrt(jnp.maximum(s**2 - 4.0 * k10 * k21, 0.0))
        return jnp.stack([0.5 * (s + disc), 0.5 * (s - disc)])

    A = rate_matrix(params, structure)
    c = structure.central
    return jnp.sort(-jnp.real(jnp.linalg.eigvals(A[c:, c:])))[::-1]


def secondary_parameters(
    params: PKParams,
    structure: PKStructure = ONE_COMPARTMENT,
) -> dict[str, Float[Array, ""]]:
    """Derived quantities a pharmacometrics report is expected to contain.

    Half-lives are ``ln 2`` over the disposition rate constants from
    :func:`disposition_rate_constants`; the slowest is the terminal
    half-life.
    """
    lam = disposition_rate_constants(params, structure)
    out: dict[str, Float[Array, ""]] = {
        "k10": params.CL / params.V1,
        "half_lives": jnp.log(2.0) / lam,
        "t_half_terminal": jnp.log(2.0) / lam[-1],
        "Vss": params.V1
        + (params.V2 if structure.n_peripheral >= 1 else 0.0)
        + (params.V3 if structure.n_peripheral >= 2 else 0.0),
    }
    return out
