"""Five brain-body system models, and whether a recording set can tell them apart.

The confound work in :mod:`vpjax.validation.statespace_recovery` asks
where an input acts: on the neural drive (``beta``) or on the
flow-inducing signal (``beta_vasc``).  That is a question about one
arrow.  This module asks the prior question -- which way the arrows
between brain and body point at all -- because several answers predict
the same amplitudes, and a study that does not separate them can report
a brain effect the data never established.

The model adds one slow autonomic state ``a`` to the augmented Balloon,
standing for sympathetic/vagal tone and observed through heart rate::

    dz/dt = -z / tau_z + beta * u      + g_bp * a
    ds/dt =  z - kappa * s - gamma * (f - 1) + beta_vasc * u
    df/dt =  s
    dv/dt = (f - fout) / tau
    dq/dt = (f * E / E0 - fout * q / v) / tau
    da/dt = -a / tau_a + beta_a * u    + g_bb * z

Two coupling gains define the topology -- ``g_bb`` brain to body,
``g_bp`` body to brain -- and three gains say where the input enters:
``beta`` on the drive, ``beta_a`` on the autonomic state, ``beta_vasc``
on the flow-inducing signal.  The five topologies in
:data:`TOPOLOGIES` are corners of that space; see
``docs/brain_body_system_models.md`` for what each one claims.

The autonomic state is deliberately the *slow* one.  The existing
body-to-brain path in :mod:`vpjax.cardiac.baroreceptor`
(``modulate_neural_drive``) gates the drive by cardiac *phase*, which is
a within-cycle effect; ``g_bp`` here is the tonic path by which a
sustained autonomic change shifts cortical excitability over minutes,
which is the one confusable with a drug effect.

Nothing here is specific to any compound.  As with
``drug_shaped_input``, the input is a *shape* applied at a named entry
point, with its magnitude and time course supplied by the caller.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vpjax._types import BalloonParams
from vpjax.hemodynamics.bold import BOLDParams

# State layout
Z, S, F, V, Q, A = 0, 1, 2, 3, 4, 5
N_STATE = 6

_FLOOR = 1e-3

#: Gains that can be nonzero, in a fixed order.
GAIN_NAMES = ("beta", "beta_a", "beta_vasc", "g_bb", "g_bp")

#: Dynamical constants that are estimated in practice.  ``gamma``,
#: ``alpha`` and ``E0`` are excluded for the reason given in
#: :mod:`vpjax.validation.statespace_recovery`: they are not estimable
#: from a single run and fixing them is the honest choice.
CORE_NAMES = ("kappa", "tau", "tau_z", "tau_a")

#: Free loadings a channel may carry, and the state each scales.
#:
#: A real channel reports its state up to an unknown gain, so it is
#: tempting to treat unit loadings as an optimistic assumption.  For the
#: fast channel it is not needed: the drive's scale is pinned by the
#: Balloon, where ``z`` enters ``ds/dt`` with a coefficient of exactly
#: one, so ``gain_fast`` is estimable alongside everything else.
#:
#: The autonomic state has no such anchor.  ``a`` appears only through
#: ``g_bp`` (into the drive) and through its own channel loading, so
#: rescaling ``a -> c a`` while sending ``beta_a -> c beta_a``,
#: ``g_bp -> g_bp / c`` and ``gain_hr -> gain_hr / c`` leaves every
#: observation untouched.  That is an exact invariance, so a free
#: ``gain_hr`` always costs exactly one flat direction, at any run
#: length and with any number of channels.
#:
#: The consequence is that ``gain_hr = 1`` is the *identifying
#: convention* rather than an optimistic approximation: it defines the
#: autonomic state in the units of the series actually recorded, which
#: is what makes ``beta_a`` and ``g_bp`` mean anything.  Fixing it costs
#: no generality and is the only way the body arm is estimable at all.
LOADING_NAMES = {"fast": "gain_fast", "hr": "gain_hr"}

#: Observable channels, with the noise SD each is scaled by.  Scaling
#: by noise makes the Jacobian's singular values *detectable* units, so
#: the numeric rank reflects what a recording can actually resolve
#: rather than the arbitrary units each channel happens to carry.
CHANNELS = ("fast", "bold", "hr")


#: Default closed-loop gain for the ``loop`` topology.  See
#: :func:`loop_gain` for why it must stay below one.
DEFAULT_LOOP_GAIN = 0.25


def loop_gain(p: dict) -> float:
    """Closed-loop gain of the brain-body loop, ``g_bb g_bp tau_z tau_a``.

    Linearising the ``(z, a)`` subsystem about rest gives the Jacobian
    ``[[-1/tau_z, g_bp], [g_bb, -1/tau_a]]``, whose determinant is
    ``(1 - g_bb g_bp tau_z tau_a) / (tau_z tau_a)``.  So this quantity
    reaching one drives the determinant to zero, parks an eigenvalue at
    the origin and the loop integrates without bound: the BOLD runs to
    minus infinity rather than returning to baseline.

    This is a real constraint on the topology space and not an artefact
    of the parameterisation.  A brain-body loop with positive gain in
    both directions is only stable while the product of the two gains,
    scaled by both time constants, stays below one -- and the closer it
    runs to one, the more the loop amplifies a given input (by
    ``1 / (1 - L)`` at steady state).  Any nominal point chosen for the
    loop topology has to state where it sits relative to that boundary,
    because the answer governs both the excursion and how nonlinear the
    response is.
    """
    return float(
        p["g_bb"] * p["g_bp"] * p["tau_z"] * p["tau_a"]
    )


def nominal(target_loop_gain: float = DEFAULT_LOOP_GAIN) -> dict[str, float]:
    """Nominal operating point.

    The Balloon constants are the Stephan 2007 DCM values carried by
    :class:`~vpjax._types.BalloonParams`.  ``tau_a`` of 10 s is the
    order of a sustained autonomic adjustment, slower than the drive
    and far slower than a cardiac cycle.

    The gains are **matched, not measured**.  Each entry gain is set so
    a unit-peak input produces a unit steady-state excursion in the
    state it targets -- ``beta = 1 / tau_z``, ``beta_a = 1 / tau_a`` --
    and ``beta_vasc = 1`` is the value that matched ``beta = 0.5`` to the
    same 7.87% peak-to-peak BOLD excursion in
    ``scripts/autonomic_confound_case.py``.  Matching is what makes a
    failure to separate two topologies a statement about the
    *recording* rather than about one route having been handed a larger
    signal.

    The two coupling gains are derived from *target_loop_gain*, splitting
    it evenly between the directions: ``g_bb = sqrt(L) / tau_a`` and
    ``g_bp = sqrt(L) / tau_z``, so :func:`loop_gain` returns ``L``.  The
    even split is the neutral choice -- nothing known says either
    direction is the stronger -- and ``L`` must stay below one for the
    reason given in :func:`loop_gain`.
    """
    if not 0.0 <= target_loop_gain < 1.0:
        raise ValueError(
            f"target_loop_gain must be in [0, 1); {target_loop_gain} is at or "
            "past the stability boundary -- see loop_gain()"
        )
    bp = BalloonParams()
    tau_z, tau_a = 2.0, 10.0
    c = float(np.sqrt(target_loop_gain))
    return {
        "kappa": float(bp.kappa),
        "tau": float(bp.tau),
        "tau_z": tau_z,
        "tau_a": tau_a,
        "gamma": float(bp.gamma),
        "alpha": float(bp.alpha),
        "E0": float(bp.E0),
        # Entry gains: a unit input gives a unit excursion in its target.
        "beta": 1.0 / tau_z,
        "beta_a": 1.0 / tau_a,
        "beta_vasc": 1.0,
        # Coupling gains: even split of the closed-loop gain.
        "g_bb": c / tau_a,
        "g_bp": c / tau_z,
        # Channel loadings.  Unity is a *normalisation*, not a guess --
        # see LOADING_NAMES.
        "gain_fast": 1.0,
        "gain_hr": 1.0,
    }


#: Which gains are nonzero in each topology.  Everything not listed is
#: held at exactly zero, which is what makes these nested models.
TOPOLOGIES: dict[str, tuple[str, ...]] = {
    # input -> brain, body follows.  What a drug study usually claims.
    "efferent": ("beta", "g_bb"),
    # input -> body, brain follows.  The "it is all just heart rate" rival.
    "afferent": ("beta_a", "g_bp"),
    # input -> both, no coupling.  Correlated without causation.
    "common": ("beta", "beta_a"),
    # input -> brain inside a closed loop.  Physiologically likely.
    "loop": ("beta", "g_bb", "g_bp"),
    # no neural effect at all; BOLD moves through the vascular route.
    "measurement": ("beta_a", "beta_vasc"),
}


def brain_body_drift(p: dict):
    """``drift(x, u)`` for the six-state brain-body model.

    *p* carries every name in :data:`CORE_NAMES` and :data:`GAIN_NAMES`
    plus ``gamma``, ``alpha`` and ``E0``.  Any gain may be zero, which
    removes that arrow.
    """
    def drift(x, u):
        z, s, f, v, q, a = x
        f = jnp.maximum(f, _FLOOR)
        v = jnp.maximum(v, _FLOOR)
        fout = jnp.power(v, 1.0 / p["alpha"])
        extraction = 1.0 - jnp.power(1.0 - p["E0"], 1.0 / f)
        return jnp.stack([
            -z / p["tau_z"] + p["beta"] * u + p["g_bp"] * a,
            z - p["kappa"] * s - p["gamma"] * (f - 1.0) + p["beta_vasc"] * u,
            s,
            (f - fout) / p["tau"],
            (f * extraction / p["E0"] - fout * q / v) / p["tau"],
            -a / p["tau_a"] + p["beta_a"] * u + p["g_bb"] * z,
        ])
    return drift


def baseline_state() -> jnp.ndarray:
    """Resting state: zero drive and tone, unit flow, volume and content."""
    return jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])


def observe(
    x,
    bold_params: BOLDParams | None = None,
    gain_fast: float = 1.0,
    gain_hr: float = 1.0,
):
    """``(fast, bold, hr)`` from a state trajectory of shape ``(T, 6)``.

    ``fast`` is the drive itself, standing for a band-limited EEG power
    envelope rather than raw EEG, exactly as in
    :mod:`vpjax.validation.statespace_recovery`; ``hr`` is the autonomic
    state.  Both are the quantity a real channel reports up to a
    loading, which is why the identifiability question below is asked
    about the dynamical parameters and not about those loadings.
    """
    bp = BOLDParams() if bold_params is None else bold_params
    v = jnp.maximum(x[:, V], _FLOOR)
    q = x[:, Q]
    bold = bp.V0 * (
        bp.k1 * (1.0 - q) + bp.k2 * (1.0 - q / v) + bp.k3 * (1.0 - v)
    )
    return gain_fast * x[:, Z], bold, gain_hr * x[:, A]


def integrate(p: dict, u: jnp.ndarray, dt: float):
    """Heun integration of the noise-free model over a given input series."""
    drift = brain_body_drift(p)

    def step(x, u_pair):
        u0, u1 = u_pair
        k1 = drift(x, u0)
        k2 = drift(x + dt * k1, u1)
        return x + 0.5 * dt * (k1 + k2), x

    pairs = (u[:-1], u[1:])
    _, traj = jax.lax.scan(step, baseline_state(), pairs)
    return traj


def input_shape(t: np.ndarray, onset: float, rise: float, decay: float) -> np.ndarray:
    """Unit-peak input shape: zero before *onset*, then a Bateman rise and decay.

    A shape, not a pharmacokinetic model of any compound -- the same
    object as ``statespace_recovery.drug_shaped_input``, evaluated here
    on a supplied grid so the same series can drive the ODE and the
    observations.
    """
    tau_r, tau_d = float(rise), float(decay)
    s = np.maximum(np.asarray(t, dtype=float) - float(onset), 0.0)
    raw = np.exp(-s / tau_d) - np.exp(-s / tau_r)
    peak = np.max(raw) if np.max(raw) > 0 else 1.0
    return raw / peak


def duty_cycle_shape(
    t: np.ndarray,
    period: float,
    on_fraction: float = 5.0 / 16.0,
    onset: float = 0.0,
) -> np.ndarray:
    """Unit-amplitude on/off square wave of *period* seconds.

    The perturbation-arm counterpart to :func:`input_shape`.  A block
    perturbation puts all its energy near zero frequency, where it
    competes with drift and with the slow envelope the drug arm is
    looking for; a duty-cycled one concentrates its energy at ``1 /
    period`` and its harmonics; their separation from nuisance signals
    must be checked. The default
    ``on_fraction`` is 5 s on in 16, matching the cycle used by the
    TMS heart-brain-coupling literature (Dijkstra et al. 2024,
    entrainment read at 1/16 = 0.0625 Hz); see
    ``docs/periodic_perturbation.md``.

    This is an externally applied, externally timed perturbation shape.
    It carries no compound and no pharmacokinetics.
    """
    if not period > 0:
        raise ValueError(f"period must be positive, not {period}")
    if not 0.0 < on_fraction < 1.0:
        raise ValueError(f"on_fraction must lie in (0, 1), not {on_fraction}")
    s = np.asarray(t, dtype=float) - float(onset)
    phase = np.mod(s, float(period)) / float(period)
    return np.where((s >= 0.0) & (phase < float(on_fraction)), 1.0, 0.0)


def drive_series(t: np.ndarray, drive: str = "bateman", **kw) -> np.ndarray:
    """Dispatch to a named drive shape, so callers carry a string."""
    if drive == "bateman":
        return input_shape(
            t,
            kw.get("onset", 120.0),
            kw.get("rise", 60.0),
            kw.get("decay", 420.0),
        )
    if drive == "duty":
        return duty_cycle_shape(
            t,
            kw.get("period", 16.0),
            kw.get("on_fraction", 5.0 / 16.0),
            kw.get("onset", 0.0),
        )
    raise ValueError(f"unknown drive {drive!r}; expected 'bateman' or 'duty'")


# ---------------------------------------------------------------------------
# Step 0: can a recording set resolve a topology's parameters at all?
# ---------------------------------------------------------------------------

def topology_params(
    topology: str,
    overrides: dict | None = None,
    target_loop_gain: float = DEFAULT_LOOP_GAIN,
) -> dict:
    """Nominal parameters with only *topology*'s gains nonzero."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; have {tuple(TOPOLOGIES)}")
    p = nominal(target_loop_gain)
    active = TOPOLOGIES[topology]
    for g in GAIN_NAMES:
        if g not in active:
            p[g] = 0.0
    p.update(overrides or {})
    return p


def free_names(topology: str, loadings: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Parameters a fit would estimate for *topology*: its gains plus the time
    constants the topology gives meaning to.

    A time constant is only offered when its state is actually driven.
    In the ``measurement`` topology nothing reaches the neural drive --
    ``beta`` and ``g_bp`` are both zero -- so ``z`` never leaves rest and
    ``tau_z`` has no effect on any observation.  Leaving it in the free
    set would report a flat direction that says nothing about the
    recording: the parameter does not exist in that model.  Omitting it
    keeps every reported rank deficiency a statement about what the
    channels can resolve.
    """
    gains = TOPOLOGIES[topology]
    driven = {
        "tau_z": ("beta", "g_bp"),
        "tau_a": ("beta_a", "g_bb"),
    }
    names = tuple(
        n for n in CORE_NAMES
        if n not in driven or any(g in gains for g in driven[n])
    )
    # A loading on a channel whose state is never driven is as
    # meaningless as that state's time constant, so it is dropped on the
    # same rule.
    anchor = {"fast": ("beta", "g_bp"), "hr": ("beta_a", "g_bb")}
    extra = tuple(
        LOADING_NAMES[c] for c in loadings
        if c in LOADING_NAMES and any(g in gains for g in anchor[c])
    )
    return names + gains + extra


def topology_identifiability(
    topology: str,
    observers: tuple[str, ...] = CHANNELS,
    duration: float = 900.0,
    dt: float = 0.1,
    tr: float = 2.0,
    fast_dt: float = 2.0,
    hr_dt: float = 1.0,
    onset: float = 120.0,
    rise: float = 60.0,
    decay: float = 420.0,
    snr: float = 3.0,
    loadings: tuple[str, ...] = (),
    overrides: dict | None = None,
    target_loop_gain: float = DEFAULT_LOOP_GAIN,
    rtol: float = 1e-5,
) -> dict:
    """Local structural identifiability of *topology* from *observers*.

    Answers the prerequisite question: **if this topology is the truth,
    can this recording set resolve its parameters at all?**  A rank
    deficiency here is not a power problem.  It means some combination
    of parameters is flat to first order no matter how long the run, so
    no amount of data rescues it and the recording set has to change.

    Parameters are tested as *multipliers* on the nominal values and the
    Jacobian is taken at a multiplier of one, so each column is a
    relative sensitivity -- the same quantity a log-scale fit sees, with
    no need to take the log of a gain that some other topology holds at
    zero.

    Every channel is divided by its own noise SD, and those SDs are set
    so each channel carries the same *snr* against its own
    peak-to-peak excursion under this topology.  That is deliberate: it
    makes a failure to separate parameters a statement about the
    structure of the recording set, not an artefact of one channel
    being quieter than another in whatever units it happens to use.  A
    per-channel SNR sweep is a separate question.

    Returns the dict from
    :func:`vpjax.identifiability.check_local_identifiability` plus
    ``topology``, ``observers``, ``names`` and ``noise_sd``.
    """
    from vpjax.identifiability import check_local_identifiability

    bad = [o for o in observers if o not in CHANNELS]
    if bad:
        raise ValueError(f"unknown observer(s) {bad}; have {CHANNELS}")
    if not observers:
        raise ValueError("at least one observer is required")

    p0 = topology_params(topology, overrides, target_loop_gain)
    names = free_names(topology, loadings)
    t = np.arange(0.0, duration, dt)
    u = jnp.asarray(input_shape(t, onset, rise, decay))
    steps = {
        "fast": max(int(round(fast_dt / dt)), 1),
        "bold": max(int(round(tr / dt)), 1),
        "hr": max(int(round(hr_dt / dt)), 1),
    }

    def channels(mult):
        p = dict(p0)
        for i, n in enumerate(names):
            p[n] = p0[n] * mult[i]
        traj = integrate(p, u, dt)
        fast, bold, hr = observe(
            traj, gain_fast=p["gain_fast"], gain_hr=p["gain_hr"]
        )
        return {"fast": fast, "bold": bold, "hr": hr}

    ones = jnp.ones(len(names))
    ref = channels(ones)
    # Noise SD per channel: equal SNR against this topology's own
    # excursion.  A channel the topology does not move has no scale of
    # its own, so it falls back to unit SD and contributes nothing,
    # which is the correct behaviour rather than a division by zero.
    noise_sd = {}
    for c in observers:
        ptp = float(jnp.ptp(ref[c]))
        noise_sd[c] = ptp / snr if ptp > 1e-12 else 1.0

    def forward_fn(mult):
        ch = channels(mult)
        return jnp.concatenate([
            jnp.ravel(ch[c][:: steps[c]]) / noise_sd[c] for c in observers
        ])

    result = check_local_identifiability(
        forward_fn, ones, names=names, rtol=rtol, jac="fwd",
    )
    result["topology"] = topology
    result["observers"] = tuple(observers)
    result["names"] = names
    result["noise_sd"] = noise_sd
    result["loop_gain"] = loop_gain(p0)
    result["loadings"] = tuple(loadings)
    return result


# ---------------------------------------------------------------------------
# Step 1: can two topologies be told apart?
# ---------------------------------------------------------------------------

#: Dynamical and noise parameters a fit estimates whatever the topology.
#:
#: Unlike the deterministic test above, a filtered model gives ``z`` and
#: ``a`` process noise, so both states fluctuate even where the input
#: never reaches them.  Those fluctuations pass through the Balloon, so
#: ``tau_z`` and ``q_z`` are informed by BOLD in every topology --
#: including ``measurement``, where the *input* never touches the drive
#: but spontaneous drive still does.  That is why this set is fixed
#: while :func:`free_names` prunes by what the input drives: the two
#: answer different questions.
FIT_CORE = ("kappa", "tau", "tau_z", "q_z", "tau_a", "q_a")


def simulate(
    topology: str,
    duration: float = 900.0,
    dt: float = 0.01,
    tr: float = 2.0,
    fast_dt: float = 2.0,
    hr_dt: float = 1.0,
    onset: float = 120.0,
    rise: float = 60.0,
    decay: float = 420.0,
    q_z: float = 0.01,
    q_a: float = 0.01,
    snr: float = 3.0,
    seed: int = 0,
    target_loop_gain: float = DEFAULT_LOOP_GAIN,
    overrides: dict | None = None,
    drive: str = "bateman",
    period: float = 16.0,
    on_fraction: float = 5.0 / 16.0,
) -> dict:
    """A noisy run generated by *topology*.

    Process noise enters the drive and the autonomic state; every
    observation channel gets Gaussian noise at *snr* against that
    channel's own noise-free peak-to-peak excursion, so no channel is
    privileged by the units it happens to use.  A channel the topology
    never moves gets unit SD and so carries noise only, which is the
    correct representation of a recording that saw nothing.
    """
    p = topology_params(topology, overrides, target_loop_gain)
    p = dict(p, q_z=q_z, q_a=q_a)
    t = np.arange(0.0, duration, dt)
    u = jnp.asarray(
        drive_series(
            t,
            drive,
            onset=onset,
            rise=rise,
            decay=decay,
            period=period,
            on_fraction=on_fraction,
        )
    )

    # Noise-free excursions set the observation noise, so the same
    # topology is scaled identically however many seeds are drawn.
    clean = integrate(p, u, dt)
    ref = dict(zip(CHANNELS, observe(clean)))
    noise_sd = {}
    for c in CHANNELS:
        ptp = float(jnp.ptp(ref[c]))
        noise_sd[c] = ptp / snr if ptp > 1e-12 else 1.0

    drift = brain_body_drift(p)
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    dw = rng.normal(size=(t.size - 1, 2)) * sqrt_dt
    sd = jnp.asarray([np.sqrt(q_z), np.sqrt(q_a)])

    def step(x, inp):
        u0, u1, w = inp
        k1 = drift(x, u0)
        k2 = drift(x + dt * k1, u1)
        x = x + 0.5 * dt * (k1 + k2)
        kick = jnp.zeros(N_STATE).at[Z].set(sd[0] * w[0]).at[A].set(sd[1] * w[1])
        return x + kick, x

    _, traj = jax.lax.scan(
        step, baseline_state(), (u[:-1], u[1:], jnp.asarray(dw))
    )
    fast, bold, hr = observe(traj)
    t_traj = t[:-1]

    out = {
        "topology": topology,
        "truth": p,
        "t_fine": t_traj,
        "x": np.asarray(traj),
        "u_fine": np.asarray(u[:-1]),
        "noise_sd": noise_sd,
        "snr": snr,
    }
    series = {"fast": (fast, fast_dt), "bold": (bold, tr), "hr": (hr, hr_dt)}
    for c, (y, every) in series.items():
        k = max(int(round(every / dt)), 1)
        # Three series per channel, and the distinction matters:
        #   noisefree_<c>  the deterministic response, no noise of any
        #                  kind -- this is what set the noise SD above;
        #   latent_<c>     the realised state, so process noise but not
        #                  observation noise;
        #   <c>            what a recording would hold.
        latent = np.asarray(y)[::k]
        out[f"t_{c}"] = t_traj[::k]
        out[f"noisefree_{c}"] = np.asarray(ref[c])[::k]
        out[f"latent_{c}"] = latent
        out[c] = latent + rng.normal(scale=noise_sd[c], size=latent.size)
    return out


def make_grid(sim: dict, observers: tuple[str, ...] = CHANNELS, max_dt: float = 0.25):
    """Observation grid over the channels in *observers*, in CHANNELS order."""
    from vpjax.statespace import build_observation_grid

    bad = [o for o in observers if o not in CHANNELS]
    if bad:
        raise ValueError(f"unknown observer(s) {bad}; have {CHANNELS}")
    streams = {
        c: {
            "time_s": sim[f"t_{c}"],
            "values": sim[c],
            "noise_sd": sim["noise_sd"][c],
        }
        for c in CHANNELS if c in observers
    }
    if not streams:
        raise ValueError("at least one observer is required")
    return build_observation_grid(streams, max_dt=max_dt)


def build_model(
    grid,
    sim: dict,
    topology: str,
    fit_names: tuple[str, ...] | None = None,
    jitter: float = 1e-8,
):
    """``build(theta)`` fitting *topology* to *sim*, whatever generated it.

    Every gain the fitted topology does not carry is held at **exactly
    zero**, never at the value that generated the data.  That is the
    point of the comparison: a model is being asked to account for the
    run using only the arrows it claims exist, which is what a real
    analysis does when it assumes a topology.  Holding an omitted gain
    at truth would smuggle the right answer into the wrong model.

    ``gamma``, ``alpha`` and ``E0`` stay fixed for the reason given in
    :mod:`vpjax.validation.statespace_recovery`.
    """
    names = fit_names if fit_names is not None else FIT_CORE + TOPOLOGIES[topology]
    active = set(TOPOLOGIES[topology])
    truth = sim["truth"]
    free = {n: i for i, n in enumerate(names)}
    cols = {c: grid.channels.index(c) for c in grid.channels}
    n_obs = len(grid.channels)
    inputs = jnp.asarray(
        np.interp(np.asarray(grid.t), sim["t_fine"], sim["u_fine"])
    )
    bp = BOLDParams()

    def build(theta):
        def get(name):
            return theta[free[name]] if name in free else truth[name]

        p = {n: get(n) for n in ("kappa", "tau", "tau_z", "tau_a")}
        p.update(gamma=truth["gamma"], alpha=truth["alpha"], E0=truth["E0"])
        # Omitted gains are zero, not truth.
        for g in GAIN_NAMES:
            p[g] = get(g) if g in active else 0.0
        p["gain_fast"] = 1.0
        p["gain_hr"] = 1.0
        drift = brain_body_drift(p)

        def f(t, x, args):
            u = args[1] if isinstance(args, tuple) else 0.0
            return drift(x, u)

        def h(t, x, args):
            v = jnp.maximum(x[V], _FLOOR)
            q = x[Q]
            bold = bp.V0 * (
                bp.k1 * (1.0 - q) + bp.k2 * (1.0 - q / v) + bp.k3 * (1.0 - v)
            )
            vals = {"fast": x[Z], "bold": bold, "hr": x[A]}
            out = jnp.zeros(n_obs)
            for c, col in cols.items():
                out = out.at[col].set(vals[c])
            return out

        Qm = (
            jnp.zeros((N_STATE, N_STATE))
            .at[Z, Z].set(get("q_z"))
            .at[A, A].set(get("q_a"))
            + jitter * jnp.eye(N_STATE)
        )
        return {
            "f": f,
            "h": h,
            "Q": Qm,
            "m0": baseline_state(),
            "P0": jnp.diag(jnp.array([1.0, 0.1, 0.1, 0.01, 0.01, 1.0])),
            "inputs": inputs,
        }

    return build


def fit_topology(
    sim: dict,
    topology: str,
    observers: tuple[str, ...] = CHANNELS,
    restarts: int = 3,
    max_steps: int = 200,
    seed: int = 1,
    max_dt: float = 0.25,
    init_scale: float = 0.5,
    fit_names: tuple[str, ...] | None = None,
    init: dict[str, float] | None = None,
) -> dict:
    """Fit *topology* to *sim*, with every gain it omits held at zero.

    Returns the log-likelihood, the parameter count, AIC, the estimates
    and the identifiability verdict.  AIC is reported because the
    topologies do not all carry the same number of gains, and most pairs
    of them are **not nested**: ``efferent`` sits inside ``loop``, but
    ``afferent`` and ``common`` sit inside neither, so a likelihood
    ratio is only a test for some of the pairs and AIC is what makes the
    rest comparable.  Where a pair *is* nested the likelihood ratio is
    the sharper statistic and should be preferred.

    ``init`` optionally overrides named starting values in physical units;
    omitted names retain the default start. Unknown, nonfinite and nonpositive
    values are rejected. Set ``restarts=0`` to test a specific start alone.
    ``standard_error`` is on the log scale (approximately relative SE),
    while ``standard_error_absolute`` is its delta-method physical-unit value.

    Starting values are a fixed fraction of the generating ones rather
    than the generating ones themselves, so a fit has to travel; and the
    same fraction is used for every topology so no model is handed a
    better start than its rivals.
    """
    from vpjax.statespace import fit_statespace, parameter_uncertainty

    grid = make_grid(sim, observers, max_dt=max_dt)
    names = fit_names if fit_names is not None else FIT_CORE + TOPOLOGIES[topology]
    build = build_model(grid, sim, topology, fit_names=names)
    truth = sim["truth"]
    # A gain this topology carries but the data did not generate has a
    # truth of zero, which is useless as a starting value; fall back to
    # the nominal magnitude so the fit starts somewhere it can move from.
    ref = nominal()
    initial = np.array([
        init_scale * (truth[n] if truth.get(n, 0.0) != 0.0 else ref[n])
        for n in names
    ])
    if init is not None:
        unknown = set(init) - set(names)
        if unknown:
            raise ValueError(f"init contains parameters not fitted: {sorted(unknown)}")
        for n, value in init.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"init[{n!r}] must be finite and strictly positive")
            initial[names.index(n)] = value
    out = fit_statespace(
        build, grid, init=jnp.asarray(initial),
        restarts=restarts, seed=seed, max_steps=max_steps,
    )
    unc = parameter_uncertainty(build, grid, out["log_theta"])
    k = len(names)
    ll = float(out["log_likelihood"])
    return {
        "topology": topology,
        "observers": tuple(observers),
        "names": names,
        "n_params": k,
        "log_likelihood": ll,
        "aic": 2.0 * k - 2.0 * ll,
        "estimate": {n: float(v) for n, v in zip(names, out["theta"])},
        "initial": dict(zip(names, map(float, initial))),
        "success": bool(out["success"]),
        "best_start": int(out["best_start"]),
        "all_log_likelihoods": np.asarray(out["all_log_likelihoods"]).tolist(),
        "standard_error_scale": "log_parameter",
        "standard_error": {
            n: float(v) for n, v in zip(names, unc["standard_error"])
        },
        "standard_error_absolute": {
            n: float(v * s) for n, v, s in
            zip(names, out["theta"], unc["standard_error"])
        },
        "positive_definite": bool(unc["positive_definite"]),
        "information_eigenvalues": np.asarray(unc["eigenvalues"]).tolist(),
        "correlation": np.asarray(unc["correlation"]).tolist(),
        "identifiable": bool(unc["identifiable"]),
        "curvature_ratio": float(unc["curvature_ratio"]),
    }
