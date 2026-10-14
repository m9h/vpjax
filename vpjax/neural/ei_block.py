"""The gnf-jax linearised E/I field as a state-space latent block.

Every Laplacian eigenmode of a graph Wilson-Cowan field, linearised about
its homogeneous steady state, is an independent two-state
Ornstein-Uhlenbeck process (Aqil et al. 2021, PLoS Comput Biol; Aqil
2026, Netw Neurosci; ported in gnf-jax as ``linearise``).  Stacked over
modes those processes are a linear-Gaussian latent that the
local-linearisation filter in :mod:`vpjax.statespace` propagates exactly,
with gradients through every parameter, so the block can be fitted
jointly with the haemodynamics without any neural-mass ODE code of its
own.

Why this rather than a free loading on an envelope: the E and I temporal
spectra are closed-form functions of the same parameters that set the
dynamics, so what an EEG band is predicted to do under a parameter change
is a *consequence* of the fitted model, not a sign the fit is free to
choose.  ``docs/ei_autonomic_coupling_plan.md`` says why that matters and
``docs/ei_spectral_observable.md`` says what pattern is worth testing.
Two limits, stated there and repeated here: a single-region use
(``eigs = [0.]``) keeps only the constant mode, so the field's spatial
structure is gone until a graph is supplied; and a linearised two-state
oscillator has one resonance per mode, so it cannot by itself produce an
opposite-sign two-band (theta-down, gamma-up) pattern.  It supplies the
latent E/I dynamics and a resonance/instability readout; the two-band
prediction still comes from a circuit run.

Framing.  An external time course enters at a *named entry point*
(``p``, ``q`` or one of the four coupling gains), linearised about the
steady state.  The block has no notion of any compound; what the time
course is, where it enters and with what provenance are the caller's.

State layout is ``x = [E_0, I_0, E_1, I_1, ...]`` (mode-major), in the
modal coordinates of whatever eigenbasis produced ``eigs``.  Time is in
seconds throughout vpjax, so ``tau_e``/``tau_i`` are seconds here; the
gnf-jax defaults of 1.0 give sub-hertz dynamics, and an alpha-band
resonance needs time constants of a few tens of milliseconds.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    import gnfjax  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "vpjax.neural.ei_block needs gnf-jax; install the 'ei' extra "
        "(uv pip install 'vpjax[ei]') or gnf-jax from github.com/m9h/gnf-jax"
    ) from exc

from gnfjax import (
    GNFParams,
    jacobian_eigenvalues,
    linearise,
    mode_covariance,
    steady_states,
    temporal_spectrum,
)

ENTRY_POINTS = ("p", "q", "alpha_ee", "alpha_ie", "alpha_ei", "alpha_ii")


class EIBlock(NamedTuple):
    """A linearised E/I field ready for the filter.

    ``A`` is ``(N, 2, 2)`` with ``dx_k = -A_k x_k dt + sqrt(D) dW``; ``D`` is
    the ``(2, 2)`` noise intensity shared by all modes; ``ess``/``iss`` the
    steady state the linearisation was taken about; ``n_valid`` how many
    admissible steady states gnf-jax found (zero means the parameters have
    none and every quantity here is NaN, which the likelihood will reject).
    """

    params: GNFParams
    eigs: jnp.ndarray
    A: jnp.ndarray
    D: jnp.ndarray
    ess: jnp.ndarray
    iss: jnp.ndarray
    kernel: str
    n_valid: jnp.ndarray

    @property
    def n_modes(self) -> int:
        return int(self.eigs.shape[0])

    @property
    def n_state(self) -> int:
        return 2 * self.n_modes

    # -- dynamics ---------------------------------------------------------

    def jacobian(self) -> jnp.ndarray:
        """Drift Jacobian ``(2N, 2N)``: block-diagonal ``-A_k``."""
        return jax.scipy.linalg.block_diag(*[-self.A[k] for k in range(self.n_modes)])

    @property
    def Q(self) -> jnp.ndarray:
        """Continuous-time process-noise covariance ``(2N, 2N)``."""
        return jnp.kron(jnp.eye(self.n_modes), self.D)

    @property
    def m0(self) -> jnp.ndarray:
        return jnp.zeros(self.n_state)

    def stationary_covariance(self) -> jnp.ndarray:
        """Closed-form Lyapunov solution per mode, block-diagonal ``(2N, 2N)``."""
        S = mode_covariance(self.A, self.D)
        return jax.scipy.linalg.block_diag(*[S[k] for k in range(self.n_modes)])

    def input_vector(self, entry: str, mode_weights=None) -> jnp.ndarray:
        """Drift gain ``(2N,)`` of a unit external time course at *entry*.

        Linearised about the steady state: a perturbation ``u`` of ``p``
        moves the E row by ``a u / tau_e`` with ``a = d_e E*(1 - d_e E*)``
        the logistic slope, of ``q`` the I row by ``b u / tau_i``, and of a
        coupling gain ``alpha_xy`` the x row by the slope times the
        presynaptic steady-state activity ``y*``.  *mode_weights* ``(N,)``
        are the modal coefficients of the input's spatial pattern; the
        default is unit weight on the constant mode (the one with
        eigenvalue closest to zero), i.e. a spatially uniform input.  With
        an orthonormal eigenbasis from ``gnfjax.laplacian_eigensystem`` a
        uniform pattern of amplitude one has coefficient ``sqrt(N)`` on
        that mode; pass it explicitly when the absolute scale matters.
        """
        if entry not in ENTRY_POINTS:
            raise ValueError(f"entry must be one of {ENTRY_POINTS}, got {entry!r}")
        p = self.params
        a = p.d_e * self.ess * (1.0 - p.d_e * self.ess)
        b = p.d_i * self.iss * (1.0 - p.d_i * self.iss)
        row = {
            "p": (a / p.tau_e, 0.0),
            "q": (0.0, b / p.tau_i),
            "alpha_ee": (a * self.ess / p.tau_e, 0.0),
            "alpha_ie": (-a * self.iss / p.tau_e, 0.0),
            "alpha_ei": (0.0, b * self.ess / p.tau_i),
            "alpha_ii": (0.0, -b * self.iss / p.tau_i),
        }[entry]
        if mode_weights is None:
            w = jnp.zeros(self.n_modes).at[jnp.argmax(self.eigs)].set(1.0)
        else:
            w = jnp.asarray(mode_weights, dtype=float)
        return (w[:, None] * jnp.asarray(row)[None, :]).reshape(-1)

    def drift(self, x, u=0.0, entry: str | None = None, mode_weights=None) -> jnp.ndarray:
        """``dx/dt`` for the stacked state, with an optional external time course."""
        xm = x.reshape(self.n_modes, 2)
        dx = -jnp.einsum("kij,kj->ki", self.A, xm).reshape(-1)
        if entry is not None:
            dx = dx + u * self.input_vector(entry, mode_weights)
        return dx

    # -- readouts ---------------------------------------------------------

    def excitatory(self, x) -> jnp.ndarray:
        """Per-mode E coefficients ``(N,)``."""
        return x.reshape(self.n_modes, 2)[:, 0]

    def inhibitory(self, x) -> jnp.ndarray:
        return x.reshape(self.n_modes, 2)[:, 1]

    def eigenvalues(self) -> jnp.ndarray:
        """Per-mode Jacobian eigenvalues ``(N, 2)``, complex."""
        return jacobian_eigenvalues(self.params, self.eigs, self.ess, self.iss, self.kernel)

    def distance_to_instability(self) -> jnp.ndarray:
        """``-max Re(lambda)``; positive is stable."""
        return -jnp.max(self.eigenvalues().real)

    def resonance_hz(self) -> jnp.ndarray:
        """Per-mode damped frequency ``|Im lambda| / 2 pi`` (zero when overdamped)."""
        return jnp.abs(self.eigenvalues()[:, 0].imag) / (2.0 * jnp.pi)

    def spectrum(self, freq_hz) -> tuple[jnp.ndarray, jnp.ndarray]:
        """One-sided power spectral density per mode, ``(n_f, N)`` each for E and I.

        Normalised so that ``integral over f >= 0`` returns the stationary
        variance of :meth:`stationary_covariance` (checked in the tests).
        """
        omega = 2.0 * jnp.pi * jnp.asarray(freq_hz, dtype=float)
        s_e, s_i = temporal_spectrum(self.params, self.eigs, self.ess, self.iss, omega, self.kernel)
        return 2.0 * s_e, 2.0 * s_i


def band_power(block: EIBlock, f_lo: float, f_hi: float, n: int = 1024) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-mode E and I power in ``[f_lo, f_hi]`` Hz, ``(N,)`` each, by trapezoid.

    This is the derived band observable: a function of the block's
    parameters, differentiable through them, with no free sign.
    """
    f = jnp.linspace(f_lo, f_hi, n)
    s_e, s_i = block.spectrum(f)
    return jnp.trapezoid(s_e, f, axis=0), jnp.trapezoid(s_i, f, axis=0)


def linear_ei_block(
    params: GNFParams,
    eigs=None,
    kernel: str = "gaussian",
    steady: tuple[float, float] | None = None,
    n_guesses: int = 30,
) -> EIBlock:
    """Linearise *params* on *eigs* about a homogeneous steady state.

    *eigs* defaults to the single constant mode (a single-region model).
    With *steady* unspecified the admissible steady states are found by
    gnf-jax's multi-start Newton solve and the first is used; when there
    are several the choice is a modelling decision, so pass *steady*
    explicitly in that case.  Everything is jit- and grad-compatible: the
    steady-state solve is a fixed-length ``lax.scan``.
    """
    eigs = jnp.array([0.0]) if eigs is None else jnp.asarray(eigs, dtype=float)
    if steady is None:
        sols, ok = steady_states(params, n_guesses=n_guesses)
        n_valid = jnp.sum(ok)
        pick = sols[jnp.argmax(ok)]
        pick = jnp.where(n_valid > 0, pick, jnp.nan)
        ess, iss = pick[0], pick[1]
    else:
        ess, iss = jnp.asarray(steady[0], dtype=float), jnp.asarray(steady[1], dtype=float)
        n_valid = jnp.asarray(1)
    A, D = linearise(params, eigs, ess, iss, kernel)
    return EIBlock(params, eigs, A, D, ess, iss, kernel, n_valid)


def statespace_spec(
    block: EIBlock,
    entry: str | None = None,
    mode_weights=None,
    jitter: float = 0.0,
) -> dict:
    """``f``, ``Q``, ``m0``, ``P0`` for :func:`vpjax.statespace.ll_filter`.

    ``f(t, x, args)`` follows the Diffrax order; when the filter is given
    ``inputs`` it calls ``f`` with ``args = (static, u_k)`` and ``u_k`` is
    applied at *entry*.  ``P0`` is the stationary covariance, so the filter
    starts in the block's own steady state.  Compose with a haemodynamic
    model by concatenating states and feeding :meth:`EIBlock.excitatory`
    (or a loading over modes) into the flow-inducing signal.
    """
    def f(t, x, args):
        u = args[1] if isinstance(args, tuple) else 0.0
        return block.drift(x, u, entry, mode_weights)

    return {
        "f": f,
        "Q": block.Q + jitter * jnp.eye(block.n_state),
        "m0": block.m0,
        "P0": block.stationary_covariance(),
    }
