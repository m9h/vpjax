"""Longitudinal Balloon-Windkessel fitting across multiple sessions.

For a subject scanned at S waves with ages (a_1, ..., a_S), fit a single
set of per-region parameters that varies with age::

    p(s, r) = p_0(r) + β_p(r) · (a_s - age_center)

for each parameter ``p`` in ``fit_names`` (default: kappa, tau).  The
inverse problem couples all S sessions through a shared (baseline,
slope) per region, so β captures the within-subject vascular aging
rate that a single-wave fit cannot identify.

Implementation: sessions are zero-padded to a common length so they
share a JIT graph.  We then JIT a *per-session* gradient kernel that
takes the current theta and one session's (stim, bold, mask, age) and
returns ``∂L_session/∂theta``.  At each optimisation step we call this
kernel S times (Python for-loop) and sum the gradients, then apply a
NaN-safe momentum update.  This keeps the compiled graph the same
size as :func:`vpjax.hemodynamics.inversion.fit_balloon_bold_batch`
(a single ``solve_balloon`` vmapped over R) and avoids the BPTT-tape
blowup of putting ``solve_balloon`` inside a ``jax.lax.scan``.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from vpjax._types import BalloonParams
from vpjax.hemodynamics.balloon import solve_balloon
from vpjax.hemodynamics.bold import BOLDParams, observe_bold
from vpjax.hemodynamics.cvr import extract_global_stimulus
from vpjax.hemodynamics.inversion import (
    _BALLOON_ALL_NAMES,
    _BALLOON_BOUNDS,
    _BALLOON_DEFAULT_FIT,
)


def _make_session_balloon_params(
    baselines: Float[Array, "P"],
    slopes: Float[Array, "P"],
    age_offset: Float[Array, ""],
    base_params: BalloonParams,
    fit_names: tuple[str, ...],
) -> BalloonParams:
    """Build BalloonParams for one (region, session) combo."""
    vals = {name: getattr(base_params, name) for name in _BALLOON_ALL_NAMES}
    for i, name in enumerate(fit_names):
        lo, hi = _BALLOON_BOUNDS[name]
        vals[name] = jnp.clip(baselines[i] + slopes[i] * age_offset, lo, hi)
    return BalloonParams(**vals)


def _pad_sessions(
    bold_per_session: Sequence[np.ndarray],
    stimulus_per_session: Sequence[Array],
    sub: int,
) -> tuple[Array, Array, Array, int]:
    """Stack S sessions into shape-aligned arrays + per-timepoint mask.

    Returns
    -------
    bold_stacked : (S, R, N_max)
    stim_stacked : (S, T_max)         where T_max = N_max * sub
    mask_stacked : (S, N_max)         1 where the timepoint is real
    N_max : int
    """
    Ns = [int(b.shape[1]) for b in bold_per_session]
    N_max = max(Ns)
    T_max = N_max * sub
    R = bold_per_session[0].shape[0]
    S = len(bold_per_session)

    bold_stacked = np.zeros((S, R, N_max), dtype=np.float32)
    stim_stacked = np.zeros((S, T_max), dtype=np.float32)
    mask_stacked = np.zeros((S, N_max), dtype=np.float32)

    for s in range(S):
        bs = bold_per_session[s]
        ts = np.asarray(stimulus_per_session[s])
        ns = bs.shape[1]
        ts_len = min(ts.shape[0], T_max)
        bold_stacked[s, :, :ns] = bs
        stim_stacked[s, :ts_len] = ts[:ts_len]
        mask_stacked[s, :ns] = 1.0

    return (
        jnp.asarray(bold_stacked),
        jnp.asarray(stim_stacked),
        jnp.asarray(mask_stacked),
        N_max,
    )


def fit_balloon_longitudinal(
    bold_per_session: Sequence[Float[np.ndarray, "R N_s"]],
    stimulus_per_session: Sequence[Float[Array, "T_s"]],
    ages: Sequence[float],
    tr: float,
    dt: float = 0.1,
    age_center: float = 60.0,
    fit_names: tuple[str, ...] = _BALLOON_DEFAULT_FIT,
    balloon_params: BalloonParams | None = None,
    bold_params: BOLDParams | None = None,
    n_steps: int = 200,
    learning_rate: float = 1.0,
) -> dict[str, np.ndarray]:
    """Fit per-region (baseline, age-slope) Balloon params across sessions.

    Parameters
    ----------
    bold_per_session : sequence of S arrays, each shape (R, N_s).
        Per-region mean BOLD (fractional change).  R must be identical
        across sessions; N_s may vary (sessions are zero-padded
        internally to ``max(N_s)``).
    stimulus_per_session : S stimuli on the ODE grid (length
        ``N_s * round(tr/dt)``).  Typically built with
        :func:`vpjax.hemodynamics.cvr.extract_global_stimulus`.
    ages : per-session ages (years), length S.
    tr, dt : repetition time and ODE timestep, shared across sessions.
    age_center : centring constant; β is interpreted as Δp per year of
        deviation from this age (default 60 — DLBS mid-cohort).
    fit_names : Balloon parameters to make age-dependent (default
        ``("kappa", "tau")``).  All others held at their values in
        ``balloon_params``.
    n_steps, learning_rate : optimiser controls (NaN-safe momentum step).

    Returns
    -------
    Dict::

        baselines : (R, P)        – per-region p_0 for each fit_name
        slopes : (R, P)           – per-region β_p
        loss : scalar             – mean squared residual averaged over
                                     valid timepoints, sessions, regions
        bold_predicted : list of (R, N_s) – per-session fitted BOLD
        ages : (S,)               – ages used (echoed for downstream)
    """
    if balloon_params is None:
        balloon_params = BalloonParams()
    if bold_params is None:
        bold_params = BOLDParams()

    S = len(bold_per_session)
    if len(stimulus_per_session) != S or len(ages) != S:
        raise ValueError(
            f"S inconsistent: bold={S}, stim={len(stimulus_per_session)}, "
            f"ages={len(ages)}"
        )
    if S == 0:
        raise ValueError("at least one session is required")

    R = bold_per_session[0].shape[0]
    if any(b.shape[0] != R for b in bold_per_session):
        raise ValueError("All sessions must have the same number of regions R")

    P = len(fit_names)
    Ns = [int(b.shape[1]) for b in bold_per_session]
    sub = int(round(tr / dt))

    bold_stacked, stim_stacked, mask_stacked, N_max = _pad_sessions(
        bold_per_session, stimulus_per_session, sub,
    )
    age_offsets = jnp.asarray(
        [float(a) - age_center for a in ages], dtype=jnp.float32,
    )

    base_init = jnp.array(
        [float(getattr(balloon_params, n)) for n in fit_names], dtype=jnp.float32,
    )
    theta_init = jnp.concatenate([
        jnp.broadcast_to(base_init, (R, P)),
        jnp.zeros((R, P), dtype=jnp.float32),
    ], axis=1)  # (R, 2P)

    def per_region_session_pred(theta_r, age_off, stim_s):
        baselines = theta_r[:P]
        slopes = theta_r[P:]
        bp = _make_session_balloon_params(
            baselines, slopes, age_off, balloon_params, fit_names,
        )
        _, traj = solve_balloon(bp, stim_s, dt=dt)
        return observe_bold(traj, bold_params)[::sub][:N_max]

    def session_loss(theta, age_off, stim_s, bold_s, mask_s):
        # theta: (R, 2P), bold_s: (R, N_max), mask_s: (N_max,)
        preds = jax.vmap(
            per_region_session_pred, in_axes=(0, None, None),
        )(theta, age_off, stim_s)  # (R, N_max)
        sq = (preds - bold_s) ** 2
        denom = jnp.maximum(mask_s.sum(), 1.0)
        return ((sq * mask_s[None, :]).sum(axis=1) / denom).mean()

    @jax.jit
    def session_loss_and_grad(theta, age_off, stim_s, bold_s, mask_s):
        return jax.value_and_grad(session_loss)(
            theta, age_off, stim_s, bold_s, mask_s,
        )

    @jax.jit
    def apply_update(theta, vel, g_total):
        safe = jnp.all(jnp.isfinite(g_total))
        vel = jnp.where(safe, 0.9 * vel + 0.1 * g_total, vel)
        theta = jnp.where(safe, theta - learning_rate * vel, theta)
        return theta, vel

    theta = theta_init
    velocity = jnp.zeros_like(theta)
    for _ in range(n_steps):
        g_accum = jnp.zeros_like(theta)
        for s in range(S):
            _, g_s = session_loss_and_grad(
                theta, age_offsets[s], stim_stacked[s],
                bold_stacked[s], mask_stacked[s],
            )
            g_accum = g_accum + g_s
        g_accum = g_accum / S
        theta, velocity = apply_update(theta, velocity, g_accum)

    baselines_final = np.asarray(theta[:, :P])
    slopes_final = np.asarray(theta[:, P:])
    final_loss = float(
        sum(
            float(session_loss(
                theta, age_offsets[s], stim_stacked[s],
                bold_stacked[s], mask_stacked[s],
            ))
            for s in range(S)
        )
        / S
    )

    # Reconstruct per-session predicted BOLD on the un-padded grids.
    bold_pred_per_session: list[np.ndarray] = []
    for s in range(S):
        preds_R_Nmax = jax.vmap(
            per_region_session_pred, in_axes=(0, None, None),
        )(theta, age_offsets[s], stim_stacked[s])
        bold_pred_per_session.append(np.asarray(preds_R_Nmax)[:, :Ns[s]])

    return {
        "baselines": baselines_final,
        "slopes": slopes_final,
        "fit_names": np.asarray(fit_names),
        "loss": np.float32(final_loss),
        "bold_predicted": bold_pred_per_session,
        "ages": np.asarray(ages, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Convenience: extract region means from 4D volumes, then fit
# ---------------------------------------------------------------------------

def _region_means_one_session(
    bold_4d: Float[np.ndarray, "X Y Z N"],
    region_volume: Float[np.ndarray, "X Y Z"],
    region_ids: np.ndarray,
    min_voxels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-region fractional-change BOLD time series for one session."""
    bold = np.asarray(bold_4d)
    region_volume = np.asarray(region_volume)
    n_t = bold.shape[-1]

    ts: list[np.ndarray] = []
    kept: list[int] = []
    for rid in region_ids:
        mask = region_volume == rid
        if int(mask.sum()) < min_voxels:
            continue
        x = bold[mask].mean(axis=0)
        baseline = x.mean()
        if baseline <= 0:
            continue
        ts.append(((x - baseline) / baseline).astype(np.float32))
        kept.append(int(rid))

    if not kept:
        return np.zeros((0, n_t), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.stack(ts, axis=0), np.asarray(kept, dtype=np.int32)


def fit_balloon_longitudinal_from_volumes(
    bold_4d_per_session: Sequence[Float[np.ndarray, "X Y Z N"]],
    brain_mask_per_session: Sequence[Float[np.ndarray, "X Y Z"]],
    region_volume_per_session: Sequence[Float[np.ndarray, "X Y Z"]],
    ages: Sequence[float],
    tr: float,
    dt: float = 0.1,
    age_center: float = 60.0,
    stimulus_per_session: Sequence[Float[Array, "T_s"]] | None = None,
    fit_names: tuple[str, ...] = _BALLOON_DEFAULT_FIT,
    balloon_params: BalloonParams | None = None,
    bold_params: BOLDParams | None = None,
    n_steps: int = 200,
    learning_rate: float = 1.0,
    min_voxels: int = 5,
) -> dict[str, np.ndarray]:
    """Fit longitudinal Balloon directly from 4D BOLD + per-session label maps.

    Region IDs that don't appear in *every* session (or fall below
    ``min_voxels`` in any session) are dropped — the fit needs the same
    set of regions across waves.

    If ``stimulus_per_session`` is None we derive each session's
    stimulus from the global mean BOLD via
    :func:`vpjax.hemodynamics.cvr.extract_global_stimulus` (data-driven
    CO2 proxy).

    Returns the same dict as :func:`fit_balloon_longitudinal`, plus
    ``region_ids`` (R,) listing the labels that were actually fit.
    """
    S = len(bold_4d_per_session)
    if not (len(brain_mask_per_session) == len(region_volume_per_session) == len(ages) == S):
        raise ValueError("All per-session inputs must have the same length")

    candidate_ids: set[int] | None = None
    for rv in region_volume_per_session:
        ids_s = set(int(x) for x in np.unique(np.asarray(rv)) if x > 0)
        candidate_ids = ids_s if candidate_ids is None else candidate_ids & ids_s
    if not candidate_ids:
        raise ValueError("No region label is present in all sessions")
    candidate_ids_arr = np.asarray(sorted(candidate_ids), dtype=np.int32)

    per_session_ts: list[np.ndarray] = []
    per_session_kept: list[np.ndarray] = []
    for bold_4d, rv in zip(bold_4d_per_session, region_volume_per_session):
        ts, kept = _region_means_one_session(
            bold_4d, rv, candidate_ids_arr, min_voxels=min_voxels,
        )
        per_session_ts.append(ts)
        per_session_kept.append(kept)

    common = per_session_kept[0]
    for k in per_session_kept[1:]:
        common = np.intersect1d(common, k)
    if common.size == 0:
        raise ValueError("No region met min_voxels in every session")

    aligned: list[np.ndarray] = []
    for ts, kept in zip(per_session_ts, per_session_kept):
        idx = np.searchsorted(kept, common)
        aligned.append(ts[idx])

    if stimulus_per_session is None:
        stimuli: list[Array] = []
        for bold_4d, mask in zip(bold_4d_per_session, brain_mask_per_session):
            stim, _ = extract_global_stimulus(bold_4d, mask, tr=tr, dt=dt)
            stimuli.append(stim)
    else:
        stimuli = list(stimulus_per_session)

    out = fit_balloon_longitudinal(
        bold_per_session=aligned,
        stimulus_per_session=stimuli,
        ages=ages,
        tr=tr,
        dt=dt,
        age_center=age_center,
        fit_names=fit_names,
        balloon_params=balloon_params,
        bold_params=bold_params,
        n_steps=n_steps,
        learning_rate=learning_rate,
    )
    out["region_ids"] = common
    return out


__all__ = [
    "fit_balloon_longitudinal",
    "fit_balloon_longitudinal_from_volumes",
]
