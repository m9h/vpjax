"""Assemble asynchronous recordings into one masked observation grid.

Simultaneous EEG-fMRI produces streams whose sampling rates differ by
three orders of magnitude, and which do not share sample instants even
where their rates divide evenly.  The filter in
:mod:`vpjax.statespace.filters` handles that directly, provided the
data arrive as a single grid of times with a presence mask per channel.
Building that grid is this module's only job.

Nothing is interpolated or imputed.  A grid point at which only the EEG
channels are present carries a mask of zero for the BOLD channels, and
the filter simply does not update those channels there.  Upsampling the
slow modality instead would feed the estimator correlated pseudo-data
and shrink its uncertainty by the upsampling factor.

Noise variances must be supplied per stream.  They are not estimated
from the data here and have no default, for the same reason
:func:`vpjax.autonomic.infer_shared_drive` insists on them: the
posterior width is a direct function of this assumption, so it belongs
with the analysis that states it.

References
----------
Deneux T, Faugeras O (2010) NeuroImage 49:1668-1681
    Multimodal state-space estimation with asynchronous observations
Valdes-Sosa PA et al. (2009) Hum Brain Mapp 30:2701-2721
    "Model driven EEG/fMRI fusion of brain oscillations"
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


class ObservationGrid(eqx.Module):
    """A union-of-timestamps grid with per-channel presence masks.

    Attributes
    ----------
    t        : grid times, shape (K,), strictly increasing
    y        : observations, shape (K, M); NaN where absent
    mask     : 1.0 where present, 0.0 where absent, shape (K, M)
    r_diag   : observation noise variances, shape (M,)
    channels : channel names, length M — static
    """

    t: Float[Array, "K"]
    y: Float[Array, "K M"]
    mask: Float[Array, "K M"]
    r_diag: Float[Array, "M"]
    channels: tuple[str, ...] = eqx.field(static=True)

    @property
    def n_present(self) -> int:
        return int(np.asarray(self.mask).sum())

    def coverage(self) -> dict[str, float]:
        """Fraction of grid points at which each channel is present.

        Worth looking at before fitting: a channel present at 0.1% of
        grid points contributes 0.1% of the observations, however many
        samples it has in its own time base.
        """
        m = np.asarray(self.mask)
        return {
            name: float(m[:, i].mean()) for i, name in enumerate(self.channels)
        }


def _validate_stream(name, stream):
    if "time_s" not in stream or "values" not in stream:
        raise ValueError(f"stream {name!r} needs 'time_s' and 'values'")
    if "noise_sd" not in stream:
        raise ValueError(
            f"stream {name!r} needs an explicit 'noise_sd'; the posterior "
            "width depends on it and it is not estimated here"
        )

    t = np.asarray(stream["time_s"], dtype=float)
    v = np.asarray(stream["values"], dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError(f"{name}: time_s must be a non-empty 1-D array")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f"{name}: time_s must be finite and strictly increasing")
    if v.ndim == 1:
        v = v[:, None]
    if v.ndim != 2 or v.shape[0] != t.shape[0]:
        raise ValueError(f"{name}: values must be (T,) or (T, C) matching time_s")

    n_chan = v.shape[1]
    sd = np.atleast_1d(np.asarray(stream["noise_sd"], dtype=float))
    if sd.size == 1:
        sd = np.repeat(sd, n_chan)
    if sd.shape != (n_chan,):
        raise ValueError(f"{name}: noise_sd must be scalar or one per channel")
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError(f"{name}: noise_sd must be positive and finite")

    valid = stream.get("valid")
    valid = (
        np.ones(v.shape, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if valid.ndim == 1:
        valid = valid[:, None]
    if valid.shape != v.shape:
        raise ValueError(f"{name}: valid must broadcast to the shape of values")

    names = (
        (name,)
        if n_chan == 1
        else tuple(f"{name}[{i}]" for i in range(n_chan))
    )
    return t, v, valid & np.isfinite(v), sd, names


def build_observation_grid(
    streams: dict[str, dict],
    max_dt: float | None = None,
    t0: float | None = None,
    t1: float | None = None,
    tol: int = 9,
) -> ObservationGrid:
    """Merge asynchronous streams into one :class:`ObservationGrid`.

    Parameters
    ----------
    streams : mapping from name to a dict with

              ``time_s``   sample times, shape (T,), strictly increasing
              ``values``   shape (T,) or (T, C)
              ``noise_sd`` scalar or one per channel; required
              ``valid``    optional boolean mask of the same shape as
                           ``values``; non-finite values are dropped
                           regardless

    max_dt  : if given, subdivide any grid interval longer than this.
              Set it when one modality is much slower than the dynamics
              being estimated: the LL propagator is exact for the
              linearized system but the linearization point goes stale
              over a long step, and a 2 s TR is a long step for a
              hemodynamic ODE.  The inserted points carry no
              observations, so they cost compute, not information.
    t0, t1  : clip the grid to this closed interval.  The default spans
              every stream; pass these to restrict it to the window
              where all modalities are present, if an analysis needs
              that.
    tol     : decimal places used to collapse near-identical timestamps

    Returns
    -------
    ObservationGrid
    """
    if not streams:
        raise ValueError("no streams given")

    prepared = {
        name: _validate_stream(name, stream) for name, stream in streams.items()
    }

    # The grid spans the union of the streams, not their intersection: a
    # channel absent at an instant is already represented by its mask, so
    # truncating to the overlap would discard usable samples at the edges
    # -- EEG recorded before the first volume, say.  Pass t0/t1 to clip.
    starts = [t[0] for t, *_ in prepared.values()]
    stops = [t[-1] for t, *_ in prepared.values()]
    lo = min(starts) if t0 is None else float(t0)
    hi = max(stops) if t1 is None else float(t1)
    if not hi > lo:
        raise ValueError("the requested time window is empty")
    if max(starts) >= min(stops):
        raise ValueError(
            "streams share no common instant; check that they are on one "
            "clock, since this usually means a missing trigger offset "
            "rather than genuinely disjoint recordings"
        )

    marks = [
        np.round(t[(t >= lo) & (t <= hi)], tol) for t, *_ in prepared.values()
    ]
    grid = np.unique(np.concatenate(marks))

    if max_dt is not None:
        if not np.isfinite(max_dt) or max_dt <= 0:
            raise ValueError("max_dt must be positive and finite")
        filled = [grid[:1]]
        for a, b in zip(grid[:-1], grid[1:]):
            n_sub = int(np.ceil((b - a) / max_dt))
            filled.append(np.round(np.linspace(a, b, n_sub + 1)[1:], tol))
        grid = np.unique(np.concatenate(filled))

    channels: list[str] = []
    for _, _, _, _, names in prepared.values():
        channels.extend(names)
    k, m = grid.size, len(channels)

    y = np.full((k, m), np.nan)
    mask = np.zeros((k, m))
    r_diag = np.empty(m)

    col = 0
    for name, (t, v, valid, sd, names) in prepared.items():
        keep = (t >= lo) & (t <= hi)
        idx = np.searchsorted(grid, np.round(t[keep], tol))
        if idx.size and not np.allclose(grid[idx], t[keep], atol=10.0**-tol):
            raise ValueError(f"{name}: sample times did not land on the grid")
        if np.unique(idx).size != idx.size:
            raise ValueError(
                f"{name}: two samples collapse onto one grid point at "
                f"tol={tol}; increase tol or decimate the stream"
            )
        width = len(names)
        y[idx, col : col + width] = v[keep]
        mask[idx, col : col + width] = valid[keep].astype(float)
        r_diag[col : col + width] = sd**2
        col += width

    # A grid point with nothing present contributes only propagation.
    y = np.where(mask > 0, y, np.nan)

    return ObservationGrid(
        t=jnp.asarray(grid),
        y=jnp.asarray(y),
        mask=jnp.asarray(mask),
        r_diag=jnp.asarray(r_diag),
        channels=tuple(channels),
    )


def regular_grid(
    t_start: float,
    t_stop: float,
    dt: float,
    streams: dict[str, dict],
    tol: int = 9,
) -> ObservationGrid:
    """Resample stream *times* onto a regular grid without touching values.

    Each sample is assigned to its nearest grid point and kept there.
    Use this only when the exact sample instants are not trusted to
    better than ``dt`` anyway; it discards sub-``dt`` timing, which for
    EEG is usually the information of interest.  Prefer
    :func:`build_observation_grid`.
    """
    grid = np.round(np.arange(t_start, t_stop + dt * 1e-6, dt), tol)
    snapped = {}
    for name, stream in streams.items():
        t, v, valid, sd, _ = _validate_stream(name, stream)
        idx = np.clip(np.searchsorted(grid, t, side="left"), 0, grid.size - 1)
        left = np.clip(idx - 1, 0, grid.size - 1)
        pick = np.where(
            np.abs(grid[left] - t) <= np.abs(grid[idx] - t), left, idx
        )
        keep = np.zeros(grid.size, dtype=bool)
        first = np.full(grid.size, -1)
        for j, p in enumerate(pick):
            if first[p] < 0:
                first[p] = j
                keep[p] = True
        sel = first[keep]
        snapped[name] = {
            "time_s": grid[keep],
            "values": v[sel],
            "valid": valid[sel],
            "noise_sd": sd,
        }
    return build_observation_grid(snapped, tol=tol)
