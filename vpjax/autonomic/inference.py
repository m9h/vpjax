"""A reduced linear-Gaussian shared-drive model, not a PK/PD model.

Observations y_k(t)=loading_k*x(t)+noise_k must be baseline-standardized.
The latent drive follows a random walk with diffusion process_sd² per second.
Fixed signed loadings make the latent scale explicit. Sensor noise and process
noise are user assumptions; returned uncertainty is conditional on them.
No claim of receptor occupancy, sympathetic/vagal identification, sensor
response delays or causal neural/vascular separation is made.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np


def load_observations(path):
    """Read neurophys-observations/1 without importing neurojax or MNE."""
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        if manifest["schema"] != "neurophys-observations/1":
            raise ValueError("unsupported observation schema")
        return {
            item["name"]: {
                "time_s": data[item["key"] + "_time_s"],
                "values": data[item["key"] + "_values"],
                "valid": data[item["key"] + "_valid"],
                "unit": item["unit"],
                "metadata": item["metadata"],
            }
            for item in manifest["streams"]
        }


def infer_shared_drive(
    streams, grid_s, *, loadings, noise_sd, process_sd=0.1, prior_sd=1.0
):
    """Exact scalar Kalman filter/RTS smoother on union of sensor timestamps.

    Streams may be asynchronous and have missing samples. Observations outside
    grid bounds are excluded, so separate recording clocks must be reconciled
    by the caller. No temporal interpolation or imputation is needed. Each
    selected stream must have an explicit loading and positive noise SD.
    Returns mean/SD on grid_s, plus the assumptions and valid sample counts.
    """
    grid = np.asarray(grid_s, dtype=float)
    if (
        grid.ndim != 1
        or not len(grid)
        or not np.isfinite(grid).all()
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("grid_s must be finite, nonempty and strictly increasing")
    if (
        not np.isfinite([process_sd, prior_sd]).all()
        or process_sd <= 0
        or prior_sd <= 0
    ):
        raise ValueError("process_sd and prior_sd must be positive and finite")
    if not streams or set(streams) != set(loadings) or set(streams) != set(noise_sd):
        raise ValueError("streams, loadings and noise_sd must have identical keys")
    prepared = {}
    for name, stream in streams.items():
        if stream.get("unit", "1") != "1":
            raise ValueError(
                f"baseline-standardized dimensionless units required: {name}"
            )
        t, y = np.asarray(stream["time_s"], float), np.asarray(stream["values"], float)
        valid = np.asarray(stream["valid"], bool)
        if t.ndim != 1 or t.shape != y.shape or t.shape != valid.shape:
            raise ValueError(f"matching 1D arrays required for {name}")
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError(f"timestamps must be finite and increasing: {name}")
        h, sd = loadings[name], noise_sd[name]
        if not np.isfinite([h, sd]).all() or h == 0 or sd <= 0:
            raise ValueError(
                "nonzero finite loadings and positive finite noise required"
            )
        use = valid & np.isfinite(y) & (t >= grid[0]) & (t <= grid[-1])
        prepared[name] = (t[use], y[use], h, sd)
    times = np.unique(np.concatenate([grid] + [p[0] for p in prepared.values()]))
    information = np.zeros(len(times))
    weighted = np.zeros(len(times))
    for t, y, h, sd in prepared.values():
        indices = np.searchsorted(times, t)
        np.add.at(information, indices, h * h / sd**2)
        np.add.at(weighted, indices, h * y / sd**2)
    q = np.r_[0.0, np.diff(times)] * process_sd**2

    def update(carry, inputs):
        mean, variance = carry
        process_var, info, value = inputs
        predicted_var = variance + process_var
        posterior_var = 1.0 / (1.0 / predicted_var + info)
        posterior_mean = posterior_var * (mean / predicted_var + value)
        return (posterior_mean, posterior_var), (posterior_mean, posterior_var)

    _, (means, variances) = jax.lax.scan(
        update,
        (jnp.array(0.0), jnp.array(prior_sd**2)),
        (jnp.asarray(q), jnp.asarray(information), jnp.asarray(weighted)),
    )

    def smooth(carry, inputs):
        next_mean, next_var = carry
        mean, variance, next_q = inputs
        gain = variance / (variance + next_q)
        smoothed_mean = mean + gain * (next_mean - mean)
        smoothed_var = variance + gain**2 * (next_var - variance - next_q)
        return (smoothed_mean, smoothed_var), (smoothed_mean, smoothed_var)

    _, (sm, sv) = jax.lax.scan(
        smooth,
        (means[-1], variances[-1]),
        (means[:-1], variances[:-1], jnp.asarray(q[1:])),
        reverse=True,
    )
    mean = jnp.concatenate([sm, means[-1:]])
    variance = jnp.concatenate([sv, variances[-1:]])
    index = np.searchsorted(times, grid)
    return {
        "time_s": jnp.asarray(grid),
        "mean": mean[index],
        "sd": jnp.sqrt(jnp.maximum(variance[index], 0.0)),
        "loadings": dict(loadings),
        "noise_sd": dict(noise_sd),
        "process_sd": process_sd,
        "prior_sd": prior_sd,
        "valid_counts": {name: len(p[0]) for name, p in prepared.items()},
    }
