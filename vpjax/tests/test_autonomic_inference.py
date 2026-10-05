"""Recovery, missing observations, and conditional uncertainty."""

import numpy as np
import pytest

from vpjax.autonomic.inference import infer_shared_drive


def test_joint_recovery_with_asynchronous_observations():
    grid = np.arange(80.0)
    truth = np.sin(grid / 12.0)
    streams = {
        "hr": {"time_s": grid[::2], "values": truth[::2], "valid": np.ones(40, bool)},
        "eda": {
            "time_s": grid[1::3],
            "values": 2 * truth[1::3],
            "valid": np.ones(27, bool),
        },
    }
    result = infer_shared_drive(
        streams,
        grid,
        loadings={"hr": 1.0, "eda": 2.0},
        noise_sd={"hr": 0.1, "eda": 0.2},
        process_sd=0.2,
    )
    assert np.sqrt(np.mean((np.asarray(result["mean"]) - truth) ** 2)) < 0.04
    assert np.all(np.asarray(result["sd"]) > 0)
    masked = {
        name: dict(stream, valid=np.zeros_like(stream["valid"]))
        for name, stream in streams.items()
    }
    prior = infer_shared_drive(
        masked,
        grid,
        loadings={"hr": 1.0, "eda": 2.0},
        noise_sd={"hr": 0.1, "eda": 0.2},
        process_sd=0.2,
    )
    assert np.mean(prior["sd"]) > np.mean(result["sd"])


def test_missing_values_ignored_and_invalid_noise_rejected():
    streams = {
        "hr": {
            "time_s": np.array([0.0, 1.0]),
            "values": np.array([1.0, np.nan]),
            "valid": np.array([True, False]),
        }
    }
    result = infer_shared_drive(
        streams, np.arange(3.0), loadings={"hr": 1.0}, noise_sd={"hr": 0.1}
    )
    assert np.isfinite(result["mean"]).all()
    with pytest.raises(ValueError, match="positive"):
        infer_shared_drive(
            streams, np.arange(3.0), loadings={"hr": 1.0}, noise_sd={"hr": 0.0}
        )


def test_smoother_matches_dense_gaussian_posterior():
    grid = np.array([0.0, 0.5, 2.0, 4.0])
    observations = {
        "hr": {
            "time_s": grid,
            "values": np.array([1.0, 2.0, -1.0, 0.5]),
            "valid": np.array([True, False, True, True]),
        }
    }
    result = infer_shared_drive(
        observations,
        grid,
        loadings={"hr": -2.0},
        noise_sd={"hr": 0.5},
        process_sd=0.3,
        prior_sd=2.0,
    )
    covariance = 4.0 + 0.3**2 * np.minimum.outer(grid, grid)
    selected = np.flatnonzero(observations["hr"]["valid"])
    design = -2.0 * np.eye(4)[selected]
    precision = np.linalg.inv(covariance) + design.T @ design / 0.5**2
    posterior_cov = np.linalg.inv(precision)
    expected = (
        posterior_cov @ design.T @ observations["hr"]["values"][selected] / 0.5**2
    )
    np.testing.assert_allclose(result["mean"], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result["sd"], np.sqrt(np.diag(posterior_cov)), rtol=1e-5)
