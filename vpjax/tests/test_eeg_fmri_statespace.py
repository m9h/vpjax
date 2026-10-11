"""Tests for the recorded-run state-space fit, on simulated runs."""

import numpy as np
import pytest

from vpjax.validation.eeg_fmri_statespace import (
    BOLD_ONLY_NAMES,
    ONE_STATE_NAMES,
    PARAM_NAMES,
    bold_fractional,
    default_init,
    fit_run,
    format_run,
    run_grid,
    run_model,
    standardise_envelope,
)
from vpjax.validation.statespace_recovery import simulate


@pytest.fixture(scope="module")
def run():
    """A short simulated run with an EEG-specific fast component added."""
    sim = simulate(duration=120.0, eeg_snr=3.0, seed=0)
    rng = np.random.default_rng(1)
    n = np.zeros(sim["t_eeg"].size)
    for k in range(1, n.size):
        n[k] = 0.7 * n[k - 1] + rng.normal(scale=0.05)
    env = np.exp(sim["eeg"] + n)       # positive envelope, log-linear in the drive
    return sim, env


class TestPreparation:
    def test_bold_fractional_is_detrended_and_relative(self):
        ts = 1000.0 + 5.0 * np.arange(100) + np.sin(np.arange(100))
        y = bold_fractional(ts, drop=3)
        assert y.size == 97
        assert abs(y.mean()) < 1e-6
        assert y.std() < 1e-3

    def test_standardise_envelope_window_and_scale(self, run):
        sim, env = run
        t, x = standardise_envelope(env, sim["t_eeg"], 10.0, 50.0)
        assert t.min() >= 10.0 and t.max() <= 50.0
        assert abs(x.mean()) < 1e-9 and abs(x.std() - 1.0) < 1e-9

    def test_grid_noise_placeholders_are_overridden(self, run):
        sim, env = run
        grid = run_grid(sim["t_bold"], sim["bold"], {"eeg": (sim["t_eeg"], env)})
        assert set(grid.channels) == {"bold", "eeg"}
        init = default_init(sim["bold"], ("eeg",))
        build = run_model(grid, PARAM_NAMES, {}, nuisance=True)
        spec = build(np.array([init[n] for n in PARAM_NAMES]))
        assert "r_diag" in spec and spec["Q"].shape == (6, 6)
        assert spec["m0"].shape == (6,) and spec["P0"].shape == (6, 6)


class TestModelShapes:
    def test_one_state_model_is_five_dimensional(self, run):
        sim, env = run
        grid = run_grid(sim["t_bold"], sim["bold"], {"eeg": (sim["t_eeg"], env)})
        init = default_init(sim["bold"], ("eeg",), nuisance=False)
        build = run_model(grid, ONE_STATE_NAMES, {}, nuisance=False)
        spec = build(np.array([init[n] for n in ONE_STATE_NAMES]))
        assert spec["Q"].shape == (5, 5)

    def test_bold_only_has_no_eeg_parameters(self, run):
        sim, _ = run
        grid = run_grid(sim["t_bold"], sim["bold"])
        init = default_init(sim["bold"], ())
        assert not any(k.startswith(("gain_", "tau_n_")) for k in init)
        build = run_model(grid, BOLD_ONLY_NAMES, {})
        spec = build(np.array([init[n] for n in BOLD_ONLY_NAMES]))
        assert spec["r_diag"].shape == (1,)

    def test_missing_parameter_is_an_error(self, run):
        sim, env = run
        grid = run_grid(sim["t_bold"], sim["bold"], {"eeg": (sim["t_eeg"], env)})
        with pytest.raises(ValueError, match="neither free nor fixed"):
            run_model(grid, ("kappa", "tau"), {})

    def test_fixed_parameters_are_used(self, run):
        sim, _ = run
        grid = run_grid(sim["t_bold"], sim["bold"])
        names = ("kappa", "tau")
        fixed = {"tau_z": 2.0, "q_z": 0.01, "r_bold": 1e-6}
        spec = run_model(grid, names, fixed)(np.array([0.65, 0.98]))
        assert float(spec["Q"][0, 0]) == pytest.approx(0.01, abs=1e-7)
        assert float(spec["r_diag"][0]) == pytest.approx(1e-6, rel=1e-6)


class TestFit:
    def test_bold_only_fit_reports_everything(self, run):
        sim, _ = run
        r = fit_run(sim["t_bold"], sim["bold"], restarts=0, max_steps=5)
        assert r["fit_names"] == BOLD_ONLY_NAMES
        assert r["nuisance"] is False and r["sign"] == {}
        assert set(r["standard_error"]) == set(BOLD_ONLY_NAMES)
        assert "bold" in r["diagnostics"]["per_channel"]
        assert isinstance(format_run("x", r), str)

    def test_joint_fit_tries_both_signs(self, run):
        sim, env = run
        t, x = standardise_envelope(env, sim["t_eeg"], 0.0, 120.0)
        r = fit_run(sim["t_bold"], sim["bold"], t, x, restarts=0, max_steps=5)
        assert r["nuisance"] is True
        assert set(r["log_likelihood_by_sign"]) == {"eeg:+1", "eeg:-1"}
        assert r["sign"]["eeg"] in (-1.0, 1.0)
        assert set(r["fit_names"]) == set(PARAM_NAMES)
        assert {"bold", "eeg"} <= set(r["diagnostics"]["per_channel"])

    @pytest.mark.slow
    def test_joint_fit_prefers_the_true_sign(self):
        # The loading's sign is decided only through the BOLD, so it needs
        # a run long enough to hold a few hundred volumes' worth of
        # evidence; 120 s (60 volumes) is not, 360 s is by ~100 log-lik.
        sim = simulate(duration=360.0, eeg_snr=3.0, seed=0)
        rng = np.random.default_rng(1)
        n = np.zeros(sim["t_eeg"].size)
        for k in range(1, n.size):
            n[k] = 0.7 * n[k - 1] + rng.normal(scale=0.05)
        env = np.exp(sim["eeg"] + n)
        t, x = standardise_envelope(env, sim["t_eeg"], 0.0, 360.0)
        r = fit_run(sim["t_bold"], sim["bold"], t, x, restarts=2, max_steps=150)
        # The envelope was built with a positive loading on the drive.
        assert r["sign"]["eeg"] == 1.0
        assert r["log_likelihood_by_sign"]["eeg:+1"] > r["log_likelihood_by_sign"]["eeg:-1"] + 10
        assert abs(r["diagnostics"]["per_channel"]["eeg"]["variance"] - 1.0) < 0.3
