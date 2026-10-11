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


class TestRebin:
    def test_averages_into_coarser_bins(self):
        from vpjax.validation.eeg_fmri_statespace import rebin
        t = np.arange(0, 10, 0.1) + 0.05
        v = np.arange(t.size, dtype=float)
        t2, v2 = rebin(t, v, 0.5)
        assert t2.size == 20 and np.isclose(t2[0], 0.25)
        assert np.isclose(v2[0], np.mean(v[:5]))

    def test_no_op_when_already_coarse(self):
        from vpjax.validation.eeg_fmri_statespace import rebin
        t = np.arange(0, 10, 0.5)
        t2, v2 = rebin(t, t, 0.5)
        assert t2.size == t.size


class TestVascularInput:
    def test_names_and_state_dimension(self, run):
        from vpjax.validation.eeg_fmri_statespace import param_names, sample_inputs
        sim, env = run
        names = param_names(("eeg",), True, ("resp",))
        assert names[-3:] == ("beta_resp", "tau_in_resp", "r_bold")
        grid = run_grid(sim["t_bold"], sim["bold"], {"eeg": (sim["t_eeg"], env)})
        t_u = np.arange(0.0, 120.0, 1.0)
        u = np.sin(2 * np.pi * t_u / 30.0)
        in_names, ug = sample_inputs(grid, {"resp": (t_u, u)})
        assert in_names == ("resp",) and ug.shape == (np.asarray(grid.t).size, 1)
        assert abs(ug.mean()) < 0.1 and abs(ug.std() - 1.0) < 0.1
        init = default_init(sim["bold"], ("eeg",), True, ("resp",))
        build = run_model(grid, names, {}, inputs={"resp": (t_u, u)})
        spec = build(np.array([init[n] for n in names]))
        assert spec["Q"].shape == (7, 7) and spec["inputs"].shape == ug.shape

    def test_input_moves_the_flow_signal(self, run):
        import jax.numpy as jnp
        from vpjax.validation.eeg_fmri_statespace import param_names
        sim, _ = run
        grid = run_grid(sim["t_bold"], sim["bold"])
        names = param_names((), True, ("resp",))
        t_u = np.arange(0.0, 120.0, 1.0)
        build = run_model(grid, names, {}, inputs={"resp": (t_u, np.sin(t_u))})
        init = default_init(sim["bold"], (), True, ("resp",))
        spec = build(np.array([init[n] for n in names]))
        x = jnp.concatenate([jnp.array([0.0, 0.0, 1.0, 1.0, 1.0]), jnp.array([1.0])])  # r = 1
        dx = spec["f"](0.0, x, (None, jnp.array([0.0])))
        assert float(dx[1]) == pytest.approx(init["beta_resp"], rel=1e-5)
        assert float(dx[5]) == pytest.approx(-1.0 / init["tau_in_resp"], rel=1e-5)

    def test_fit_run_accepts_inputs(self, run):
        sim, _ = run
        t_u = np.arange(0.0, 120.0, 1.0)
        r = fit_run(sim["t_bold"], sim["bold"], inputs={"resp": (t_u, np.sin(t_u / 5))},
                    restarts=0, max_steps=3)
        assert r["inputs"] == ("resp",)
        assert "beta_resp" in r["estimate"] and r["sign"]["resp"] in (-1.0, 1.0)
        assert set(r["log_likelihood_by_sign"]) == {"resp:+1", "resp:-1"}


class TestNoiseFloor:
    def test_collapsed_noise_is_refit_at_floor(self, run):
        sim, env = run
        t, x = standardise_envelope(env, sim["t_eeg"], 0.0, 120.0)
        # Force the first fit to a collapsed noise variance by starting it
        # there with no optimisation steps, then check the refit path.
        init = default_init(sim["bold"], ("eeg",), True)
        init["r_eeg"] = 1e-6
        r = fit_run(sim["t_bold"], sim["bold"], t, x, init=init, sign=1.0,
                    restarts=0, max_steps=1, noise_floor=1e-2)
        assert r["noise_floored"] == ["eeg"]
        assert r["collapsed_noise"]["eeg"] < 1e-2
        assert "r_eeg" not in r["fit_names"] and r["fixed"]["r_eeg"] == 1e-2

    def test_floor_can_be_disabled(self, run):
        sim, env = run
        t, x = standardise_envelope(env, sim["t_eeg"], 0.0, 120.0)
        init = default_init(sim["bold"], ("eeg",), True)
        init["r_eeg"] = 1e-6
        r = fit_run(sim["t_bold"], sim["bold"], t, x, init=init, sign=1.0,
                    restarts=0, max_steps=1, noise_floor=None)
        assert r["noise_floored"] == [] and "r_eeg" in r["fit_names"]


class TestPriors:
    def test_log_prior_is_gaussian_in_log_space(self):
        from vpjax.validation.eeg_fmri_statespace import make_log_prior
        import jax.numpy as jnp
        names = ("kappa", "tau", "q_z")
        lp = make_log_prior(names, {"tau": (0.0, 0.5)})
        assert float(lp(jnp.log(jnp.array([1.0, 1.0, 1.0])))) == 0.0
        assert float(lp(jnp.log(jnp.array([1.0, np.e, 1.0])))) == pytest.approx(-2.0)
        assert make_log_prior(names, None) is None
        assert make_log_prior(names, {"beta_x": (0.0, 1.0)}) is None

    def test_prior_pulls_an_unconstrained_parameter(self, run):
        sim, _ = run
        # Fix everything but tau and give tau a tight prior far from the
        # likelihood's preference; the MAP must sit near the prior.
        from vpjax.validation.eeg_fmri_statespace import BOLD_ONLY_NAMES
        init = default_init(sim["bold"], ())
        fixed = {n: init[n] for n in BOLD_ONLY_NAMES if n != "tau"}
        r = fit_run(sim["t_bold"], sim["bold"], fixed=fixed, restarts=0, max_steps=60,
                    priors={"tau": (float(np.log(5.0)), 0.05)})
        assert abs(np.log(r["estimate"]["tau"]) - np.log(5.0)) < 0.2
        assert r["priors"] == {"tau": [float(np.log(5.0)), 0.05]}
