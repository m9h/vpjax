"""Tests for the simulated EEG-fMRI recovery validation.

The expensive part of this module is the fitting, so the tests are
split: cheap, deterministic checks that the simulation and the forward
model are self-consistent, and one slow recovery run marked so it can
be deselected.

The sharpest cheap check is that the filter whitens the innovations
*at the true parameters*. That isolates the model and the filter from
the optimizer: if whiteness holds at the truth but a fit does not find
it, the problem is the search, not the estimator.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vpjax.statespace import ll_filter, residual_diagnostics, rts_smoother
from vpjax.validation.statespace_recovery import (
    Q as Q_IDX,
    V as V_IDX,
    Z as Z_IDX,
    _FIT_NAMES,
    _build_model,
    augmented_drift,
    augmented_observation,
    baseline_state,
    compare_modalities,
    fit_simulation,
    format_comparison,
    make_grid,
    simulate,
)


@pytest.fixture(scope="module")
def sim():
    return simulate(duration=90.0, seed=0)


class TestForwardModel:
    def test_baseline_is_a_fixed_point(self):
        """With no drive the resting state must not move."""
        drift = augmented_drift(0.65, 0.98, 2.0, 0.41, 0.32, 0.34)
        dx = drift(0.0, baseline_state(), None)
        # Tolerance is float32 round-off in the dHb balance, where
        # f*E(f)/E0 and fout*q/v each evaluate to 1 at baseline.
        assert jnp.allclose(dx, 0.0, atol=1e-6)

    def test_baseline_gives_zero_bold(self):
        observe = augmented_observation()
        y = observe(0.0, baseline_state(), None)
        assert float(y[0]) == pytest.approx(0.0)
        assert float(y[1]) == pytest.approx(0.0, abs=1e-12)

    def test_positive_drive_raises_flow_then_bold(self):
        """A step drive must increase flow, and dHb must fall."""
        drift = augmented_drift(0.65, 0.98, 1e6, 0.41, 0.32, 0.34)
        x = baseline_state().at[Z_IDX].set(0.5)
        from vpjax.statespace import ll_mean_step

        for _ in range(200):
            x, _ = ll_mean_step(drift, jnp.array(0.0), x, jnp.array(0.02))
        assert float(x[V_IDX]) > 1.0      # volume expands
        assert float(x[Q_IDX]) < 1.0      # deoxyhemoglobin washes out
        assert float(augmented_observation()(0.0, x, None)[1]) > 0.0

    def test_drift_survives_unphysiological_states(self):
        """The filter's mean is unconstrained; the drift must stay finite."""
        drift = augmented_drift(0.65, 0.98, 2.0, 0.41, 0.32, 0.34)
        for bad in ([0.0, 0.0, -5.0, -5.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]):
            dx = drift(0.0, jnp.array(bad), None)
            assert jnp.all(jnp.isfinite(dx))


class TestSimulation:
    def test_shapes_and_rates(self, sim):
        assert sim["t_eeg"].size == 901        # 90 s at 0.1 s
        assert sim["t_bold"].size == 46        # 90 s at TR 2 s
        assert sim["x"].shape[1] == 5

    def test_drive_variance_matches_the_ou_stationary_value(self, sim):
        """Var[z] should be q_z tau_z / 2 for the OU drive."""
        want = sim["truth"]["q_z"] * sim["truth"]["tau_z"] / 2.0
        got = float(np.var(sim["x"][:, Z_IDX]))
        assert got == pytest.approx(want, rel=0.3)

    def test_bold_is_a_plausible_magnitude(self, sim):
        """Fractional BOLD change should be percent-scale, not unit-scale."""
        assert 0.001 < float(np.std(sim["bold"])) < 0.2

    def test_grid_drops_eeg_when_asked(self, sim):
        both = make_grid(sim)
        bold_only = make_grid(sim, use_eeg=False)
        assert set(both.channels) == {"bold", "eeg"}
        assert bold_only.channels == ("bold",)
        assert float(bold_only.mask.sum()) == 46.0
        assert float(both.mask.sum()) == 901.0 + 46.0

    def test_max_dt_bounds_the_propagation_step(self, sim):
        g = make_grid(sim, max_dt=0.25)
        assert float(jnp.max(jnp.diff(g.t))) <= 0.25 + 1e-9


class TestFilterAtTruth:
    def _run(self, sim, use_eeg):
        grid = make_grid(sim, use_eeg=use_eeg)
        build = _build_model(grid, sim)
        theta = jnp.array([sim["truth"][n] for n in _FIT_NAMES])
        spec = build(theta)
        spec.setdefault("r_diag", grid.r_diag)
        res = ll_filter(t=grid.t, y=grid.y, mask=grid.mask, **spec)
        return grid, res

    def test_innovations_are_white_at_the_true_parameters(self, sim):
        grid, res = self._run(sim, use_eeg=True)
        d = residual_diagnostics(res, channels=grid.channels)
        for name in grid.channels:
            c = d["per_channel"][name]
            assert float(c["variance"]) == pytest.approx(1.0, abs=0.45), name
            # Tolerance scaled to the sampling error of a lag-1 estimate,
            # ~1/sqrt(n_pairs): the BOLD channel has only ~46 samples in a
            # 90 s run, so a fixed threshold would be testing noise.
            tol = max(3.0 / np.sqrt(float(c["n_pairs"])), 0.1)
            assert abs(float(c["lag1"])) < tol, (name, float(c["lag1"]), tol)

    def test_filter_tracks_the_simulated_drive(self, sim):
        """The estimated drive should correlate strongly with the truth."""
        grid, res = self._run(sim, use_eeg=True)
        idx = np.searchsorted(np.asarray(grid.t), np.round(sim["t_eeg"], 9))
        est = np.asarray(res.mean)[idx, Z_IDX]
        true = sim["x"][:: int(round(0.1 / 0.01)), Z_IDX][: est.size]
        assert np.corrcoef(est, true)[0, 1] > 0.9

    def test_smoother_improves_the_drive_estimate(self, sim):
        grid, res = self._run(sim, use_eeg=True)
        sm = rts_smoother(res)
        idx = np.searchsorted(np.asarray(grid.t), np.round(sim["t_eeg"], 9))
        true = sim["x"][:: int(round(0.1 / 0.01)), Z_IDX]
        true = true[: idx.size]
        f_err = np.abs(np.asarray(res.mean)[idx, Z_IDX] - true).mean()
        s_err = np.abs(np.asarray(sm.mean)[idx, Z_IDX] - true).mean()
        assert s_err <= f_err

    def test_truth_beats_perturbed_parameters(self, sim):
        """The likelihood must prefer the true parameters locally."""
        grid = make_grid(sim, use_eeg=True)
        build = _build_model(grid, sim)
        from vpjax.statespace import innovation_log_likelihood

        truth = jnp.array([sim["truth"][n] for n in _FIT_NAMES])
        ll_true = innovation_log_likelihood(build, jnp.log(truth), grid)
        for i in range(len(_FIT_NAMES)):
            for mult in (0.6, 1.6):
                ll = innovation_log_likelihood(
                    build, jnp.log(truth.at[i].multiply(mult)), grid
                )
                assert float(ll_true) > float(ll), (_FIT_NAMES[i], mult)

    def test_bold_only_cannot_whiten_this_model(self, sim):
        """The scientific claim: 46 volumes do not constrain the drive.

        Stated as a test so that it fails loudly if a future change to
        the model or the filter makes BOLD alone sufficient -- which
        would undercut the case for recording EEG at all.
        """
        grid, res = self._run(sim, use_eeg=False)
        d = residual_diagnostics(res, channels=grid.channels)
        both_grid, both_res = self._run(sim, use_eeg=True)
        d_both = residual_diagnostics(both_res, channels=both_grid.channels)
        # At the true parameters both are calibrated; the difference is
        # in how much the data constrain the state, not in the fit.
        assert float(d["per_channel"]["bold"]["n"]) == 46.0
        assert float(d_both["per_channel"]["eeg"]["n"]) == 901.0
        # The drive posterior must be much tighter with EEG present.
        var_bold = float(jnp.mean(res.cov[:, Z_IDX, Z_IDX]))
        var_both = float(jnp.mean(both_res.cov[:, Z_IDX, Z_IDX]))
        assert var_both < 0.5 * var_bold


class TestAmplitudeLimit:
    """Encode the measured validity limit of the local linearization.

    The BOLD observation is nonlinear in v and q, so a first-order
    filter degrades as the fluctuation grows.  These tests pin the
    boundary: calibrated at resting-state amplitude, visibly biased at
    four times it.  If a future change moves that boundary, these fail
    and the table in the module docstring needs updating.
    """

    def _bold_innovation_variance(self, q_z):
        sim = simulate(duration=180.0, q_z=q_z, seed=0)
        grid = make_grid(sim, use_eeg=True)
        build = _build_model(grid, sim)
        theta = jnp.array([sim["truth"][n] for n in _FIT_NAMES])
        spec = build(theta)
        spec.setdefault("r_diag", grid.r_diag)
        res = ll_filter(t=grid.t, y=grid.y, mask=grid.mask, **spec)
        d = residual_diagnostics(res, channels=grid.channels)
        return (
            float(d["per_channel"]["bold"]["variance"]),
            float(d["per_channel"]["eeg"]["variance"]),
            float(np.std(sim["bold"])),
        )

    def test_calibrated_at_resting_state_amplitude(self):
        bold_var, eeg_var, bold_sd = self._bold_innovation_variance(0.01)
        assert bold_sd < 0.025
        assert bold_var == pytest.approx(1.0, abs=0.25)
        assert eeg_var == pytest.approx(1.0, abs=0.1)

    def test_biased_at_large_amplitude(self):
        bold_var, eeg_var, bold_sd = self._bold_innovation_variance(0.5)
        assert bold_sd > 0.03
        assert bold_var > 1.5          # the nonlinear channel degrades
        assert eeg_var == pytest.approx(1.0, abs=0.1)   # the linear one does not


@pytest.mark.slow
class TestRecovery:
    def test_recovers_parameters_with_restarts(self, sim):
        r = fit_simulation(sim, use_eeg=True, restarts=8, seed=1)
        assert r["relative_error"]["kappa"] < 0.25
        assert r["relative_error"]["tau"] < 0.25
        pc = r["diagnostics"]["per_channel"]
        assert abs(pc["eeg"]["lag1"]) < 0.25

    def test_comparison_reports_both_conditions(self, sim):
        c = compare_modalities(sim, restarts=4)
        assert set(c) == {
            "bold_only", "both", "error_ratio",
            "calibrated", "identifiable", "comparable",
        }
        # comparable must agree with the per-arm identifiability verdicts,
        # since the error ratios are only meaningful when both hold.
        assert c["comparable"] == all(c["identifiable"].values())
        assert c["bold_only"]["channels"] == ("bold",)
        assert set(c["both"]["channels"]) == {"bold", "eeg"}
        text = format_comparison(c)
        assert "err ratio" in text and "innovation whiteness" in text


class TestDrugInput:
    """A deterministic input through the drive, with its gain estimated."""

    def test_shape_is_unit_peak_and_causal(self):
        from vpjax.validation.statespace_recovery import drug_shaped_input
        u = drug_shaped_input(onset=30.0, rise=10.0, decay=60.0)
        t = np.arange(0.0, 300.0, 0.5)
        v = u(t)
        assert np.all(v[t < 30.0] == 0.0)
        assert np.isclose(v.max(), 1.0)
        assert v[-1] < 0.1

    def test_shape_rejects_bad_timescales(self):
        from vpjax.validation.statespace_recovery import drug_shaped_input
        with pytest.raises(ValueError):
            drug_shaped_input(0.0, 60.0, 10.0)

    def test_simulation_carries_input_and_beta(self):
        from vpjax.validation.statespace_recovery import (
            default_fit_names, drug_shaped_input, simulate,
        )
        u = drug_shaped_input(20.0, 10.0, 40.0)
        sim = simulate(duration=120.0, drive_input=u, beta=0.3, seed=0)
        assert sim["truth"]["beta"] == 0.3
        assert sim["u_fine"].shape == sim["t_fine"].shape
        assert default_fit_names(sim)[-1] == "beta"
        # The input pushes the drive well above its resting fluctuations.
        assert sim["x"][:, 0].max() > 5 * sim["drive_sd"]

    def test_without_input_beta_is_not_exposed(self):
        from vpjax.validation.statespace_recovery import (
            _build_model, default_fit_names, make_grid, simulate,
        )
        sim = simulate(duration=60.0, seed=0)
        assert "beta" not in default_fit_names(sim)
        with pytest.raises(ValueError, match="only estimable"):
            _build_model(make_grid(sim), sim, ("kappa", "beta"))

    def test_fit_subset_fixes_the_rest(self):
        from vpjax.validation.statespace_recovery import fit_simulation, simulate
        sim = simulate(duration=60.0, seed=0)
        r = fit_simulation(sim, restarts=0, max_steps=5, fit_names=("kappa", "tau"))
        assert r["fit_names"] == ("kappa", "tau")
        assert set(r["estimate"]) == {"kappa", "tau"}
        assert set(r["standard_error"]) == {"kappa", "tau"}

    @pytest.mark.slow
    def test_beta_recovered_with_eeg(self):
        from vpjax.validation.statespace_recovery import (
            drug_shaped_input, fit_simulation, simulate,
        )
        # Same configuration as scripts/simultaneity_case.py's drug arm:
        # a slow input over a 6-minute run, where the gain is identified
        # at a relative SE near 0.1 with the fast channel present.
        u = drug_shaped_input(60.0, 30.0, 240.0)
        sim = simulate(duration=360.0, drive_input=u, beta=0.5, seed=0)
        r = fit_simulation(sim, use_eeg=True, restarts=4)
        assert r["identifiable"]
        assert r["standard_error"]["beta"] < 0.2
        assert r["relative_error"]["beta"] < 3 * r["standard_error"]["beta"]
