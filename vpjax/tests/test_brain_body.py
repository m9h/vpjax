"""The five brain-body topologies, their stability bound and what resolves them."""

import jax.numpy as jnp
import numpy as np
import pytest

from vpjax.validation.brain_body import (
    A,
    CORE_NAMES,
    DEFAULT_LOOP_GAIN,
    FIT_CORE,
    TOPOLOGIES,
    Z,
    free_names,
    input_shape,
    integrate,
    loop_gain,
    nominal,
    observe,
    simulate,
    topology_identifiability,
    topology_params,
)


def _run(topology, duration=600.0, dt=0.1, **kw):
    t = np.arange(0.0, duration, dt)
    u = jnp.asarray(input_shape(t, 120.0, 60.0, 300.0))
    return integrate(topology_params(topology, **kw), u, dt)


class TestTopologyStructure:
    def test_measurement_topology_never_moves_the_drive(self):
        """No neural effect means z stays exactly at rest."""
        traj = _run("measurement")
        assert float(jnp.ptp(traj[:, Z])) < 1e-9
        # ...while the BOLD still moves, which is the whole problem.
        _, bold, _ = observe(traj)
        assert float(jnp.ptp(bold)) > 0.01

    def test_efferent_moves_brain_before_body(self):
        traj = _run("efferent")
        z, _, a = traj[:, Z], None, traj[:, A]
        assert float(jnp.ptp(z)) > 0.5 and float(jnp.ptp(a)) > 0.1
        # The body peaks later than the brain when the brain drives it.
        assert int(jnp.argmax(a)) > int(jnp.argmax(z))

    def test_afferent_moves_body_before_brain(self):
        traj = _run("afferent")
        z, a = traj[:, Z], traj[:, A]
        assert float(jnp.ptp(z)) > 0.1 and float(jnp.ptp(a)) > 0.5
        assert int(jnp.argmax(z)) > int(jnp.argmax(a))

    def test_common_drive_has_no_coupling(self):
        p = topology_params("common")
        assert p["g_bb"] == 0.0 and p["g_bp"] == 0.0

    def test_free_names_drops_a_time_constant_with_no_driven_state(self):
        """tau_z is meaningless when nothing reaches the drive."""
        assert "tau_z" not in free_names("measurement")
        assert "tau_a" in free_names("measurement")
        for name in ("efferent", "afferent", "common", "loop"):
            assert set(CORE_NAMES) <= set(free_names(name))


class TestLoopStability:
    def test_loop_gain_is_the_product_of_both_gains_and_both_times(self):
        p = nominal(target_loop_gain=0.36)
        assert loop_gain(p) == pytest.approx(0.36, rel=1e-6)

    def test_the_even_split_gives_each_direction_the_same_share(self):
        p = nominal(target_loop_gain=0.25)
        assert p["g_bb"] * p["tau_a"] == pytest.approx(p["g_bp"] * p["tau_z"])

    def test_a_loop_gain_at_or_past_one_is_refused(self):
        with pytest.raises(ValueError, match="stability boundary"):
            nominal(target_loop_gain=1.0)

    def test_the_loop_amplifies_by_one_over_one_minus_the_gain(self):
        """The closed-loop excursion over the open-loop one is 1/(1-L).

        This is the check that the stability analysis in ``loop_gain``
        describes the integrated model and not just its linearisation on
        paper.
        """
        L = 0.25
        closed = _run("loop", target_loop_gain=L)
        open_loop = _run("efferent", target_loop_gain=L)
        ratio = float(jnp.ptp(closed[:, Z]) / jnp.ptp(open_loop[:, Z]))
        assert ratio == pytest.approx(1.0 / (1.0 - L), rel=0.02)

    def test_a_marginal_loop_runs_away(self):
        """At a loop gain of one the drive integrates instead of returning."""
        p = topology_params("loop")
        p["g_bb"] = 1.0 / p["tau_a"]
        p["g_bp"] = 1.0 / p["tau_z"]
        assert loop_gain(p) == pytest.approx(1.0)
        t = np.arange(0.0, 600.0, 0.1)
        u = jnp.asarray(input_shape(t, 120.0, 60.0, 300.0))
        traj = integrate(p, u, 0.1)
        # The drive ends far from rest rather than decaying back to it.
        assert abs(float(traj[-1, Z])) > 5.0


class TestIdentifiability:
    """Which recording sets resolve a topology's own parameters."""

    @pytest.mark.parametrize("topology", sorted(TOPOLOGIES))
    def test_physiology_confers_identifiability_and_bold_alone_does_not(self, topology):
        kw = dict(duration=600.0, decay=300.0)
        bold_only = topology_identifiability(topology, observers=("bold",), **kw)
        with_hr = topology_identifiability(topology, observers=("bold", "hr"), **kw)
        assert not bold_only["is_identifiable"]
        assert bold_only["rank"] < bold_only["n_params"]
        assert with_hr["is_identifiable"], topology
        assert with_hr["rank"] == with_hr["n_params"]

    def test_the_fast_channel_does_not_rescue_the_rank(self):
        """It improves conditioning, but the body arm stays unobserved."""
        kw = dict(duration=600.0, decay=300.0)
        without = topology_identifiability("efferent", observers=("bold",), **kw)
        with_fast = topology_identifiability(
            "efferent", observers=("bold", "fast"), **kw
        )
        assert with_fast["rank"] == without["rank"]

    def test_bold_alone_fixes_only_the_loop_product(self):
        """The flat direction trades the two coupling gains against each other."""
        r = topology_identifiability(
            "loop", observers=("bold",), duration=600.0, decay=300.0
        )
        assert r["rank"] == r["n_params"] - 1
        (flat,) = r["collinear_sets"]
        loaded = {n: c for n, c in zip(flat["params"], flat["coefficients"])}
        assert set(loaded) == {"g_bb", "g_bp"}
        # Equal and opposite: only the product is determined.
        assert loaded["g_bb"] == pytest.approx(-loaded["g_bp"], rel=0.05)

    def test_an_unknown_observer_is_refused(self):
        with pytest.raises(ValueError, match="unknown observer"):
            topology_identifiability("efferent", observers=("bold", "eyebrow"))

    def test_an_unknown_topology_is_refused(self):
        with pytest.raises(ValueError, match="unknown topology"):
            topology_params("telepathy")

    def test_the_reported_loop_gain_is_zero_for_the_open_topologies(self):
        for name in ("efferent", "afferent", "common", "measurement"):
            assert loop_gain(topology_params(name)) == 0.0
        assert loop_gain(topology_params("loop")) == pytest.approx(DEFAULT_LOOP_GAIN)


class TestChannelLoadings:
    """A free loading on the drive is estimable; one on the body state is not.

    The drive's scale is pinned by the Balloon, where ``z`` enters
    ``ds/dt`` with a coefficient of exactly one.  The autonomic state has
    no such anchor, so rescaling it can be absorbed by its own gains and
    its channel loading together -- an exact invariance, and therefore a
    flat direction that no amount of data removes.
    """

    ALL = ("bold", "fast", "hr")

    @pytest.mark.parametrize("topology", sorted(TOPOLOGIES))
    def test_a_free_drive_loading_stays_identifiable(self, topology):
        r = topology_identifiability(
            topology, observers=self.ALL, loadings=("fast",),
            duration=600.0, decay=300.0,
        )
        assert r["is_identifiable"], (topology, r.get("collinear_sets"))

    @pytest.mark.parametrize("topology", sorted(TOPOLOGIES))
    def test_a_free_body_loading_always_costs_exactly_one_direction(self, topology):
        r = topology_identifiability(
            topology, observers=self.ALL, loadings=("hr",),
            duration=600.0, decay=300.0,
        )
        assert not r["is_identifiable"]
        assert r["rank"] == r["n_params"] - 1
        assert len(r["collinear_sets"]) == 1

    @pytest.mark.parametrize("topology", sorted(TOPOLOGIES))
    def test_the_flat_direction_is_the_rescaling_of_the_body_state(self, topology):
        """The null space matches the analytic invariance, term for term.

        Sending ``a -> c a`` leaves every observation unchanged provided
        the gains that *drive* ``a`` scale up with it and the gains that
        *read* ``a`` scale down.  On the log scale that is a vector of
        ``+1`` on each driving gain and ``-1`` on each reading one, so
        the null direction must load exactly those parameters with
        exactly those relative signs.
        """
        r = topology_identifiability(
            topology, observers=self.ALL, loadings=("hr",),
            duration=600.0, decay=300.0,
        )
        (flat,) = r["collinear_sets"]
        loaded = dict(zip(flat["params"], flat["coefficients"]))
        active = TOPOLOGIES[topology]
        # Gains writing into a, and gains reading out of it.
        drives = [g for g in ("beta_a", "g_bb") if g in active]
        reads = [g for g in ("g_bp",) if g in active] + ["gain_hr"]
        assert set(loaded) == set(drives + reads)
        # One overall sign is arbitrary; fix it from the first driver.
        s = np.sign(loaded[drives[0]])
        for g in drives:
            assert np.sign(loaded[g]) == s
        for g in reads:
            assert np.sign(loaded[g]) == -s
        # Equal magnitudes: the invariance weights every term the same.
        mags = [abs(v) for v in loaded.values()]
        assert max(mags) == pytest.approx(min(mags), rel=0.05)
        assert max(mags) == pytest.approx(1.0 / np.sqrt(len(loaded)), rel=0.05)

    def test_a_loading_on_an_undriven_state_is_not_offered(self):
        """``measurement`` never drives the drive, so it has no drive loading."""
        assert "gain_fast" not in free_names("measurement", ("fast",))
        assert "gain_hr" in free_names("measurement", ("hr",))
        assert "gain_fast" in free_names("efferent", ("fast",))


class TestSimulation:
    def test_each_channel_is_noised_against_its_own_excursion(self):
        sim = simulate("efferent", duration=400.0, decay=200.0, snr=3.0, seed=0)
        for c in ("fast", "bold", "hr"):
            # Against the *deterministic* excursion, not the realised
            # one: process noise must not feed back into the noise scale.
            ptp = float(np.ptp(sim[f"noisefree_{c}"]))
            assert sim["noise_sd"][c] == pytest.approx(ptp / 3.0, rel=0.05)
            assert np.ptp(sim[f"latent_{c}"]) >= np.ptp(sim[f"noisefree_{c}"])

    def test_a_channel_the_topology_never_moves_falls_back_to_unit_noise(self):
        """measurement leaves the drive at rest, so its channel is noise only."""
        sim = simulate("measurement", duration=400.0, decay=200.0, seed=0)
        assert float(np.ptp(sim["noisefree_fast"])) < 1e-9
        assert sim["noise_sd"]["fast"] == 1.0
        # The realised channel still moves, because the drive has its
        # own process noise -- it just carries no input-driven signal.
        assert float(np.ptp(sim["latent_fast"])) > 0.0

    def test_the_same_seed_reproduces_the_run(self):
        kw = dict(duration=300.0, decay=150.0, seed=7)
        a_ = simulate("loop", **kw)
        b_ = simulate("loop", **kw)
        assert np.allclose(a_["bold"], b_["bold"])

    def test_process_noise_moves_the_drive_without_an_input(self):
        """q_z alone fluctuates z, which is why tau_z stays informed."""
        sim = simulate("measurement", duration=400.0, decay=200.0, q_z=0.05, seed=0)
        assert float(np.std(sim["x"][:, Z])) > 1e-3


class TestOmittedGainsAreZero:
    """The invariant the whole confusion matrix rests on."""

    def test_a_fitted_topology_ignores_a_gain_it_does_not_carry(self):
        """Fitting 'efferent' to 'loop' data must not use the body-to-brain arm.

        Checked on the drift itself rather than through a fit: with the
        autonomic state held away from rest, a model that carried
        ``g_bp`` would show it in ``dz/dt``.  ``efferent`` does not, so
        that row must be exactly the decay of the drive.
        """
        import jax.numpy as jnp
        from vpjax.validation.brain_body import build_model, make_grid

        sim = simulate("loop", duration=300.0, decay=150.0, seed=0)
        assert sim["truth"]["g_bp"] > 0.0
        grid = make_grid(sim)
        names = FIT_CORE + TOPOLOGIES["efferent"]
        build = build_model(grid, sim, "efferent")
        spec = build(jnp.array([sim["truth"][n] for n in names]))
        x = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 3.0])   # a well away from rest
        dx = spec["f"](0.0, x, (None, 0.0))
        assert float(dx[Z]) == pytest.approx(0.0, abs=1e-12)

        # ...whereas the loop model, same data, does use it.
        names_l = FIT_CORE + TOPOLOGIES["loop"]
        spec_l = build_model(grid, sim, "loop")(
            jnp.array([sim["truth"][n] for n in names_l])
        )
        assert float(spec_l["f"](0.0, x, (None, 0.0))[Z]) > 0.1

    def test_an_unknown_observer_is_refused_by_the_grid(self):
        from vpjax.validation.brain_body import make_grid

        sim = simulate("efferent", duration=200.0, decay=100.0, seed=0)
        with pytest.raises(ValueError, match="unknown observer"):
            make_grid(sim, ("bold", "eyebrow"))


class TestFitBookkeeping:
    def test_aic_follows_from_the_likelihood_and_the_count(self):
        from vpjax.validation.brain_body import fit_topology

        sim = simulate("efferent", duration=300.0, decay=150.0, seed=0)
        r = fit_topology(sim, "efferent", restarts=1, max_steps=40)
        assert r["aic"] == pytest.approx(
            2.0 * r["n_params"] - 2.0 * r["log_likelihood"]
        )
        assert r["n_params"] == len(FIT_CORE) + len(TOPOLOGIES["efferent"])


class TestExplicitInitialization:
    def test_named_start_reaches_estimator_and_errors_keep_their_units(self, monkeypatch):
        import vpjax.statespace as ss
        from vpjax.validation.brain_body import fit_topology

        sim = simulate("efferent", duration=10.0, dt=0.1)
        names = ("tau_a", "g_bb")
        captured = {}

        def fake_fit(build, grid, init, **kw):
            captured["init"] = np.asarray(init)
            captured.update(kw)
            return dict(theta=init, log_theta=jnp.log(init), log_likelihood=1.0,
                        success=True, best_start=0, all_log_likelihoods=jnp.array([1.0]))

        def fake_uncertainty(*args):
            return dict(standard_error=jnp.array([0.2, 0.3]),
                        positive_definite=True, eigenvalues=jnp.array([1., 2.]),
                        correlation=jnp.eye(2), identifiable=True, curvature_ratio=0.5)

        monkeypatch.setattr(ss, "fit_statespace", fake_fit)
        monkeypatch.setattr(ss, "parameter_uncertainty", fake_uncertainty)
        r = fit_topology(sim, "efferent", fit_names=names,
                         init={"g_bb": 0.2}, restarts=0)
        assert captured["init"] == pytest.approx([sim["truth"]["tau_a"] * 0.5, 0.2])
        assert captured["restarts"] == 0
        assert r["standard_error"]["g_bb"] == pytest.approx(0.3)
        assert r["standard_error_absolute"]["g_bb"] == pytest.approx(0.06)
        assert r["standard_error_scale"] == "log_parameter"

    @pytest.mark.parametrize("init", [{"typo": 1}, {"g_bb": 0},
                                      {"g_bb": -1}, {"g_bb": np.nan},
                                      {"g_bb": np.inf}])
    def test_invalid_starts_are_rejected(self, init):
        from vpjax.validation.brain_body import fit_topology
        sim = simulate("efferent", duration=10.0, dt=0.1)
        with pytest.raises(ValueError, match="init"):
            fit_topology(sim, "efferent", init=init)


class TestPeriodicDrive:
    def test_schedule_is_zero_before_onset_and_repeats_on_the_clock(self):
        from vpjax.validation.brain_body import duty_cycle_shape, drive_series
        times = np.array([-1, 0, 4, 5, 15, 16, 20, 21], dtype=float) + 3
        expected = [0, 1, 1, 0, 0, 1, 1, 0]
        assert duty_cycle_shape(times, 16, onset=3).tolist() == expected
        assert drive_series(times, "duty", period=16, onset=3).tolist() == expected

    @pytest.mark.parametrize("kw", [{"period": 0}, {"period": -1},
                                    {"period": 16, "on_fraction": 0},
                                    {"period": 16, "on_fraction": 1}])
    def test_invalid_schedule_is_rejected(self, kw):
        from vpjax.validation.brain_body import duty_cycle_shape
        with pytest.raises(ValueError):
            duty_cycle_shape(np.array([0.]), **kw)
