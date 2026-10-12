"""The five brain-body topologies, their stability bound and what resolves them."""

import jax.numpy as jnp
import numpy as np
import pytest

from vpjax.validation.brain_body import (
    A,
    CORE_NAMES,
    DEFAULT_LOOP_GAIN,
    TOPOLOGIES,
    Z,
    free_names,
    input_shape,
    integrate,
    loop_gain,
    nominal,
    observe,
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
