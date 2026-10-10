"""Tests for the PK/PD subpackage.

The linear compartment models have closed-form solutions independent of
the matrix-exponential propagator used in the implementation, so most
of these tests check the solver against algebra rather than against
itself.  The remaining structural checks — mass balance, superposition,
dose proportionality — are the ones that catch a mis-signed or
mis-scaled rate constant, which is the characteristic PK bug: it leaves
the curve looking entirely plausible.
"""

import jax
import jax.numpy as jnp
import pytest

from vpjax.pharmacokinetics import (
    ONE_COMPARTMENT,
    ORAL_ONE_COMPARTMENT,
    THREE_COMPARTMENT,
    TWO_COMPARTMENT,
    EffectSiteParams,
    HillParams,
    MichaelisMentenParams,
    PKParams,
    ResidualErrorParams,
    bolus,
    build_schedule,
    combine,
    concentration,
    effect_site_concentration,
    fit_pk_subject,
    hill_effect,
    infusion,
    initial_guess,
    occupancy,
    rate_matrix,
    regional_drive,
    repeated,
    residual_sd,
    secondary_parameters,
    solve_linear_pk,
    solve_nonlinear_pk,
    weighted_residuals,
)


class TestOneCompartment:
    def test_iv_bolus_matches_closed_form(self):
        """C(t) = D/V1 · exp(-CL/V1 · t) for a single IV bolus."""
        p = PKParams(CL=jnp.array(2.0), V1=jnp.array(10.0))
        dose = 100.0
        t = jnp.linspace(0.0, 10.0, 21)
        got = concentration(p, bolus(dose), t, ONE_COMPARTMENT)
        want = (dose / 10.0) * jnp.exp(-(2.0 / 10.0) * t)
        assert jnp.allclose(got, want, rtol=2e-5, atol=1e-6)

    def test_infusion_approaches_steady_state(self):
        """A long infusion plateaus at Css = rate / CL."""
        p = PKParams(CL=jnp.array(2.0), V1=jnp.array(10.0))
        rate = 4.0
        duration = 200.0
        reg = infusion(rate * duration, duration)
        t = jnp.array([duration])
        css = concentration(p, reg, t, ONE_COMPARTMENT)[0]
        assert float(css) == pytest.approx(rate / 2.0, rel=1e-3)

    def test_dose_proportionality(self):
        """A linear model scales exactly with dose."""
        p = PKParams(CL=jnp.array(1.5), V1=jnp.array(8.0))
        t = jnp.linspace(0.1, 12.0, 15)
        c1 = concentration(p, bolus(50.0), t, ONE_COMPARTMENT)
        c2 = concentration(p, bolus(150.0), t, ONE_COMPARTMENT)
        assert jnp.allclose(c2, 3.0 * c1, rtol=2e-5)

    def test_superposition_of_repeated_doses(self):
        """Repeated dosing equals the sum of time-shifted single doses."""
        p = PKParams(CL=jnp.array(1.0), V1=jnp.array(5.0))
        interval, n = 4.0, 3
        t = jnp.linspace(0.0, 20.0, 41)
        multi = concentration(p, repeated(20.0, interval, n), t, ONE_COMPARTMENT)

        single = jnp.zeros_like(t)
        for i in range(n):
            shifted = t - i * interval
            contrib = (20.0 / 5.0) * jnp.exp(-(1.0 / 5.0) * jnp.clip(shifted, 0.0))
            single = single + jnp.where(shifted >= 0.0, contrib, 0.0)

        assert jnp.allclose(multi, single, rtol=2e-5, atol=1e-6)

    def test_loading_dose_plus_infusion(self):
        """A bolus sized V1·Css with a matched infusion holds concentration flat."""
        cl, v1, css = 2.0, 10.0, 3.0
        p = PKParams(CL=jnp.array(cl), V1=jnp.array(v1))
        duration = 30.0
        reg = combine(
            bolus(v1 * css),
            infusion(cl * css * duration, duration),
        )
        t = jnp.linspace(0.0, duration, 16)
        c = concentration(p, reg, t, ONE_COMPARTMENT)
        assert jnp.allclose(c, css, rtol=2e-5)


class TestMultiCompartment:
    def test_mass_conserved_without_elimination(self):
        """With CL = 0 the total amount in the body is constant."""
        p = PKParams(
            CL=jnp.array(0.0), V1=jnp.array(10.0),
            Q2=jnp.array(3.0), V2=jnp.array(25.0),
        )
        reg = bolus(100.0)
        t = jnp.linspace(0.0, 50.0, 26)
        sched = build_schedule(reg, t)
        amounts = solve_linear_pk(p, reg, sched, TWO_COMPARTMENT)
        total = jnp.sum(amounts, axis=1)
        assert jnp.allclose(total, 100.0, rtol=2e-5)

    def test_two_compartment_matches_biexponential(self):
        """The propagator reproduces the textbook hybrid-constant solution."""
        cl, v1, q, v2 = 2.0, 10.0, 3.0, 25.0
        p = PKParams(
            CL=jnp.array(cl), V1=jnp.array(v1),
            Q2=jnp.array(q), V2=jnp.array(v2),
        )
        k10, k12, k21 = cl / v1, q / v1, q / v2
        s = k10 + k12 + k21
        disc = jnp.sqrt(s**2 - 4.0 * k10 * k21)
        alpha, beta = 0.5 * (s + disc), 0.5 * (s - disc)

        dose = 100.0
        t = jnp.linspace(0.0, 40.0, 41)
        a = dose / v1 * (alpha - k21) / (alpha - beta)
        b = dose / v1 * (k21 - beta) / (alpha - beta)
        want = a * jnp.exp(-alpha * t) + b * jnp.exp(-beta * t)

        got = concentration(p, bolus(dose), t, TWO_COMPARTMENT)
        assert jnp.allclose(got, want, rtol=2e-5, atol=1e-6)

    def test_terminal_half_life_is_the_slowest(self):
        p = PKParams(
            CL=jnp.array(2.0), V1=jnp.array(10.0),
            Q2=jnp.array(3.0), V2=jnp.array(60.0),
        )
        sec = secondary_parameters(p, TWO_COMPARTMENT)
        hl = sec["half_lives"]
        assert float(sec["t_half_terminal"]) == pytest.approx(float(jnp.max(hl)))
        assert float(hl[0]) < float(hl[-1])

    def test_vss_is_sum_of_volumes(self):
        p = PKParams(V1=jnp.array(10.0), V2=jnp.array(20.0), V3=jnp.array(40.0))
        sec = secondary_parameters(p, THREE_COMPARTMENT)
        assert float(sec["Vss"]) == pytest.approx(70.0)

    def test_three_compartment_is_triexponential(self):
        """Three distinct disposition rate constants, all positive."""
        p = PKParams(
            CL=jnp.array(2.0), V1=jnp.array(10.0),
            Q2=jnp.array(3.0), V2=jnp.array(25.0),
            Q3=jnp.array(0.6), V3=jnp.array(120.0),
        )
        A = rate_matrix(p, THREE_COMPARTMENT)
        lam = jnp.real(jnp.linalg.eigvals(A))
        assert A.shape == (3, 3)
        assert jnp.all(lam < 0.0)
        assert jnp.unique(jnp.round(lam, 6)).size == 3


class TestAbsorption:
    def test_oral_matches_bateman_function(self):
        """First-order absorption into a one-compartment body."""
        cl, v1, ka, f = 2.0, 10.0, 1.2, 0.8
        p = PKParams(
            CL=jnp.array(cl), V1=jnp.array(v1),
            ka=jnp.array(ka), F=jnp.array(f),
        )
        dose = 100.0
        k = cl / v1
        t = jnp.linspace(0.0, 20.0, 41)
        want = (f * dose * ka) / (v1 * (ka - k)) * (jnp.exp(-k * t) - jnp.exp(-ka * t))
        got = concentration(p, bolus(dose, compartment=0), t, ORAL_ONE_COMPARTMENT)
        assert jnp.allclose(got, want, rtol=2e-5, atol=1e-6)

    def test_oral_starts_at_zero_and_peaks_later(self):
        p = PKParams(CL=jnp.array(2.0), V1=jnp.array(10.0), ka=jnp.array(1.0))
        t = jnp.linspace(0.0, 20.0, 101)
        c = concentration(p, bolus(100.0, compartment=0), t, ORAL_ONE_COMPARTMENT)
        assert float(c[0]) == pytest.approx(0.0, abs=1e-12)
        assert int(jnp.argmax(c)) > 0


class TestNonlinearPath:
    def test_diffrax_path_matches_closed_form_without_saturation(self):
        """With mm=None the Diffrax path must reproduce the exact solution."""
        p = PKParams(
            CL=jnp.array(2.0), V1=jnp.array(10.0),
            Q2=jnp.array(3.0), V2=jnp.array(25.0),
        )
        reg = bolus(100.0)
        t = jnp.linspace(0.0, 20.0, 21)
        sched = build_schedule(reg, t)
        exact = solve_linear_pk(p, reg, sched, TWO_COMPARTMENT)
        numeric = solve_nonlinear_pk(p, reg, sched, TWO_COMPARTMENT, mm=None)
        assert jnp.allclose(numeric, exact, rtol=1e-4, atol=1e-6)

    def test_saturable_elimination_slows_clearance(self):
        """Adding Michaelis-Menten elimination cannot raise the concentration."""
        p = PKParams(CL=jnp.array(1.0), V1=jnp.array(10.0))
        reg = bolus(200.0)
        t = jnp.linspace(0.0, 20.0, 21)
        sched = build_schedule(reg, t)
        linear = solve_nonlinear_pk(p, reg, sched, ONE_COMPARTMENT, mm=None)
        mm = MichaelisMentenParams(Vmax=jnp.array(5.0), Km=jnp.array(2.0))
        sat = solve_nonlinear_pk(p, reg, sched, ONE_COMPARTMENT, mm=mm)
        assert jnp.all(sat[1:, 0] < linear[1:, 0])


class TestPharmacodynamics:
    def test_hill_bounds_and_midpoint(self):
        p = HillParams(ec50=jnp.array(5.0), gamma=jnp.array(1.5))
        assert float(hill_effect(jnp.array(0.0), p)) == pytest.approx(0.0)
        assert float(hill_effect(jnp.array(5.0), p)) == pytest.approx(0.5)
        assert float(hill_effect(jnp.array(1e6), p)) == pytest.approx(1.0, abs=1e-6)

    def test_occupancy_is_monotone_and_bounded(self):
        c = jnp.linspace(0.0, 100.0, 50)
        occ = occupancy(c, ec50=jnp.array(10.0), gamma=2.0)
        assert jnp.all(jnp.diff(occ) >= 0.0)
        assert jnp.all((occ >= 0.0) & (occ <= 1.0))

    def test_higher_gamma_is_steeper(self):
        """At twice EC50 a steeper Hill coefficient gives higher occupancy."""
        lo = occupancy(jnp.array(20.0), ec50=jnp.array(10.0), gamma=1.0)
        hi = occupancy(jnp.array(20.0), ec50=jnp.array(10.0), gamma=4.0)
        assert float(hi) > float(lo)

    def test_hill_gradient_finite_at_zero(self):
        g = jax.grad(lambda c: hill_effect(c, HillParams(gamma=jnp.array(2.0))))
        assert jnp.isfinite(g(jnp.array(0.0)))

    def test_effect_site_lags_plasma(self):
        """Ce peaks after Cp and is lower at the peak — the hysteresis."""
        p = PKParams(CL=jnp.array(2.0), V1=jnp.array(10.0), ka=jnp.array(1.5))
        t = jnp.linspace(0.0, 30.0, 301)
        cp = concentration(p, bolus(100.0, compartment=0), t, ORAL_ONE_COMPARTMENT)
        ce = effect_site_concentration(cp, dt=0.1, params=EffectSiteParams(ke0=jnp.array(0.3)))
        assert int(jnp.argmax(ce)) > int(jnp.argmax(cp))
        assert float(jnp.max(ce)) < float(jnp.max(cp))

    def test_effect_site_tracks_constant_plasma(self):
        """Under a constant Cp the effect site equilibrates to it."""
        cp = jnp.full((400,), 3.0)
        ce = effect_site_concentration(cp, dt=0.1, params=EffectSiteParams(ke0=jnp.array(0.5)))
        assert float(ce[-1]) == pytest.approx(3.0, rel=1e-3)

    def test_regional_drive_preserves_mean(self):
        occ = jnp.linspace(0.0, 1.0, 10)
        density = jnp.array([0.5, 1.0, 2.0, 4.0])
        drive = regional_drive(occ, density)
        assert drive.shape == (4, 10)
        assert jnp.allclose(jnp.mean(drive, axis=0), occ, rtol=2e-5)


class TestErrorModel:
    def test_sd_grows_with_prediction(self):
        p = ResidualErrorParams(sigma_add=jnp.array(0.1), sigma_prop=jnp.array(0.2))
        sd = residual_sd(jnp.array([0.0, 10.0]), p)
        assert float(sd[0]) == pytest.approx(0.1)
        assert float(sd[1]) == pytest.approx(jnp.sqrt(0.01 + 4.0), rel=1e-4)

    def test_weighting_equalises_relative_error(self):
        """A constant 10% error gives equal weighted residuals at any level."""
        p = ResidualErrorParams(sigma_add=jnp.array(0.0), sigma_prop=jnp.array(0.1))
        pred = jnp.array([1.0, 100.0])
        obs = pred * 1.1
        r = weighted_residuals(pred, obs, p)
        assert jnp.allclose(r, r[0], rtol=2e-5)


class TestEstimation:
    def test_recovers_one_compartment_parameters(self):
        true = PKParams(CL=jnp.array(2.5), V1=jnp.array(12.0))
        reg = bolus(100.0)
        t = jnp.linspace(0.25, 24.0, 24)
        data = concentration(true, reg, t, ONE_COMPARTMENT)

        res = fit_pk_subject(
            data, t, reg, ONE_COMPARTMENT,
            init_params=PKParams(CL=jnp.array(1.0), V1=jnp.array(20.0)),
        )
        assert float(res["CL"]) == pytest.approx(2.5, rel=1e-2)
        assert float(res["V1"]) == pytest.approx(12.0, rel=1e-2)

    def test_recovers_two_compartment_parameters(self):
        true = PKParams(
            CL=jnp.array(2.0), V1=jnp.array(10.0),
            Q2=jnp.array(3.0), V2=jnp.array(30.0),
        )
        reg = bolus(100.0)
        t = jnp.concatenate([jnp.linspace(0.1, 2.0, 12), jnp.linspace(3.0, 48.0, 20)])
        data = concentration(true, reg, t, TWO_COMPARTMENT)

        res = fit_pk_subject(
            data, t, reg, TWO_COMPARTMENT,
            init_params=PKParams(
                CL=jnp.array(1.0), V1=jnp.array(6.0),
                Q2=jnp.array(1.0), V2=jnp.array(50.0),
            ),
            max_steps=512,
        )
        for name, want in [("CL", 2.0), ("V1", 10.0), ("Q2", 3.0), ("V2", 30.0)]:
            assert float(res[name]) == pytest.approx(want, rel=5e-2)

    def test_fit_tolerates_noise(self):
        true = PKParams(CL=jnp.array(2.5), V1=jnp.array(12.0))
        reg = bolus(100.0)
        t = jnp.linspace(0.25, 24.0, 40)
        clean = concentration(true, reg, t, ONE_COMPARTMENT)
        noise = 1.0 + 0.05 * jax.random.normal(jax.random.PRNGKey(0), clean.shape)
        res = fit_pk_subject(
            clean * noise, t, reg, ONE_COMPARTMENT,
            init_params=PKParams(CL=jnp.array(1.0), V1=jnp.array(20.0)),
        )
        assert float(res["CL"]) == pytest.approx(2.5, rel=0.1)
        assert float(res["V1"]) == pytest.approx(12.0, rel=0.1)

    def test_initial_guess_is_right_order_of_magnitude(self):
        true = PKParams(CL=jnp.array(2.5), V1=jnp.array(12.0))
        t = jnp.linspace(0.05, 30.0, 60)
        data = concentration(true, bolus(100.0), t, ONE_COMPARTMENT)
        guess = initial_guess(data, t, dose=100.0)
        assert 0.3 < float(guess.CL) / 2.5 < 3.0
        assert 0.3 < float(guess.V1) / 12.0 < 3.0


class TestJaxTransforms:
    def test_differentiable_in_clearance(self):
        t = jnp.linspace(0.5, 10.0, 10)
        reg = bolus(100.0)
        sched = build_schedule(reg, t)

        def auc(log_cl):
            p = PKParams(CL=jnp.exp(log_cl), V1=jnp.array(10.0))
            c = solve_linear_pk(p, reg, sched, ONE_COMPARTMENT)[sched.obs_idx, 0] / 10.0
            return jnp.sum(c)

        g = jax.grad(auc)(jnp.log(2.0))
        assert jnp.isfinite(g)
        assert float(g) < 0.0  # faster clearance lowers exposure

    def test_jit_and_vmap_over_parameters(self):
        t = jnp.linspace(0.5, 10.0, 10)
        reg = bolus(100.0)
        sched = build_schedule(reg, t)

        @jax.jit
        def curve(cl):
            p = PKParams(CL=cl, V1=jnp.array(10.0))
            return solve_linear_pk(p, reg, sched, ONE_COMPARTMENT)[sched.obs_idx, 0] / 10.0

        cls = jnp.array([1.0, 2.0, 4.0])
        out = jax.vmap(curve)(cls)
        assert out.shape == (3, 10)
        # Higher clearance gives lower concentration at every time point.
        assert jnp.all(jnp.diff(out, axis=0) < 0.0)


class TestPopulation:
    def test_map_fit_recovers_population_median(self):
        pytest.importorskip("numpyro")
        from vpjax.pharmacokinetics.nlme import fit_population_map

        key = jax.random.PRNGKey(1)
        n_subj = 12
        reg = bolus(100.0)
        t = jnp.linspace(0.25, 24.0, 12)

        eta = 0.25 * jax.random.normal(key, (n_subj, 2))
        cls = 2.5 * jnp.exp(eta[:, 0])
        v1s = 12.0 * jnp.exp(eta[:, 1])
        data = jnp.stack([
            concentration(PKParams(CL=c, V1=v), reg, t, ONE_COMPARTMENT)
            for c, v in zip(cls, v1s)
        ])

        out = fit_population_map(
            data, t, reg, ONE_COMPARTMENT, ("CL", "V1"),
            n_steps=3000, prior_median={"CL": 1.0, "V1": 20.0},
        )
        log_pop = out["params"]["log_theta_pop_auto_loc"]
        cl_hat, v1_hat = float(jnp.exp(log_pop[0])), float(jnp.exp(log_pop[1]))
        assert cl_hat == pytest.approx(float(jnp.exp(jnp.mean(jnp.log(cls)))), rel=0.25)
        assert v1_hat == pytest.approx(float(jnp.exp(jnp.mean(jnp.log(v1s)))), rel=0.25)
        assert float(out["losses"][-1]) < float(out["losses"][0])

    def test_prior_predictive_runs(self):
        pytest.importorskip("numpyro")
        from numpyro.infer import Predictive

        from vpjax.pharmacokinetics.nlme import population_model

        reg = bolus(100.0)
        t = jnp.linspace(0.25, 24.0, 8)
        sched = build_schedule(reg, t)
        pred = Predictive(population_model, num_samples=16)
        draws = pred(
            jax.random.PRNGKey(0),
            conc_obs=None,
            regimen=reg,
            schedule=sched,
            structure=ONE_COMPARTMENT,
            fit_names=("CL", "V1"),
        )
        assert draws["obs"].shape[0] == 16
        assert jnp.all(jnp.isfinite(draws["pred"]))
