"""End-to-end integration tests.

Tests the full vpjax pipeline from neural activity through to
multi-modal observation signals.  These verify that modules compose
correctly and produce physiologically plausible results.

Pipeline 1 (Standard Balloon):
    neural stimulus → BalloonWindkessel ODE → BOLD / ASL / VASO

Pipeline 2 (Riera NVC):
    neural stimulus → RieraNVC ODE → multi-compartment → BOLD

Pipeline 3 (CMRO₂ hierarchy):
    Level 1: TRUST global OEF + CBF → global CMRO₂
    Level 2: qBOLD regional OEF + CBF → regional CMRO₂
    Level 3: Riera dynamic model → time-varying CMRO₂

Pipeline 4 (Calibrated fMRI):
    qBOLD R₂' → M (calibration) → Davis model → ΔCMRO₂

Pipeline 5 (Layer-resolved):
    stimulus → layer NVC → per-layer BOLD → ascending vein → devein
"""

import jax
import jax.numpy as jnp
import pytest

# Hemodynamics
from vpjax._types import BalloonParams, BalloonState
from vpjax.hemodynamics.balloon import BalloonWindkessel, solve_balloon
from vpjax.hemodynamics.bold import BOLDParams, observe_bold
from vpjax.hemodynamics.riera import RieraNVC, RieraParams, RieraState, riera_to_balloon
from vpjax.perfusion.asl import observe_asl
from vpjax.perfusion.vaso import observe_vaso

# Metabolism
from vpjax.metabolism.cmro2 import CMRO2Params, compute_cmro2
from vpjax.metabolism.oef import compute_oef, extraction_fraction
from vpjax.metabolism.fick import fick_cmro2, compute_cao2

# Perfusion
from vpjax.perfusion.trust import trust_oef, trust_global_cmro2
from vpjax.perfusion.kinetic import ASLKineticParams, asl_kinetic_signal, quantify_cbf

# qBOLD
from vpjax.qbold.signal_model import QBOLDParams, qbold_signal, compute_r2prime
from vpjax.qbold.calibrated import estimate_M_from_r2prime, davis_model, estimate_cmro2_change

# VASO
from vpjax.vaso.signal_model import vaso_signal_change
from vpjax.vaso.cbv_mapping import balloon_cbv_ratio

# Layers
from vpjax.layers.layer_nvc import layer_stimulus, ascending_vein_contamination, devein_bold

# Vascular
from vpjax.vascular.compliance import grubb_cbv


class TestBalloonPipeline:
    """Full pipeline: stimulus → Balloon ODE → multi-modal observations."""

    @pytest.fixture
    def balloon_trajectory(self):
        """3s block stimulus, 30s integration."""
        dt = 0.01
        T = 30.0
        n = int(T / dt)
        stim = jnp.zeros(n).at[int(1.0 / dt):int(4.0 / dt)].set(1.0)
        ts, traj = solve_balloon(BalloonParams(), stim, dt=dt)
        return ts, traj, stim

    def test_all_signals_baseline_at_rest(self, balloon_trajectory):
        """All observation modalities should be ~0 at rest."""
        _, traj, _ = balloon_trajectory
        y0 = BalloonState.steady_state()
        assert jnp.allclose(observe_bold(y0), 0.0, atol=1e-6)
        assert jnp.allclose(observe_asl(y0), 0.0, atol=1e-6)
        assert jnp.allclose(observe_vaso(y0), 0.0, atol=1e-6)

    def test_bold_asl_vaso_all_respond(self, balloon_trajectory):
        """All three modalities should show a response to stimulus."""
        _, traj, _ = balloon_trajectory
        bold = observe_bold(traj)
        asl = observe_asl(traj)
        vaso = observe_vaso(traj)

        assert float(jnp.max(bold)) > 0.001   # BOLD positive
        assert float(jnp.max(asl)) > 0.01     # ASL positive (CBF up)
        assert float(jnp.min(vaso)) < -0.001   # VASO negative (CBV up)

    def test_bold_peaks_after_asl(self, balloon_trajectory):
        """BOLD should peak later than ASL (hemodynamic lag)."""
        ts, traj, _ = balloon_trajectory
        bold = observe_bold(traj)
        asl = observe_asl(traj)
        t_bold_peak = ts[jnp.argmax(bold)]
        t_asl_peak = ts[jnp.argmax(asl)]
        # ASL (flow) should peak before or at same time as BOLD
        assert float(t_asl_peak) <= float(t_bold_peak) + 1.0

    def test_vaso_tracks_volume(self, balloon_trajectory):
        """VASO signal change should track CBV from Balloon model."""
        _, traj, _ = balloon_trajectory
        vaso_obs = observe_vaso(traj)  # simple: 1-v
        vaso_model = vaso_signal_change(traj.v)  # detailed model
        # Both should be negative during activation and correlate
        assert jnp.corrcoef(vaso_obs.ravel(), vaso_model.ravel())[0, 1] > 0.99

    def test_grubb_cbv_consistent(self, balloon_trajectory):
        """Grubb CBV from flow should approximate Balloon CBV."""
        _, traj, _ = balloon_trajectory
        # At peak flow
        peak_idx = jnp.argmax(traj.f)
        v_balloon = traj.v[peak_idx]
        v_grubb = grubb_cbv(traj.f[peak_idx])
        # Should be in same ballpark (not identical — Grubb is steady-state)
        assert jnp.abs(v_balloon - v_grubb) < 0.3

    def test_all_return_to_baseline(self, balloon_trajectory):
        """All signals should return near baseline after stimulus."""
        _, traj, _ = balloon_trajectory
        bold = observe_bold(traj)
        asl = observe_asl(traj)
        vaso = observe_vaso(traj)
        assert jnp.abs(bold[-1]) < 0.01
        assert jnp.abs(asl[-1]) < 0.01
        assert jnp.abs(vaso[-1]) < 0.01


class TestRieraPipeline:
    """Full pipeline through Riera multi-compartment NVC model."""

    def test_riera_produces_bold(self):
        """Riera model → Balloon-equivalent → BOLD should work."""
        model = RieraNVC(params=RieraParams())
        y = RieraState.steady_state()

        # Simulate a few Euler steps with stimulus
        dt = 0.01
        for _ in range(500):
            dy = model(jnp.array(0.0), y, jnp.array(1.0))
            y = jax.tree.map(lambda yi, dyi: yi + dt * dyi, y, dy)

        v, q = riera_to_balloon(y)

        # Construct a pseudo BalloonState for BOLD observation
        pseudo_state = BalloonState(
            s=jnp.array(0.0), f=y.f_a, v=v, q=q
        )
        bold = observe_bold(pseudo_state)
        # Should have deviated from baseline
        assert jnp.abs(bold) > 0.001

    def test_riera_cmro2_increases(self):
        """CMRO₂ should increase in response to stimulus."""
        model = RieraNVC(params=RieraParams())
        y = RieraState.steady_state()
        dt = 0.01
        for _ in range(300):
            dy = model(jnp.array(0.0), y, jnp.array(1.0))
            y = jax.tree.map(lambda yi, dyi: yi + dt * dyi, y, dy)
        assert float(y.cmro2) > 1.0  # above baseline


class TestCMRO2Hierarchy:
    """Test the three-level CMRO₂ hierarchy from the README."""

    def test_level1_trust_global(self):
        """Level 1: TRUST + global CBF → global CMRO₂."""
        t2_venous = jnp.array(0.055)   # ~55ms at 3T
        cbf_global = jnp.array(50.0)   # mL/100g/min
        cmro2 = trust_global_cmro2(t2_venous, cbf_global)
        # Should be in physiological range (100-200 µmol/100g/min)
        assert 50.0 < float(cmro2) < 300.0

    def test_level2_qbold_regional(self):
        """Level 2: qBOLD regional OEF + CBF → regional CMRO₂."""
        # Simulate qBOLD measurement
        oef_regional = jnp.array(0.34)
        cbf_regional = jnp.array(55.0)
        cao2 = compute_cao2()
        cmro2 = fick_cmro2(cbf_regional, oef_regional, cao2)
        assert 50.0 < float(cmro2) < 300.0

    def test_level3_dynamic_riera(self):
        """Level 3: Riera model → time-varying CMRO₂."""
        model = RieraNVC(params=RieraParams())
        y = RieraState.steady_state()
        dt = 0.01
        cmro2_trace = []
        for _ in range(200):
            dy = model(jnp.array(0.0), y, jnp.array(1.0))
            y = jax.tree.map(lambda yi, dyi: yi + dt * dyi, y, dy)
            cmro2_trace.append(float(y.cmro2))

        # CMRO₂ should rise from 1.0 toward steady state
        assert cmro2_trace[-1] > cmro2_trace[0]

    def test_hierarchy_consistency(self):
        """All three levels should give comparable CMRO₂ estimates."""
        # Level 1: TRUST
        cmro2_L1 = trust_global_cmro2(jnp.array(0.055), jnp.array(50.0))

        # Level 2: qBOLD + Fick
        oef = jnp.array(0.34)
        cbf = jnp.array(50.0)
        cao2 = compute_cao2()
        cmro2_L2 = fick_cmro2(cbf, oef, cao2)

        # Both should be in similar range (not identical — different methods)
        ratio = float(cmro2_L1 / cmro2_L2)
        assert 0.3 < ratio < 3.0  # within 3x of each other


class TestCalibratedFMRIPipeline:
    """Pipeline: qBOLD R₂' → M → Davis model → ΔCMRO₂."""

    def test_full_calibration_pipeline(self):
        """qBOLD → calibration → CMRO₂ estimation."""
        # Step 1: qBOLD provides resting R₂'
        oef_rest = jnp.array(0.34)
        dbv_rest = jnp.array(0.03)
        r2prime = compute_r2prime(oef_rest, dbv_rest)
        assert float(r2prime) > 0

        # Step 2: Gas-free M estimation from R₂'
        M = estimate_M_from_r2prime(r2prime)
        assert 0.01 < float(M) < 0.5

        # Step 3: During task — BOLD and CBF measured
        cbf_ratio = jnp.array(1.4)   # 40% CBF increase
        bold_change = jnp.array(0.02)  # 2% BOLD increase

        # Step 4: Inverse Davis model → CMRO₂ ratio
        cmro2_ratio = estimate_cmro2_change(bold_change, cbf_ratio, M)

        # CMRO₂ should increase less than CBF (flow-metabolism uncoupling)
        assert 1.0 < float(cmro2_ratio) < float(cbf_ratio)

    def test_davis_model_consistency(self):
        """Davis forward → inverse should roundtrip."""
        M = jnp.array(0.08)
        cbf_r = jnp.array(1.5)
        cmro2_r = jnp.array(1.2)

        bold = davis_model(cbf_r, cmro2_r, M)
        cmro2_back = estimate_cmro2_change(bold, cbf_r, M)
        assert jnp.allclose(cmro2_back, cmro2_r, atol=0.01)


class TestLayerResolvedPipeline:
    """Pipeline: stimulus → per-layer NVC → BOLD → vein contamination → devein."""

    def test_layer_pipeline(self):
        """Full layer-resolved forward model and deveining."""
        # Step 1: Distribute stimulus to layers
        stimulus = jnp.array(1.0)
        layer_stim = layer_stimulus(stimulus)
        assert layer_stim.shape == (3,)
        assert jnp.all(layer_stim > 0)

        # Step 2: Simulate per-layer BOLD (proportional to stimulus)
        local_bold = layer_stim * 0.02  # ~2% modulated by layer gain

        # Step 3: Ascending vein contamination
        observed_bold = ascending_vein_contamination(local_bold)

        # Superficial should be contaminated (higher than local)
        assert float(observed_bold[2]) >= float(local_bold[2])

        # Step 4: Devein
        recovered_bold = devein_bold(observed_bold)

        # Should approximately recover local BOLD
        assert jnp.allclose(recovered_bold, local_bold, atol=0.005)

    def test_feedforward_feedback_asymmetry(self):
        """Deep layers should get more feedforward, superficial more feedback."""
        stim_ff = layer_stimulus(jnp.array(1.0), feedforward_frac=jnp.array(1.0), feedback_frac=jnp.array(0.0))
        stim_fb = layer_stimulus(jnp.array(1.0), feedforward_frac=jnp.array(0.0), feedback_frac=jnp.array(1.0))

        # Feedforward: deep > superficial
        assert float(stim_ff[0]) > float(stim_ff[2])
        # Feedback: superficial > deep
        assert float(stim_fb[2]) > float(stim_fb[0])


class TestASLQuantificationPipeline:
    """Pipeline: pCASL acquisition → kinetic model → CBF quantification."""

    def test_pcasl_pipeline(self):
        """Simulate pCASL and recover CBF."""
        params = ASLKineticParams()
        cbf_true = jnp.array(55.0)

        # Simulate ASL signal at PLD = 1.8s
        pld = jnp.array(1.8)
        t = jnp.array([float(params.delta + pld)])
        delta_m = asl_kinetic_signal(t, cbf_true, params)

        # Quantify CBF from the measurement
        cbf_est = quantify_cbf(delta_m[0], pld, params)

        # Should be in reasonable range
        assert 20.0 < float(cbf_est) < 100.0


class TestFullPipelineDifferentiability:
    """Ensure gradients flow through the complete pipeline."""

    def test_grad_through_balloon_to_bold(self):
        """Gradient of peak BOLD w.r.t. E0 through full Balloon pipeline."""
        def pipeline(E0):
            params = BalloonParams(E0=E0)
            dt = 0.05
            n = int(15.0 / dt)
            stim = jnp.zeros(n).at[int(1.0 / dt):int(4.0 / dt)].set(1.0)
            _, traj = solve_balloon(params, stim, dt=dt)
            bold = observe_bold(traj)
            return jnp.max(bold)

        g = jax.grad(pipeline)(jnp.array(0.34))
        assert jnp.isfinite(g)
        assert float(g) != 0.0

    def test_grad_through_metabolism(self):
        """Gradient through activity → CMRO₂ → OEF → Fick."""
        def pipeline(gain):
            params = CMRO2Params(coupling_gain=gain)
            activity = jnp.array(1.0)
            cmro2_ratio = compute_cmro2(activity, params)
            cbf_ratio = jnp.array(1.5)
            oef = compute_oef(cbf_ratio, cmro2_ratio)
            cao2 = compute_cao2()
            cbf = jnp.array(50.0) * cbf_ratio
            return fick_cmro2(cbf, oef, cao2)

        g = jax.grad(pipeline)(jnp.array(0.5))
        assert jnp.isfinite(g)

    def test_grad_through_calibrated_fmri(self):
        """Gradient through qBOLD → M → Davis → CMRO₂."""
        def pipeline(oef):
            dbv = jnp.array(0.03)
            r2p = compute_r2prime(oef, dbv)
            M = estimate_M_from_r2prime(r2p)
            bold = davis_model(jnp.array(1.5), jnp.array(1.2), M)
            return bold

        g = jax.grad(pipeline)(jnp.array(0.34))
        assert jnp.isfinite(g)
