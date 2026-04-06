"""Tests for multi-compartment glymphatic model (SIMULA-inspired).

Extends the simple CSF coupling with arterial PVS, ECS, and venous PVS
compartments with solute transport (diffusion + advection).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


class TestMultiCompartmentGlymphatic:
    """Multi-compartment solute transport through perivascular spaces."""

    def test_compartment_params(self):
        from vpjax.sleep.glymphatic import GlymphaticParams
        p = GlymphaticParams()
        assert float(p.kappa_pvs_art) > 0  # arterial PVS permeability
        assert float(p.kappa_ecs) > 0       # ECS permeability
        assert float(p.D_eff) > 0            # effective diffusion

    def test_state_initial(self):
        """Initial solute concentration should be settable."""
        from vpjax.sleep.glymphatic import GlymphaticState
        state = GlymphaticState.with_solute(
            c_pvs_art=1.0, c_ecs=0.5, c_pvs_ven=0.0
        )
        assert float(state.c_pvs_art) == 1.0
        assert float(state.c_ecs) == 0.5
        assert float(state.c_pvs_ven) == 0.0

    def test_solute_flows_art_to_ecs(self):
        """Solute should flow from arterial PVS → ECS."""
        from vpjax.sleep.glymphatic import GlymphaticState, GlymphaticODE, GlymphaticParams
        model = GlymphaticODE(params=GlymphaticParams())
        state = GlymphaticState.with_solute(c_pvs_art=1.0, c_ecs=0.0, c_pvs_ven=0.0)
        dy = model(state, vasomotion_drive=0.0)
        # ECS concentration should increase
        assert float(dy.c_ecs) > 0
        # Arterial PVS should decrease
        assert float(dy.c_pvs_art) < 0

    def test_solute_flows_ecs_to_ven(self):
        """Solute should flow from ECS → venous PVS for clearance."""
        from vpjax.sleep.glymphatic import GlymphaticState, GlymphaticODE, GlymphaticParams
        model = GlymphaticODE(params=GlymphaticParams())
        state = GlymphaticState.with_solute(c_pvs_art=0.0, c_ecs=1.0, c_pvs_ven=0.0)
        dy = model(state, vasomotion_drive=0.0)
        assert float(dy.c_pvs_ven) > 0

    def test_vasomotion_enhances_clearance(self):
        """Vasomotion drive should increase solute transport."""
        from vpjax.sleep.glymphatic import GlymphaticState, GlymphaticODE, GlymphaticParams
        model = GlymphaticODE(params=GlymphaticParams())
        state = GlymphaticState.with_solute(c_pvs_art=0.0, c_ecs=1.0, c_pvs_ven=0.0)
        dy_no_vaso = model(state, vasomotion_drive=0.0)
        dy_vaso = model(state, vasomotion_drive=1.0)
        # Venous PVS clearance should be faster with vasomotion
        assert float(dy_vaso.c_pvs_ven) > float(dy_no_vaso.c_pvs_ven)

    def test_clearance_simulation(self):
        """Full simulation: solute in ECS should clear over time."""
        from vpjax.sleep.glymphatic import simulate_clearance, GlymphaticParams
        params = GlymphaticParams()
        t = jnp.linspace(0, 3600, 360)  # 1 hour, 10s steps
        # Vasomotion during NREM
        vaso_drive = 0.5 * jnp.ones_like(t)
        result = simulate_clearance(
            c_ecs_init=1.0, t=t, vasomotion_drive=vaso_drive, params=params
        )
        # ECS concentration should decrease
        assert float(result.c_ecs[-1]) < float(result.c_ecs[0])
        # Venous PVS should have accumulated solute
        assert float(result.c_pvs_ven[-1]) > 0

    def test_no_vasomotion_slower_clearance(self):
        """Without vasomotion, clearance should be slower."""
        from vpjax.sleep.glymphatic import simulate_clearance, GlymphaticParams
        params = GlymphaticParams()
        t = jnp.linspace(0, 3600, 360)
        result_vaso = simulate_clearance(1.0, t, 0.5 * jnp.ones_like(t), params)
        result_none = simulate_clearance(1.0, t, jnp.zeros_like(t), params)
        # With vasomotion: more clearance (lower ECS at end)
        assert float(result_vaso.c_ecs[-1]) < float(result_none.c_ecs[-1])

    def test_differentiable(self):
        from vpjax.sleep.glymphatic import GlymphaticState, GlymphaticODE, GlymphaticParams
        model = GlymphaticODE(params=GlymphaticParams())
        def loss(kappa):
            p = GlymphaticParams(kappa_pvs_art=kappa)
            m = GlymphaticODE(params=p)
            state = GlymphaticState.with_solute(c_pvs_art=0.0, c_ecs=1.0, c_pvs_ven=0.0)
            dy = m(state, vasomotion_drive=0.5)
            return dy.c_pvs_ven
        g = jax.grad(loss)(jnp.array(1e-4))
        assert jnp.isfinite(g)
