"""Tests for the layers package."""

import jax
import jax.numpy as jnp
import pytest

from vpjax.layers.layering import equivolume_depths, assign_layers, layer_mask
from vpjax.layers.profiles import sample_profile, normalize_profile
from vpjax.layers.iron_myelin import IronMyelinParams, decompose_r2star_qsm, r2star_from_iron_myelin, chi_from_iron_myelin
from vpjax.layers.layer_nvc import LayerNVCParams, layer_stimulus, ascending_vein_contamination, devein_bold


class TestLayering:
    def test_equidistant_layers(self):
        """Without curvature, layers should be equidistant."""
        bounds = equivolume_depths(3)
        expected = jnp.array([0.0, 1/3, 2/3, 1.0])
        assert jnp.allclose(bounds, expected, atol=1e-4)

    def test_assign_layers_basic(self):
        """Assign depths to correct layers."""
        depth = jnp.array([0.1, 0.4, 0.7, 0.95])
        layers = assign_layers(depth, 3)
        assert int(layers[0]) == 0  # deep
        assert int(layers[1]) == 1  # middle
        assert int(layers[2]) == 2  # superficial
        assert int(layers[3]) == 2  # superficial

    def test_layer_count(self):
        """Equivolume depths should have n_layers+1 boundaries."""
        bounds = equivolume_depths(5)
        assert bounds.shape == (6,)


class TestProfiles:
    def test_sample_profile_shape(self):
        """Profile should have correct number of depth bins."""
        volume = jnp.ones(100)
        depth = jnp.linspace(0, 1, 100)
        centers, profile = sample_profile(volume, depth, n_depths=10)
        assert centers.shape == (10,)
        assert profile.shape == (10,)

    def test_constant_volume(self):
        """Constant input should give constant profile."""
        volume = jnp.ones(100) * 5.0
        depth = jnp.linspace(0, 1, 100)
        _, profile = sample_profile(volume, depth, n_depths=10)
        assert jnp.allclose(profile, 5.0, atol=0.5)

    def test_normalize_minmax(self):
        profile = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = normalize_profile(profile, "minmax")
        assert jnp.allclose(normed[0], 0.0, atol=1e-6)
        assert jnp.allclose(normed[-1], 1.0, atol=1e-6)


class TestIronMyelin:
    def test_decompose_roundtrip(self):
        """iron/myelin → R2*/χ → iron/myelin roundtrip."""
        params = IronMyelinParams()
        iron = jnp.array(0.5)
        myelin = jnp.array(0.3)
        r2star = r2star_from_iron_myelin(iron, myelin, params)
        chi = chi_from_iron_myelin(iron, myelin, params)
        iron_back, myelin_back = decompose_r2star_qsm(r2star, chi, params)
        assert jnp.allclose(iron_back, iron, atol=0.05)
        assert jnp.allclose(myelin_back, myelin, atol=0.05)

    def test_iron_positive_chi(self):
        """Iron should produce positive susceptibility."""
        params = IronMyelinParams()
        chi = chi_from_iron_myelin(jnp.array(1.0), jnp.array(0.0), params)
        assert float(chi) > float(params.chi_tissue)

    def test_myelin_negative_chi(self):
        """Myelin should produce negative (diamagnetic) susceptibility."""
        params = IronMyelinParams()
        chi = chi_from_iron_myelin(jnp.array(0.0), jnp.array(1.0), params)
        assert float(chi) < float(params.chi_tissue)


class TestLayerNVC:
    def test_layer_stimulus_shape(self):
        """Layer stimulus should have L columns."""
        stim = layer_stimulus(jnp.array(1.0))
        assert stim.shape == (3,)

    def test_deep_stronger_feedforward(self):
        """Deep layers should receive more feedforward drive."""
        stim = layer_stimulus(jnp.array(1.0))
        # Default: deep layer (0) has higher coupling gain
        assert float(stim[0]) >= float(stim[2])

    def test_ascending_vein_contaminates(self):
        """Ascending veins should contaminate superficial layers."""
        local = jnp.array([1.0, 0.0, 0.0])  # signal only in deep layer
        observed = ascending_vein_contamination(local)
        # Superficial layers should have some contamination
        assert float(observed[2]) > 0.0

    def test_devein_recovers(self):
        """Deveining should approximately recover local BOLD."""
        local = jnp.array([1.0, 0.5, 0.2])
        contaminated = ascending_vein_contamination(local)
        recovered = devein_bold(contaminated)
        assert jnp.allclose(recovered, local, atol=0.1)
