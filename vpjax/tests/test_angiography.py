"""Tests for vascular/angiography.py — TOF → vpjax model parameters.

Tests work WITHOUT vmtk (using numpy/scipy fallbacks).
"""

import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# VesselTree data structure
# ---------------------------------------------------------------------------

class TestVesselTree:

    def test_from_arrays(self):
        from vpjax.vascular.angiography import VesselTree
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        radii = np.array([5.0, 4.0, 3.0], dtype=np.float32)
        branch_ids = np.array([0, 0, 0], dtype=np.int32)
        tree = VesselTree(points=points, radii=radii, branch_ids=branch_ids)
        assert tree.n_points == 3
        assert tree.n_branches == 1

    def test_branch_lengths(self):
        from vpjax.vascular.angiography import VesselTree
        points = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0],
        ], dtype=np.float32)
        radii = np.ones(5, dtype=np.float32)
        branch_ids = np.zeros(5, dtype=np.int32)
        tree = VesselTree(points=points, radii=radii, branch_ids=branch_ids)
        assert abs(tree.branch_lengths()[0] - 4.0) < 0.01

    def test_branch_mean_radii(self):
        from vpjax.vascular.angiography import VesselTree
        points = np.zeros((4, 3), dtype=np.float32)
        radii = np.array([10.0, 8.0, 4.0, 2.0], dtype=np.float32)
        branch_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        tree = VesselTree(points=points, radii=radii, branch_ids=branch_ids)
        mr = tree.branch_mean_radii()
        assert abs(mr[0] - 9.0) < 0.01
        assert abs(mr[1] - 3.0) < 0.01


# ---------------------------------------------------------------------------
# TOF → VascularParams
# ---------------------------------------------------------------------------

class TestToVascularParams:
    """VesselTree.to_vascular_params() should produce region-specific
    VascularParams from measured vessel morphometry."""

    def test_returns_vascular_params(self):
        from vpjax.vascular.angiography import VesselTree
        from vpjax.vascular.geometry import VascularParams
        points = np.array([
            [0, 0, 0], [100, 0, 0],
            [0, 0, 0], [0, 80, 0],
        ], dtype=np.float32)
        radii = np.array([15.0, 15.0, 3.0, 3.0], dtype=np.float32)
        branch_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        tree = VesselTree(points=points, radii=radii, branch_ids=branch_ids)
        params = tree.to_vascular_params(tissue_volume_mm3=1.0)
        assert isinstance(params, VascularParams)
        assert float(params.radius_art) > 0

    def test_measured_vs_defaults_differ(self):
        """Subject-specific params should differ from literature defaults."""
        from vpjax.vascular.angiography import VesselTree
        from vpjax.vascular.geometry import VascularParams
        points = np.array([
            [0, 0, 0], [300, 0, 0],  # long artery
            [0, 0, 0], [0, 50, 0],   # short capillary
        ], dtype=np.float32)
        radii = np.array([20.0, 20.0, 2.0, 2.0], dtype=np.float32)
        branch_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        tree = VesselTree(points=points, radii=radii, branch_ids=branch_ids)
        measured = tree.to_vascular_params()
        defaults = VascularParams()
        # Measured artery radius (20) should differ from default (15)
        assert float(measured.radius_art) != float(defaults.radius_art)


# ---------------------------------------------------------------------------
# TOF → Balloon/Riera transit times
# ---------------------------------------------------------------------------

class TestTransitTimes:
    """Extract subject-specific transit times from vessel morphometry."""

    def test_balloon_params_from_tree(self):
        """Should produce BalloonParams with subject-specific tau."""
        from vpjax.vascular.angiography import VesselTree, balloon_params_from_tree
        # Arteries: 200µm long, veins: 300µm long
        points = np.array([
            [0, 0, 0], [200, 0, 0],   # artery
            [0, 0, 0], [0, 300, 0],    # vein
        ], dtype=np.float32)
        radii = np.array([15.0, 15.0, 25.0, 25.0], dtype=np.float32)
        branch_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        tree = VesselTree(points=points, radii=radii, branch_ids=branch_ids)
        params = balloon_params_from_tree(tree)
        from vpjax._types import BalloonParams
        assert isinstance(params, BalloonParams)
        assert float(params.tau) > 0
        assert jnp.isfinite(params.tau)

    def test_riera_params_from_tree(self):
        """Should produce RieraParams with per-compartment transit times."""
        from vpjax.vascular.angiography import VesselTree, riera_params_from_tree
        points = np.array([
            [0, 0, 0], [150, 0, 0],   # artery
            [0, 0, 0], [0, 80, 0],    # capillary
            [0, 0, 0], [0, 0, 250],   # vein
        ], dtype=np.float32)
        radii = np.array([15.0, 15.0, 3.0, 3.0, 25.0, 25.0], dtype=np.float32)
        branch_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        tree = VesselTree(points=points, radii=radii, branch_ids=branch_ids)
        params = riera_params_from_tree(tree)
        from vpjax.hemodynamics.riera import RieraParams
        assert isinstance(params, RieraParams)
        # Each compartment should have distinct transit time
        assert float(params.tau_a) > 0
        assert float(params.tau_c) > 0
        assert float(params.tau_v) > 0


# ---------------------------------------------------------------------------
# TOF → qBOLD cylinder geometry
# ---------------------------------------------------------------------------

class TestQBOLDGeometry:
    """Vessel geometry for the qBOLD static dephasing model."""

    def test_dbv_from_tree(self):
        """Should estimate DBV (deoxygenated blood volume) from venous morphometry."""
        from vpjax.vascular.angiography import VesselTree, estimate_dbv_from_tree
        points = np.array([
            [0, 0, 0], [100, 0, 0],   # artery (not deoxy)
            [0, 0, 0], [0, 80, 0],    # vein (deoxy)
            [0, 0, 0], [0, 0, 60],    # vein (deoxy)
        ], dtype=np.float32)
        radii = np.array([15.0, 15.0, 20.0, 20.0, 18.0, 18.0], dtype=np.float32)
        branch_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        tree = VesselTree(points=points, radii=radii, branch_ids=branch_ids)
        dbv = estimate_dbv_from_tree(tree, tissue_volume_mm3=1.0)
        assert float(dbv) > 0
        assert float(dbv) < 0.5  # physically reasonable


# ---------------------------------------------------------------------------
# TOF → regional CBV₀ baseline prior
# ---------------------------------------------------------------------------

class TestCBVPrior:
    """Morphometric CBV₀ from vessel geometry as Balloon model prior."""

    def test_cbv0_from_tree(self):
        from vpjax.vascular.angiography import VesselTree, estimate_cbv0_from_tree
        points = np.array([
            [0, 0, 0], [200, 0, 0],
            [0, 0, 0], [0, 100, 0],
            [0, 0, 0], [0, 0, 150],
        ], dtype=np.float32)
        radii = np.array([12.0, 12.0, 3.0, 3.0, 20.0, 20.0], dtype=np.float32)
        branch_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        tree = VesselTree(points=points, radii=radii, branch_ids=branch_ids)
        cbv0 = estimate_cbv0_from_tree(tree, tissue_volume_mm3=1.0)
        assert float(cbv0) > 0
        assert float(cbv0) < 0.3  # physically reasonable

    def test_cbv0_as_vaso_prior(self):
        """CBV₀ from morphometry should be usable as VASOParams.CBV0."""
        from vpjax.vascular.angiography import VesselTree, estimate_cbv0_from_tree
        from vpjax.vaso.signal_model import VASOParams
        points = np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float32)
        radii = np.array([10.0, 10.0], dtype=np.float32)
        branch_ids = np.array([0, 0], dtype=np.int32)
        tree = VesselTree(points=points, radii=radii, branch_ids=branch_ids)
        cbv0 = estimate_cbv0_from_tree(tree, tissue_volume_mm3=1.0)
        vaso_params = VASOParams(CBV0=jnp.array(float(cbv0)))
        assert float(vaso_params.CBV0) > 0


# ---------------------------------------------------------------------------
# Skeletonization & radius estimation (numpy/scipy fallback)
# ---------------------------------------------------------------------------

class TestSkeletonize:

    def test_skeletonize_tube(self):
        from vpjax.vascular.angiography import skeletonize_segmentation
        vol = np.zeros((40, 40, 80), dtype=np.uint8)
        cy, cz, radius = 20, 20, 5
        for x in range(5, 75):
            for y in range(cy - radius, cy + radius + 1):
                for z in range(cz - radius, cz + radius + 1):
                    if (y - cy)**2 + (z - cz)**2 <= radius**2:
                        vol[y, z, x] = 1
        points = skeletonize_segmentation(vol)
        assert points.shape[1] == 3
        assert points.shape[0] > 5


class TestRadiusEstimation:

    def test_radius_from_distance_transform(self):
        from vpjax.vascular.angiography import estimate_radii
        vol = np.zeros((30, 30, 40), dtype=np.uint8)
        cy, cz, radius = 15, 15, 3
        for x in range(5, 35):
            for y in range(cy - radius, cy + radius + 1):
                for z in range(cz - radius, cz + radius + 1):
                    if (y - cy)**2 + (z - cz)**2 <= radius**2:
                        vol[y, z, x] = 1
        centerline = np.array([[cy, cz, x] for x in range(10, 30)], dtype=np.float32)
        radii = estimate_radii(vol, centerline)
        assert radii.shape == (20,)
        assert abs(np.mean(radii) - 3.0) < 1.0
